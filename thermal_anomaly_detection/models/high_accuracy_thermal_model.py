"""
High-Accuracy Thermal Anomaly Detection Model
============================================

A state-of-the-art deep learning model for thermal anomaly detection using
Transformer-based architecture with temporal fusion capabilities.

Features:
- Swin Transformer backbone for high accuracy
- PatchCore anomaly detection head
- ConvLSTM for temporal fusion (video sequences)
- Mixed precision training for A100 optimization
- Real-time inference capabilities

Author: Thermal Anomaly Detection System
Date: 2025-10-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """Patch embedding layer for thermal images"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 1, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias"""
    
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get pair-wise relative position bias
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block for thermal feature extraction"""
    
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class PatchCoreAnomalyHead(nn.Module):
    """PatchCore-inspired anomaly detection head for pixel-level scoring"""
    
    def __init__(self, input_dim: int, memory_bank_size: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.memory_bank_size = memory_bank_size
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
        )
        
        # Memory bank for normal features
        self.register_buffer('memory_bank', torch.randn(memory_bank_size, input_dim // 4))
        self.register_buffer('initialized', torch.tensor(False))
        
    def forward(self, features, training_phase=False):
        B, L, C = features.shape
        
        # Project features
        projected = self.projection(features)  # B, L, C//4
        
        if training_phase and not self.initialized:
            # Initialize memory bank with normal features during training
            with torch.no_grad():
                flat_features = projected.view(-1, projected.size(-1))
                if flat_features.size(0) >= self.memory_bank_size:
                    indices = torch.randperm(flat_features.size(0))[:self.memory_bank_size]
                    self.memory_bank.copy_(flat_features[indices])
                    self.initialized.fill_(True)
        
        # Compute anomaly scores using distance to memory bank
        anomaly_scores = []
        for i in range(B):
            feature_batch = projected[i]  # L, C//4
            distances = torch.cdist(feature_batch, self.memory_bank)  # L, memory_bank_size
            min_distances, _ = torch.min(distances, dim=1)  # L
            anomaly_scores.append(min_distances)
        
        anomaly_scores = torch.stack(anomaly_scores)  # B, L
        return anomaly_scores


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for temporal fusion"""
    
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class HighAccuracyThermalModel(nn.Module):
    """
    High-Accuracy Thermal Anomaly Detection Model
    
    Features:
    - Swin Transformer backbone for spatial feature extraction
    - PatchCore anomaly detection head for pixel-level scoring
    - ConvLSTM for temporal fusion (video sequences)
    - Mixed precision support for A100 optimization
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 1,
                 embed_dim: int = 96,
                 depths: List[int] = [2, 2, 6, 2],
                 num_heads: List[int] = [3, 6, 12, 24],
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_path_rate: float = 0.1,
                 use_temporal: bool = False,
                 temporal_length: int = 8):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_temporal = use_temporal
        self.temporal_length = temporal_length
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = self._make_layer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            )
            self.layers.append(layer)
            
            # Patch merging
            if i_layer < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.LayerNorm(int(embed_dim * 2 ** i_layer)),
                    nn.Linear(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** (i_layer + 1)), bias=False)
                )
                self.layers.append(downsample)
        
        # Final normalization
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (len(depths) - 1)))
        
        # Anomaly detection head
        self.anomaly_head = PatchCoreAnomalyHead(
            input_dim=int(embed_dim * 2 ** (len(depths) - 1)),
            memory_bank_size=1000
        )
        
        # Temporal fusion (ConvLSTM)
        if use_temporal:
            self.temporal_fusion = ConvLSTMCell(
                input_dim=1,
                hidden_dim=32,
                kernel_size=3
            )
        
        # Output projection for segmentation
        self.output_proj = nn.Sequential(
            nn.Conv2d(int(embed_dim * 2 ** (len(depths) - 1)), 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _make_layer(self, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, drop_path):
        blocks = []
        for i in range(depth):
            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path
                )
            )
        return nn.Sequential(*blocks)
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    @autocast()
    def forward(self, x, temporal_state=None, training_phase=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W) for temporal
            temporal_state: Previous hidden state for temporal fusion
            training_phase: Whether in training phase for memory bank initialization
        
        Returns:
            Dict containing anomaly scores and segmentation maps
        """
        if self.use_temporal and x.dim() == 5:
            # Handle temporal input (B, T, C, H, W)
            B, T, C, H, W = x.shape
            outputs = []
            
            for t in range(T):
                frame = x[:, t]  # B, C, H, W
                frame_output = self._forward_single(frame, training_phase)
                outputs.append(frame_output)
            
            # Aggregate temporal outputs
            anomaly_scores = torch.stack([out['anomaly_scores'] for out in outputs], dim=1)  # B, T, L
            segmentation_maps = torch.stack([out['segmentation_map'] for out in outputs], dim=1)  # B, T, H, W
            
            # Apply temporal fusion
            temporal_fused = self._apply_temporal_fusion(segmentation_maps, temporal_state)
            
            return {
                'anomaly_scores': anomaly_scores.mean(dim=1),  # Average over time
                'segmentation_map': temporal_fused,
                'temporal_outputs': outputs
            }
        else:
            # Handle single frame input
            return self._forward_single(x, training_phase)
    
    def _forward_single(self, x, training_phase=False):
        """Forward pass for single frame"""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, L, C
        x = x + self.absolute_pos_embed
        
        # Store original spatial dimensions
        H_patches, W_patches = self.patches_resolution
        
        # Apply Swin Transformer layers
        for layer in self.layers:
            if isinstance(layer, nn.Sequential):  # Swin blocks
                for block in layer:
                    x = block(x, H_patches, W_patches)
            else:  # Downsampling
                x = layer(x)
                H_patches, W_patches = H_patches // 2, W_patches // 2
        
        # Final normalization
        x = self.norm(x)  # B, L, C
        
        # Anomaly detection
        anomaly_scores = self.anomaly_head(x, training_phase)  # B, L
        
        # Reshape for segmentation output
        feature_map = x.transpose(1, 2).view(B, -1, H_patches, W_patches)
        
        # Generate segmentation map
        segmentation_map = self.output_proj(feature_map)  # B, 1, H', W'
        segmentation_map = F.interpolate(segmentation_map, size=(H, W), mode='bilinear', align_corners=False)
        segmentation_map = segmentation_map.squeeze(1)  # B, H, W
        
        return {
            'anomaly_scores': anomaly_scores,
            'segmentation_map': segmentation_map,
            'features': x
        }
    
    def _apply_temporal_fusion(self, segmentation_maps, temporal_state):
        """Apply ConvLSTM for temporal fusion"""
        B, T, H, W = segmentation_maps.shape
        
        if temporal_state is None:
            h_t = torch.zeros(B, 32, H, W, device=segmentation_maps.device)
            c_t = torch.zeros(B, 32, H, W, device=segmentation_maps.device)
        else:
            h_t, c_t = temporal_state
        
        outputs = []
        for t in range(T):
            input_t = segmentation_maps[:, t:t+1]  # B, 1, H, W
            h_t, c_t = self.temporal_fusion(input_t, (h_t, c_t))
            outputs.append(h_t)
        
        # Final output projection
        temporal_fused = torch.stack(outputs, dim=1).mean(dim=1)  # B, 32, H, W
        temporal_fused = F.adaptive_avg_pool2d(temporal_fused, (H, W))
        temporal_fused = torch.mean(temporal_fused, dim=1)  # B, H, W
        
        return temporal_fused


def create_model(config: Dict) -> HighAccuracyThermalModel:
    """
    Create high-accuracy thermal anomaly detection model
    
    Args:
        config: Configuration dictionary
    
    Returns:
        HighAccuracyThermalModel instance
    """
    model = HighAccuracyThermalModel(
        img_size=config.get('img_size', 224),
        patch_size=config.get('patch_size', 4),
        in_chans=config.get('in_chans', 1),
        embed_dim=config.get('embed_dim', 96),
        depths=config.get('depths', [2, 2, 6, 2]),
        num_heads=config.get('num_heads', [3, 6, 12, 24]),
        window_size=config.get('window_size', 7),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        qkv_bias=config.get('qkv_bias', True),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_temporal=config.get('use_temporal', False),
        temporal_length=config.get('temporal_length', 8)
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    config = {
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 1,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'use_temporal': False
    }
    
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    with torch.no_grad():
        output = model(x, training_phase=True)
        print(f"Output shapes:")
        print(f"  Anomaly scores: {output['anomaly_scores'].shape}")
        print(f"  Segmentation map: {output['segmentation_map'].shape}")