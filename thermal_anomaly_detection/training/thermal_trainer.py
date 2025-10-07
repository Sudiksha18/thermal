"""
High-Accuracy Thermal Anomaly Detection Training Script
======================================================

Features:
- Mixed precision training for A100 optimization
- Advanced loss functions for anomaly detection
- Comprehensive metrics (F1, ROC-AUC, PR-AUC)
- Model checkpointing and early stopping
- TensorBoard logging
- GPU memory optimization

Author: Thermal Anomaly Detection System
Date: 2025-10-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import os
import time
import hashlib
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.high_accuracy_thermal_model import create_model, HighAccuracyThermalModel
from data.thermal_dataset import create_thermal_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in anomaly detection"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined loss function for thermal anomaly detection"""
    
    def __init__(self, 
                 focal_weight: float = 0.5,
                 dice_weight: float = 0.3,
                 consistency_weight: float = 0.2,
                 alpha: float = 1.0,
                 gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        segmentation_map = outputs['segmentation_map']
        anomaly_scores = outputs['anomaly_scores']
        
        # Focal loss on segmentation
        focal = self.focal_loss(segmentation_map, targets)
        
        # Dice loss on segmentation
        dice = self.dice_loss(segmentation_map, targets)
        
        # Consistency loss between anomaly scores and segmentation
        # Aggregate anomaly scores to match segmentation resolution
        B, L = anomaly_scores.shape
        H, W = segmentation_map.shape[-2:]
        
        # Reshape anomaly scores to spatial dimensions
        patch_size = int(np.sqrt(L))
        if patch_size * patch_size == L:
            anomaly_spatial = anomaly_scores.view(B, patch_size, patch_size)
            anomaly_spatial = F.interpolate(
                anomaly_spatial.unsqueeze(1), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            
            consistency = self.mse_loss(torch.sigmoid(segmentation_map), anomaly_spatial)
        else:
            consistency = torch.tensor(0.0, device=segmentation_map.device)
        
        # Combined loss
        total_loss = (self.focal_weight * focal + 
                     self.dice_weight * dice + 
                     self.consistency_weight * consistency)
        
        return {
            'total_loss': total_loss,
            'focal_loss': focal,
            'dice_loss': dice,
            'consistency_loss': consistency
        }


class MetricsCalculator:
    """Calculate comprehensive metrics for thermal anomaly detection"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.segmentations = []
    
    def update(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor):
        """Update metrics with batch results"""
        segmentation_map = torch.sigmoid(outputs['segmentation_map']).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        self.predictions.extend(segmentation_map.flatten())
        self.targets.extend(targets_np.flatten())
        self.segmentations.append(segmentation_map)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(predictions) == 0:
            return {
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }
        
        # Binary predictions for F1 score
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        try:
            f1 = f1_score(targets, binary_preds, average='binary', zero_division=0)
            roc_auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
            pr_auc = average_precision_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
            
            # Additional metrics
            tp = np.sum((binary_preds == 1) & (targets == 1))
            fp = np.sum((binary_preds == 1) & (targets == 0))
            fn = np.sum((binary_preds == 0) & (targets == 1))
            tn = np.sum((binary_preds == 0) & (targets == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            return {
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0
            }
        
        return {
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }


class ThermalTrainer:
    """
    High-accuracy thermal anomaly detection trainer
    
    Features:
    - Mixed precision training
    - Advanced optimization
    - Comprehensive logging
    - Model checkpointing
    """
    
    def __init__(self,
                 model: HighAccuracyThermalModel,
                 train_loader,
                 val_loader,
                 config: Dict,
                 device: torch.device,
                 log_dir: str = './logs',
                 checkpoint_dir: str = './checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: Thermal anomaly detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Training device
            log_dir: Logging directory
            checkpoint_dir: Checkpoint directory
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(log_dir)
        
        # Setup training components
        self._setup_training()
        
        # Metrics
        self.train_metrics = MetricsCalculator(device)
        self.val_metrics = MetricsCalculator(device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        
    def _setup_training(self):
        """Setup training components"""
        # Loss function
        self.criterion = CombinedLoss(
            focal_weight=self.config.get('focal_weight', 0.5),
            dice_weight=self.config.get('dice_weight', 0.3),
            consistency_weight=self.config.get('consistency_weight', 0.2),
            alpha=self.config.get('focal_alpha', 1.0),
            gamma=self.config.get('focal_gamma', 2.0)
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('t_0', 10),
            T_mult=self.config.get('t_mult', 2),
            eta_min=self.config.get('eta_min', 1e-6)
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Gradient clipping
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            thermal = batch['thermal'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(thermal, training_phase=True)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict['total_loss']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs, labels)
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), 
                                     self.current_epoch * num_batches + batch_idx)
                
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        
        # Log epoch metrics
        self.writer.add_scalar('Train/EpochLoss', avg_loss, self.current_epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', metric_value, self.current_epoch)
        
        return {'loss': avg_loss, **metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                thermal = batch['thermal'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast():
                    outputs = self.model(thermal, training_phase=False)
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(outputs, labels)
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        
        # Log epoch metrics
        self.writer.add_scalar('Val/EpochLoss', avg_loss, self.current_epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', metric_value, self.current_epoch)
        
        return {'loss': avg_loss, **metrics}
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation score: {metrics.get('f1_score', 0.0):.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int):
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {train_metrics['roc_auc']:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_score']:.4f}, "
                  f"ROC-AUC: {val_metrics['roc_auc']:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}")
            
            # Check for best model
            current_score = val_metrics['f1_score']
            is_best = current_score > self.best_val_score
            
            if is_best:
                self.best_val_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 10):
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation F1 score: {self.best_val_score:.4f}")
        
        # Close writer
        self.writer.close()


def calculate_model_hash(model_path: str) -> str:
    """Calculate SHA-256 hash of model file"""
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    """Main training function"""
    # Configuration
    config = {
        # Model configuration
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 1,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'use_temporal': False,
        
        # Training configuration
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,
        
        # Loss configuration
        'focal_weight': 0.5,
        'dice_weight': 0.3,
        'consistency_weight': 0.2,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        
        # Scheduler configuration
        't_0': 10,
        't_mult': 2,
        'eta_min': 1e-6,
        
        # Other
        'patience': 15,
        'log_interval': 50,
        'num_workers': 4,
        
        # Data paths (to be configured based on your data)
        'train_paths': [],  # Add your training data paths
        'val_paths': [],    # Add your validation data paths
        'train_labels': [], # Add your training labels paths
        'val_labels': [],   # Add your validation labels paths
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    if config['train_paths']:  # Only if data paths are provided
        train_loader, val_loader, _ = create_thermal_dataloaders(config)
        print(f"Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        # Create trainer
        trainer = ThermalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            log_dir='./logs/thermal_training',
            checkpoint_dir='./checkpoints/thermal_model'
        )
        
        # Start training
        trainer.train(config['num_epochs'])
        
        # Calculate model hash
        best_model_path = './checkpoints/thermal_model/best_model.pth'
        if os.path.exists(best_model_path):
            model_hash = calculate_model_hash(best_model_path)
            print(f"Model SHA-256 hash: {model_hash}")
            
            # Save hash to file
            with open('./checkpoints/thermal_model/model_hash.txt', 'w') as f:
                f.write(f"SHA-256: {model_hash}\n")
                f.write(f"Model: best_model.pth\n")
                f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    else:
        print("No data paths provided. Please configure train_paths and val_paths in the config.")
        print("Model architecture created successfully.")


if __name__ == "__main__":
    main()