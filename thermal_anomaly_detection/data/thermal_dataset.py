"""
Thermal Dataset Loader for High-Accuracy Anomaly Detection
=========================================================

Supports loading thermal data from various formats:
- .he5 datasets (Landsat-8/9, FLAME-2/3, KAIST, MIRSat-QL)
- .tif thermal images
- Video streams for temporal analysis

Features:
- Minimal preprocessing to preserve temperature data
- GPU-optimized data loading
- Automatic normalization and augmentation
- Support for both single images and video sequences

Author: Thermal Anomaly Detection System
Date: 2025-10-07
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import h5py
import rasterio
from rasterio.windows import Window
import os
import glob
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ThermalDataset(Dataset):
    """
    High-performance thermal dataset loader with minimal preprocessing
    
    Supports:
    - .he5 files (HDF5 format for satellite data)
    - .tif thermal images
    - Video sequences for temporal analysis
    - Real-time data augmentation
    """
    
    def __init__(self,
                 data_paths: List[str],
                 labels_paths: Optional[List[str]] = None,
                 img_size: int = 224,
                 temporal_length: int = 8,
                 use_temporal: bool = False,
                 augmentation: bool = True,
                 normalize: bool = True,
                 temperature_range: Tuple[float, float] = (273.15, 373.15),  # Kelvin range
                 cache_data: bool = False):
        """
        Initialize thermal dataset
        
        Args:
            data_paths: List of paths to thermal data files
            labels_paths: List of paths to ground truth labels (optional)
            img_size: Target image size for resizing
            temporal_length: Number of frames for temporal sequences
            use_temporal: Whether to load temporal sequences
            augmentation: Whether to apply data augmentation
            normalize: Whether to normalize temperature data
            temperature_range: Expected temperature range in Kelvin
            cache_data: Whether to cache loaded data in memory
        """
        self.data_paths = data_paths
        self.labels_paths = labels_paths
        self.img_size = img_size
        self.temporal_length = temporal_length
        self.use_temporal = use_temporal
        self.normalize = normalize
        self.temperature_range = temperature_range
        self.cache_data = cache_data
        
        # Data cache
        self.data_cache = {} if cache_data else None
        
        # Setup augmentation pipeline
        self.augmentation = self._setup_augmentation() if augmentation else None
        
        # Scan and validate data files
        self.valid_files = self._scan_files()
        
        print(f"Loaded {len(self.valid_files)} valid thermal data files")
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup thermal-specific augmentation pipeline"""
        return A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.7
            ),
            
            # Thermal-specific augmentations (preserve temperature relationships)
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(0.01, 0.03), p=0.3),
            
            # Normalization
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2()
        ])
    
    def _scan_files(self) -> List[Dict]:
        """Scan and validate data files"""
        valid_files = []
        
        for i, data_path in enumerate(self.data_paths):
            try:
                # Check file existence and format
                if not os.path.exists(data_path):
                    print(f"Warning: File not found: {data_path}")
                    continue
                
                file_info = {
                    'data_path': data_path,
                    'label_path': self.labels_paths[i] if self.labels_paths else None,
                    'format': self._detect_format(data_path),
                    'index': i
                }
                
                # Validate file can be loaded
                if self._validate_file(file_info):
                    valid_files.append(file_info)
                else:
                    print(f"Warning: Invalid file: {data_path}")
                    
            except Exception as e:
                print(f"Error processing {data_path}: {e}")
                continue
        
        return valid_files
    
    def _detect_format(self, file_path: str) -> str:
        """Detect thermal data format"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.he5', '.h5', '.hdf5']:
            return 'he5'
        elif ext in ['.tif', '.tiff']:
            return 'tif'
        elif ext in ['.mp4', '.avi', '.mov']:
            return 'video'
        else:
            return 'unknown'
    
    def _validate_file(self, file_info: Dict) -> bool:
        """Validate that file can be loaded"""
        try:
            if file_info['format'] == 'he5':
                with h5py.File(file_info['data_path'], 'r') as f:
                    # Check for common thermal datasets in HDF5
                    thermal_keys = ['thermal', 'temperature', 'TIR', 'LST', 'Band_10', 'Band_11']
                    found_thermal = any(key in f.keys() for key in thermal_keys)
                    return found_thermal
            
            elif file_info['format'] == 'tif':
                with rasterio.open(file_info['data_path']) as src:
                    return src.count >= 1 and src.width > 0 and src.height > 0
            
            elif file_info['format'] == 'video':
                cap = cv2.VideoCapture(file_info['data_path'])
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                return frame_count > 0
            
            return False
            
        except Exception:
            return False
    
    def __len__(self) -> int:
        return len(self.valid_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load thermal data item"""
        file_info = self.valid_files[idx]
        
        # Check cache first
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        # Load thermal data
        thermal_data = self._load_thermal_data(file_info)
        
        # Load labels if available
        labels = self._load_labels(file_info) if file_info['label_path'] else None
        
        # Process data
        processed_data = self._process_thermal_data(thermal_data, labels)
        
        # Cache if enabled
        if self.cache_data:
            self.data_cache[idx] = processed_data
        
        return processed_data
    
    def _load_thermal_data(self, file_info: Dict) -> np.ndarray:
        """Load thermal data based on format"""
        data_path = file_info['data_path']
        format_type = file_info['format']
        
        if format_type == 'he5':
            return self._load_he5_data(data_path)
        elif format_type == 'tif':
            return self._load_tif_data(data_path)
        elif format_type == 'video':
            return self._load_video_data(data_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _load_he5_data(self, file_path: str) -> np.ndarray:
        """Load thermal data from HDF5/HE5 files"""
        with h5py.File(file_path, 'r') as f:
            # Common thermal dataset names in satellite data
            thermal_keys = [
                'thermal', 'temperature', 'TIR', 'LST',
                'Band_10', 'Band_11',  # Landsat thermal bands
                'FLAME_L2_Fire_Mask',  # FLAME dataset
                'thermal_infrared'  # General thermal
            ]
            
            # Find thermal data
            thermal_data = None
            for key in thermal_keys:
                if key in f.keys():
                    thermal_data = f[key][:]
                    break
                
                # Search in groups
                for group_name in f.keys():
                    if isinstance(f[group_name], h5py.Group):
                        if key in f[group_name].keys():
                            thermal_data = f[group_name][key][:]
                            break
                
                if thermal_data is not None:
                    break
            
            if thermal_data is None:
                # Fallback: use first available dataset
                def find_first_dataset(group):
                    for item in group.keys():
                        if isinstance(group[item], h5py.Dataset):
                            return group[item][:]
                        elif isinstance(group[item], h5py.Group):
                            result = find_first_dataset(group[item])
                            if result is not None:
                                return result
                    return None
                
                thermal_data = find_first_dataset(f)
            
            if thermal_data is None:
                raise ValueError(f"No thermal data found in {file_path}")
            
            # Convert to float32 and handle common data issues
            thermal_data = thermal_data.astype(np.float32)
            
            # Handle fill values and invalid data
            if hasattr(thermal_data, 'fill_value'):
                thermal_data[thermal_data == thermal_data.fill_value] = np.nan
            
            # Handle common invalid values
            thermal_data[thermal_data <= 0] = np.nan
            thermal_data[thermal_data > 1000] = np.nan  # Unrealistic temperatures
            
            return thermal_data
    
    def _load_tif_data(self, file_path: str) -> np.ndarray:
        """Load thermal data from TIFF files"""
        with rasterio.open(file_path) as src:
            # Read thermal band(s)
            if src.count == 1:
                thermal_data = src.read(1)
            else:
                # For multi-band images, take the first band or thermal-specific band
                thermal_data = src.read(1)  # Assume first band is thermal
            
            thermal_data = thermal_data.astype(np.float32)
            
            # Handle nodata values
            if src.nodata is not None:
                thermal_data[thermal_data == src.nodata] = np.nan
            
            return thermal_data
    
    def _load_video_data(self, file_path: str) -> np.ndarray:
        """Load thermal video data"""
        cap = cv2.VideoCapture(file_path)
        frames = []
        
        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale if needed (assume thermal is single channel)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames.append(frame.astype(np.float32))
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {file_path}")
        
        # Stack frames
        thermal_data = np.stack(frames, axis=0)  # T, H, W
        
        # Sample temporal sequence if using temporal mode
        if self.use_temporal and len(frames) > self.temporal_length:
            # Sample evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, self.temporal_length, dtype=int)
            thermal_data = thermal_data[indices]
        
        return thermal_data
    
    def _load_labels(self, file_info: Dict) -> Optional[np.ndarray]:
        """Load ground truth labels"""
        if file_info['label_path'] is None:
            return None
        
        label_path = file_info['label_path']
        
        try:
            # Try different formats
            if label_path.endswith('.tif') or label_path.endswith('.tiff'):
                with rasterio.open(label_path) as src:
                    labels = src.read(1).astype(np.float32)
            elif label_path.endswith('.png') or label_path.endswith('.jpg'):
                labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            else:
                # Try numpy format
                labels = np.load(label_path).astype(np.float32)
            
            # Normalize to 0-1
            if labels.max() > 1:
                labels = labels / 255.0
            
            return labels
            
        except Exception as e:
            print(f"Warning: Could not load labels from {label_path}: {e}")
            return None
    
    def _process_thermal_data(self, thermal_data: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """Process thermal data for model input"""
        
        # Handle temporal dimension
        if self.use_temporal and thermal_data.ndim == 3:
            # T, H, W format
            processed_frames = []
            processed_labels = []
            
            for t in range(thermal_data.shape[0]):
                frame = thermal_data[t]
                frame_labels = labels[t] if labels is not None and labels.ndim == 3 else labels
                
                processed_frame, processed_label = self._process_single_frame(frame, frame_labels)
                processed_frames.append(processed_frame)
                if processed_label is not None:
                    processed_labels.append(processed_label)
            
            # Stack temporal frames
            thermal_tensor = torch.stack(processed_frames, dim=0)  # T, C, H, W
            labels_tensor = torch.stack(processed_labels, dim=0) if processed_labels else None
            
            return {
                'thermal': thermal_tensor,
                'labels': labels_tensor,
                'temporal': True
            }
        
        else:
            # Single frame processing
            thermal_tensor, labels_tensor = self._process_single_frame(thermal_data, labels)
            
            return {
                'thermal': thermal_tensor,
                'labels': labels_tensor,
                'temporal': False
            }
    
    def _process_single_frame(self, thermal_data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process single thermal frame"""
        
        # Handle NaN values
        if np.isnan(thermal_data).any():
            thermal_data = np.nan_to_num(thermal_data, nan=np.nanmean(thermal_data))
        
        # Temperature normalization (preserve physical meaning)
        if self.normalize:
            # Clip to reasonable temperature range
            thermal_data = np.clip(thermal_data, self.temperature_range[0], self.temperature_range[1])
            
            # Normalize to 0-1
            thermal_data = (thermal_data - self.temperature_range[0]) / (self.temperature_range[1] - self.temperature_range[0])
        
        # Resize to target size
        if thermal_data.shape != (self.img_size, self.img_size):
            thermal_data = cv2.resize(thermal_data, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Process labels if available
        if labels is not None:
            if labels.shape != (self.img_size, self.img_size):
                labels = cv2.resize(labels, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentation
        if self.augmentation is not None:
            if labels is not None:
                augmented = self.augmentation(image=thermal_data, mask=labels)
                thermal_tensor = augmented['image']
                labels_tensor = torch.from_numpy(augmented['mask']).float()
            else:
                # Add channel dimension for albumentations
                thermal_3d = np.stack([thermal_data] * 3, axis=-1)
                augmented = self.augmentation(image=thermal_3d)
                thermal_tensor = augmented['image'][0:1]  # Take only first channel
                labels_tensor = None
        else:
            # Convert to tensor manually
            thermal_tensor = torch.from_numpy(thermal_data).unsqueeze(0).float()  # Add channel dimension
            labels_tensor = torch.from_numpy(labels).float() if labels is not None else None
        
        return thermal_tensor, labels_tensor


class ThermalDataModule:
    """
    Data module for thermal anomaly detection with GPU optimization
    """
    
    def __init__(self,
                 train_paths: List[str],
                 val_paths: List[str],
                 test_paths: Optional[List[str]] = None,
                 train_labels: Optional[List[str]] = None,
                 val_labels: Optional[List[str]] = None,
                 test_labels: Optional[List[str]] = None,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 img_size: int = 224,
                 use_temporal: bool = False,
                 temporal_length: int = 8,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2):
        """
        Initialize thermal data module
        
        Args:
            train_paths: Training data file paths
            val_paths: Validation data file paths
            test_paths: Test data file paths (optional)
            train_labels: Training labels file paths (optional)
            val_labels: Validation labels file paths (optional)
            test_labels: Test labels file paths (optional)
            batch_size: Batch size for data loading
            num_workers: Number of worker processes
            img_size: Target image size
            use_temporal: Whether to use temporal sequences
            temporal_length: Length of temporal sequences
            pin_memory: Whether to pin memory for GPU transfer
            prefetch_factor: Number of batches to prefetch
        """
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.use_temporal = use_temporal
        self.temporal_length = temporal_length
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        train_dataset = ThermalDataset(
            data_paths=self.train_paths,
            labels_paths=self.train_labels,
            img_size=self.img_size,
            use_temporal=self.use_temporal,
            temporal_length=self.temporal_length,
            augmentation=True,
            normalize=True
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        val_dataset = ThermalDataset(
            data_paths=self.val_paths,
            labels_paths=self.val_labels,
            img_size=self.img_size,
            use_temporal=self.use_temporal,
            temporal_length=self.temporal_length,
            augmentation=False,
            normalize=True
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader"""
        if self.test_paths is None:
            return None
        
        test_dataset = ThermalDataset(
            data_paths=self.test_paths,
            labels_paths=self.test_labels,
            img_size=self.img_size,
            use_temporal=self.use_temporal,
            temporal_length=self.temporal_length,
            augmentation=False,
            normalize=True
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False
        )


def create_thermal_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create thermal data loaders from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = ThermalDataModule(
        train_paths=config['train_paths'],
        val_paths=config['val_paths'],
        test_paths=config.get('test_paths'),
        train_labels=config.get('train_labels'),
        val_labels=config.get('val_labels'),
        test_labels=config.get('test_labels'),
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        img_size=config.get('img_size', 224),
        use_temporal=config.get('use_temporal', False),
        temporal_length=config.get('temporal_length', 8),
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2)
    )
    
    train_loader = data_module.get_train_dataloader()
    val_loader = data_module.get_val_dataloader()
    test_loader = data_module.get_test_dataloader()
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import tempfile
    
    # Create dummy thermal data for testing
    dummy_thermal = np.random.rand(224, 224).astype(np.float32) * 100 + 273.15  # Kelvin
    dummy_labels = (np.random.rand(224, 224) > 0.9).astype(np.float32)  # Sparse anomalies
    
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_thermal:
        with rasterio.open(tmp_thermal.name, 'w', driver='GTiff', height=224, width=224, count=1, dtype='float32') as dst:
            dst.write(dummy_thermal, 1)
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_labels:
            with rasterio.open(tmp_labels.name, 'w', driver='GTiff', height=224, width=224, count=1, dtype='float32') as dst:
                dst.write(dummy_labels, 1)
            
            # Test dataset
            config = {
                'train_paths': [tmp_thermal.name],
                'val_paths': [tmp_thermal.name],
                'train_labels': [tmp_labels.name],
                'val_labels': [tmp_labels.name],
                'batch_size': 2,
                'num_workers': 0,
                'img_size': 224,
                'use_temporal': False
            }
            
            train_loader, val_loader, test_loader = create_thermal_dataloaders(config)
            
            # Test loading
            for batch in train_loader:
                print(f"Batch thermal shape: {batch['thermal'].shape}")
                print(f"Batch labels shape: {batch['labels'].shape}")
                print(f"Thermal range: {batch['thermal'].min():.3f} - {batch['thermal'].max():.3f}")
                break
            
            print("Dataset loading test successful!")