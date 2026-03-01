"""
Data Loader for fastMRI Breast DCE Dataset

FastMRIBreastDataLoader: Loads .h5 files, handles multi-coil data
FastMRISliceDataset: PyTorch Dataset that yields single-coil-combined k-space slices ready for the RL environment

Key features over the original:
- Root-Sum-of-Squares (RSS) coil combination
- Proper normalization
- PyTorch Dataset interface for DataLoader integration
- Train/val/test splitting by volume
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FastMRIBreastDataLoader:
    """
    Loads raw fastMRI Breast DCE .h5 files.
    
    Each .h5 file contains:
        - kspace: shape (num_slices, num_coils, height, width) complex
        - Various metadata attributes
    """
    
    def __init__(self, data_path: str, max_slices: int = 10):
        """
        Args:
            data_path: Path to directory containing .h5 files.
            max_slices: Max slices to load per volume (memory control).
        """
        self.data_path = Path(data_path)
        self.max_slices = max_slices
        self.file_list = self._get_file_list()
        logger.info(f"Found {len(self.file_list)} volumes in {data_path}")
    
    def _get_file_list(self) -> List[Path]:
        """Get sorted list of .h5 files."""
        if not self.data_path.exists():
            logger.warning(f"Dataset path {self.data_path} not found.")
            return []
        
        h5_files = sorted(self.data_path.glob("*.h5"))
        if not h5_files:
            logger.warning(f"No .h5 files found in {self.data_path}")
        return h5_files
    
    def load_volume(self, file_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Load a single volume.
        
        Returns:
            Dictionary with:
                'kspace': complex array, shape (slices, coils, H, W)
                'metadata': dict of HDF5 attributes
                'filename': str
        """
        if file_idx >= len(self.file_list):
            raise IndexError(f"File index {file_idx} out of range (total: {len(self.file_list)})")
        
        file_path = self.file_list[file_idx]
        logger.info(f"Loading volume: {file_path.name}")
        
        with h5py.File(file_path, 'r') as f:
            kspace = f['kspace'][:]
            
            metadata = {}
            for key in f.attrs.keys():
                metadata[key] = f.attrs[key]
            
            # Limit slices for memory
            if kspace.shape[0] > self.max_slices:
                kspace = kspace[:self.max_slices]
            
            logger.info(f"Loaded k-space shape: {kspace.shape}, dtype: {kspace.dtype}")
        
        return {
            'kspace': kspace,
            'metadata': metadata,
            'filename': file_path.name,
        }
    
    @staticmethod
    def rss_combine(kspace_multicoil: np.ndarray) -> np.ndarray:
        """
        Combine multi-coil k-space data via RSS in image domain.
        
        Args:
            kspace_multicoil: Complex array of shape (coils, H, W).
            
        Returns:
            RSS-combined magnitude image of shape (H, W).
        """
        # IFFT each coil
        images = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace_multicoil, axes=(-2, -1)),
                axes=(-2, -1), norm="ortho"
            ),
            axes=(-2, -1)
        )
        # RSS combination
        return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))
    
    @staticmethod
    def combine_to_single_coil_kspace(kspace_multicoil: np.ndarray) -> np.ndarray:
        """
        Combine multi-coil k-space into single-coil-equivalent k-space.
        
        Process: multi-coil → RSS image → FFT → single-coil k-space.
        This is what we feed to the RL environment.
        
        Args:
            kspace_multicoil: Complex array of shape (coils, H, W).
            
        Returns:
            Complex k-space array of shape (H, W).
        """
        # Get RSS-combined image
        rss_image = FastMRIBreastDataLoader.rss_combine(kspace_multicoil)
        
        # Convert back to k-space (now single-coil equivalent)
        kspace_single = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(rss_image, axes=(-2, -1)),
                axes=(-2, -1), norm="ortho"
            ),
            axes=(-2, -1)
        )
        return kspace_single


class FastMRISliceDataset(Dataset):
    """
    PyTorch Dataset yielding single k-space slices for the RL environment.
    
    Each item is a complex numpy array of shape (H, W) representing
    the single-coil-equivalent k-space of one slice.
    """
    
    def __init__(self,
                 data_path: str,
                 split: str = "train",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 max_slices_per_volume: int = 10,
                 target_shape: Optional[Tuple[int, int]] = None):
        """
        Args:
            data_path: Path to directory with .h5 files.
            split: One of "train", "val", "test".
            train_ratio: Fraction of volumes for training.
            val_ratio: Fraction of volumes for validation.
            max_slices_per_volume: Max slices to extract per volume.
            target_shape: If provided, center-crop or pad to this shape.
        """
        self.loader = FastMRIBreastDataLoader(data_path, max_slices_per_volume)
        self.target_shape = target_shape
        self.split = split
        
        # Split volumes by index
        num_volumes = len(self.loader.file_list)
        num_train = int(train_ratio * num_volumes)
        num_val = int(val_ratio * num_volumes)
        
        if split == "train":
            volume_indices = list(range(0, num_train))
        elif split == "val":
            volume_indices = list(range(num_train, num_train + num_val))
        elif split == "test":
            volume_indices = list(range(num_train + num_val, num_volumes))
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Pre-index all slices: list of (volume_idx, slice_idx)
        self.slice_index = []
        for vol_idx in volume_indices:
            try:
                with h5py.File(self.loader.file_list[vol_idx], 'r') as f:
                    num_slices = min(f['kspace'].shape[0], max_slices_per_volume)
                    for s in range(num_slices):
                        self.slice_index.append((vol_idx, s))
            except Exception as e:
                logger.warning(f"Error indexing volume {vol_idx}: {e}")
        
        logger.info(f"FastMRISliceDataset [{split}]: {len(self.slice_index)} slices "
                     f"from {len(volume_indices)} volumes")
    
    def __len__(self) -> int:
        return len(self.slice_index)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single k-space slice.
        
        Returns:
            Dict with:
                'kspace': complex numpy array of shape (H, W)
                'volume_idx': int
                'slice_idx': int
        """
        vol_idx, slice_idx = self.slice_index[idx]
        file_path = self.loader.file_list[vol_idx]
        
        with h5py.File(file_path, 'r') as f:
            kspace_multicoil = f['kspace'][slice_idx]  # (coils, H, W)
        
        # Combine to single-coil equivalent
        kspace = self.loader.combine_to_single_coil_kspace(kspace_multicoil)
        
        # Center crop / pad to target shape if needed
        if self.target_shape is not None:
            kspace = self._center_crop_or_pad(kspace, self.target_shape)
        
        return {
            'kspace': kspace,
            'volume_idx': vol_idx,
            'slice_idx': slice_idx,
        }
    
    @staticmethod
    def _center_crop_or_pad(data: np.ndarray,
                             target_shape: Tuple[int, int]) -> np.ndarray:
        """Center crop or zero-pad to target shape."""
        H, W = data.shape
        tH, tW = target_shape
        
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Compute crop/pad offsets
        sh = max(0, (H - tH) // 2)
        sw = max(0, (W - tW) // 2)
        dh = max(0, (tH - H) // 2)
        dw = max(0, (tW - W) // 2)
        
        copy_h = min(H, tH)
        copy_w = min(W, tW)
        
        result[dh:dh + copy_h, dw:dw + copy_w] = data[sh:sh + copy_h, sw:sw + copy_w]
        
        return result


# Convenience: create a DataLoader that feeds k-space slices
def create_kspace_dataloader(data_path: str,
                              split: str = "train",
                              batch_size: int = 1,
                              num_workers: int = 0,
                              target_shape: Optional[Tuple[int, int]] = None,
                              **kwargs) -> DataLoader:
    """
    Create a DataLoader for k-space slices.
    
    Note: batch_size is typically 1 for RL (one slice per env reset),
    but can be >1 for supervised U-Net pre-training.
    """
    dataset = FastMRISliceDataset(
        data_path=data_path,
        split=split,
        target_shape=target_shape,
        **kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        # Custom collate: don't convert complex to tensor (keep numpy)
        collate_fn=_numpy_collate,
    )


def _numpy_collate(batch):
    """Custom collate that keeps numpy arrays (complex data can't be torch tensors easily)."""
    return {
        'kspace': np.stack([item['kspace'] for item in batch]),
        'volume_idx': [item['volume_idx'] for item in batch],
        'slice_idx': [item['slice_idx'] for item in batch],
    }