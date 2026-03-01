"""
MRI/K-Space Transform Utilities

Helper functions for:
- Forward and inverse FFT (with proper shifting)
- Complex data handling (complex -> 2-channel, and back)
- Root-Sum-of-Squares (RSS) multi-coil combination
- K-space normalization
- Zero-filled reconstruction
"""

import numpy as np
import torch
from typing import Union, Tuple


# Fourier Transform Helpers
def fft2c(image: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D FFT: image domain → k-space.
    
    Applies fftshift -> fft2 -> ifftshift (centered convention used in MRI).
    
    Args:
        image: Real or complex tensor of shape (..., H, W).
        
    Returns:
        Complex k-space tensor of shape (..., H, W).
    """
    return torch.fft.fftshift(
        torch.fft.fft2(
            torch.fft.ifftshift(image, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1)
    )


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    """
    Centered 2D inverse FFT: k-space -> image domain.
    
    Applies fftshift -> ifft2 -> ifftshift (centered convention used in MRI).
    
    Args:
        kspace: Complex k-space tensor of shape (..., H, W).
        
    Returns:
        Complex image tensor of shape (..., H, W).
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(
            torch.fft.ifftshift(kspace, dim=(-2, -1)),
            dim=(-2, -1), norm="ortho"
        ),
        dim=(-2, -1)
    )


def fft2c_numpy(image: np.ndarray) -> np.ndarray:
    """Centered 2D FFT for numpy arrays."""
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(image, axes=(-2, -1)),
            axes=(-2, -1), norm="ortho"
        ),
        axes=(-2, -1)
    )


def ifft2c_numpy(kspace: np.ndarray) -> np.ndarray:
    """Centered 2D inverse FFT for numpy arrays."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1), norm="ortho"
        ),
        axes=(-2, -1)
    )


# Complex Data Handling

def complex_to_channels(data: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor to 2-channel real tensor (real, imag).
    
    Args:
        data: Complex tensor of shape (..., H, W).
        
    Returns:
        Real tensor of shape (..., 2, H, W).
    """
    return torch.stack([data.real, data.imag], dim=-3)


def channels_to_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Convert 2-channel real tensor back to complex.
    
    Args:
        data: Real tensor of shape (..., 2, H, W).
        
    Returns:
        Complex tensor of shape (..., H, W).
    """
    return torch.complex(data[..., 0, :, :], data[..., 1, :, :])


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute magnitude of complex tensor.
    
    Args:
        data: Complex tensor or 2-channel real tensor.
        
    Returns:
        Magnitude tensor.
    """
    if torch.is_complex(data):
        return data.abs()
    # Assume 2-channel real
    return (data[..., 0, :, :] ** 2 + data[..., 1, :, :] ** 2).sqrt()


# Multi-Coil Combination

def rss_combine(data: torch.Tensor, coil_dim: int = -3) -> torch.Tensor:
    """
    Root-Sum-of-Squares (RSS) coil combination.
    
    Combines multi-coil images into a single composite image.
    
    Args:
        data: Complex or real tensor with a coil dimension.
              E.g., shape (batch, coils, H, W) for complex,
              or (batch, coils, 2, H, W) for 2-channel real.
        coil_dim: Dimension along which to combine coils.
        
    Returns:
        RSS-combined image with coil dimension removed.
    """
    if torch.is_complex(data):
        return torch.sqrt((data.abs() ** 2).sum(dim=coil_dim))
    else:
        # Assume magnitude images
        return torch.sqrt((data ** 2).sum(dim=coil_dim))


def rss_combine_numpy(data: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    """RSS coil combination for numpy arrays."""
    return np.sqrt(np.sum(np.abs(data) ** 2, axis=coil_axis))


# Normalization

def normalize_kspace(kspace: torch.Tensor, 
                     method: str = "max") -> Tuple[torch.Tensor, dict]:
    """
    Normalize k-space data.
    
    Args:
        kspace: Complex k-space tensor.
        method: "max" (divide by max magnitude) or "std" (zero mean, unit std).
        
    Returns:
        Tuple of (normalized kspace, normalization_params for denormalization).
    """
    if method == "max":
        max_val = kspace.abs().max()
        return kspace / (max_val + 1e-8), {"method": "max", "max_val": max_val}
    elif method == "std":
        mean_val = kspace.mean()
        std_val = kspace.std()
        return (kspace - mean_val) / (std_val + 1e-8), {
            "method": "std", "mean": mean_val, "std": std_val
        }
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Normalize image to [0, 1] range."""
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val + 1e-8)


# Zero-filled Reconstruction

def zero_filled_recon(kspace: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
    """
    Zero-filled reconstruction from undersampled k-space.
    
    This is the simplest reconstruction: apply mask, then inverse FFT.
    Used as a channel in the RL state representation.
    
    Args:
        kspace: Fully-sampled complex k-space, shape (..., H, W).
        mask: Binary sampling mask, shape (..., H) or (..., H, 1) or (..., H, W).
        
    Returns:
        Magnitude image from zero-filled reconstruction, shape (..., H, W).
    """
    # Expand mask to match kspace shape if needed
    if mask.dim() < kspace.dim() or mask.shape[-1] != kspace.shape[-1]:
        # mask is per-line (H,) → expand to (H, W)
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1).expand_as(kspace)
        elif mask.shape[-1] == 1:
            mask = mask.expand_as(kspace)
    
    undersampled = kspace * mask
    image = ifft2c(undersampled)
    return image.abs()


def zero_filled_recon_numpy(kspace: np.ndarray, 
                             mask: np.ndarray) -> np.ndarray:
    """Zero-filled reconstruction for numpy arrays."""
    if mask.ndim == 1:
        mask = mask[:, np.newaxis]
    undersampled = kspace * mask
    image = ifft2c_numpy(undersampled)
    return np.abs(image)


# Sampling Mask Utilities

def create_random_mask(num_lines: int, 
                       acceleration: float,
                       center_fraction: float = 0.08,
                       seed: int = None) -> np.ndarray:
    """
    Create a random undersampling mask.
    
    Args:
        num_lines: Total number of phase-encoding lines.
        acceleration: Acceleration factor.
        center_fraction: Fraction of center lines to always include.
        seed: Random seed for reproducibility.
        
    Returns:
        Boolean mask of shape (num_lines,).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    mask = np.zeros(num_lines, dtype=bool)
    
    # Always sample center
    num_center = int(center_fraction * num_lines)
    center_start = (num_lines - num_center) // 2
    mask[center_start:center_start + num_center] = True
    
    # Randomly sample remaining lines
    num_total_samples = int(num_lines / acceleration)
    num_outer_samples = max(0, num_total_samples - num_center)
    outer_indices = np.where(~mask)[0]
    
    if num_outer_samples > 0 and len(outer_indices) > 0:
        chosen = rng.choice(outer_indices, size=min(num_outer_samples, len(outer_indices)),
                            replace=False)
        mask[chosen] = True
    
    return mask


def create_variable_density_mask(num_lines: int,
                                  acceleration: float,
                                  center_fraction: float = 0.08,
                                  pdf_power: float = 2.0,
                                  seed: int = None) -> np.ndarray:
    """
    Creates a variable-density undersampling mask.
    
    Sampling probability decays as a power-law from center to periphery.
    
    Args:
        num_lines: Total number of phase-encoding lines.
        acceleration: Acceleration factor.
        center_fraction: Fraction of center lines to always include.
        pdf_power: Power-law exponent (higher = more center-concentrated).
        seed: Random seed.
        
    Returns:
        Boolean mask of shape (num_lines,).
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    mask = np.zeros(num_lines, dtype=bool)
    
    # Always sample center
    num_center = int(center_fraction * num_lines)
    center_start = (num_lines - num_center) // 2
    mask[center_start:center_start + num_center] = True
    
    # Variable-density probability for outer lines
    center = num_lines / 2
    distances = np.abs(np.arange(num_lines) - center) / center  # Normalized [0, 1]
    probabilities = (1 - distances) ** pdf_power
    probabilities[mask] = 0  # Don't re-sample center
    probabilities /= probabilities.sum() + 1e-8
    
    # Sample based on density
    num_total_samples = int(num_lines / acceleration)
    num_outer_samples = max(0, num_total_samples - num_center)
    outer_indices = np.where(~mask)[0]
    
    if num_outer_samples > 0 and len(outer_indices) > 0:
        outer_probs = probabilities[outer_indices]
        outer_probs /= outer_probs.sum() + 1e-8
        chosen = rng.choice(outer_indices,
                            size=min(num_outer_samples, len(outer_indices)),
                            replace=False, p=outer_probs)
        mask[chosen] = True
    
    return mask


def create_equispaced_mask(num_lines: int,
                            acceleration: float,
                            center_fraction: float = 0.08) -> np.ndarray:
    """
    Create an equispaced undersampling mask.
    
    Samples every Nth line with a fully-sampled center region.
    
    Args:
        num_lines: Total number of phase-encoding lines.
        acceleration: Acceleration factor (approximate).
        center_fraction: Fraction of center lines to always include.
        
    Returns:
        Boolean mask of shape (num_lines,).
    """
    mask = np.zeros(num_lines, dtype=bool)
    
    # Always sample center
    num_center = int(center_fraction * num_lines)
    center_start = (num_lines - num_center) // 2
    mask[center_start:center_start + num_center] = True
    
    # Equispaced outer sampling
    step = max(1, int(acceleration))
    mask[::step] = True
    
    return mask
