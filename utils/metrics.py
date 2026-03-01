"""
Image Quality Metrics for MRI Reconstruction

Provides PSNR, SSIM, and Total Variation computations used for:
1. Reward calculation in the RL environment
2. Evaluation of reconstruction quality
3. Comparison against baselines
"""

import numpy as np
import torch
from typing import Optional, Union


def psnr(target: torch.Tensor, prediction: torch.Tensor,
         data_range: Optional[float] = None) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio. PSNR = 10 * log10(data_range^2 / MSE)
    
    Args:
        target: Ground truth image, shape (..., H, W).
        prediction: Reconstructed image, shape (..., H, W).
        data_range: Maximum possible pixel value. If None, uses target max.
        
    Returns:
        PSNR value in dB (scalar tensor).
    """
    if data_range is None:
        data_range = target.max()
    
    mse = torch.mean((target - prediction) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    
    return 10.0 * torch.log10(data_range ** 2 / mse)


def psnr_numpy(target: np.ndarray, prediction: np.ndarray,
               data_range: Optional[float] = None) -> float:
    """Compute PSNR for numpy arrays."""
    if data_range is None:
        data_range = target.max()
    
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10.0 * np.log10(data_range ** 2 / mse)


# SSIM — Structural Similarity Index
def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel for SSIM computation."""
    kernel_1d = _gaussian_kernel_1d(size, sigma)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def ssim(target: torch.Tensor, prediction: torch.Tensor,
         data_range: Optional[float] = None,
         win_size: int = 7,
         k1: float = 0.01,
         k2: float = 0.03) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) using a Gaussian-weighted window for local statistics.
    
    Args:
        target: Ground truth image, shape (B, 1, H, W) or (1, H, W) or (H, W).
        prediction: Reconstructed image, same shape as target.
        data_range: Max pixel value. If None, uses target max.
        win_size: Size of the Gaussian window.
        k1: SSIM constant (luminance).
        k2: SSIM constant (contrast).
        
    Returns:
        Mean SSIM value (scalar tensor).
    """
    if data_range is None:
        data_range = target.max()
    
    # Ensure 4D: (B, C, H, W)
    ndim_orig = target.dim()
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
        prediction = prediction.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
    
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2
    
    # Gaussian window
    kernel = _gaussian_kernel_2d(win_size, sigma=1.5)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, win, win)
    kernel = kernel.to(target.device, dtype=target.dtype)
    
    channels = target.shape[1]
    kernel = kernel.expand(channels, -1, -1, -1)
    
    # Compute local statistics using convolution
    pad = win_size // 2
    mu_target = torch.nn.functional.conv2d(target, kernel, padding=pad, groups=channels)
    mu_pred = torch.nn.functional.conv2d(prediction, kernel, padding=pad, groups=channels)
    
    mu_target_sq = mu_target ** 2
    mu_pred_sq = mu_pred ** 2
    mu_target_pred = mu_target * mu_pred
    
    sigma_target_sq = torch.nn.functional.conv2d(
        target ** 2, kernel, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_sq = torch.nn.functional.conv2d(
        prediction ** 2, kernel, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_pred = torch.nn.functional.conv2d(
        target * prediction, kernel, padding=pad, groups=channels) - mu_target_pred
    
    # SSIM formula
    numerator = (2 * mu_target_pred + C1) * (2 * sigma_target_pred + C2)
    denominator = (mu_target_sq + mu_pred_sq + C1) * (sigma_target_sq + sigma_pred_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


def ssim_numpy(target: np.ndarray, prediction: np.ndarray,
               data_range: Optional[float] = None) -> float:
    """
    Compute SSIM for numpy arrays.
    
    Falls back to scikit-image implementation for reliability.
    """
    try:
        from skimage.metrics import structural_similarity
        if data_range is None:
            data_range = target.max() - target.min()
        return structural_similarity(target, prediction, data_range=data_range)
    except ImportError:
        # Fallback: convert to torch and use our implementation
        t = torch.from_numpy(target.astype(np.float32))
        p = torch.from_numpy(prediction.astype(np.float32))
        dr = data_range if data_range is not None else float(target.max())
        return ssim(t, p, data_range=dr).item()



# Total Variation
def total_variation(image: torch.Tensor) -> torch.Tensor:
    """
    Compute Total Variation (TV) of an image.
    
    TV = sum of absolute differences between adjacent pixels (anisotropic TV).
    Used as a smoothness / regularization term in the reward.
    
    Args:
        image: Image tensor, shape (..., H, W).
        
    Returns:
        Total variation value (scalar tensor).
    """
    # Horizontal and vertical differences
    diff_h = torch.abs(image[..., :, 1:] - image[..., :, :-1])
    diff_v = torch.abs(image[..., 1:, :] - image[..., :-1, :])
    
    return diff_h.mean() + diff_v.mean()


def total_variation_numpy(image: np.ndarray) -> float:
    """Compute Total Variation for numpy arrays."""
    diff_h = np.abs(image[..., :, 1:] - image[..., :, :-1])
    diff_v = np.abs(image[..., 1:, :] - image[..., :-1, :])
    return float(diff_h.mean() + diff_v.mean())


# Composite Reward Function
def compute_reward(target: torch.Tensor,
                   prediction: torch.Tensor,
                   ssim_weight: float = 1.0,
                   tv_weight: float = 0.1,
                   step_penalty: float = 0.01,
                   num_steps: int = 0,
                   data_range: Optional[float] = None) -> dict:
    """
    Compute the composite reward for the RL environment.
    reward = α * SSIM(target, pred) + β * (1 - TV(pred)/TV(target)) - γ * num_steps
    
    Args:
        target: Ground truth image.
        prediction: Reconstructed image.
        ssim_weight: Weight α for SSIM component.
        tv_weight: Weight β for TV component.
        step_penalty: Penalty γ per step taken.
        num_steps: Number of steps taken in the episode.
        data_range: Max pixel value for SSIM.
        
    Returns:
        Dict with 'total', 'ssim', 'tv', 'step_penalty', 'psnr' keys.
    """
    # SSIM component
    ssim_val = ssim(target, prediction, data_range=data_range)
    
    # TV component: reward for matching target's smoothness
    tv_pred = total_variation(prediction)
    tv_target = total_variation(target)
    tv_ratio = tv_pred / (tv_target + 1e-8)
    # Penalize deviation from 1.0 (perfect match)
    tv_component = 1.0 - torch.abs(tv_ratio - 1.0).clamp(max=1.0)
    
    # Step penalty
    step_cost = step_penalty * num_steps
    
    # Total reward
    total = ssim_weight * ssim_val + tv_weight * tv_component - step_cost
    
    # Also compute PSNR for logging (not part of reward)
    psnr_val = psnr(target, prediction, data_range=data_range)
    
    return {
        'total': total,
        'ssim': ssim_val,
        'tv_component': tv_component,
        'tv_raw': tv_pred,
        'step_penalty': step_cost,
        'psnr': psnr_val,
    }
