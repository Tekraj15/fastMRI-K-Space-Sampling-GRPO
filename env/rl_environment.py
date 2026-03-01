"""
MRI K-Space Sampling Environment (Gymnasium)

State Space:
    Multi-channel 2D tensor of shape (3, H, W):
        Channel 0: Partial k-space magnitude (normalized, log-scaled)
        Channel 1: Binary sampling mask (broadcast to 2D)
        Channel 2: Zero-filled IFFT reconstruction (normalized)

Action Space:
    Discrete(H) — index of the next phase-encoding line to sample.
    Actions for already-sampled lines are masked to -inf before softmax
    in the policy network.

Reward:
    Sparse, episode-level hybrid reward:
        R = α·SSIM(I_gt, I_recon) + β·TV_match - γ·num_steps
    where TV_match penalizes deviation of reconstruction TV from target TV.

The environment accepts a pluggable `reconstructor` callable:
    - Default: IFFT (zero-filled reconstruction)
    - Production: Frozen U-Net
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KSpaceSamplingEnv(gym.Env):
    """    
    The agent sequentially selects phase-encoding lines to sample.
    After the episode ends (budget exhausted or max steps reached),
    a reconstruction is performed and the reward is computed.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self,
                 kspace_height: int = 256,
                 kspace_width: int = 256,
                 max_steps: int = 50,
                 center_fraction: float = 0.08,
                 reward_ssim_weight: float = 1.0,
                 reward_tv_weight: float = 0.1,
                 reward_step_penalty: float = 0.01,
                 sparse_reward: bool = True,
                 reconstructor: Optional[Callable] = None,
                 render_mode: Optional[str] = None):
        """
        Args:
            kspace_height: Number of phase-encoding lines (action space size).
            kspace_width: Number of frequency-encoding columns.
            max_steps: Maximum lines the agent can select per episode.
            center_fraction: Fraction of center lines auto-sampled at reset.
            reward_ssim_weight: Weight α for SSIM in reward.
            reward_tv_weight: Weight β for TV match in reward.
            reward_step_penalty: Penalty γ per step.
            sparse_reward: If True, reward is given only at episode end.
            reconstructor: Callable(undersampled_kspace, mask) → image.
                           Defaults to IFFT zero-filled reconstruction.
            render_mode: "human" or "rgb_array".
        """
        super().__init__()
        
        self.kspace_height = kspace_height
        self.kspace_width = kspace_width
        self.max_steps = max_steps
        self.center_fraction = center_fraction
        self.sparse_reward = sparse_reward
        self.render_mode = render_mode
        
        # Reward weights
        self.reward_ssim_weight = reward_ssim_weight
        self.reward_tv_weight = reward_tv_weight
        self.reward_step_penalty = reward_step_penalty
        
        # Reconstructor (pluggable: IFFT now, U-Net later)
        self._reconstructor = reconstructor or self._default_ifft_reconstructor
        
        # ---- Spaces ----
        # Action: select a phase-encoding line index
        self.action_space = spaces.Discrete(kspace_height)
        
        # Observation: 3-channel image (partial_kspace, mask, zero_filled_recon)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, kspace_height, kspace_width),
            dtype=np.float32,
        )
        
        # ---- Internal state (set in reset()) ----
        self._target_kspace: Optional[np.ndarray] = None   # Complex (H, W)
        self._target_image: Optional[np.ndarray] = None      # Ground truth image
        self._mask: Optional[np.ndarray] = None               # Bool (H,)
        self._sampled_kspace: Optional[np.ndarray] = None     # Complex (H, W)
        self._step_count: int = 0
        self._action_history: list = []
        
        logger.info(
            f"KSpaceSamplingEnv initialized: "
            f"shape=({kspace_height},{kspace_width}), "
            f"max_steps={max_steps}, center_frac={center_fraction}"
        )
    
    # ==================================================================
    # Gymnasium API
    # ==================================================================
    
    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Options:
            target_kspace (np.ndarray): Complex k-space array of shape (H, W).
                If not provided, a synthetic phantom is generated.
        
        Returns:
            (observation, info) per Gymnasium API.
        """
        super().reset(seed=seed)
        
        # Get target k-space from options or generate synthetic
        if options is not None and "target_kspace" in options:
            self._target_kspace = options["target_kspace"].copy()
        else:
            self._target_kspace = self._generate_synthetic_kspace()
        
        assert self._target_kspace.shape == (self.kspace_height, self.kspace_width), \
            f"K-space shape mismatch: {self._target_kspace.shape} vs ({self.kspace_height}, {self.kspace_width})"
        
        # Pre-compute ground truth image
        self._target_image = np.abs(self._ifft2c(self._target_kspace))
        
        # Initialize mask: auto-sample center lines
        self._mask = np.zeros(self.kspace_height, dtype=bool)
        num_center = max(1, int(self.center_fraction * self.kspace_height))
        center_start = (self.kspace_height - num_center) // 2
        self._mask[center_start:center_start + num_center] = True
        
        # Initialize sampled k-space with center lines
        self._sampled_kspace = np.zeros_like(self._target_kspace)
        self._sampled_kspace[self._mask] = self._target_kspace[self._mask]
        
        # Reset episode tracking
        self._step_count = 0
        self._action_history = []
        
        obs = self._build_observation()
        info = self._build_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: sample the selected k-space line.
        
        Args:
            action: Index of the phase-encoding line to sample.
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self._target_kspace is not None, "Call reset() before step()"
        
        # Record action
        self._action_history.append(action)
        self._step_count += 1
        
        # Apply action (sample the line if not already sampled)
        already_sampled = self._mask[action]
        if not already_sampled:
            self._mask[action] = True
            self._sampled_kspace[action, :] = self._target_kspace[action, :]
        
        # Check termination
        terminated = self._step_count >= self.max_steps
        truncated = False
        
        # Check if all lines are sampled (early termination)
        if self._mask.all():
            terminated = True
        
        # Compute reward
        if self.sparse_reward:
            # Sparse: reward only at episode end
            if terminated:
                reward = self._compute_episode_reward()
            else:
                reward = 0.0
                # Small penalty for re-sampling an already-sampled line
                if already_sampled:
                    reward = -0.1
        else:
            # Dense: reward at every step
            reward = self._compute_step_reward(already_sampled)
        
        obs = self._build_observation()
        info = self._build_info()
        
        if terminated:
            # Add final metrics to info
            info.update(self._compute_final_metrics())
        
        return obs, reward, terminated, truncated, info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get the valid action mask.
        
        Returns:
            Boolean array of shape (H,) where True = valid (unsampled) action.
        """
        return ~self._mask.copy()
    
    # ==================================================================
    # Observation Building
    # ==================================================================
    
    def _build_observation(self) -> np.ndarray:
        """
        Build the 3-channel observation tensor.
        
        Channel 0: Log-magnitude of partial k-space (normalized to [0, 1])
        Channel 1: Binary sampling mask (2D broadcast)
        Channel 2: Zero-filled reconstruction magnitude (normalized to [0, 1])
        """
        # Channel 0: Partial k-space magnitude (log scale for better dynamic range)
        kspace_mag = np.log1p(np.abs(self._sampled_kspace))
        max_mag = np.log1p(np.abs(self._target_kspace)).max()
        ch0 = (kspace_mag / (max_mag + 1e-8)).astype(np.float32)
        
        # Channel 1: Sampling mask broadcast to 2D
        ch1 = np.tile(self._mask[:, np.newaxis], (1, self.kspace_width)).astype(np.float32)
        
        # Channel 2: Zero-filled IFFT reconstruction (normalized)
        recon = np.abs(self._ifft2c(self._sampled_kspace))
        max_recon = self._target_image.max()
        ch2 = (recon / (max_recon + 1e-8)).astype(np.float32)
        
        obs = np.stack([ch0, ch1, ch2], axis=0)  # (3, H, W)
        return np.clip(obs, 0.0, 1.0)
    
    # ==================================================================
    # Reward Computation
    # ==================================================================
    
    def _compute_episode_reward(self) -> float:
        """
        Compute the sparse episode-level reward.
        
        R = α·SSIM + β·TV_match - γ·num_steps
        """
        from utils.metrics import ssim, total_variation
        
        # Reconstruct using the pluggable reconstructor
        recon_image = self._reconstructor(self._sampled_kspace, self._mask)
        target = self._target_image
        
        # Convert to torch for metrics
        t_target = torch.from_numpy(target).float()
        t_recon = torch.from_numpy(recon_image).float()
        
        # SSIM
        ssim_val = ssim(t_target, t_recon, data_range=t_target.max()).item()
        
        # TV match: penalize deviation from ground truth TV
        tv_target = total_variation(t_target).item()
        tv_recon = total_variation(t_recon).item()
        tv_ratio = tv_recon / (tv_target + 1e-8)
        tv_match = max(0.0, 1.0 - abs(tv_ratio - 1.0))
        
        # Step penalty
        step_cost = self.reward_step_penalty * self._step_count
        
        reward = (self.reward_ssim_weight * ssim_val
                  + self.reward_tv_weight * tv_match
                  - step_cost)
        
        return float(reward)
    
    def _compute_step_reward(self, already_sampled: bool) -> float:
        """Compute per-step reward for dense reward mode."""
        if already_sampled:
            return -0.1  # Penalty for wasting a step
        
        # Small incremental reward based on energy captured
        energy_ratio = (np.sum(np.abs(self._sampled_kspace) ** 2) /
                        (np.sum(np.abs(self._target_kspace) ** 2) + 1e-8))
        return 0.01 * energy_ratio - self.reward_step_penalty
    
    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute final evaluation metrics at episode end."""
        from utils.metrics import psnr, ssim as ssim_fn
        
        recon_image = self._reconstructor(self._sampled_kspace, self._mask)
        target = self._target_image
        
        t_target = torch.from_numpy(target).float()
        t_recon = torch.from_numpy(recon_image).float()
        
        return {
            "final_psnr": psnr(t_target, t_recon, data_range=t_target.max()).item(),
            "final_ssim": ssim_fn(t_target, t_recon, data_range=t_target.max()).item(),
            "sampling_ratio": self._mask.sum() / len(self._mask),
            "acceleration_factor": len(self._mask) / max(1, self._mask.sum()),
            "num_steps": self._step_count,
            "num_unique_lines": int(self._mask.sum()),
            "action_history": self._action_history.copy(),
        }
    
    # ==================================================================
    # Reconstruction
    # ==================================================================
    
    def _default_ifft_reconstructor(self, kspace: np.ndarray,
                                     mask: np.ndarray) -> np.ndarray:
        """Default zero-filled IFFT reconstruction."""
        return np.abs(self._ifft2c(kspace))
    
    def set_reconstructor(self, reconstructor: Callable):
        """
        Switch the reconstruction method (e.g., plug in frozen U-Net).
        
        Args:
            reconstructor: Callable(kspace, mask) → image (numpy).
        """
        self._reconstructor = reconstructor
        logger.info(f"Reconstructor updated to: {reconstructor}")
    
    # FFT Helpers (numpy, no torch dependency for env speed)
    
    @staticmethod
    def _ifft2c(kspace: np.ndarray) -> np.ndarray:
        """Centered inverse 2D FFT."""
        return np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace, axes=(-2, -1)),
                axes=(-2, -1), norm="ortho"
            ),
            axes=(-2, -1)
        )
    
    @staticmethod
    def _fft2c(image: np.ndarray) -> np.ndarray:
        """Centered forward 2D FFT."""
        return np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(image, axes=(-2, -1)),
                axes=(-2, -1), norm="ortho"
            ),
            axes=(-2, -1)
        )
    
    # Synthetic Data (for testing without real data)
    
    def _generate_synthetic_kspace(self) -> np.ndarray:
        """
        Generate a synthetic Shepp-Logan-like phantom in k-space.
        
        Used for environment testing before real data is available.
        """
        H, W = self.kspace_height, self.kspace_width
        image = np.zeros((H, W), dtype=float)
        
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        
        # Outer ellipse (body)
        r_outer = ((x - cx) / (W * 0.4)) ** 2 + ((y - cy) / (H * 0.35)) ** 2
        image[r_outer <= 1.0] = 0.6
        
        # Inner brighter region (tissue)
        r_inner = ((x - cx) / (W * 0.25)) ** 2 + ((y - cy) / (H * 0.2)) ** 2
        image[r_inner <= 1.0] = 0.9
        
        # Small bright lesion (simulating a tumor)
        lesion_cy, lesion_cx = cy + H // 8, cx + W // 10
        r_lesion = ((x - lesion_cx) / (W * 0.05)) ** 2 + ((y - lesion_cy) / (H * 0.04)) ** 2
        image[r_lesion <= 1.0] = 1.2
        
        # Small dark region
        dark_cy, dark_cx = cy - H // 10, cx - W // 8
        r_dark = ((x - dark_cx) / (W * 0.04)) ** 2 + ((y - dark_cy) / (H * 0.05)) ** 2
        image[r_dark <= 1.0] = 0.3
        
        # Add slight texture noise for realism
        rng = np.random.RandomState(42)
        image += rng.randn(H, W) * 0.02
        image = np.clip(image, 0, None)
        
        # Convert to k-space
        return self._fft2c(image)
    
    # Info & Rendering
    
    def _build_info(self) -> Dict[str, Any]:
        """Build the info dictionary."""
        return {
            "step_count": self._step_count,
            "sampling_ratio": self._mask.sum() / len(self._mask),
            "num_sampled_lines": int(self._mask.sum()),
            "action_mask": self.get_action_mask(),
        }
    
    def render(self):
        """Render the current environment state."""
        if self.render_mode is None:
            return
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Ground truth
        axes[0].imshow(self._target_image, cmap='gray')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        # Sampling mask
        mask_2d = np.tile(self._mask[:, np.newaxis], (1, self.kspace_width))
        axes[1].imshow(mask_2d, cmap='Blues')
        axes[1].set_title(f'Mask ({self._mask.sum()}/{len(self._mask)} lines)')
        axes[1].axis('off')
        
        # K-space (log magnitude)
        axes[2].imshow(np.log1p(np.abs(self._sampled_kspace)), cmap='gray')
        axes[2].set_title('Sampled K-space')
        axes[2].axis('off')
        
        # Reconstruction
        recon = self._reconstructor(self._sampled_kspace, self._mask)
        axes[3].imshow(recon, cmap='gray')
        axes[3].set_title(f'Reconstruction (step {self._step_count})')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if self.render_mode == "human":
            plt.show()
            return None
        elif self.render_mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.close(fig)
            return img[:, :, :3]


# Factory function for creating env from config

def make_env(config=None, **kwargs) -> KSpaceSamplingEnv:
    """
    Create a KSpaceSamplingEnv from config or kwargs.
    
    Args:
        config: Config object (uses config.env fields).
        **kwargs: Override any env parameter.
        
    Returns:
        Configured KSpaceSamplingEnv instance.
    """
    if config is not None:
        env_kwargs = {
            "kspace_height": config.env.kspace_height,
            "kspace_width": config.env.kspace_width,
            "max_steps": config.env.max_steps,
            "center_fraction": config.env.center_fraction,
            "reward_ssim_weight": config.env.reward_ssim_weight,
            "reward_tv_weight": config.env.reward_tv_weight,
            "reward_step_penalty": config.env.reward_step_penalty,
            "sparse_reward": config.env.sparse_reward,
        }
        env_kwargs.update(kwargs)
    else:
        env_kwargs = kwargs
    
    return KSpaceSamplingEnv(**env_kwargs)