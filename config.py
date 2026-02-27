"""
Global Configs for GRPO K-Space Sampling
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_root: str = "data/raw"
    dataset_name: str = "fastmri_breast"
    max_slices_per_volume: int = 10
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EnvironmentConfig:
    """MRI k-space sampling env configs."""
    # K-space dimensions
    kspace_height: int = 256
    kspace_width: int = 256
    
    # Episode parameters
    max_steps: int = 50
    target_acceleration: float = 8.0  
    center_fraction: float = 0.08     
    
    # State space: 3-channel tensor (partial kspace, mask, zero-filled recon)
    state_channels: int = 3
    
    # Reward weights (SSIM + TV + step penalty)
    reward_ssim_weight: float = 1.0
    reward_tv_weight: float = 0.1
    reward_step_penalty: float = 0.01
    
    # use sparse episode-level reward 
    sparse_reward: bool = True


@dataclass
class ReconstructionConfig:
    """U-Net reconstruction network configuration."""
    # Architecture
    in_channels: int = 1       # Input: magnitude of undersampled kspace
    out_channels: int = 1      # Output: reconstructed magnitude image
    num_pool_layers: int = 4
    channels: int = 32         # Base channel count (doubles each layer)
    drop_prob: float = 0.0
    
    # Pretrained checkpoint
    checkpoint_path: Optional[str] = None
    freeze_weights: bool = True
    
    # Training (only if pre-training from scratch)
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    pretrain_batch_size: int = 16


@dataclass
class PolicyConfig:
    """Policy network (ResNet-18 + MLP) configuration."""
    # Encoder
    encoder_name: str = "resnet18"
    pretrained_encoder: bool = True  # ImageNet pretrained
    encoder_input_channels: int = 3  # Matches state_channels
    
    # MLP head
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    mlp_dropout: float = 0.1
    
    # Output: number of k-space lines (set dynamically from env)
    num_actions: int = 256


@dataclass
class GRPOConfig:
    """GRPO algorithm hyperparameters."""
    # Group sampling
    group_size: int = 16          # G: number of trajectories per group
    
    # Clipped surrogate objective
    clip_epsilon: float = 0.2     # ε: clipping parameter
    
    # KL divergence penalty
    kl_penalty_coeff: float = 0.04  # β: KL penalty coefficient
    kl_target: float = 0.01         # Target KL for adaptive β
    adaptive_kl: bool = True        # Whether to adapt β during training
    
    # Reference policy
    ref_policy_update_interval: int = 10  # Update π_ref every N policy updates
    ref_policy_ema_tau: float = 0.0       # 0.0 = hard copy, >0 = EMA
    
    # Entropy regularization
    entropy_coeff: float = 0.01       # λ_H: entropy bonus coefficient
    entropy_decay: float = 0.995      # Decay per epoch
    min_entropy_coeff: float = 0.001  # Floor


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 0.5
    optimizer: str = "adam"       # "adam" or "adamw"
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    
    # Training schedule
    num_epochs: int = 200
    warmup_epochs: int = 5
    eval_interval: int = 5       # Evaluate every N epochs
    checkpoint_interval: int = 10
    
    # Batch / data
    slices_per_batch: int = 4    # Number of kspace slices per batch
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_schedule: Dict[int, float] = field(default_factory=lambda: {
        0: 4.0,     # Epochs 0-49:   4× acceleration (easy)
        50: 6.0,    # Epochs 50-99:  6× acceleration (medium)
        100: 8.0,   # Epochs 100-149: 8× acceleration (hard)
        150: 12.0,  # Epochs 150+:   12× acceleration (target)
    })
    
    # Parallelization
    num_rollout_workers: int = 4  # Number of parallel trajectory workers
    use_ray: bool = False          # Use Ray for parallelization
    
    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    # Reproducibility
    seed: int = 42


@dataclass
class LoggingConfig:
    """Experiment tracking / logging configuration."""
    project_name: str = "grpo-kspace-sampling"
    entity: Optional[str] = None   # W&B team/user
    run_name: Optional[str] = None
    use_wandb: bool = True
    log_interval: int = 1          # Log every N epochs
    
    # Directories
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"
    visualization_dir: str = "results/visualizations"


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    recon: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def save(self, path: str):
        """Save config to YAML file."""
        import dataclasses
        
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**data.get('data', {})),
            env=EnvironmentConfig(**data.get('env', {})),
            recon=ReconstructionConfig(**data.get('recon', {})),
            policy=PolicyConfig(**data.get('policy', {})),
            grpo=GRPOConfig(**data.get('grpo', {})),
            training=TrainingConfig(**data.get('training', {})),
            logging=LoggingConfig(**data.get('logging', {})),
        )
    
    def get_device(self) -> str:
        """Resolve 'auto' device to best available."""
        import torch
        if self.training.device != "auto":
            return self.training.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# Default configuration instance
def get_default_config() -> Config:
    """Get a Config instance with all default values."""
    return Config()


if __name__ == "__main__":
    config = get_default_config()
    print("Default Configs")
    config.save("/dev/stdout")
