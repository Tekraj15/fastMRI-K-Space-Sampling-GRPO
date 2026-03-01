"""
Experiment Logging Utilities
=============================

Provides a unified logging interface wrapping W&B and local file logging.
Used throughout the training pipeline for metrics, artifacts, and visualizations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Unified experiment logger supporting W&B and local file logging.
    
    Usage:
        logger = ExperimentLogger(config)
        logger.log_metrics({"loss": 0.5, "ssim": 0.8}, step=10)
        logger.log_image("mask", mask_array, step=10)
        logger.finish()
    """
    
    def __init__(self, config, run_name: Optional[str] = None):
        """
        Initialize the experiment logger.
        
        Args:
            config: LoggingConfig or full Config object.
            run_name: Optional override for W&B run name.
        """
        from config import LoggingConfig
        
        if hasattr(config, 'logging'):
            self.log_config = config.logging
            self.full_config = config
        else:
            self.log_config = config
            self.full_config = None
        
        self.use_wandb = self.log_config.use_wandb
        self.run = None
        
        # Create local log directories
        self.log_dir = Path(self.log_config.log_dir)
        self.checkpoint_dir = Path(self.log_config.checkpoint_dir)
        self.viz_dir = Path(self.log_config.visualization_dir)
        
        for d in [self.log_dir, self.checkpoint_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Local metrics file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.jsonl"
        
        # Initialize W&B if enabled
        if self.use_wandb:
            self._init_wandb(run_name)
        
        logger.info(f"ExperimentLogger initialized. W&B: {self.use_wandb}, "
                     f"Log dir: {self.log_dir}")
    
    def _init_wandb(self, run_name: Optional[str] = None):
        """Initialize Weights & Biases."""
        try:
            import wandb
            
            name = run_name or self.log_config.run_name
            if name is None:
                name = f"grpo_{datetime.now().strftime('%m%d_%H%M')}"
            
            # Convert config to dict for W&B
            wandb_config = {}
            if self.full_config is not None:
                import dataclasses
                wandb_config = dataclasses.asdict(self.full_config)
            
            self.run = wandb.init(
                project=self.log_config.project_name,
                entity=self.log_config.entity,
                name=name,
                config=wandb_config,
                reinit=True,
            )
            logger.info(f"W&B initialized: {self.run.url}")
            
        except ImportError:
            logger.warning("wandb not installed. Falling back to local logging only.")
            self.use_wandb = False
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Falling back to local logging.")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log scalar metrics.
        
        Args:
            metrics: Dictionary of metric name → value.
            step: Global step number.
        """
        # Local logging
        record = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        # W&B logging
        if self.use_wandb and self.run is not None:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_image(self, key: str, image, step: Optional[int] = None,
                  caption: Optional[str] = None):
        """
        Log an image (numpy array or matplotlib figure).
        
        Args:
            key: Image identifier.
            image: numpy array or matplotlib Figure.
            step: Global step number.
            caption: Optional caption.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Save locally
        save_path = self.viz_dir / f"{key}_step{step}.png"
        if hasattr(image, 'savefig'):
            image.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(image)
        else:
            plt.imsave(str(save_path), image, cmap='gray')
        
        # W&B logging
        if self.use_wandb and self.run is not None:
            import wandb
            wandb.log({key: wandb.Image(str(save_path), caption=caption)}, step=step)
    
    def log_histogram(self, key: str, values, step: Optional[int] = None):
        """Log a histogram of values."""
        if self.use_wandb and self.run is not None:
            import wandb
            wandb.log({key: wandb.Histogram(values)}, step=step)
    
    def save_checkpoint(self, state_dict: Dict[str, Any], name: str,
                         epoch: int, metrics: Optional[Dict] = None):
        """
        Save a model checkpoint.
        
        Args:
            state_dict: Model state dict (or full checkpoint dict).
            name: Checkpoint name prefix.
            epoch: Current epoch.
            metrics: Optional metrics to save alongside.
        """
        import torch
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': state_dict,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        path = self.checkpoint_dir / f"{name}_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
        # Also save as 'best' if this is the best so far
        if metrics:
            best_path = self.checkpoint_dir / f"{name}_best.pt"
            # Simple: always overwrite best (caller should check before calling)
            torch.save(checkpoint, best_path)
        
        # Log to W&B
        if self.use_wandb and self.run is not None:
            import wandb
            wandb.save(str(path))
    
    def finish(self):
        """Finalize logging (close W&B run)."""
        if self.use_wandb and self.run is not None:
            import wandb
            wandb.finish()
            logger.info("W&B run finished.")
