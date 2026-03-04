"""
Evaluates three static k-space sampling strategies against the environment:
1. Random Uniform Sampling
2. Variable-Density Sampling  
3. Equispaced Sampling
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.rl_environment import KSpaceSamplingEnv
from utils.metrics import psnr, ssim, total_variation, psnr_numpy, ssim_numpy
from utils.transforms import (
    create_random_mask, 
    create_variable_density_mask, 
    create_equispaced_mask,
    ifft2c_numpy,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# =========================================================================
# Baseline Strategies
# =========================================================================

class BaselineSampler:
    """Base class for static sampling strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_action_sequence(self, num_lines: int, budget: int,
                            center_fraction: float = 0.08,
                            seed: int = 0) -> List[int]:
        """
        Return an ordered list of line indices to sample.
        
        Args:
            num_lines: Total phase-encoding lines.
            budget: How many lines the agent is allowed to pick
                    (excluding auto-sampled center).
            center_fraction: Fraction of center lines that are auto-sampled.
            seed: Random seed.
            
        Returns:
            List of line indices (those NOT already center-sampled).
        """
        raise NotImplementedError


class RandomSampler(BaselineSampler):
    """Uniformly random sampling of k-space lines."""
    
    def __init__(self):
        super().__init__("Random")
    
    def get_action_sequence(self, num_lines, budget, center_fraction=0.08, seed=0):
        rng = np.random.RandomState(seed)
        
        # Determine which lines are already center-sampled
        num_center = max(1, int(center_fraction * num_lines))
        center_start = (num_lines - num_center) // 2
        center_lines = set(range(center_start, center_start + num_center))
        
        # Available outer lines
        available = [i for i in range(num_lines) if i not in center_lines]
        rng.shuffle(available)
        
        return available[:budget]


class VariableDensitySampler(BaselineSampler):
    """Variable-density sampling — higher probability near center."""
    
    def __init__(self, pdf_power: float = 2.0):
        super().__init__("Variable-Density")
        self.pdf_power = pdf_power
    
    def get_action_sequence(self, num_lines, budget, center_fraction=0.08, seed=0):
        rng = np.random.RandomState(seed)
        
        num_center = max(1, int(center_fraction * num_lines))
        center_start = (num_lines - num_center) // 2
        center_lines = set(range(center_start, center_start + num_center))
        
        available = np.array([i for i in range(num_lines) if i not in center_lines])
        
        # Probability proportional to closeness to center
        center = num_lines / 2.0
        distances = np.abs(available - center) / center  # [0, 1]
        probs = (1.0 - distances) ** self.pdf_power
        probs /= probs.sum()
        
        chosen = rng.choice(available, size=min(budget, len(available)),
                            replace=False, p=probs)
        return chosen.tolist()


class EquispacedSampler(BaselineSampler):
    """Equispaced (uniform grid) sampling."""
    
    def __init__(self):
        super().__init__("Equispaced")
    
    def get_action_sequence(self, num_lines, budget, center_fraction=0.08, seed=0):
        num_center = max(1, int(center_fraction * num_lines))
        center_start = (num_lines - num_center) // 2
        center_lines = set(range(center_start, center_start + num_center))
        
        available = [i for i in range(num_lines) if i not in center_lines]
        
        if budget >= len(available):
            return available
        
        # Pick equispaced indices from the available lines
        step = max(1, len(available) // budget)
        # Add offset from seed for variety
        offset = seed % step
        chosen = available[offset::step][:budget]
        
        return chosen


# =========================================================================
# Benchmark Runner
# =========================================================================

def run_baseline_episode(env: KSpaceSamplingEnv,
                         sampler: BaselineSampler,
                         budget: int,
                         target_kspace: np.ndarray = None,
                         seed: int = 0) -> Dict:
    """
    Run a single episode with a static baseline sampler.
    
    Args:
        env: The k-space sampling environment.
        sampler: A BaselineSampler.
        budget: Number of lines to sample (agent steps).
        target_kspace: Optional real k-space data.
        seed: Random seed.
        
    Returns:
        Dictionary with metrics and episode info.
    """
    options = {"target_kspace": target_kspace} if target_kspace is not None else None
    obs, info = env.reset(seed=seed, options=options)
    
    # Get the action sequence from the sampler
    actions = sampler.get_action_sequence(
        num_lines=env.kspace_height,
        budget=budget,
        center_fraction=env.center_fraction,
        seed=seed,
    )
    
    # Step through the actions
    total_reward = 0.0
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    # If episode didn't terminate naturally, force a termination to get metrics
    if not terminated and not truncated:
        # Run remaining steps with dummy actions to reach termination
        while not terminated and not truncated:
            valid = np.where(env.get_action_mask())[0]
            if len(valid) == 0:
                break
            obs, reward, terminated, truncated, info = env.step(valid[0])
            total_reward += reward
    
    result = {
        "sampler": sampler.name,
        "budget": budget,
        "seed": seed,
        "total_reward": total_reward,
    }
    
    # Add final metrics if present
    for key in ["final_psnr", "final_ssim", "sampling_ratio", 
                "acceleration_factor", "num_steps", "num_unique_lines"]:
        if key in info:
            result[key] = info[key]
    
    return result


def run_benchmark(kspace_height: int = 128,
                  kspace_width: int = 128,
                  acceleration_factors: List[float] = None,
                  num_seeds: int = 5,
                  target_kspace: np.ndarray = None,
                  output_dir: str = None) -> List[Dict]:
    """
    Run the full baseline benchmark across strategies and accelerations.
    
    Args:
        kspace_height: Height of k-space.
        kspace_width: Width of k-space.
        acceleration_factors: List of acceleration factors to test.
        num_seeds: Number of random seeds for statistical significance.
        target_kspace: Optional real k-space data.
        output_dir: Directory to save results.
        
    Returns:
        List of result dictionaries.
    """
    if acceleration_factors is None:
        acceleration_factors = [4.0, 8.0, 12.0]
    
    samplers = [
        RandomSampler(),
        VariableDensitySampler(pdf_power=2.0),
        EquispacedSampler(),
    ]
    
    all_results = []
    
    for accel in acceleration_factors:
        # Budget = total lines / accel factor (minus center lines)
        num_center = max(1, int(0.08 * kspace_height))
        total_budget = int(kspace_height / accel)
        agent_budget = max(1, total_budget - num_center)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Acceleration: {accel}x | Total lines: {total_budget} | "
                     f"Agent budget: {agent_budget}")
        logger.info(f"{'='*60}")
        
        for sampler in samplers:
            psnr_vals = []
            ssim_vals = []
            reward_vals = []
            
            for seed in range(num_seeds):
                env = KSpaceSamplingEnv(
                    kspace_height=kspace_height,
                    kspace_width=kspace_width,
                    max_steps=agent_budget,
                    center_fraction=0.08,
                    sparse_reward=True,
                )
                
                result = run_baseline_episode(
                    env, sampler, agent_budget,
                    target_kspace=target_kspace, seed=seed,
                )
                
                all_results.append(result)
                
                if "final_psnr" in result:
                    psnr_vals.append(result["final_psnr"])
                if "final_ssim" in result:
                    ssim_vals.append(result["final_ssim"])
                reward_vals.append(result["total_reward"])
            
            # Log summary
            if psnr_vals:
                logger.info(
                    f"  {sampler.name:20s} | "
                    f"PSNR: {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB | "
                    f"SSIM: {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f} | "
                    f"Reward: {np.mean(reward_vals):.4f} ± {np.std(reward_vals):.4f}"
                )
    
    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        results_file = out_path / "baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_file}")
        
        # Also save a summary table
        summary = _create_summary_table(all_results)
        summary_file = out_path / "baseline_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)
        logger.info(f"Summary saved to {summary_file}")
    
    return all_results


def _create_summary_table(results: List[Dict]) -> str:
    """Create a formatted summary table from benchmark results."""
    import itertools
    
    lines = []
    lines.append("=" * 80)
    lines.append("STATIC BASELINES BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by acceleration factor
    accel_groups = {}
    for r in results:
        # Infer acceleration from budget and sampling_ratio
        accel = r.get("acceleration_factor", "?")
        key = f"{float(accel):.1f}x" if isinstance(accel, (int, float)) else str(accel)
        if key not in accel_groups:
            accel_groups[key] = {}
        
        sampler = r["sampler"]
        if sampler not in accel_groups[key]:
            accel_groups[key][sampler] = {"psnr": [], "ssim": [], "reward": []}
        
        if "final_psnr" in r:
            accel_groups[key][sampler]["psnr"].append(r["final_psnr"])
        if "final_ssim" in r:
            accel_groups[key][sampler]["ssim"].append(r["final_ssim"])
        accel_groups[key][sampler]["reward"].append(r["total_reward"])
    
    for accel_key, samplers in sorted(accel_groups.items()):
        lines.append(f"\nAcceleration: {accel_key}")
        lines.append("-" * 70)
        lines.append(f"{'Sampler':20s} | {'PSNR (dB)':15s} | {'SSIM':15s} | {'Reward':15s}")
        lines.append("-" * 70)
        
        for sampler_name, metrics in sorted(samplers.items()):
            psnr_str = (f"{np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f}" 
                         if metrics['psnr'] else "N/A")
            ssim_str = (f"{np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}" 
                         if metrics['ssim'] else "N/A")
            rew_str = f"{np.mean(metrics['reward']):.4f} ± {np.std(metrics['reward']):.4f}"
            
            lines.append(f"{sampler_name:20s} | {psnr_str:15s} | {ssim_str:15s} | {rew_str:15s}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run static baselines benchmark")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to fastMRI .h5 data. If not provided, uses synthetic.")
    parser.add_argument("--output", type=str, default="results/baselines",
                        help="Output directory for results.")
    parser.add_argument("--kspace_height", type=int, default=128,
                        help="K-space height (default 128 for speed).")
    parser.add_argument("--kspace_width", type=int, default=128,
                        help="K-space width.")
    parser.add_argument("--accelerations", type=float, nargs="+", 
                        default=[4.0, 8.0, 12.0],
                        help="Acceleration factors to test.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds.")
    args = parser.parse_args()
    
    target_kspace = None
    
    # Load real data if provided
    if args.data_path:
        try:
            from env.data_loader import FastMRIBreastDataLoader
            loader = FastMRIBreastDataLoader(args.data_path)
            vol = loader.load_volume(0)
            # Use first slice, combine coils
            kspace_mc = vol['kspace'][0]  # (coils, H, W)
            target_kspace = loader.combine_to_single_coil_kspace(kspace_mc)
            
            # Update dimensions to match
            args.kspace_height, args.kspace_width = target_kspace.shape
            logger.info(f"Using real data: shape={target_kspace.shape}")
        except Exception as e:
            logger.warning(f"Could not load real data: {e}. Using synthetic.")
    
    results = run_benchmark(
        kspace_height=args.kspace_height,
        kspace_width=args.kspace_width,
        acceleration_factors=args.accelerations,
        num_seeds=args.seeds,
        target_kspace=target_kspace,
        output_dir=args.output,
    )
    
    logger.info(f"\nBenchmark complete. {len(results)} total runs.")


if __name__ == "__main__":
    main()
