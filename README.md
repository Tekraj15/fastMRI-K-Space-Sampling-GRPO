
# Deep RL for Spatio-Temporal Medical Imaging Analysis: Adaptive K-Space Sampling via GRPO

## Problem Statement
Dynamic Magnetic Resonance Imaging (MRI) is a cornerstone of medical diagnostics, particularly for assessing complex conditions like breast cancer. However, its clinical utility is often hampered by long acquisition times[cite: 4]. To accelerate scanning, k-space is typically undersampled, which transforms the reconstruction into a mathematically ill-posed inverse problem[cite: 19, 20]. While Deep Reinforcement Learning (DRL) is ideal for learning adaptive sequential sampling policies, current state-of-the-art methods (like PPO) suffer from significant computational overhead and memory bottlenecks due to their reliance on an explicit "critic" network in high-dimensional medical imaging spaces[cite: 6, 28, 29].

## Introduction
This project introduces a novel framework for adaptive k-space sampling using **Group Relative Policy Optimization (GRPO)**. Originating from large language model training, GRPO is a cutting-edge, critic-less RL algorithm that estimates policy advantages relative to a group of sampled trajectories rather than an absolute value function[cite: 31, 32, 33]. By applying this to MRI acquisition, we aim to dynamically focus on the most informative regions of k-space[cite: 24], accelerating scan times while preserving diagnostic image quality. 

## Objective
To develop a computationally efficient, stable, and intelligent medical image acquisition agent using GRPO. The agent will learn a policy $\pi(a|s)$ to sequentially select k-space lines, optimizing for high acceleration factors without the memory overhead of traditional actor-critic DRL architectures.

## Methodology
Our framework formulates the sequential k-space acquisition as a Markov Decision Process (MDP)[cite: 90]:
* **State Space ($S$):** A multi-channel tensor combining the partially acquired k-space, the binary sampling mask, and a prior zero-filled reconstruction[cite: 94, 95, 96].
* **Action Space ($A$):** The agent's choice of the next un-sampled phase-encoding line in k-space[cite: 97, 98].
* **Reward Function ($R$):** A sparse, episode-level hybrid reward incorporating Structural Similarity Index (SSIM) for perceptual quality, Total Variation (TV) for smoothness, and a step penalty for efficiency[cite: 101, 103, 105, 106].
* **Optimization:** The GRPO agent generates $G$ candidate sampling trajectories per state. The advantage of each trajectory is calculated relative to the group's empirical mean and standard deviation, updating the policy via a clipped surrogate objective with KL-divergence regularization[cite: 75, 77, 128, 140].
* **Reconstruction:** A pre-trained, frozen U-Net architecture reconstructs the final image from the undersampled k-space data[cite: 152, 153].

## Dataset
This project is targeting **Breast Cancer MRI data**. (Accessed form https://fastmri.med.nyu.edu/ by manually downloading.)

Note: The standard NYU fastMRI dataset primarily features knee, brain, and prostate scans, we will adapt fastMRI-style data loaders to process publicly available breast MRI datasets, ensuring complex k-space data and multi-coil sensitivities are properly formatted.

## Expected Results
* **Superior Efficiency:** The GRPO agent is expected to reach peak performance in significantly fewer training epochs and with a much lower memory footprint compared to a baseline PPO agent.
* **High Acceleration:** Achieving high acceleration factors (e.g., 8x, 12x) while maintaining diagnostic-quality image reconstructions.
* **Adaptive Strategies:** The agent is expected to learn physically meaningful sampling masks, prioritizing low-frequency central lines before adaptively selecting high-frequency peripheral lines based on specific anatomical content.

## Evaluation Metrics
The agent's performance will be benchmarked against static sampling methods (Random, Variable-Density) and a PPO baseline using:
1.  **Reconstruction Quality:** Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
2.  **Efficiency:** Target Acceleration Factor.

## Technical Frameworks
* **Deep Learning:** PyTorch
* **Reinforcement Learning:** Ray RLlib (for parallelized group trajectory sampling)
* **Medical Imaging:** fastMRI API, MONAI
* **Experiment Tracking:** Weights & Biases (W&B)

## Project Structure
```text
fastMRI-K-Space-Sampling-GRPO/
├── agents/                  # GRPO agent logic, policy networks (ResNet+MLP), and RLlib configs
├── data/                    # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── envs/                    # Custom Gymnasium MDP environments
│   ├── data_loader.py       # fastMRI / Breast MRI data ingestion
│   └── mri_env.py           # The simulated MRI acquisition environment
├── models/                  # Pre-trained reconstruction networks
│   └── unet_reconstructor.py# Frozen U-Net for reward calculation
├── notebooks/               # Jupyter notebooks for EDA and mask visualization
├── scripts/                 # Entry points for execution
│   ├── train.py             # Main RL training loop
│   └── evaluate.py          # Script for testing and metric generation
├── utils/                   # Helper functions
│   ├── metrics.py           # PSNR, SSIM, TV calculators
│   └── transforms.py        # Fourier transforms and complex data handling
├── results/                 # Checkpoints, tensorboard logs, and generated masks
├── config.py                # Global hyperparameters (Group size, learning rates)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation