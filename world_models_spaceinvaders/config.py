"""
World Models Configuration
Hyperparameters and settings for training World Models on Space Invaders
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VAEConfig:
    """VAE (Vision Model) configuration"""
    input_channels: int = 3
    input_size: int = 64  # 64x64 input images
    latent_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kl_weight: float = 0.0001  # β-VAE weight
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 30


@dataclass
class MDNRNNConfig:
    """MDN-RNN (Memory Model) configuration"""
    latent_dim: int = 32
    action_dim: int = 6  # Space Invaders has 6 actions
    hidden_size: int = 256
    num_layers: int = 1
    num_mixtures: int = 5
    temperature: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    sequence_length: int = 100
    num_epochs: int = 30


@dataclass
class ControllerConfig:
    """Controller (Policy) configuration"""
    latent_dim: int = 32
    rnn_hidden_size: int = 256
    hidden_size: int = 128  # Increased from 64
    action_dim: int = 6
    learning_rate: float = 1e-3
    gamma: float = 0.99
    entropy_weight: float = 0.02  # Increased for better exploration


@dataclass
class TrainingConfig:
    """Overall training configuration"""
    # Environment
    env_name: str = "ALE/SpaceInvaders-v5"
    frame_size: int = 64
    frame_skip: int = 4
    
    # Data collection
    num_rollouts: int = 100
    max_episode_steps: int = 1000
    
    # Training phases
    vae_epochs: int = 30
    mdn_epochs: int = 30
    controller_epochs: int = 100
    
    # Dream training
    dream_rollouts: int = 16
    dream_steps: int = 100
    
    # Evaluation
    eval_episodes: int = 10
    record_video: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    video_dir: str = "videos"
    data_dir: str = "data"
    
    # Wandb
    wandb_project: str = "world-models-spaceinvaders"
    wandb_entity: Optional[str] = None
    
    # Random seed
    seed: int = 42


# Default configurations
vae_config = VAEConfig()
mdn_config = MDNRNNConfig()
controller_config = ControllerConfig()
training_config = TrainingConfig()
