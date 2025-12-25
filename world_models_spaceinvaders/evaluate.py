"""
Evaluation Script
Evaluate trained World Models agent and record videos
"""

import os
import argparse
import torch
import numpy as np
import wandb
from tqdm import tqdm
import json

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller, WorldModelAgent
from utils.wrappers import make_env


def evaluate(
    agent: WorldModelAgent,
    env,
    device: torch.device,
    num_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True
) -> dict:
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained WorldModelAgent
        env: Gymnasium environment
        device: Torch device
        num_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    agent.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating")
    
    for ep in iterator:
        obs, info = env.reset()
        agent.reset(1, device)
        
        total_reward = 0
        steps = 0
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.act(obs_tensor, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            iterator.set_postfix({
                'reward': total_reward,
                'length': steps,
                'avg_reward': np.mean(episode_rewards)
            })
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def record_video(
    agent: WorldModelAgent,
    env_name: str,
    frame_size: int,
    device: torch.device,
    video_folder: str,
    num_episodes: int = 5,
    deterministic: bool = True
) -> list:
    """
    Record videos of trained agent.
    
    Returns:
        List of video file paths
    """
    os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with video recording
    env = make_env(
        env_name,
        frame_size=frame_size,
        record_video=True,
        video_folder=video_folder,
        episode_trigger=lambda x: True,  # Record all episodes
        render_mode="rgb_array"
    )
    
    agent.eval()
    video_files = []
    
    print(f"Recording {num_episodes} episodes to {video_folder}")
    
    for ep in tqdm(range(num_episodes), desc="Recording"):
        obs, info = env.reset()
        agent.reset(1, device)
        
        total_reward = 0
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.act(obs_tensor, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Episode {ep+1}: Reward = {total_reward}")
    
    env.close()
    
    # Find recorded video files
    for f in os.listdir(video_folder):
        if f.endswith('.mp4'):
            video_files.append(os.path.join(video_folder, f))
    
    return video_files


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.video_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"evaluation_{args.seed}",
            config=vars(args)
        )
    
    # Load VAE
    print("Loading VAE...")
    vae = VAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    vae_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'vae_best.pt'),
        map_location=device,
        weights_only=False
    )
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    
    # Load MDN-RNN
    print("Loading MDN-RNN...")
    mdn_rnn = MDNRNN(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures
    ).to(device)
    
    mdn_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'mdn_rnn_best.pt'),
        map_location=device,
        weights_only=False
    )
    mdn_rnn.load_state_dict(mdn_checkpoint['model_state_dict'])
    mdn_rnn.eval()
    
    # Load Controller
    print("Loading Controller...")
    controller = Controller(
        latent_dim=args.latent_dim,
        rnn_hidden_size=args.hidden_size,
        hidden_size=args.controller_hidden_size,
        action_dim=args.action_dim
    ).to(device)
    
    controller_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'controller_best.pt'),
        map_location=device,
        weights_only=False
    )
    controller.load_state_dict(controller_checkpoint['controller_state_dict'])
    controller.eval()
    
    # Create agent
    agent = WorldModelAgent(vae, mdn_rnn, controller)
    
    print("\nModels loaded successfully!")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"MDN-RNN parameters: {sum(p.numel() for p in mdn_rnn.parameters()):,}")
    print(f"Controller parameters: {sum(p.numel() for p in controller.parameters()):,}")
    
    # Create evaluation environment
    env = make_env(args.env_name, frame_size=args.frame_size)
    
    # Evaluate
    print("\n=== Evaluation ===")
    eval_results = evaluate(
        agent, env, device,
        num_episodes=args.num_episodes,
        deterministic=True
    )
    
    print(f"\nResults:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Min Reward:  {eval_results['min_reward']:.2f}")
    print(f"  Max Reward:  {eval_results['max_reward']:.2f}")
    print(f"  Mean Length: {eval_results['mean_length']:.2f}")
    
    env.close()
    
    # Record videos
    if args.record_video:
        print("\n=== Recording Videos ===")
        video_files = record_video(
            agent,
            args.env_name,
            args.frame_size,
            device,
            args.video_dir,
            num_episodes=min(5, args.num_episodes)
        )
        print(f"Recorded {len(video_files)} videos")
    
    # Log to wandb
    if not args.no_wandb:
        wandb.log({
            'eval/mean_reward': eval_results['mean_reward'],
            'eval/std_reward': eval_results['std_reward'],
            'eval/min_reward': eval_results['min_reward'],
            'eval/max_reward': eval_results['max_reward'],
            'eval/mean_length': eval_results['mean_length']
        })
        
        # Log videos
        if args.record_video and video_files:
            for i, video_path in enumerate(video_files[:3]):  # Log up to 3 videos
                try:
                    wandb.log({f"video/episode_{i}": wandb.Video(video_path)})
                except Exception as e:
                    print(f"Could not log video {video_path}: {e}")
        
        wandb.finish()
    
    # Save results
    results_path = os.path.join(args.checkpoint_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy values to Python types
        save_results = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in eval_results.items()
        }
        save_results = {
            k: float(v) if isinstance(v, np.floating) else v
            for k, v in save_results.items()
        }
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate World Models Agent")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Model architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--controller-hidden-size", type=int, default=128)  # Increased from 64
    
    # Evaluation
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--record-video", action="store_true", default=True)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)
