"""
Comparison Script for World Models vs Model-Free Baselines
Compares World Models with DDQN, SAC, and PPO on Space Invaders
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller, WorldModelAgent
from baselines.ddqn import DDQNAgent
from baselines.ppo import PPOAgent
from baselines.sac import SACAgent
from utils.wrappers import make_env


def evaluate_agent(env, agent, device, num_episodes=10, agent_type="world_models"):
    """
    Evaluate an agent on the environment.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent (WorldModelAgent, DDQNAgent, PPOAgent, or SACAgent)
        device: Torch device
        num_episodes: Number of evaluation episodes
        agent_type: Type of agent for proper handling
        
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for ep in tqdm(range(num_episodes), desc=f"Evaluating {agent_type}"):
        obs, info = env.reset()
        
        # Reset agent state if needed
        if agent_type == "world_models":
            agent.reset(1, device)
        
        total_reward = 0
        steps = 0
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Get action based on agent type
            if agent_type == "world_models":
                action = agent.act(obs_tensor, deterministic=True)
            elif agent_type == "ddqn":
                action = agent.select_action(obs_tensor, eval_mode=True)
            elif agent_type == "ppo":
                with torch.no_grad():
                    action_tensor, _, _, _ = agent.policy.get_action(obs_tensor, deterministic=True)
                action = action_tensor.item()
            elif agent_type == "sac":
                action = agent.select_action(obs_tensor, eval_mode=True)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
    
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


def load_world_models_agent(checkpoint_dir, device, args):
    """Load trained World Models agent."""
    # Load VAE
    vae = VAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=[32, 64, 128, 256]
    ).to(device)
    
    vae_path = os.path.join(checkpoint_dir, 'vae_best.pt')
    if os.path.exists(vae_path):
        vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
        vae.load_state_dict(vae_checkpoint['model_state_dict'])
    else:
        print(f"Warning: VAE checkpoint not found at {vae_path}")
        return None
    
    vae.eval()
    
    # Load MDN-RNN
    mdn_rnn = MDNRNN(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        num_layers=1,
        num_mixtures=5
    ).to(device)
    
    mdn_path = os.path.join(checkpoint_dir, 'mdn_rnn_best.pt')
    if os.path.exists(mdn_path):
        mdn_checkpoint = torch.load(mdn_path, map_location=device, weights_only=False)
        mdn_rnn.load_state_dict(mdn_checkpoint['model_state_dict'])
    else:
        print(f"Warning: MDN-RNN checkpoint not found at {mdn_path}")
        return None
    
    mdn_rnn.eval()
    
    # Load Controller
    controller = Controller(
        latent_dim=args.latent_dim,
        rnn_hidden_size=args.hidden_size,
        hidden_size=128,
        action_dim=args.action_dim
    ).to(device)
    
    controller_path = os.path.join(checkpoint_dir, 'controller_best.pt')
    if os.path.exists(controller_path):
        controller_checkpoint = torch.load(controller_path, map_location=device, weights_only=False)
        controller.load_state_dict(controller_checkpoint['controller_state_dict'])
    else:
        print(f"Warning: Controller checkpoint not found at {controller_path}")
        return None
    
    controller.eval()
    
    return WorldModelAgent(vae, mdn_rnn, controller)


def load_ddqn_agent(checkpoint_dir, device, action_dim):
    """Load trained DDQN agent."""
    agent = DDQNAgent(action_dim=action_dim, device=device)
    
    ddqn_path = os.path.join(checkpoint_dir, 'ddqn_best.pt')
    if os.path.exists(ddqn_path):
        agent.load(ddqn_path)
        agent.policy_net.eval()
        return agent
    else:
        print(f"Warning: DDQN checkpoint not found at {ddqn_path}")
        return None


def load_ppo_agent(checkpoint_dir, device, action_dim):
    """Load trained PPO agent."""
    agent = PPOAgent(action_dim=action_dim, device=device)
    
    ppo_path = os.path.join(checkpoint_dir, 'ppo_best.pt')
    if os.path.exists(ppo_path):
        agent.load(ppo_path)
        agent.policy.eval()
        return agent
    else:
        print(f"Warning: PPO checkpoint not found at {ppo_path}")
        return None


def load_sac_agent(checkpoint_dir, device, action_dim):
    """Load trained SAC agent."""
    agent = SACAgent(action_dim=action_dim, device=device)
    
    sac_path = os.path.join(checkpoint_dir, 'sac_best.pt')
    if os.path.exists(sac_path):
        agent.load(sac_path)
        agent.policy.eval()
        return agent
    else:
        print(f"Warning: SAC checkpoint not found at {sac_path}")
        return None


def plot_comparison(results, output_dir):
    """Create comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the algorithms that have results
    algorithms = [alg for alg in results.keys() if results[alg] is not None]
    
    if len(algorithms) == 0:
        print("No results to plot!")
        return
    
    # Colors for each algorithm
    colors = {
        'World Models': '#2ecc71',  # Green
        'DDQN': '#3498db',          # Blue
        'PPO': '#e74c3c',           # Red
        'SAC': '#9b59b6'            # Purple
    }
    
    # 1. Bar chart of mean rewards with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = [results[alg]['mean_reward'] for alg in algorithms]
    stds = [results[alg]['std_reward'] for alg in algorithms]
    bar_colors = [colors.get(alg, '#95a5a6') for alg in algorithms]
    
    bars = ax.bar(algorithms, means, yerr=stds, capsize=5, color=bar_colors, 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Comparison of RL Algorithms on Space Invaders', fontsize=14, fontweight='bold')
    ax.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar.png'), dpi=150)
    plt.close()
    
    # 2. Box plot of episode rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [results[alg]['episode_rewards'] for alg in algorithms]
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    for patch, alg in zip(bp['boxes'], algorithms):
        patch.set_facecolor(colors.get(alg, '#95a5a6'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Distribution of Episode Rewards', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_boxplot.png'), dpi=150)
    plt.close()
    
    # 3. Summary table as image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    table_data = []
    for alg in algorithms:
        r = results[alg]
        table_data.append([
            alg,
            f"{r['mean_reward']:.2f} ± {r['std_reward']:.2f}",
            f"{r['min_reward']:.2f}",
            f"{r['max_reward']:.2f}",
            f"{r['mean_length']:.0f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Algorithm', 'Mean ± Std', 'Min', 'Max', 'Mean Length'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color the headers
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color the algorithm names
    for i, alg in enumerate(algorithms):
        table[(i+1, 0)].set_facecolor(colors.get(alg, '#95a5a6'))
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.title('Summary Statistics', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def print_comparison_table(results):
    """Print a formatted comparison table to console."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS: World Models vs Model-Free Baselines")
    print("="*80)
    
    # Header
    print(f"\n{'Algorithm':<15} {'Mean±Std':<20} {'Min':<10} {'Max':<10} {'Avg Length':<12}")
    print("-"*70)
    
    # Sort by mean reward
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1]['mean_reward'],
        reverse=True
    )
    
    for alg, r in sorted_results:
        print(f"{alg:<15} {r['mean_reward']:>7.2f} ± {r['std_reward']:<8.2f} "
              f"{r['min_reward']:<10.2f} {r['max_reward']:<10.2f} {r['mean_length']:<12.0f}")
    
    print("-"*70)
    
    # Highlight winner
    if sorted_results:
        winner = sorted_results[0][0]
        print(f"\n🏆 BEST ALGORITHM: {winner} with mean reward {sorted_results[0][1]['mean_reward']:.2f}")
    
    print("="*80 + "\n")


def main(args):
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    # Create environment
    env = make_env(
        args.env_name, 
        frame_size=args.frame_size,
        record_video=args.record_video,
        video_folder=args.video_dir
    )
    action_dim = env.action_space.n
    print(f"Environment: {args.env_name}")
    print(f"Action space: {action_dim} actions")
    
    # Initialize wandb for comparison
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    results = {}
    
    # =====================================================================
    # 1. Evaluate World Models
    # =====================================================================
    print("\n" + "="*50)
    print("Evaluating World Models Agent")
    print("="*50)
    
    world_models_agent = load_world_models_agent(args.checkpoint_dir, device, args)
    if world_models_agent is not None:
        world_models_agent.eval()
        results['World Models'] = evaluate_agent(
            env, world_models_agent, device, 
            num_episodes=args.num_episodes,
            agent_type="world_models"
        )
        print(f"World Models: {results['World Models']['mean_reward']:.2f} ± "
              f"{results['World Models']['std_reward']:.2f}")
    else:
        print("World Models agent not available")
        results['World Models'] = None
    
    # =====================================================================
    # 2. Evaluate DDQN
    # =====================================================================
    print("\n" + "="*50)
    print("Evaluating DDQN Agent")
    print("="*50)
    
    ddqn_agent = load_ddqn_agent(args.checkpoint_dir, device, action_dim)
    if ddqn_agent is not None:
        results['DDQN'] = evaluate_agent(
            env, ddqn_agent, device,
            num_episodes=args.num_episodes,
            agent_type="ddqn"
        )
        print(f"DDQN: {results['DDQN']['mean_reward']:.2f} ± "
              f"{results['DDQN']['std_reward']:.2f}")
    else:
        print("DDQN agent not available")
        results['DDQN'] = None
    
    # =====================================================================
    # 3. Evaluate PPO
    # =====================================================================
    print("\n" + "="*50)
    print("Evaluating PPO Agent")
    print("="*50)
    
    ppo_agent = load_ppo_agent(args.checkpoint_dir, device, action_dim)
    if ppo_agent is not None:
        results['PPO'] = evaluate_agent(
            env, ppo_agent, device,
            num_episodes=args.num_episodes,
            agent_type="ppo"
        )
        print(f"PPO: {results['PPO']['mean_reward']:.2f} ± "
              f"{results['PPO']['std_reward']:.2f}")
    else:
        print("PPO agent not available")
        results['PPO'] = None
    
    # =====================================================================
    # 4. Evaluate SAC
    # =====================================================================
    print("\n" + "="*50)
    print("Evaluating SAC Agent")
    print("="*50)
    
    sac_agent = load_sac_agent(args.checkpoint_dir, device, action_dim)
    if sac_agent is not None:
        results['SAC'] = evaluate_agent(
            env, sac_agent, device,
            num_episodes=args.num_episodes,
            agent_type="sac"
        )
        print(f"SAC: {results['SAC']['mean_reward']:.2f} ± "
              f"{results['SAC']['std_reward']:.2f}")
    else:
        print("SAC agent not available")
        results['SAC'] = None
    
    env.close()
    
    # =====================================================================
    # Print and save comparison results
    # =====================================================================
    print_comparison_table(results)
    
    # Create visualization
    plot_comparison(results, args.output_dir)
    
    # Save results to JSON
    results_to_save = {}
    for alg, r in results.items():
        if r is not None:
            results_to_save[alg] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in r.items()
                if not isinstance(v, list)  # Skip lists for JSON
            }
            results_to_save[alg]['episode_rewards'] = list(map(float, r['episode_rewards']))
            results_to_save[alg]['episode_lengths'] = list(map(float, r['episode_lengths']))
    
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Log to wandb
    if not args.no_wandb:
        for alg, r in results.items():
            if r is not None:
                wandb.log({
                    f'{alg}/mean_reward': r['mean_reward'],
                    f'{alg}/std_reward': r['std_reward'],
                    f'{alg}/max_reward': r['max_reward']
                })
        
        # Log plots
        for img_name in ['comparison_bar.png', 'comparison_boxplot.png', 'comparison_table.png']:
            img_path = os.path.join(args.output_dir, img_name)
            if os.path.exists(img_path):
                wandb.log({img_name.replace('.png', ''): wandb.Image(img_path)})
        
        wandb.finish()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare World Models with Model-Free Baselines")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Model architecture (for World Models)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=256)
    
    # Evaluation
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to evaluate each agent")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    parser.add_argument("--record-video", action="store_true", help="Record videos of evaluation")
    parser.add_argument("--video-dir", type=str, default="comparison_videos")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)
