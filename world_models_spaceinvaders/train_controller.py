"""
Controller Training Script
Train the policy controller using REINFORCE with baseline
Trains both in "dream" (MDN-RNN simulation) and real environment
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from collections import deque

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller, WorldModelAgent
from utils.wrappers import make_env
from config import controller_config, training_config


def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    R = 0
    
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = r + gamma * R
        returns.insert(0, R)
    
    return returns


def shape_reward_for_movement(reward: float, action: int, movement_bonus: float = 0.1) -> float:
    """
    Add reward shaping to encourage movement.
    
    Space Invaders actions:
    0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHTFIRE, 5: LEFTFIRE
    
    Movement actions (2,3,4,5) get a small bonus.
    Staying still (0,1) gets a small penalty.
    """
    movement_actions = {2, 3, 4, 5}  # RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    
    if action in movement_actions:
        return reward + movement_bonus
    else:
        # Small penalty for not moving (but still allow firing)
        return reward - (movement_bonus * 0.5)


def train_controller_dream(
    agent: WorldModelAgent,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_rollouts: int = 16,
    rollout_length: int = 100,
    gamma: float = 0.99,
    entropy_weight: float = 0.01
) -> dict:
    """
    Train controller in the dream (MDN-RNN rollouts).
    
    Uses REINFORCE with baseline (value function).
    """
    agent.train()
    agent.controller.train()
    
    all_log_probs = []
    all_values = []
    all_returns = []
    all_entropy = []
    
    total_reward = 0
    
    for _ in range(num_rollouts):
        # Start from random initial latent
        z = torch.randn(1, agent.vae.latent_dim, device=device)
        hidden = agent.mdn_rnn.get_initial_hidden(1, device)
        
        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []
        
        for step in range(rollout_length):
            h = hidden[0][-1]  # Last layer hidden state
            
            # Get action from controller
            action_probs, value = agent.controller(z, h)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Step in MDN-RNN (dream)
            z_next, reward, done, hidden = agent.mdn_rnn.step(z, action, hidden)
            
            # Apply movement reward shaping
            shaped_reward = shape_reward_for_movement(reward.squeeze().item(), action.item())
            
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(shaped_reward)
            dones.append(done.squeeze().item() > 0.5)
            entropies.append(entropy)
            
            z = z_next
            
            if done.squeeze().item() > 0.5:
                break
        
        # Compute returns
        returns = compute_returns(rewards, dones, gamma)
        
        all_log_probs.extend(log_probs)
        all_values.extend(values)
        all_returns.extend(returns)
        all_entropy.extend(entropies)
        total_reward += sum(rewards)
    
    # Convert to tensors
    log_probs = torch.stack(all_log_probs)
    values = torch.stack(all_values)
    returns = torch.tensor(all_returns, device=device, dtype=torch.float32)
    entropies = torch.stack(all_entropy)
    
    # Normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute losses
    advantages = returns - values.detach()
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    entropy_loss = -entropies.mean()
    
    loss = policy_loss + 0.5 * value_loss + entropy_weight * entropy_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.controller.parameters(), 0.5)
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': -entropy_loss.item(),
        'dream_reward': total_reward / num_rollouts
    }


def evaluate_real_env(
    agent: WorldModelAgent,
    env,
    device: torch.device,
    num_episodes: int = 5,
    deterministic: bool = True
) -> dict:
    """Evaluate controller in real environment."""
    agent.eval()
    
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
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
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'max_reward': np.max(episode_rewards)
    }


def train_controller_real(
    agent: WorldModelAgent,
    env,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_episodes: int = 10,
    gamma: float = 0.99,
    entropy_weight: float = 0.01
) -> dict:
    """Train controller with real environment rollouts."""
    agent.train()
    agent.controller.train()
    
    all_log_probs = []
    all_values = []
    all_returns = []
    all_entropy = []
    all_entropy = []
    total_rewards = []  # Shaped rewards
    raw_rewards = []    # Raw game scores
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset(1, device)
        
        log_probs = []
        values = []
        rewards = []
        episode_raw_reward = 0
        dones = []
        entropies = []
        
        while True:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Encode observation
            z = agent.vae.get_latent(obs_tensor, deterministic=True)
            h = agent.hidden[0][-1] if agent.hidden is not None else torch.zeros(1, agent.mdn_rnn.hidden_size, device=device)
            
            # Get action
            action_probs, value = agent.controller(z, h)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Step in environment
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            episode_raw_reward += reward
            
            # Apply movement reward shaping
            shaped_reward = shape_reward_for_movement(reward, action.item())
            
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(shaped_reward)
            dones.append(done)
            entropies.append(entropy)
            
            # Update RNN hidden state
            with torch.no_grad():
                _, _, _, _, _, agent.hidden = agent.mdn_rnn(
                    z.unsqueeze(1),
                    action.unsqueeze(1),
                    agent.hidden
                )
            
            obs = next_obs
            
            if done:
                break
        
        # Compute returns
        returns = compute_returns(rewards, dones, gamma)
        
        all_log_probs.extend(log_probs)
        all_values.extend(values)
        all_returns.extend(returns)
        all_entropy.extend(entropies)
        all_entropy.extend(entropies)
        total_rewards.append(sum(rewards))
        raw_rewards.append(episode_raw_reward)
    
    if len(all_log_probs) == 0:
        return {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'value_loss': 0,
            'entropy': 0,
            'real_reward': 0,
            'raw_reward': 0
        }
    
    # Convert to tensors
    log_probs = torch.stack(all_log_probs)
    values = torch.stack(all_values)
    returns = torch.tensor(all_returns, device=device, dtype=torch.float32)
    entropies = torch.stack(all_entropy)
    
    # Normalize returns
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Compute losses
    advantages = returns - values.detach()
    
    policy_loss = -(log_probs * advantages).mean()
    value_loss = nn.functional.mse_loss(values, returns)
    entropy_loss = -entropies.mean()
    
    loss = policy_loss + 0.5 * value_loss + entropy_weight * entropy_loss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.controller.parameters(), 0.5)
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': -entropy_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': -entropy_loss.item(),
        'real_reward': np.mean(total_rewards),
        'raw_reward': np.mean(raw_rewards)
    }


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"controller_training_{args.seed}",
            config=vars(args)
        )
    
    # Load VAE
    print("Loading trained VAE...")
    vae = VAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    vae_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'vae_best.pt'),
        map_location=device
    )
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE loaded!")
    
    # Load MDN-RNN
    print("Loading trained MDN-RNN...")
    mdn_rnn = MDNRNN(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures
    ).to(device)
    
    mdn_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'mdn_rnn_best.pt'),
        map_location=device
    )
    mdn_rnn.load_state_dict(mdn_checkpoint['model_state_dict'])
    mdn_rnn.eval()
    for param in mdn_rnn.parameters():
        param.requires_grad = False
    print("MDN-RNN loaded!")
    
    # Create Controller
    controller = Controller(
        latent_dim=args.latent_dim,
        rnn_hidden_size=args.hidden_size,
        hidden_size=args.controller_hidden_size,
        action_dim=args.action_dim
    ).to(device)
    
    print(f"Controller parameters: {sum(p.numel() for p in controller.parameters()):,}")
    
    # Create agent
    agent = WorldModelAgent(vae, mdn_rnn, controller)
    
    # Optimizer (only for controller)
    optimizer = optim.Adam(controller.parameters(), lr=args.learning_rate)
    
    # Create environment for evaluation
    env = make_env(args.env_name, frame_size=args.frame_size)
    
    # Training loop
    best_reward = -float('inf')
    reward_history = deque(maxlen=100)
    
    print("\n=== Training Controller ===")
    print(f"Phase 1: Dream training for {args.dream_epochs} epochs")
    print(f"Phase 2: Real environment fine-tuning for {args.real_epochs} epochs")
    
    # Phase 1: Dream training
    for epoch in range(1, args.dream_epochs + 1):
        metrics = train_controller_dream(
            agent, optimizer, device,
            num_rollouts=args.dream_rollouts,
            rollout_length=args.dream_length,
            gamma=args.gamma,
            entropy_weight=args.entropy_weight
        )
        
        # Evaluate in real environment periodically
        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate_real_env(agent, env, device, num_episodes=5)
            reward_history.append(eval_metrics['mean_reward'])
            
            if not args.no_wandb:
                wandb.log({
                    'epoch': epoch,
                    'phase': 'dream',
                    'controller/policy_loss': metrics['policy_loss'],
                    'controller/value_loss': metrics['value_loss'],
                    'controller/entropy': metrics['entropy'],
                    'controller/dream_reward': metrics['dream_reward'],
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/max_reward': eval_metrics['max_reward'],
                    'eval/mean_length': eval_metrics['mean_length']
                })
            
            print(f"Epoch {epoch}: Dream reward: {metrics['dream_reward']:.2f}, "
                  f"Real reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                torch.save({
                    'epoch': epoch,
                    'controller_state_dict': controller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': best_reward,
                }, os.path.join(args.checkpoint_dir, 'controller_best.pt'))
                print(f"  -> New best model saved!")
        else:
            if not args.no_wandb:
                wandb.log({
                    'epoch': epoch,
                    'phase': 'dream',
                    'controller/policy_loss': metrics['policy_loss'],
                    'controller/value_loss': metrics['value_loss'],
                    'controller/entropy': metrics['entropy'],
                    'controller/dream_reward': metrics['dream_reward']
                })
    
    # Phase 2: Real environment fine-tuning
    print("\n=== Phase 2: Real Environment Fine-tuning ===")
    
    for epoch in range(1, args.real_epochs + 1):
        metrics = train_controller_real(
            agent, env, optimizer, device,
            num_episodes=args.real_episodes,
            gamma=args.gamma,
            entropy_weight=args.entropy_weight
        )
        
        # Evaluate
        if epoch % args.eval_interval == 0:
            eval_metrics = evaluate_real_env(agent, env, device, num_episodes=5)
            reward_history.append(eval_metrics['mean_reward'])
            
            if not args.no_wandb:
                wandb.log({
                    'epoch': args.dream_epochs + epoch,
                    'phase': 'real',
                    'controller/policy_loss': metrics['policy_loss'],
                    'controller/value_loss': metrics['value_loss'],
                    'controller/entropy': metrics['entropy'],
                    'controller/real_reward': metrics['real_reward'],
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/max_reward': eval_metrics['max_reward'],
                    'eval/mean_length': eval_metrics['mean_length']
                })
            
            print(f"Epoch {epoch}: Train reward (Shaped): {metrics['real_reward']:.2f}, Raw: {metrics['raw_reward']:.2f}, "
                  f"Eval reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            
            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                torch.save({
                    'epoch': args.dream_epochs + epoch,
                    'controller_state_dict': controller.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': best_reward,
                }, os.path.join(args.checkpoint_dir, 'controller_best.pt'))
                print(f"  -> New best model saved!")
    
    env.close()
    
    # Save final model
    torch.save({
        'epoch': args.dream_epochs + args.real_epochs,
        'controller_state_dict': controller.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward': best_reward,
    }, os.path.join(args.checkpoint_dir, 'controller_final.pt'))
    
    print(f"\nController training complete!")
    print(f"Best reward: {best_reward:.2f}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Controller for World Models")
    
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
    
    # Training - Dream phase
    parser.add_argument("--dream-epochs", type=int, default=200)  # Increased from 100
    parser.add_argument("--dream-rollouts", type=int, default=16)
    parser.add_argument("--dream-length", type=int, default=100)
    
    # Training - Real environment phase
    parser.add_argument("--real-epochs", type=int, default=100)  # Increased from 50
    parser.add_argument("--real-episodes", type=int, default=10)  # Increased from 5
    
    # Training - General
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-weight", type=float, default=0.05)  # Increased for more exploration
    parser.add_argument("--eval-interval", type=int, default=5)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)
