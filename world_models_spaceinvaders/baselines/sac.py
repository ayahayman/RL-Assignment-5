"""
SAC Baseline for Comparison
Soft Actor-Critic (Discrete) implementation for Space Invaders
"""

import os
import sys
import argparse
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import wandb
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_env


# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    """Soft Q-Network with CNN encoder for discrete actions."""
    
    def __init__(self, input_channels: int = 3, action_dim: int = 6):
        super(SoftQNetwork, self).__init__()
        
        # CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size(input_channels, 64)
        
        # Q-network head
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def _get_conv_out_size(self, input_channels: int, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.conv(dummy)
            return out.view(1, -1).shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        q_values = self.fc(conv_out)
        return q_values


class PolicyNetwork(nn.Module):
    """Policy Network for discrete SAC."""
    
    def __init__(self, input_channels: int = 3, action_dim: int = 6):
        super(PolicyNetwork, self).__init__()
        
        # CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size(input_channels, 64)
        
        # Policy head (outputs action probabilities)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def _get_conv_out_size(self, input_channels: int, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.conv(dummy)
            return out.view(1, -1).shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(conv_out)
        return logits
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = None
        else:
            dist = Categorical(probs)
            action = dist.sample()
            # Log probability
            log_prob = dist.log_prob(action)
        
        return action, probs, log_prob
    
    def evaluate(self, x: torch.Tensor):
        """Get action probabilities and log probabilities for all actions."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        # Add small epsilon for numerical stability
        log_probs = torch.log(probs + 1e-8)
        return probs, log_probs


class SACAgent:
    """Soft Actor-Critic Agent for discrete action spaces."""
    
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_interval: int = 1
    ):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.updates = 0
        
        # Networks
        self.policy = PolicyNetwork(input_channels=3, action_dim=action_dim).to(device)
        self.q1 = SoftQNetwork(input_channels=3, action_dim=action_dim).to(device)
        self.q2 = SoftQNetwork(input_channels=3, action_dim=action_dim).to(device)
        self.q1_target = SoftQNetwork(input_channels=3, action_dim=action_dim).to(device)
        self.q2_target = SoftQNetwork(input_channels=3, action_dim=action_dim).to(device)
        
        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        
        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy: -log(1/|A|) = log(|A|) for discrete actions
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> int:
        with torch.no_grad():
            action, _, _ = self.policy.get_action(state, deterministic=eval_mode)
        return action.item()
    
    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        
        batch = self.buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Handle next states (None for terminal states)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                       device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Compute Q targets
        with torch.no_grad():
            # Get next action probabilities and log probs
            next_probs = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            next_log_probs = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            
            if non_final_next_states.size(0) > 0:
                next_probs[non_final_mask], next_log_probs[non_final_mask] = \
                    self.policy.evaluate(non_final_next_states)
            
            # Compute target Q values
            q1_next = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            q2_next = torch.zeros(self.batch_size, self.action_dim, device=self.device)
            
            if non_final_next_states.size(0) > 0:
                q1_next[non_final_mask] = self.q1_target(non_final_next_states)
                q2_next[non_final_mask] = self.q2_target(non_final_next_states)
            
            min_q_next = torch.min(q1_next, q2_next)
            
            # V(s') = sum_a π(a|s') * (Q(s', a) - α * log π(a|s'))
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            
            # Target: r + γ * V(s') * (1 - done)
            q_target = reward_batch + self.gamma * v_next * (1 - done_batch)
        
        # Update Q-networks
        q1_values = self.q1(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        q2_values = self.q2(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        
        q1_loss = F.mse_loss(q1_values, q_target)
        q2_loss = F.mse_loss(q2_values, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update policy
        probs, log_probs = self.policy.evaluate(state_batch)
        
        with torch.no_grad():
            q1_pi = self.q1(state_batch)
            q2_pi = self.q2(state_batch)
            min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Policy loss: E_a[α * log π(a|s) - Q(s, a)]
        policy_loss = (probs * (self.alpha * log_probs - min_q_pi)).sum(dim=1).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = 0.0
        if self.auto_alpha:
            # Entropy: H(π) = -sum_a π(a|s) * log π(a|s)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            
            alpha_loss = -(self.log_alpha * (self.target_entropy - entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()
        
        # Soft update target networks
        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss,
            'alpha': self.alpha,
            'entropy': -(probs * log_probs).sum(dim=1).mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha': self.alpha,
            'updates': self.updates
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])
        if checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
        self.alpha = checkpoint['alpha']
        self.updates = checkpoint['updates']


def train_sac(args):
    """Train SAC agent."""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"sac_baseline_{args.seed}",
            config=vars(args)
        )
    
    # Create environment
    env = make_env(args.env_name, frame_size=args.frame_size)
    
    # Create agent
    agent = SACAgent(
        action_dim=env.action_space.n,
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval
    )
    
    total_params = (
        sum(p.numel() for p in agent.policy.parameters()) +
        sum(p.numel() for p in agent.q1.parameters()) +
        sum(p.numel() for p in agent.q2.parameters())
    )
    print(f"SAC parameters: {total_params:,}")
    
    # Training loop
    episode_rewards = []
    best_reward = -float('inf')
    total_steps = 0
    
    for episode in tqdm(range(args.num_episodes), desc="Training SAC"):
        obs, info = env.reset()
        state = torch.FloatTensor(obs)
        total_reward = 0
        episode_metrics = {'q1_loss': 0, 'q2_loss': 0, 'policy_loss': 0}
        steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state.unsqueeze(0).to(device))
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_steps += 1
            
            # Store transition
            next_state = torch.FloatTensor(next_obs) if not done else None
            agent.buffer.push(state, action, next_state, reward, done)
            
            # Update agent (start after initial exploration)
            if total_steps >= args.learning_starts:
                for _ in range(args.gradient_steps):
                    metrics = agent.update()
                    if metrics:
                        for k, v in metrics.items():
                            if k in episode_metrics:
                                episode_metrics[k] += v
                        steps += 1
            
            state = torch.FloatTensor(next_obs)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        
        # Logging
        if not args.no_wandb:
            log_dict = {
                'episode': episode,
                'reward': total_reward,
                'avg_reward_100': avg_reward,
                'total_steps': total_steps,
                'buffer_size': len(agent.buffer),
                'alpha': agent.alpha
            }
            if steps > 0:
                log_dict.update({k: v / steps for k, v in episode_metrics.items()})
            wandb.log(log_dict)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.2f}, "
                  f"Avg(100) = {avg_reward:.2f}, Alpha = {agent.alpha:.4f}")
        
        # Save best model
        if avg_reward > best_reward and episode > 100:
            best_reward = avg_reward
            agent.save(os.path.join(args.checkpoint_dir, 'sac_best.pt'))
            print(f"  -> New best model saved! Avg reward: {best_reward:.2f}")
        
        # Periodic save
        if episode % 500 == 0:
            agent.save(os.path.join(args.checkpoint_dir, f'sac_episode_{episode}.pt'))
    
    env.close()
    
    # Save final model
    agent.save(os.path.join(args.checkpoint_dir, 'sac_final.pt'))
    
    print(f"\nTraining complete! Best average reward: {best_reward:.2f}")
    
    if not args.no_wandb:
        wandb.finish()
    
    return episode_rewards


def evaluate_sac(args):
    """Evaluate trained SAC agent."""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = make_env(
        args.env_name,
        frame_size=args.frame_size,
        record_video=True,
        video_folder=args.video_dir
    )
    
    # Create and load agent
    agent = SACAgent(action_dim=env.action_space.n, device=device)
    agent.load(os.path.join(args.checkpoint_dir, 'sac_best.pt'))
    agent.policy.eval()
    
    episode_rewards = []
    
    for ep in tqdm(range(args.eval_episodes), desc="Evaluating"):
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = agent.select_action(state, eval_mode=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward}")
    
    env.close()
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Baseline for Space Invaders")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Training
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", action="store_true", default=True)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-interval", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--gradient-steps", type=int, default=1)
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos/sac")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_sac(args)
    else:
        evaluate_sac(args)
