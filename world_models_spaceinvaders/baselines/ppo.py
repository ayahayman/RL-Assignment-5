"""
PPO Baseline for Comparison
Proximal Policy Optimization implementation for Space Invaders
"""

import os
import sys
import argparse
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


class ActorCritic(nn.Module):
    """Actor-Critic network with shared CNN encoder."""
    
    def __init__(self, input_channels: int = 3, action_dim: int = 6):
        super(ActorCritic, self).__init__()
        
        # Shared CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out_size(input_channels, 64)
        
        # Shared hidden layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(512, action_dim)
        
        # Critic head (value)
        # Critic head (value)
        self.critic = nn.Linear(512, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def _get_conv_out_size(self, input_channels: int, input_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.conv(dummy)
            return out.view(1, -1).shape[1]
    
    def forward(self, x: torch.Tensor):
        conv_out = self.conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        features = self.fc(conv_out)
        
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False):
        action_logits, value = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        action_logits, value = self.forward(x)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """Buffer to store rollout data for PPO updates."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get(self, device: torch.device):
        states = torch.stack(self.states).to(device)
        actions = torch.tensor(self.actions, device=device)
        log_probs = torch.tensor(self.log_probs, device=device)
        rewards = torch.tensor(self.rewards, device=device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=device, dtype=torch.float32)
        values = torch.tensor(self.values, device=device)
        
        return states, actions, log_probs, rewards, dones, values
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        input_channels: int = 3
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.policy = ActorCritic(input_channels=input_channels, action_dim=action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.buffer = RolloutBuffer()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        values = list(values) + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, device=self.device, dtype=torch.float32)
    
    def update(self, next_value: float):
        """Update policy using collected rollout data."""
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get(self.device)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), 
                                       dones.cpu().numpy(), next_value)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values_pred = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1
        
        self.buffer.clear()
        
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_ppo(args):
    """Train PPO agent."""
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"ppo_baseline_{args.seed}",
            config=vars(args)
        )
    
    env = make_env(args.env_name, frame_size=args.frame_size, clip_rewards=True, stack_frames=4)
    
    agent = PPOAgent(
        action_dim=env.action_space.n,
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        input_channels=env.observation_space.shape[0]  # Adapts to stacked frames (e.g. 12)
    )
    
    print(f"PPO parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")
    
    episode_rewards = []
    best_reward = -float('inf')
    total_steps = 0
    episode = 0
    
    pbar = tqdm(total=args.total_timesteps, desc="Training PPO")
    
    while total_steps < args.total_timesteps:
        obs, info = env.reset()
        state = torch.FloatTensor(obs)
        episode_reward = 0
        
        for step in range(args.rollout_length):
            with torch.no_grad():
                action, log_prob, _, value = agent.policy.get_action(
                    state.unsqueeze(0).to(device)
                )
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
            
            agent.buffer.add(
                state, action.item(), log_prob.item(),
                reward, done, value.item()
            )
            
            state = torch.FloatTensor(next_obs)
            total_steps += 1
            pbar.update(1)
            
            if done:
                # Use raw reward from RecordEpisodeStatistics if available
                if "episode" in info:
                    true_reward = info["episode"]["r"]
                    # If it's a tensor/array (some wrappers do this), convert to scalar
                    if hasattr(true_reward, "item"):
                        true_reward = true_reward.item()
                    elif isinstance(true_reward, (list, np.ndarray)):
                        true_reward = true_reward[0]
                    episode_rewards.append(true_reward)
                else:
                    # Fallback to summed clipped reward
                    episode_rewards.append(episode_reward)
                
                avg_reward = np.mean(episode_rewards[-100:])
                episode += 1
                
                if not args.no_wandb:
                    wandb.log({
                        'episode': episode,
                        'reward': episode_reward,
                        'avg_reward_100': avg_reward,
                        'total_steps': total_steps
                    })
                
                if avg_reward > best_reward and episode > 10:
                    best_reward = avg_reward
                    agent.save(os.path.join(args.checkpoint_dir, 'ppo_best.pt'))
                
                obs, info = env.reset()
                state = torch.FloatTensor(obs)
                episode_reward = 0
            
            if total_steps >= args.total_timesteps:
                break
        
        # Update policy
        with torch.no_grad():
            _, _, _, next_value = agent.policy.get_action(state.unsqueeze(0).to(device))
        
        metrics = agent.update(next_value.item())
        
        if not args.no_wandb:
            wandb.log({
                'loss': metrics['loss'],
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'entropy': metrics['entropy']
            })
    
    pbar.close()
    env.close()
    
    agent.save(os.path.join(args.checkpoint_dir, 'ppo_final.pt'))
    
    print(f"\nTraining complete! Best average reward: {best_reward:.2f}")
    
    if not args.no_wandb:
        wandb.finish()
    
    return episode_rewards


def evaluate_ppo(args):
    """Evaluate trained PPO agent."""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    env = make_env(
        args.env_name,
        frame_size=args.frame_size,
        record_video=True,
        video_folder=args.video_dir,
        stack_frames=4
    )
    
    agent = PPOAgent(
        action_dim=env.action_space.n, 
        device=device, 
        input_channels=env.observation_space.shape[0]
    )
    agent.load(os.path.join(args.checkpoint_dir, 'ppo_best.pt'))
    agent.policy.eval()
    
    episode_rewards = []
    
    for ep in tqdm(range(args.eval_episodes), desc="Evaluating"):
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.policy.get_action(state, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action.item())
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
    parser = argparse.ArgumentParser(description="PPO Baseline for Space Invaders")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos/ppo")
    
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(args)
    else:
        evaluate_ppo(args)
