"""
DDQN Baseline for Comparison
Double Deep Q-Network implementation for Space Invaders
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
import numpy as np
import wandb
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_env


# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
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


class DQN(nn.Module):
    """
    Deep Q-Network with CNN encoder.
    Uses dueling architecture for better value estimation.
    """
    
    def __init__(self, input_channels: int = 3, action_dim: int = 6):
        super(DQN, self).__init__()
        
        # CNN encoder
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(input_channels, 64)
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
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
        
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine value and advantage (dueling DQN)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class DDQNAgent:
    """Double DQN Agent with experience replay."""
    
    def __init__(
        self,
        action_dim: int,
        device: torch.device,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update: int = 1000
    ):
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Networks
        self.policy_net = DQN(input_channels=3, action_dim=action_dim).to(device)
        self.target_net = DQN(input_channels=3, action_dim=action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
    
    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               np.exp(-self.steps_done / self.epsilon_decay)
    
    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()
    
    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0
        
        batch = self.buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Handle next states (None for terminal states)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], 
                                       device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: Use policy net to select actions, target net to evaluate
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            with torch.no_grad():
                # Select best action using policy network
                best_actions = self.policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
                # Evaluate using target network
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze()
        
        # Compute expected Q values
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Huber loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']


def train_ddqn(args):
    """Train DDQN agent."""
    
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
            name=f"ddqn_baseline_{args.seed}",
            config=vars(args)
        )
    
    # Create environment
    env = make_env(args.env_name, frame_size=args.frame_size)
    
    # Create agent
    agent = DDQNAgent(
        action_dim=env.action_space.n,
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )
    
    print(f"DQN parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Training loop
    episode_rewards = []
    best_reward = -float('inf')
    
    for episode in tqdm(range(args.num_episodes), desc="Training DDQN"):
        obs, info = env.reset()
        state = torch.FloatTensor(obs)
        total_reward = 0
        episode_loss = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state.unsqueeze(0).to(device))
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store transition
            next_state = torch.FloatTensor(next_obs) if not done else None
            agent.buffer.push(state, action, next_state, reward, done)
            
            # Update agent
            loss = agent.update()
            episode_loss += loss
            steps += 1
            
            state = torch.FloatTensor(next_obs)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        
        # Logging
        if not args.no_wandb:
            wandb.log({
                'episode': episode,
                'reward': total_reward,
                'avg_reward_100': avg_reward,
                'epsilon': agent.epsilon,
                'loss': episode_loss / max(steps, 1),
                'buffer_size': len(agent.buffer)
            })
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.2f}, "
                  f"Avg(100) = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
        
        # Save best model
        if avg_reward > best_reward and episode > 100:
            best_reward = avg_reward
            agent.save(os.path.join(args.checkpoint_dir, 'ddqn_best.pt'))
            print(f"  -> New best model saved! Avg reward: {best_reward:.2f}")
        
        # Periodic save
        if episode % 500 == 0:
            agent.save(os.path.join(args.checkpoint_dir, f'ddqn_episode_{episode}.pt'))
    
    env.close()
    
    # Save final model
    agent.save(os.path.join(args.checkpoint_dir, 'ddqn_final.pt'))
    
    print(f"\nTraining complete! Best average reward: {best_reward:.2f}")
    
    if not args.no_wandb:
        wandb.finish()
    
    return episode_rewards


def evaluate_ddqn(args):
    """Evaluate trained DDQN agent."""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = make_env(
        args.env_name,
        frame_size=args.frame_size,
        record_video=True,
        video_folder=args.video_dir
    )
    
    # Create and load agent
    agent = DDQNAgent(action_dim=env.action_space.n, device=device)
    agent.load(os.path.join(args.checkpoint_dir, 'ddqn_best.pt'))
    agent.policy_net.eval()
    
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
    parser = argparse.ArgumentParser(description="DDQN Baseline for Space Invaders")
    
    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Training
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=int, default=100000)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update", type=int, default=1000)
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos/ddqn")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ddqn(args)
    else:
        evaluate_ddqn(args)
