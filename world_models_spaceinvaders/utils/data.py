"""
Data Collection and Dataset Classes
For collecting rollouts and preparing training data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import os
import pickle


def get_movement_biased_policy(action_space):
    """
    Create a policy that favors movement actions over staying still.
    
    Space Invaders actions:
    0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT, 4: RIGHTFIRE, 5: LEFTFIRE
    
    This policy gives 70% probability to movement actions (2,3,4,5) 
    and 30% to stationary actions (0,1).
    """
    movement_actions = [2, 3, 4, 5]  # RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    stationary_actions = [0, 1]  # NOOP, FIRE
    
    def policy(obs):
        import numpy as np
        if np.random.random() < 0.7:
            # Movement action
            return np.random.choice(movement_actions)
        else:
            # Stationary action (still need some firing)
            return np.random.choice(stationary_actions)
    
    return policy


def collect_rollouts(
    env,
    num_rollouts: int = 100,
    max_steps: int = 1000,
    policy: Optional[callable] = None,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Collect rollouts from environment.
    
    Args:
        env: Gymnasium environment
        num_rollouts: Number of episodes to collect
        max_steps: Maximum steps per episode
        policy: Optional policy function (obs -> action). If None, uses random actions.
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with 'observations', 'actions', 'rewards', 'dones' lists
    """
    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'episode_lengths': [],
        'episode_rewards': []
    }
    
    iterator = range(num_rollouts)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting rollouts")
    
    for _ in iterator:
        obs, info = env.reset()
        episode_obs = [obs]
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        for step in range(max_steps):
            if policy is not None:
                action = policy(obs)
            else:
                action = env.action_space.sample()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)
            
            if not done:
                episode_obs.append(next_obs)
            
            obs = next_obs
            
            if done:
                break
        
        # Store episode data
        data['observations'].append(np.array(episode_obs))
        data['actions'].append(np.array(episode_actions))
        data['rewards'].append(np.array(episode_rewards))
        data['dones'].append(np.array(episode_dones))
        data['episode_lengths'].append(len(episode_actions))
        data['episode_rewards'].append(sum(episode_rewards))
    
    return data


def save_rollouts(data: Dict, filepath: str):
    """Save collected rollouts to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved rollouts to {filepath}")


def load_rollouts(filepath: str) -> Dict:
    """Load rollouts from file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded rollouts from {filepath}")
    return data


class RolloutDataset(Dataset):
    """
    Dataset of individual frames for VAE training.
    
    Each item is a single observation frame.
    """
    
    def __init__(self, rollout_data: Dict):
        """
        Args:
            rollout_data: Dictionary from collect_rollouts
        """
        # Flatten all observations
        self.observations = []
        
        for episode_obs in rollout_data['observations']:
            for obs in episode_obs:
                self.observations.append(obs)
        
        self.observations = np.array(self.observations)
        print(f"RolloutDataset: {len(self.observations)} frames")
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        obs = self.observations[idx]
        return torch.FloatTensor(obs)


class SequenceDataset(Dataset):
    """
    Dataset of observation sequences for MDN-RNN training.
    
    Each item is a sequence of (z, action, reward, done, next_z).
    """
    
    def __init__(
        self,
        rollout_data: Dict,
        vae: torch.nn.Module,
        sequence_length: int = 100,
        device: str = 'cuda'
    ):
        """
        Args:
            rollout_data: Dictionary from collect_rollouts
            vae: Trained VAE model for encoding frames
            sequence_length: Length of sequences to create
            device: Device for VAE encoding
        """
        self.sequence_length = sequence_length
        self.sequences = []
        
        vae.eval()
        vae = vae.to(device)
        
        print("Encoding observations with VAE...")
        
        for i, (episode_obs, episode_actions, episode_rewards, episode_dones) in enumerate(
            tqdm(
                zip(
                    rollout_data['observations'],
                    rollout_data['actions'],
                    rollout_data['rewards'],
                    rollout_data['dones']
                ),
                total=len(rollout_data['observations']),
                desc="Encoding episodes"
            )
        ):
            if len(episode_obs) < sequence_length + 1:
                continue
            
            # Encode all frames in episode
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(episode_obs).to(device)
                latents = vae.get_latent(obs_tensor, deterministic=True)
                latents = latents.cpu().numpy()
            
            # Create sequences
            for start in range(len(episode_obs) - sequence_length):
                seq_z = latents[start:start + sequence_length]
                seq_next_z = latents[start + 1:start + sequence_length + 1]
                seq_actions = episode_actions[start:start + sequence_length]
                seq_rewards = episode_rewards[start:start + sequence_length]
                seq_dones = episode_dones[start:start + sequence_length]
                
                self.sequences.append({
                    'z': seq_z,
                    'next_z': seq_next_z,
                    'actions': seq_actions,
                    'rewards': seq_rewards,
                    'dones': seq_dones
                })
        
        print(f"SequenceDataset: {len(self.sequences)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        seq = self.sequences[idx]
        return (
            torch.FloatTensor(seq['z']),
            torch.FloatTensor(seq['next_z']),
            torch.LongTensor(seq['actions']),
            torch.FloatTensor(seq['rewards']),
            torch.FloatTensor(seq['dones'])
        )


class DreamDataset(Dataset):
    """
    Dataset for controller training from dream rollouts.
    
    Stores transitions: (z, h, action, reward, done, next_z, next_h)
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(
        self,
        z: np.ndarray,
        h: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_z: np.ndarray,
        next_h: np.ndarray,
        log_prob: float,
        value: float
    ):
        """Add a transition to the dataset."""
        transition = {
            'z': z,
            'h': h,
            'action': action,
            'reward': reward,
            'done': done,
            'next_z': next_z,
            'next_h': next_h,
            'log_prob': log_prob,
            'value': value
        }
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.max_size
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.buffer[idx]
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of transitions."""
        batch = {
            'z': [],
            'h': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        for idx in indices:
            t = self.buffer[idx]
            batch['z'].append(t['z'])
            batch['h'].append(t['h'])
            batch['actions'].append(t['action'])
            batch['rewards'].append(t['reward'])
            batch['dones'].append(t['done'])
            batch['log_probs'].append(t['log_prob'])
            batch['values'].append(t['value'])
        
        return {
            'z': torch.FloatTensor(np.array(batch['z'])),
            'h': torch.FloatTensor(np.array(batch['h'])),
            'actions': torch.LongTensor(batch['actions']),
            'rewards': torch.FloatTensor(batch['rewards']),
            'dones': torch.FloatTensor(batch['dones']),
            'log_probs': torch.FloatTensor(batch['log_probs']),
            'values': torch.FloatTensor(batch['values'])
        }


if __name__ == "__main__":
    # Test data collection
    from wrappers import make_env
    
    print("Testing data collection...")
    
    env = make_env("ALE/SpaceInvaders-v5", frame_size=64)
    
    data = collect_rollouts(env, num_rollouts=3, max_steps=100)
    
    print(f"Collected {len(data['observations'])} episodes")
    print(f"Episode lengths: {data['episode_lengths']}")
    print(f"Episode rewards: {data['episode_rewards']}")
    
    # Test dataset
    dataset = RolloutDataset(data)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    env.close()
    print("Data collection test passed!")
