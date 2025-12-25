"""
Controller (Policy Network)
Simple feed-forward network that acts in the latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, Optional


class Controller(nn.Module):
    """
    Controller network for World Models.
    
    Takes concatenated latent state z and RNN hidden state h as input,
    outputs action probabilities.
    
    Designed to be small (~1000 parameters) for efficient evolution strategies,
    but can also be trained with policy gradients.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        rnn_hidden_size: int = 256,
        hidden_size: int = 128,  # Increased from 64 for better expressiveness
        action_dim: int = 6
    ):
        super(Controller, self).__init__()
        
        self.latent_dim = latent_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Input: concatenation of z and h
        input_dim = latent_dim + rnn_hidden_size
        
        # Simple 2-layer network
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        
        # Value head for actor-critic
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize with small weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with moderate values for better initial exploration."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)  # Increased from 0.01
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through controller.
        
        Args:
            z: Latent state (batch, latent_dim)
            h: RNN hidden state (batch, rnn_hidden_size)
            
        Returns:
            action_probs: Action probabilities (batch, action_dim)
            value: State value estimate (batch, 1)
        """
        # Concatenate z and h
        x = torch.cat([z, h], dim=-1)
        
        # Forward through network
        x = F.relu(self.fc1(x))
        
        action_logits = self.fc2(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        value = self.value_head(x)
        
        return action_probs, value
    
    def get_action(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            z: Latent state
            h: RNN hidden state
            deterministic: If True, return argmax action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_probs, value = self.forward(z, h)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO-style updates.
        
        Args:
            z: Latent states
            h: RNN hidden states
            actions: Actions taken
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_probs, values = self.forward(z, h)
        
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class WorldModelAgent(nn.Module):
    """
    Complete World Model agent combining VAE, MDN-RNN, and Controller.
    """
    
    def __init__(
        self,
        vae: nn.Module,
        mdn_rnn: nn.Module,
        controller: nn.Module
    ):
        super(WorldModelAgent, self).__init__()
        
        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.controller = controller
        
        # Store RNN hidden state for online inference
        self.hidden = None
    
    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset the RNN hidden state."""
        if device is None:
            device = next(self.parameters()).device
        self.hidden = self.mdn_rnn.get_initial_hidden(batch_size, device)
    
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> int:
        """
        Select action given observation.
        
        Args:
            obs: Observation tensor (1, 3, 64, 64)
            deterministic: If True, return greedy action
            
        Returns:
            Selected action
        """
        device = obs.device
        
        # Initialize hidden state if needed
        if self.hidden is None:
            self.reset(1, device)
        
        # Encode observation to latent
        z = self.vae.get_latent(obs, deterministic=True)
        
        # Get RNN hidden state (just the h part, not c)
        h = self.hidden[0][-1]  # Last layer hidden state
        
        # Get action from controller
        action, _, _ = self.controller.get_action(z, h, deterministic)
        
        # Update RNN hidden state (predict next state but we just need hidden update)
        # We'll update properly after we know the action result
        with torch.no_grad():
            _, _, _, _, _, self.hidden = self.mdn_rnn(
                z.unsqueeze(1), 
                action.unsqueeze(1), 
                self.hidden
            )
        
        return action.item()
    
    def dream_rollout(
        self,
        initial_z: torch.Tensor,
        initial_hidden: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform rollout in the dream (using MDN-RNN as simulator).
        
        Args:
            initial_z: Starting latent state (batch, latent_dim)
            initial_hidden: Starting RNN hidden state
            num_steps: Number of steps to simulate
            deterministic: If True, use deterministic policy
            
        Returns:
            states: Latent states (batch, num_steps, latent_dim)
            actions: Actions taken (batch, num_steps)
            rewards: Predicted rewards (batch, num_steps)
            dones: Predicted done flags (batch, num_steps)
        """
        batch_size = initial_z.shape[0]
        device = initial_z.device
        
        states = []
        actions = []
        rewards = []
        dones = []
        
        z = initial_z
        hidden = initial_hidden
        
        for _ in range(num_steps):
            # Get hidden state for controller
            h = hidden[0][-1]  # Last layer hidden state
            
            # Get action
            action, _, _ = self.controller.get_action(z, h, deterministic)
            
            # Predict next state
            z_next, reward, done, hidden = self.mdn_rnn.step(z, action, hidden)
            
            states.append(z)
            actions.append(action)
            rewards.append(reward.squeeze(-1))
            dones.append(done.squeeze(-1))
            
            z = z_next
        
        states = torch.stack(states, dim=1)
        actions = torch.stack(actions, dim=1)
        rewards = torch.stack(rewards, dim=1)
        dones = torch.stack(dones, dim=1)
        
        return states, actions, rewards, dones


if __name__ == "__main__":
    # Test Controller
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = Controller(latent_dim=32, rnn_hidden_size=256, action_dim=6).to(device)
    
    batch_size = 4
    
    # Test forward pass
    z = torch.randn(batch_size, 32).to(device)
    h = torch.randn(batch_size, 256).to(device)
    
    action_probs, value = controller(z, h)
    print(f"Input z shape: {z.shape}")
    print(f"Input h shape: {h.shape}")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test action sampling
    action, log_prob, val = controller.get_action(z, h)
    print(f"\nSampled action: {action}")
    print(f"Log prob: {log_prob}")
    
    # Count parameters
    total_params = sum(p.numel() for p in controller.parameters())
    print(f"\nTotal parameters: {total_params:,}")
