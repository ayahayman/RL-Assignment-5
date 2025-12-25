"""
Mixture Density Network LSTM (MDN-RNN)
Memory model for World Models - predicts future latent states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class MDNRNN(nn.Module):
    """
    MDN-RNN: LSTM combined with Mixture Density Network head.
    
    Predicts the distribution of next latent state z_{t+1} given:
    - Current latent state z_t
    - Action a_t
    - Hidden state h_t
    
    Also predicts reward and done signal.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 6,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_mixtures: int = 5,
        temperature: float = 1.0
    ):
        super(MDNRNN, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.temperature = temperature
        
        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 32)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim + 32,  # z + action embedding
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # MDN output heads
        # For each mixture: pi (weight), mu (mean), sigma (std) for each latent dim
        self.fc_pi = nn.Linear(hidden_size, num_mixtures)  # Mixture weights
        self.fc_mu = nn.Linear(hidden_size, num_mixtures * latent_dim)  # Means
        self.fc_sigma = nn.Linear(hidden_size, num_mixtures * latent_dim)  # Stds
        
        # Reward and done prediction
        self.fc_reward = nn.Linear(hidden_size, 1)
        self.fc_done = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass through MDN-RNN.
        
        Args:
            z: Latent states, shape (batch, seq_len, latent_dim)
            actions: Actions, shape (batch, seq_len)
            hidden: Optional LSTM hidden state
            
        Returns:
            pi: Mixture weights (batch, seq_len, num_mixtures)
            mu: Mixture means (batch, seq_len, num_mixtures, latent_dim)
            sigma: Mixture stds (batch, seq_len, num_mixtures, latent_dim)
            reward: Predicted rewards (batch, seq_len, 1)
            done: Predicted done signals (batch, seq_len, 1)
            hidden: Updated hidden state
        """
        batch_size, seq_len, _ = z.shape
        
        # Embed actions
        action_embed = self.action_embed(actions)  # (batch, seq_len, 32)
        
        # Concatenate z and action embedding
        lstm_input = torch.cat([z, action_embed], dim=-1)
        
        # LSTM forward
        if hidden is None:
            lstm_out, hidden = self.lstm(lstm_input)
        else:
            lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # MDN heads
        pi = self.fc_pi(lstm_out)  # (batch, seq_len, num_mixtures)
        pi = F.softmax(pi / self.temperature, dim=-1)
        
        mu = self.fc_mu(lstm_out)  # (batch, seq_len, num_mixtures * latent_dim)
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.latent_dim)
        
        sigma = self.fc_sigma(lstm_out)
        # Clamp log_sigma to prevent explosion, then exp
        sigma = torch.clamp(sigma, min=-10, max=2)  # Limits sigma to [~0.00005, ~7.4]
        sigma = torch.exp(sigma)  # Ensure positive
        sigma = sigma.view(batch_size, seq_len, self.num_mixtures, self.latent_dim)
        
        # Reward and done
        reward = self.fc_reward(lstm_out)
        done = torch.sigmoid(self.fc_done(lstm_out))
        
        return pi, mu, sigma, reward, done, hidden
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial hidden state for LSTM."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)
    
    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from the mixture distribution.
        
        Args:
            pi: Mixture weights (batch, num_mixtures)
            mu: Mixture means (batch, num_mixtures, latent_dim)
            sigma: Mixture stds (batch, num_mixtures, latent_dim)
            
        Returns:
            Sampled latent vector (batch, latent_dim)
        """
        # Sample mixture component
        mixture_idx = torch.multinomial(pi, 1).squeeze(-1)  # (batch,)
        
        batch_size = pi.shape[0]
        batch_indices = torch.arange(batch_size, device=pi.device)
        
        # Get selected mixture parameters
        selected_mu = mu[batch_indices, mixture_idx]  # (batch, latent_dim)
        selected_sigma = sigma[batch_indices, mixture_idx]  # (batch, latent_dim)
        
        # Sample from Gaussian
        eps = torch.randn_like(selected_mu)
        z_next = selected_mu + selected_sigma * eps * self.temperature
        
        return z_next
    
    def step(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Single step prediction for dream simulation.
        
        Args:
            z: Current latent state (batch, latent_dim)
            action: Current action (batch,)
            hidden: Current hidden state
            
        Returns:
            z_next: Predicted next latent state
            reward: Predicted reward
            done: Predicted done probability
            hidden: Updated hidden state
        """
        # Add sequence dimension
        z = z.unsqueeze(1)
        action = action.unsqueeze(1)
        
        pi, mu, sigma, reward, done, hidden = self.forward(z, action, hidden)
        
        # Remove sequence dimension
        pi = pi.squeeze(1)
        mu = mu.squeeze(1)
        sigma = sigma.squeeze(1)
        reward = reward.squeeze(1)
        done = done.squeeze(1)
        
        # Sample next latent
        z_next = self.sample(pi, mu, sigma)
        
        return z_next, reward, done, hidden
    
    @staticmethod
    def mdn_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        z_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log likelihood loss for MDN.
        
        Args:
            pi: Mixture weights (batch, seq_len, num_mixtures)
            mu: Mixture means (batch, seq_len, num_mixtures, latent_dim)
            sigma: Mixture stds (batch, seq_len, num_mixtures, latent_dim)
            z_target: Target latent vectors (batch, seq_len, latent_dim)
            
        Returns:
            Negative log likelihood loss
        """
        batch_size, seq_len, num_mixtures, latent_dim = mu.shape
        
        # Expand target for broadcasting
        z_target = z_target.unsqueeze(2)  # (batch, seq_len, 1, latent_dim)
        
        # Clamp sigma for numerical stability
        sigma = torch.clamp(sigma, min=1e-4, max=10.0)
        
        # Compute log probability for each mixture component
        # log N(z | mu, sigma) = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((z - mu) / sigma)^2
        var = sigma ** 2
        log_sigma = torch.log(sigma)
        diff = z_target - mu
        
        log_prob = -0.5 * math.log(2 * math.pi) - log_sigma - 0.5 * (diff ** 2) / var
        log_prob = log_prob.sum(dim=-1)  # Sum over latent dimensions
        
        # Add log mixture weights
        log_pi = torch.log(torch.clamp(pi, min=1e-8))
        log_prob = log_prob + log_pi
        
        # Log-sum-exp for mixture (numerically stable)
        log_prob = torch.logsumexp(log_prob, dim=-1)
        
        # Negative log likelihood, clamp to prevent extreme values
        nll = -log_prob.mean()
        nll = torch.clamp(nll, min=-100, max=100)
        
        return nll
    
    def loss_function(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        reward_pred: torch.Tensor,
        done_pred: torch.Tensor,
        z_target: torch.Tensor,
        reward_target: torch.Tensor,
        done_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total MDN-RNN loss.
        
        Returns:
            total_loss, mdn_loss, reward_loss, done_loss
        """
        # MDN loss for latent prediction
        mdn_loss = self.mdn_loss(pi, mu, sigma, z_target)
        
        # Reward prediction loss (MSE)
        reward_loss = F.mse_loss(reward_pred.squeeze(-1), reward_target)
        
        # Done prediction loss (BCE)
        done_loss = F.binary_cross_entropy(done_pred.squeeze(-1), done_target.float())
        
        # Total loss
        total_loss = mdn_loss + reward_loss + done_loss
        
        return total_loss, mdn_loss, reward_loss, done_loss


if __name__ == "__main__":
    # Test MDN-RNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdn_rnn = MDNRNN(latent_dim=32, action_dim=6, hidden_size=256).to(device)
    
    batch_size = 4
    seq_len = 10
    
    # Test forward pass
    z = torch.randn(batch_size, seq_len, 32).to(device)
    actions = torch.randint(0, 6, (batch_size, seq_len)).to(device)
    
    pi, mu, sigma, reward, done, hidden = mdn_rnn(z, actions)
    
    print(f"Input z shape: {z.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Pi shape: {pi.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Sigma shape: {sigma.shape}")
    print(f"Reward shape: {reward.shape}")
    print(f"Done shape: {done.shape}")
    
    # Test single step
    z_single = torch.randn(batch_size, 32).to(device)
    action_single = torch.randint(0, 6, (batch_size,)).to(device)
    hidden_init = mdn_rnn.get_initial_hidden(batch_size, device)
    
    z_next, r, d, h = mdn_rnn.step(z_single, action_single, hidden_init)
    print(f"\nSingle step z_next shape: {z_next.shape}")
    
    # Test loss
    z_target = torch.randn(batch_size, seq_len, 32).to(device)
    reward_target = torch.randn(batch_size, seq_len).to(device)
    done_target = torch.zeros(batch_size, seq_len).to(device)
    
    total_loss, mdn_l, r_l, d_l = mdn_rnn.loss_function(
        pi, mu, sigma, reward, done, z_target, reward_target, done_target
    )
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"MDN loss: {mdn_l.item():.4f}")
    print(f"Reward loss: {r_l.item():.4f}")
    print(f"Done loss: {d_l.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in mdn_rnn.parameters())
    print(f"\nTotal parameters: {total_params:,}")
