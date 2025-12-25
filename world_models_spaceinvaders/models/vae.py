"""
Convolutional Variational Autoencoder (VAE)
Vision model for World Models - compresses frames to latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class VAE(nn.Module):
    """
    Convolutional VAE for encoding game frames into compact latent representations.
    
    Architecture:
    - Encoder: 4 conv layers (3x64x64 -> latent_dim)
    - Decoder: 4 transposed conv layers (latent_dim -> 3x64x64)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        hidden_dims: List[int] = None
    ):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        
        self.hidden_dims = hidden_dims
        
        # Build Encoder
        modules = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # After 4 conv layers with stride 2: 64 -> 32 -> 16 -> 8 -> 4
        # So final feature map is hidden_dims[-1] x 4 x 4
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)
        
        hidden_dims_reversed = hidden_dims[::-1]
        modules = []
        
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer to get back to input channels
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(hidden_dims_reversed[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims_reversed[-1], input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed image.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
            
        Returns:
            Reconstructed image of shape (batch, channels, height, width)
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var).
        
        Args:
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            recon: Reconstructed input
            x: Original input
            mu: Latent mean
            log_var: Latent log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, x, mu, log_var
    
    def get_latent(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get latent representation of input.
        
        Args:
            x: Input tensor
            deterministic: If True, return mu; else sample from distribution
            
        Returns:
            Latent representation
        """
        mu, log_var = self.encode(x)
        if deterministic:
            return mu
        return self.reparameterize(mu, log_var)
    
    @staticmethod
    def loss_function(
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        kl_weight: float = 0.0001
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            recon: Reconstructed input
            x: Original input
            mu: Latent mean
            log_var: Latent log variance
            kl_weight: Weight for KL term (β in β-VAE)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_channels=3, latent_dim=32).to(device)
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64).to(device)
    recon, _, mu, log_var = vae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent log_var shape: {log_var.shape}")
    
    # Test loss
    loss, recon_loss, kl_loss = VAE.loss_function(recon, x, mu, log_var)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Recon loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"Total parameters: {total_params:,}")
