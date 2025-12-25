"""
MDN-RNN Training Script
Train the Memory model on latent sequences
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from utils.wrappers import make_env
from utils.data import load_rollouts, SequenceDataset
from config import mdn_config, training_config


def train_mdn_rnn(
    mdn_rnn: MDNRNN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train MDN-RNN for one epoch."""
    mdn_rnn.train()
    
    total_loss = 0
    total_mdn_loss = 0
    total_reward_loss = 0
    total_done_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for z, next_z, actions, rewards, dones in pbar:
        z = z.to(device)
        next_z = next_z.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pi, mu, sigma, reward_pred, done_pred, _ = mdn_rnn(z, actions)
        
        # Compute loss
        loss, mdn_loss, reward_loss, done_loss = mdn_rnn.loss_function(
            pi, mu, sigma, reward_pred, done_pred,
            next_z, rewards, dones
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn_rnn.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mdn_loss += mdn_loss.item()
        total_reward_loss += reward_loss.item()
        total_done_loss += done_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'mdn': total_mdn_loss / num_batches,
            'reward': total_reward_loss / num_batches,
            'done': total_done_loss / num_batches
        })
    
    return {
        'loss': total_loss / num_batches,
        'mdn_loss': total_mdn_loss / num_batches,
        'reward_loss': total_reward_loss / num_batches,
        'done_loss': total_done_loss / num_batches
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
            name=f"mdn_rnn_training_{args.seed}",
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
    print("VAE loaded successfully!")
    
    # Load rollout data
    data_path = os.path.join(args.data_dir, "rollouts.pkl")
    rollout_data = load_rollouts(data_path)
    
    # Create sequence dataset
    dataset = SequenceDataset(
        rollout_data,
        vae,
        sequence_length=args.sequence_length,
        device=device
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create MDN-RNN
    mdn_rnn = MDNRNN(
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_mixtures=args.num_mixtures
    ).to(device)
    
    print(f"MDN-RNN parameters: {sum(p.numel() for p in mdn_rnn.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(mdn_rnn.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_mdn_rnn(mdn_rnn, train_loader, optimizer, device, epoch)
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'mdn_rnn/loss': metrics['loss'],
                'mdn_rnn/mdn_loss': metrics['mdn_loss'],
                'mdn_rnn/reward_loss': metrics['reward_loss'],
                'mdn_rnn/done_loss': metrics['done_loss']
            })
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': mdn_rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.checkpoint_dir, 'mdn_rnn_best.pt'))
            print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': mdn_rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
            }, os.path.join(args.checkpoint_dir, f'mdn_rnn_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': mdn_rnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics['loss'],
    }, os.path.join(args.checkpoint_dir, 'mdn_rnn_final.pt'))
    
    print("MDN-RNN training complete!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MDN-RNN for World Models")
    
    # VAE configuration (must match trained VAE)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256])
    
    # MDN-RNN architecture
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=100)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=30)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)
