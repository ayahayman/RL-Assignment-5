"""
VAE Training Script
Train the Variational Autoencoder on collected game frames
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vae import VAE
from utils.wrappers import make_env
from utils.data import collect_rollouts, RolloutDataset, save_rollouts, load_rollouts, get_movement_biased_policy
from config import vae_config, training_config


def train_vae(
    vae: VAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    kl_weight: float = 0.0001
) -> dict:
    """Train VAE for one epoch."""
    vae.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        recon, x, mu, log_var = vae(batch)
        loss, recon_loss, kl_loss = VAE.loss_function(recon, x, mu, log_var, kl_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'recon': total_recon_loss / num_batches,
            'kl': total_kl_loss / num_batches
        })
    
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'kl_loss': total_kl_loss / num_batches
    }


def visualize_reconstructions(
    vae: VAE,
    test_batch: torch.Tensor,
    device: torch.device,
    save_path: str = None
) -> np.ndarray:
    """Visualize VAE reconstructions."""
    import io
    from PIL import Image
    
    vae.eval()
    
    with torch.no_grad():
        test_batch = test_batch[:8].to(device)
        recon, _, _, _ = vae(test_batch)
    
    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(8):
        # Original
        img = test_batch[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstruction
        rec = recon[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(rec)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    # Convert to image array for wandb using BytesIO
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_array = np.array(Image.open(buf))
    buf.close()
    plt.close(fig)
    
    return img_array


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"vae_training_{args.seed}",
            config=vars(args)
        )
    
    # Collect or load data
    data_path = os.path.join(args.data_dir, "rollouts.pkl")
    
    if os.path.exists(data_path) and not args.force_collect:
        print("Loading existing rollouts...")
        rollout_data = load_rollouts(data_path)
    else:
        print("Collecting rollouts with movement-biased policy...")
        env = make_env(args.env_name, frame_size=args.frame_size)
        movement_policy = get_movement_biased_policy(env.action_space)
        rollout_data = collect_rollouts(
            env,
            num_rollouts=args.num_rollouts,
            max_steps=args.max_episode_steps,
            policy=movement_policy  # Use movement-biased policy instead of random
        )
        env.close()
        save_rollouts(rollout_data, data_path)
    
    # Create dataset and dataloader
    dataset = RolloutDataset(rollout_data)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Create VAE
    vae = VAE(
        input_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        metrics = train_vae(
            vae, train_loader, optimizer, device, epoch, args.kl_weight
        )
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'vae/loss': metrics['loss'],
                'vae/recon_loss': metrics['recon_loss'],
                'vae/kl_loss': metrics['kl_loss']
            })
            
            # Visualize reconstructions every 5 epochs
            if epoch % 5 == 0:
                test_batch = next(iter(train_loader))
                save_path = os.path.join(args.checkpoint_dir, f"recon_epoch_{epoch}.png")
                visualize_reconstructions(vae, test_batch, device, save_path=save_path)
                # Log from file path to avoid wandb temp directory issues
                wandb.log({'vae/reconstructions': wandb.Image(save_path)})
        
        # Save best model
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.checkpoint_dir, 'vae_best.pt'))
            print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['loss'],
            }, os.path.join(args.checkpoint_dir, f'vae_epoch_{epoch}.pt'))
    
    # Save final model
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics['loss'],
    }, os.path.join(args.checkpoint_dir, 'vae_final.pt'))
    
    print("VAE training complete!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for World Models")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Data collection
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--force-collect", action="store_true")
    
    # VAE architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 64, 128, 256])
    
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.0001)
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
