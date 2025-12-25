"""
World Models Training Pipeline
Complete end-to-end training for World Models on Space Invaders
"""

import os
import argparse
import subprocess
import sys
import torch
import wandb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name: str, args: list, capture_output: bool = False):
    """Run a Python script with arguments."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60 + '\n')
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result
    else:
        result = subprocess.run(cmd)
        return result


def main(args):
    # Common arguments
    common_args = [
        f"--checkpoint-dir={args.checkpoint_dir}",
        f"--seed={args.seed}",
        f"--device={args.device}",
    ]
    
    if args.no_wandb:
        common_args.append("--no-wandb")
    else:
        common_args.append(f"--wandb-project={args.wandb_project}")
    
    # Initialize wandb for the full run
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"world_models_full_training_{args.seed}",
            config=vars(args)
        )
        wandb.finish()  # We'll use individual runs for each phase
    
    # Phase 1: Train VAE
    if not args.skip_vae:
        print("\n" + "="*60)
        print("PHASE 1: Training VAE")
        print("="*60)
        
        vae_args = common_args + [
            f"--data-dir={args.data_dir}",
            f"--env-name={args.env_name}",
            f"--frame-size={args.frame_size}",
            f"--num-rollouts={args.num_rollouts}",
            f"--max-episode-steps={args.max_episode_steps}",
            f"--latent-dim={args.latent_dim}",
            f"--batch-size={args.vae_batch_size}",
            f"--learning-rate={args.vae_lr}",
            f"--kl-weight={args.kl_weight}",
            f"--num-epochs={args.vae_epochs}",
        ]
        
        if args.force_collect:
            vae_args.append("--force-collect")
        
        result = run_script("train_vae.py", vae_args)
        
        if result.returncode != 0:
            print("VAE training failed!")
            return
    
    # Phase 2: Train MDN-RNN
    if not args.skip_mdn:
        print("\n" + "="*60)
        print("PHASE 2: Training MDN-RNN")
        print("="*60)
        
        mdn_args = common_args + [
            f"--data-dir={args.data_dir}",
            f"--latent-dim={args.latent_dim}",
            f"--action-dim={args.action_dim}",
            f"--hidden-size={args.mdn_hidden_size}",
            f"--num-mixtures={args.num_mixtures}",
            f"--sequence-length={args.sequence_length}",
            f"--batch-size={args.mdn_batch_size}",
            f"--learning-rate={args.mdn_lr}",
            f"--num-epochs={args.mdn_epochs}",
        ]
        
        result = run_script("train_mdn_rnn.py", mdn_args)
        
        if result.returncode != 0:
            print("MDN-RNN training failed!")
            return
    
    # Phase 3: Train Controller
    if not args.skip_controller:
        print("\n" + "="*60)
        print("PHASE 3: Training Controller")
        print("="*60)
        
        controller_args = common_args + [
            f"--env-name={args.env_name}",
            f"--frame-size={args.frame_size}",
            f"--latent-dim={args.latent_dim}",
            f"--action-dim={args.action_dim}",
            f"--hidden-size={args.mdn_hidden_size}",
            f"--num-mixtures={args.num_mixtures}",
            f"--controller-hidden-size={args.controller_hidden_size}",
            f"--dream-epochs={args.dream_epochs}",
            f"--dream-rollouts={args.dream_rollouts}",
            f"--dream-length={args.dream_length}",
            f"--real-epochs={args.real_epochs}",
            f"--real-episodes={args.real_episodes}",
            f"--learning-rate={args.controller_lr}",
            f"--gamma={args.gamma}",
            f"--entropy-weight={args.entropy_weight}",
        ]
        
        result = run_script("train_controller.py", controller_args)
        
        if result.returncode != 0:
            print("Controller training failed!")
            return
    
    # Phase 4: Evaluation and Video Recording
    print("\n" + "="*60)
    print("PHASE 4: Evaluation and Video Recording")
    print("="*60)
    
    eval_args = [
        f"--checkpoint-dir={args.checkpoint_dir}",
        f"--video-dir={args.video_dir}",
        f"--env-name={args.env_name}",
        f"--frame-size={args.frame_size}",
        f"--latent-dim={args.latent_dim}",
        f"--action-dim={args.action_dim}",
        f"--hidden-size={args.mdn_hidden_size}",
        f"--num-mixtures={args.num_mixtures}",
        f"--controller-hidden-size={args.controller_hidden_size}",
        f"--num-episodes={args.eval_episodes}",
        f"--device={args.device}",
    ]
    
    if not args.no_wandb:
        eval_args.append(f"--wandb-project={args.wandb_project}")
    else:
        eval_args.append("--no-wandb")
    
    result = run_script("evaluate.py", eval_args)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
    print(f"Videos saved in: {args.video_dir}")
    
    if not args.no_wandb:
        print(f"\nView results at: https://wandb.ai/{args.wandb_project}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Models Complete Training Pipeline")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Data collection
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--force-collect", action="store_true")
    
    # VAE
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--vae-batch-size", type=int, default=64)
    parser.add_argument("--vae-lr", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=0.0001)
    parser.add_argument("--vae-epochs", type=int, default=30)
    
    # MDN-RNN
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--mdn-hidden-size", type=int, default=256)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--mdn-batch-size", type=int, default=32)
    parser.add_argument("--mdn-lr", type=float, default=1e-3)
    parser.add_argument("--mdn-epochs", type=int, default=30)
    
    # Controller
    parser.add_argument("--controller-hidden-size", type=int, default=128)
    parser.add_argument("--dream-epochs", type=int, default=100)
    parser.add_argument("--dream-rollouts", type=int, default=16)
    parser.add_argument("--dream-length", type=int, default=100)
    parser.add_argument("--real-epochs", type=int, default=50)
    parser.add_argument("--real-episodes", type=int, default=5)
    parser.add_argument("--controller-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--video-dir", type=str, default="videos")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Skip phases
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-mdn", action="store_true")
    parser.add_argument("--skip-controller", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Quick test mode
    parser.add_argument("--dry-run", action="store_true", help="Quick test with minimal training")
    
    args = parser.parse_args()
    
    # Apply dry-run settings
    if args.dry_run:
        print("Running in DRY-RUN mode with minimal settings...")
        args.num_rollouts = 5
        args.max_episode_steps = 100
        args.vae_epochs = 2
        args.mdn_epochs = 2
        args.dream_epochs = 5
        args.real_epochs = 2
        args.eval_episodes = 2
    
    main(args)
