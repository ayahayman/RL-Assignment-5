"""
Train All Baselines Script
Trains DDQN, PPO, and SAC baselines for comparison with World Models
"""

import os
import sys
import argparse
import subprocess


def main(args):
    """Train all baseline models."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*70)
    print("TRAINING ALL MODEL-FREE BASELINES")
    print("="*70)
    
    # Common arguments
    common_args = [
        "--env-name", args.env_name,
        "--frame-size", str(args.frame_size),
        "--checkpoint-dir", args.checkpoint_dir,
        "--seed", str(args.seed),
        "--device", args.device
    ]
    
    if args.no_wandb:
        common_args.append("--no-wandb")
    else:
        common_args.extend(["--wandb-project", args.wandb_project])
    
    # =====================================================================
    # 1. Train DDQN
    # =====================================================================
    if "ddqn" in args.algorithms or "all" in args.algorithms:
        print("\n" + "="*50)
        print("Training DDQN")
        print("="*50)
        
        ddqn_cmd = [
            sys.executable, 
            os.path.join(base_dir, "baselines", "ddqn.py"),
            "--mode", "train",
            "--num-episodes", str(args.ddqn_episodes),
            *common_args
        ]
        
        print(f"Command: {' '.join(ddqn_cmd)}")
        subprocess.run(ddqn_cmd)
    
    # =====================================================================
    # 2. Train PPO
    # =====================================================================
    if "ppo" in args.algorithms or "all" in args.algorithms:
        print("\n" + "="*50)
        print("Training PPO")
        print("="*50)
        
        ppo_cmd = [
            sys.executable,
            os.path.join(base_dir, "baselines", "ppo.py"),
            "--mode", "train",
            "--total-timesteps", str(args.ppo_timesteps),
            *common_args
        ]
        
        print(f"Command: {' '.join(ppo_cmd)}")
        subprocess.run(ppo_cmd)
    
    # =====================================================================
    # 3. Train SAC
    # =====================================================================
    if "sac" in args.algorithms or "all" in args.algorithms:
        print("\n" + "="*50)
        print("Training SAC")
        print("="*50)
        
        sac_cmd = [
            sys.executable,
            os.path.join(base_dir, "baselines", "sac.py"),
            "--mode", "train",
            "--num-episodes", str(args.sac_episodes),
            *common_args
        ]
        
        print(f"Command: {' '.join(sac_cmd)}")
        subprocess.run(sac_cmd)
    
    print("\n" + "="*70)
    print("ALL BASELINE TRAINING COMPLETE!")
    print("="*70)
    
    # =====================================================================
    # Run comparison
    # =====================================================================
    if args.run_comparison:
        print("\n" + "="*50)
        print("Running Comparison")
        print("="*50)
        
        compare_cmd = [
            sys.executable,
            os.path.join(base_dir, "compare_baselines.py"),
            "--checkpoint-dir", args.checkpoint_dir,
            "--num-episodes", str(args.eval_episodes),
            "--device", args.device
        ]
        
        if args.record_video:
            compare_cmd.append("--record-video")
            compare_cmd.extend(["--video-dir", args.video_dir])
            
        if args.no_wandb:
            compare_cmd.append("--no-wandb")
        else:
            compare_cmd.extend(["--wandb-project", args.wandb_project])
        
        print(f"Command: {' '.join(compare_cmd)}")
        subprocess.run(compare_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train All Baseline Models")
    
    # Which algorithms to train
    parser.add_argument("--algorithms", type=str, nargs="+", 
                        default=["all"],
                        choices=["all", "ddqn", "ppo", "sac"],
                        help="Which algorithms to train")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--frame-size", type=int, default=64)
    
    # Training parameters for each algorithm
    parser.add_argument("--ddqn-episodes", type=int, default=2000,
                        help="Number of episodes for DDQN training")
    parser.add_argument("--ppo-timesteps", type=int, default=500000,
                        help="Total timesteps for PPO training")
    parser.add_argument("--sac-episodes", type=int, default=2000,
                        help="Number of episodes for SAC training")
    
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of episodes for evaluation")
    parser.add_argument("--run-comparison", action="store_true", default=True,
                        help="Run comparison after training")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Skip comparison after training")
    parser.add_argument("--record-video", action="store_true",
                        help="Record videos during comparison")
    parser.add_argument("--video-dir", type=str, default="comparison_videos",
                        help="Directory to save comparison videos")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    
    # Wandb
    parser.add_argument("--wandb-project", type=str, default="world-models-spaceinvaders")
    parser.add_argument("--no-wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.no_comparison:
        args.run_comparison = False
    
    main(args)
