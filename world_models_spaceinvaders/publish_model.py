"""
Publish Model to Hugging Face Hub
Upload trained World Models to Hugging Face for sharing
"""

import os
import argparse
import json
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub import ModelCard, ModelCardData


def create_model_card(
    model_name: str,
    results: dict,
    env_name: str = "SpaceInvaders",
    wandb_url: str = None,
    github_url: str = None
) -> str:
    """Create a model card for the trained model."""
    
    card_content = f"""---
tags:
- reinforcement-learning
- world-models
- atari
- space-invaders
- pytorch
library_name: pytorch
model-index:
- name: {model_name}
  results:
  - task:
      type: reinforcement-learning
      name: Atari Games
    dataset:
      type: atari
      name: {env_name}
    metrics:
    - type: mean_reward
      value: {results.get('mean_reward', 'N/A')}
      name: Mean Reward
    - type: max_reward
      value: {results.get('max_reward', 'N/A')}
      name: Max Reward
---

# World Models for {env_name}

This model is a **World Models** implementation trained on the {env_name} Atari environment.

## Model Architecture

World Models (Ha & Schmidhuber, 2018) consists of three components:

1. **VAE (Vision Model)**: Convolutional Variational Autoencoder that compresses game frames into a compact latent representation (z ∈ ℝ³²)

2. **MDN-RNN (Memory Model)**: LSTM with Mixture Density Network that predicts future latent states and models temporal dynamics

3. **Controller**: Simple feed-forward policy network that acts in the compressed latent space

## Training Details

- **Environment**: ALE/{env_name}-v5
- **Framework**: PyTorch
- **Latent Dimension**: 32
- **MDN-RNN Hidden Size**: 256
- **Number of Gaussian Mixtures**: 5

## Results

| Metric | Value |
|--------|-------|
| Mean Reward | {results.get('mean_reward', 'N/A'):.2f} |
| Std Reward | {results.get('std_reward', 'N/A'):.2f} |
| Max Reward | {results.get('max_reward', 'N/A'):.2f} |
| Mean Episode Length | {results.get('mean_length', 'N/A'):.2f} |

## Usage

```python
import torch
from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller, WorldModelAgent

# Load models
vae = VAE(latent_dim=32)
mdn_rnn = MDNRNN(latent_dim=32, action_dim=6)
controller = Controller(latent_dim=32, rnn_hidden_size=256)

# Load weights
vae.load_state_dict(torch.load('vae_best.pt')['model_state_dict'])
mdn_rnn.load_state_dict(torch.load('mdn_rnn_best.pt')['model_state_dict'])
controller.load_state_dict(torch.load('controller_best.pt')['controller_state_dict'])

# Create agent
agent = WorldModelAgent(vae, mdn_rnn, controller)
agent.eval()

# Use in environment
obs = env.reset()
agent.reset(1, device)
action = agent.act(obs_tensor, deterministic=True)
```

## Links

"""

    if wandb_url:
        card_content += f"- 📊 [W&B Training Logs]({wandb_url})\n"
    if github_url:
        card_content += f"- 💻 [GitHub Repository]({github_url})\n"

    card_content += """
## Citation

```bibtex
@article{ha2018worldmodels,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}
```

## License

MIT License
"""
    
    return card_content


def main(args):
    print("="*60)
    print("Publishing World Models to Hugging Face Hub")
    print("="*60)
    
    # Load results if available
    results_path = os.path.join(args.checkpoint_dir, 'eval_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded evaluation results: Mean reward = {results.get('mean_reward', 'N/A')}")
    else:
        results = {}
        print("No evaluation results found.")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create repository name
    if args.repo_name:
        repo_name = args.repo_name
    else:
        repo_name = f"world-models-spaceinvaders"
    
    full_repo_name = f"{args.hf_username}/{repo_name}"
    
    print(f"\nCreating/updating repository: {full_repo_name}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=full_repo_name,
            repo_type="model",
            exist_ok=True,
            token=args.hf_token
        )
        print(f"Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Create temporary directory for upload
    upload_dir = os.path.join(args.checkpoint_dir, "hf_upload")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy model files
    import shutil
    
    files_to_copy = [
        'vae_best.pt',
        'mdn_rnn_best.pt',
        'controller_best.pt',
        'eval_results.json'
    ]
    
    for fname in files_to_copy:
        src = os.path.join(args.checkpoint_dir, fname)
        dst = os.path.join(upload_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied: {fname}")
    
    # Create model card
    model_card = create_model_card(
        model_name=repo_name,
        results=results,
        env_name="SpaceInvaders",
        wandb_url=args.wandb_url,
        github_url=args.github_url
    )
    
    with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
        f.write(model_card)
    print("  Created: README.md")
    
    # Create config file
    config = {
        "model_type": "world_models",
        "environment": "ALE/SpaceInvaders-v5",
        "vae": {
            "latent_dim": 32,
            "hidden_dims": [32, 64, 128, 256]
        },
        "mdn_rnn": {
            "hidden_size": 256,
            "num_mixtures": 5,
            "num_layers": 1
        },
        "controller": {
            "hidden_size": 64
        },
        "action_dim": 6,
        "frame_size": 64
    }
    
    with open(os.path.join(upload_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print("  Created: config.json")
    
    # Upload to Hub
    print(f"\nUploading to {full_repo_name}...")
    
    try:
        upload_folder(
            folder_path=upload_dir,
            repo_id=full_repo_name,
            repo_type="model",
            token=args.hf_token,
            commit_message="Upload World Models for Space Invaders"
        )
        print(f"\n✅ Successfully uploaded to: https://huggingface.co/{full_repo_name}")
    except Exception as e:
        print(f"Error uploading: {e}")
        return
    
    # Upload videos if available
    if args.video_dir and os.path.exists(args.video_dir):
        video_files = [f for f in os.listdir(args.video_dir) if f.endswith('.mp4')]
        if video_files:
            print(f"\nUploading {len(video_files)} videos...")
            
            video_upload_dir = os.path.join(upload_dir, 'videos')
            os.makedirs(video_upload_dir, exist_ok=True)
            
            for vf in video_files[:3]:  # Upload up to 3 videos
                src = os.path.join(args.video_dir, vf)
                dst = os.path.join(video_upload_dir, vf)
                shutil.copy2(src, dst)
            
            try:
                upload_folder(
                    folder_path=video_upload_dir,
                    repo_id=full_repo_name,
                    repo_type="model",
                    path_in_repo="videos",
                    token=args.hf_token,
                    commit_message="Upload evaluation videos"
                )
                print("  Videos uploaded successfully!")
            except Exception as e:
                print(f"  Error uploading videos: {e}")
    
    # Cleanup
    shutil.rmtree(upload_dir)
    
    print("\n" + "="*60)
    print("PUBLISHING COMPLETE!")
    print("="*60)
    print(f"\nModel available at: https://huggingface.co/{full_repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish World Models to Hugging Face Hub")
    
    # Hugging Face
    parser.add_argument("--hf-username", type=str, required=True,
                        help="Your Hugging Face username")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--repo-name", type=str, default=None,
                        help="Repository name (default: world-models-spaceinvaders)")
    
    # Links for model card
    parser.add_argument("--wandb-url", type=str, default=None,
                        help="URL to W&B training logs")
    parser.add_argument("--github-url", type=str, default=None,
                        help="URL to GitHub repository")
    
    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--video-dir", type=str, default="videos")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    if args.hf_token is None:
        args.hf_token = os.environ.get('HF_TOKEN')
        if args.hf_token is None:
            print("Error: Please provide --hf-token or set HF_TOKEN environment variable")
            exit(1)
    
    main(args)
