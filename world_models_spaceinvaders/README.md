# World Models for Space Invaders

A PyTorch implementation of the **World Models** algorithm (Ha & Schmidhuber, 2018) for the Space Invaders Atari environment.

## 🎮 Overview

World Models is a model-based reinforcement learning approach that decomposes learning into three components:

1. **VAE (Vision Model)**: Compresses high-dimensional game frames into a compact latent representation
2. **MDN-RNN (Memory Model)**: Models temporal dynamics and predicts future latent states using a Mixture Density Network
3. **Controller**: Simple policy network that acts in the compressed latent space

![World Models Architecture](https://worldmodels.github.io/assets/world_models_schematic.jpg)

## 📁 Project Structure

```
world_models_spaceinvaders/
├── models/
│   ├── vae.py           # Convolutional VAE
│   ├── mdn_rnn.py       # MDN-RNN dynamics model
│   └── controller.py    # Policy controller
├── utils/
│   ├── wrappers.py      # Gym environment wrappers
│   └── data.py          # Data collection and datasets
├── baselines/
│   ├── ddqn.py          # Double DQN baseline
│   └── ppo.py           # PPO baseline
├── train_vae.py         # VAE training script
├── train_mdn_rnn.py     # MDN-RNN training script
├── train_controller.py  # Controller training script
├── train_world_model.py # Complete training pipeline
├── evaluate.py          # Evaluation and video recording
├── publish_model.py     # Hugging Face publishing
├── config.py            # Hyperparameters
└── requirements.txt     # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd world_models_spaceinvaders

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

**Full Training Pipeline:**
```bash
python train_world_model.py --wandb-project your-project-name
```

**Quick Test (dry run):**
```bash
python train_world_model.py --dry-run --no-wandb
```

**Individual Training Steps:**
```bash
# Step 1: Train VAE
python train_vae.py --num-epochs 30

# Step 2: Train MDN-RNN
python train_mdn_rnn.py --num-epochs 30

# Step 3: Train Controller
python train_controller.py --dream-epochs 100 --real-epochs 50
```

### 3. Evaluation

```bash
python evaluate.py --num-episodes 10 --record-video
```

### 4. Publish to Hugging Face

```bash
python publish_model.py --hf-username YOUR_USERNAME --hf-token YOUR_TOKEN
```

## 📊 Model-Free Baseline Comparisons (Bonus)

Compare World Models with model-free algorithms: **DDQN**, **PPO**, and **SAC**.

### Train All Baselines

```bash
# Train all baselines and run comparison
python train_baselines.py --algorithms all

# Train specific algorithms
python train_baselines.py --algorithms ddqn sac

# Quick training for testing
python train_baselines.py --ddqn-episodes 500 --sac-episodes 500 --ppo-timesteps 100000
```

### Train Individual Baselines

```bash
# Train DDQN (Double Deep Q-Network)
python baselines/ddqn.py --mode train --num-episodes 2000

# Train PPO (Proximal Policy Optimization)
python baselines/ppo.py --mode train --total-timesteps 500000

# Train SAC (Soft Actor-Critic)
python baselines/sac.py --mode train --num-episodes 2000
```

### Compare All Algorithms

```bash
# Run comparison (requires trained models)
python compare_baselines.py --num-episodes 20

# Comparison without WandB
python compare_baselines.py --num-episodes 20 --no-wandb
```

This generates:
- **Bar chart**: Mean rewards with standard deviation
- **Box plot**: Distribution of episode rewards
- **Summary table**: Statistics for all algorithms
- **JSON results**: Detailed metrics saved to `comparison_results/`

### Evaluate Individual Baselines

```bash
# Evaluate (with video recording)
python baselines/ddqn.py --mode eval --eval-episodes 10
python baselines/ppo.py --mode eval --eval-episodes 10
python baselines/sac.py --mode eval --eval-episodes 10
```

## ⚙️ Key Hyperparameters

| Component | Parameter | Default |
|-----------|-----------|---------|
| VAE | latent_dim | 32 |
| VAE | kl_weight | 0.0001 |
| MDN-RNN | hidden_size | 256 |
| MDN-RNN | num_mixtures | 5 |
| Controller | hidden_size | 64 |
| Training | batch_size | 64 |
| Training | learning_rate | 1e-3 |

## 📈 Weights & Biases Integration

All training scripts support W&B logging:

```bash
# Login to W&B
wandb login

# Training with logging
python train_world_model.py --wandb-project world-models-spaceinvaders

# Disable logging
python train_world_model.py --no-wandb
```

## 🎥 Video Recording

Videos are automatically recorded during evaluation using Gymnasium's `RecordVideo` wrapper. Videos are saved to the `videos/` directory and can be uploaded to W&B.

## 📚 References

- [World Models Paper](https://worldmodels.github.io/) - Ha & Schmidhuber, 2018
- [ALE Documentation](https://ale.farama.org/)
- [Gymnasium Atari](https://gymnasium.farama.org/environments/atari/)

## 📄 Citation

```bibtex
@article{ha2018worldmodels,
  title={World Models},
  author={Ha, David and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:1803.10122},
  year={2018}
}
```

## 📝 License

MIT License
