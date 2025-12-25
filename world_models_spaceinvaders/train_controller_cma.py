"""
Train Controller using CMA-ES
Standard approach for World Models to avoid local optima (immobility).
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing as mp

try:
    import cma
except ImportError:
    print("Error: cma package not found. Please install with: pip install cma")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vae import VAE
from models.mdn_rnn import MDNRNN
from models.controller import Controller
from utils.wrappers import make_env

def get_reward(params, batch_id, args):
    """
    Evaluate a single controller solution (params).
    Can be run in parallel.
    """
    # Create models (load weights in each process if needed, or share)
    # Ideally we load VAE/RNN once. But for multiprocessing, we might need to reload 
    # or rely on fork. Windows uses spawn, so we must reload or pass state.
    
    device = torch.device("cpu") # Controller is small, CPU is fine for rollout
    
    # Load VAE and RNN (This is slow if done every time. 
    # Better to initialize once per worker)
    # For simplicity in this script, we'll optimistically load them.
    # A true parallel implementation would use a worker class.
    
    # ... (Implementation details: We'll implement a worker function below)
    return 0 

# Global variables for workers (Windows compatible if using init function)
g_vae = None
g_rnn = None
g_controller = None
g_env = None

def init_worker(vae_path, rnn_path, env_name, device_str):
    global g_vae, g_rnn, g_controller, g_env
    
    device = torch.device(device_str)
    
    # Load VAE
    g_vae = VAE(device=device).to(device)
    g_vae.load_state_dict(torch.load(vae_path, map_location=device))
    g_vae.eval()
    
    # Load RNN
    g_rnn = MDNRNN(input_size=32, hidden_size=256, action_size=6, num_gaussians=5, device=device).to(device)
    g_rnn.load_state_dict(torch.load(rnn_path, map_location=device))
    g_rnn.eval()
    
    # Init Controller
    # (Input: z=32 + h=256 = 288)
    g_controller = Controller(input_size=32+256, action_size=6, device=device).to(device)
    
    # Create Env
    g_env = make_env(env_name, frame_size=64)

def flatten_params(model):
    """Flatten model parameters to a single vector."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def load_params(model, flattened_params):
    """Load flattened parameters into model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(torch.from_numpy(flattened_params[offset:offset+numel]).view_as(p.data))
        offset += numel

def evaluate_worker(params):
    """Run one episode with the given controller parameters."""
    global g_vae, g_rnn, g_controller, g_env
    
    # Load params into controller
    load_params(g_controller, params)
    
    obs, info = g_env.reset()
    state = torch.FloatTensor(obs).to(g_controller.device).unsqueeze(0) # (1, 3, 64, 64)
    
    total_reward = 0
    hidden = g_rnn.init_hidden(1)
    
    done = False
    
    with torch.no_grad():
        while not done:
            # VAE encode
            mu, logvar = g_vae.encode(state)
            z = g_vae.reparameterize(mu, logvar) # (1, 32)
            
            # Controller action
            # Inputs: z + h
            # h is (dirs*layers, batch, hidden). We want last layer hidden state.
            # MDNRNN uses 1 layer. hidden is (1, 1, 256).
            h = hidden[0].squeeze(0) # (1, 256)
            
            check_input = torch.cat([z, h], dim=1)
            action = g_controller.get_action(check_input)
            
            # Env step
            next_obs, reward, terminated, truncated, info = g_env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # RNN step
            # Predict next z and reward (dreaming) - or just update state
            # We need to update RNN state given (z, a)
            action_one_hot = torch.zeros(1, 6).to(g_controller.device)
            action_one_hot[0, action] = 1
            
            # MDNRNN forward returns (pi, mu, sigma, reward, done, next_hidden)
            _, _, _, _, _, next_hidden = g_rnn(z.unsqueeze(0), action_one_hot.unsqueeze(0), hidden)
            hidden = next_hidden
            
            state = torch.FloatTensor(next_obs).to(g_controller.device).unsqueeze(0)
            
    return total_reward

def main(args):
    # Setup
    device = torch.device(args.device)
    
    # Calculate parameter count
    dummy = Controller(input_size=32+256, action_size=6)
    param_count = sum(p.numel() for p in dummy.parameters())
    print(f"Controller has {param_count} parameters")
    
    # CMA-ES options
    cma_opts = {
        'popsize': args.pop_size,
        'seed': args.seed,
        'maxiter': args.generations,
        'verbose': -9 # Suppress CMA output, we'll log ourselves
    }
    
    # Initialize CMA-ES
    if args.continue_from:
        print(f"Continuing from {args.continue_from}")
        with open(args.continue_from, 'rb') as f:
            es = pickle.load(f)
    else:
        # Start from 0 or small random
        initial_params = np.zeros(param_count) # Often start with 0 weights
        es = cma.CMAEvolutionStrategy(initial_params, args.sigma, cma_opts)
    
    print(f"Starting CMA-ES training for {args.generations} generations...")
    
    best_ever_reward = -float('inf')
    
    # Multiprocessing pool
    # Use 'spawn' for Windows/CUDA compatibility if needed, but 'spawn' is slow to init
    # For CPU-based VAE/RNN inference (small models), it's fast enough.
    mp.set_start_method('spawn', force=True)
    
    # We need to pass args to init_worker
    worker_args = (args.vae_path, args.rnn_path, args.env_name, "cpu") 
    # Use CPU for workers to avoid CUDA OOM with many processes
    
    with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=worker_args) as pool:
        
        for gen in range(args.generations):
            if es.stop():
                break
                
            # Ask for candidate solutions
            solutions = es.ask()
            
            # Evaluate solutions in parallel
            # Use 'map' to distribute work
            rewards = pool.map(evaluate_worker, solutions)
            
            # Convert to numpy
            rewards = np.array(rewards)
            
            # Tell CMA-ES the results (minimize neg reward)
            es.tell(solutions, -rewards)
            
            # Display stats
            mean = np.mean(rewards)
            max_r = np.max(rewards)
            min_r = np.min(rewards)
            
            print(f"Gen {gen+1}: Max={max_r:.1f}, Mean={mean:.1f}, Min={min_r:.1f}, Sigma={es.sigma:.3f}")
            
            # Save checkpoint
            if (gen + 1) % args.save_interval == 0:
                # Save just the best param vector
                best_params = es.result.xbest
                torch.save(best_params, os.path.join(args.checkpoint_dir, f"controller_cma_gen_{gen+1}.pt"))
                
                # Also save ES state
                with open(os.path.join(args.checkpoint_dir, "es_state.pkl"), 'wb') as f:
                    pickle.dump(es, f)
            
            if max_r > best_ever_reward:
                best_ever_reward = max_r
                # Save best
                best_params = es.result.xbest
                torch.save(best_params, os.path.join(args.checkpoint_dir, "controller_cma_best.pt"))
                print(f"  -> New best reward: {max_r}")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="checkpoints/vae_final.pt")
    parser.add_argument("--rnn-path", type=str, default="checkpoints/mdn_rnn_final.pt")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--env-name", type=str, default="ALE/SpaceInvaders-v5")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--continue-from", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=10)
    
    args = parser.parse_args()
    main(args)
