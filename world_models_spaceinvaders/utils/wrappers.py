"""
Environment Wrappers for Atari
Standard preprocessing for World Models training
"""

import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
    TransformObservation,
    ResizeObservation
)
import numpy as np
from typing import Optional
import cv2

# Register ALE environments
import ale_py
gym.register_envs(ale_py)


class ResizeAndNormalize(gym.ObservationWrapper):
    """Resize frames to target size and normalize to [0, 1]."""
    
    def __init__(self, env: gym.Env, size: int = 64):
        super().__init__(env)
        self.size = size
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, size, size),
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Resize
        obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        # Channel first (for PyTorch)
        obs = np.transpose(obs, (2, 0, 1))
        return obs


class FlattenFrameStack(gym.ObservationWrapper):
    """Flatten stacked frames from (S, C, H, W) to (S*C, H, W)."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        # (Stack, C, H, W) -> (Stack*C, H, W)
        new_shape = (old_shape[0] * old_shape[1], old_shape[2], old_shape[3])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )
        
    def observation(self, obs):
        # obs is (Stack, C, H, W). Flatten to (Stack*C, H, W)
        return obs.reshape(-1, obs.shape[-2], obs.shape[-1])


class FireResetEnv(gym.Wrapper):
    """Take FIRE action on reset for environments that require it."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Check if FIRE action exists
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every skip-th frame (frameskipping) and max over last 2."""
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    
    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        # Max over last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        
        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, 1}."""
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)


def make_env(
    env_name: str = "ALE/SpaceInvaders-v5",
    frame_size: int = 64,
    frame_skip: int = 4,
    clip_rewards: bool = False,  # Disabled by default for better reward learning
    record_video: bool = False,
    video_folder: str = "videos",
    episode_trigger: Optional[callable] = None,
    render_mode: str = None,
    stack_frames: int = 1
) -> gym.Env:
    """
    Create and wrap Atari environment for World Models training.
    
    Args:
        env_name: Gymnasium environment name
        frame_size: Size to resize frames to (frame_size x frame_size)
        frame_skip: Number of frames to skip
        clip_rewards: Whether to clip rewards
        record_video: Whether to record video
        video_folder: Folder to save videos
        episode_trigger: Function to determine which episodes to record
        render_mode: Render mode for environment
        stack_frames: Number of frames to stack (default 1)
        
    Returns:
        Wrapped Gymnasium environment
    """
    # Ensure render_mode is set if recording video
    if record_video and render_mode is None:
        render_mode = "rgb_array"
        
    # Create base environment
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    
    # Track episode statistics (raw rewards) before any clipping/processing
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Frame skipping (if not using AtariPreprocessing)
    # Note: ALE/SpaceInvaders-v5 already has frameskip built in
    # We'll skip additional preprocessing since we want RGB
    
    # Fire on reset (Space Invaders needs this)
    try:
        env = FireResetEnv(env)
    except (AssertionError, IndexError):
        pass  # Environment doesn't need fire reset
    
    # Clip rewards
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    # Resize and normalize
    env = ResizeAndNormalize(env, size=frame_size)
    
    # Record video
    if record_video:
        if episode_trigger is None:
            episode_trigger = lambda x: True  # Record all episodes
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            name_prefix="world_models"
        )
        
    # Stack frames
    if stack_frames > 1:
        env = FrameStackObservation(env, stack_size=stack_frames)
        env = FlattenFrameStack(env)
    
    return env


if __name__ == "__main__":
    # Test environment creation
    print("Testing environment creation...")
    
    env = make_env("ALE/SpaceInvaders-v5", frame_size=64)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 0:
            print(f"After step - obs shape: {obs.shape}")
    
    env.close()
    print("Environment test passed!")
