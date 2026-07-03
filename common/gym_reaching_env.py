"""Gymnasium adapter for the lightweight planar reaching environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common.reaching_env import PlanarReachingEnv, ReachingEnvConfig


class GymnasiumPlanarReachingEnv(gym.Env):
    """Gymnasium-compatible wrapper for PPO training.

    Actions are normalized to ``[-1, 1]``:
    * vanilla mode maps actions to direct torque via ``torque_limit``.
    * residual mode passes actions as residual corrections scaled internally.
    """

    metadata = {"render_modes": []}

    def __init__(self, action_mode: str = "vanilla", config: ReachingEnvConfig | None = None) -> None:
        super().__init__()
        self.env = PlanarReachingEnv(config=config, action_mode=action_mode)
        self.action_mode = action_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        observation = self.env.reset()
        return np.asarray(observation, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the wrapped environment."""
        clipped_action = np.clip(action.astype(np.float32), -1.0, 1.0)
        if self.action_mode == "vanilla":
            env_action = clipped_action * self.env.config.torque_limit
        else:
            env_action = clipped_action
        observation, reward, terminated, truncated, info = self.env.step(env_action.tolist())
        return np.asarray(observation, dtype=np.float32), float(reward), terminated, truncated, info
