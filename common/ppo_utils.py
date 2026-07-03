"""Small PPO training/evaluation helpers for Stage 06."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

from stable_baselines3 import PPO

from common.gym_reaching_env import GymnasiumPlanarReachingEnv


def train_ppo(action_mode: str, total_timesteps: int, seed: int, model_path: Path) -> PPO:
    """Train a small PPO policy and save it."""
    env = GymnasiumPlanarReachingEnv(action_mode=action_mode)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        gamma=0.98,
        device="cpu",
    )
    model.learn(total_timesteps=total_timesteps)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model


def evaluate_model(model: PPO, action_mode: str, episodes: int = 5) -> dict[str, float]:
    """Evaluate a trained model on the reaching task."""
    successes = 0
    final_errors: list[float] = []
    steps: list[int] = []
    returns: list[float] = []
    efforts: list[float] = []

    for _ in range(episodes):
        env = GymnasiumPlanarReachingEnv(action_mode=action_mode)
        observation, _ = env.reset()
        episode_return = 0.0
        effort = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            effort += float(info["effort"])
            if terminated or truncated:
                successes += int(bool(info["success"]))
                final_errors.append(float(info["error"]))
                steps.append(env.env.steps)
                returns.append(episode_return)
                efforts.append(effort)
                break

    return {
        "success_rate": successes / episodes,
        "final_error": sum(final_errors) / episodes,
        "steps": sum(steps) / episodes,
        "return": sum(returns) / episodes,
        "control_effort": sum(efforts) / episodes,
    }
