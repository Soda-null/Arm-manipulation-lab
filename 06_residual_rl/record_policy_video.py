"""Record a trained residual PPO rollout as an MP4 video."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from common.gym_reaching_env import GymnasiumPlanarReachingEnv
from common.kinematics import joint_positions_2link


def main() -> None:
    """Render the trained residual PPO policy rollout."""
    print("Stage 06: record residual PPO rollout video")

    model_path = PROJECT_ROOT / "results/policies/residual_ppo_reaching.zip"
    output_path = PROJECT_ROOT / "results/videos/residual_ppo_reaching.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = PPO.load(model_path)
    env = GymnasiumPlanarReachingEnv(action_mode="residual")
    observation, _ = env.reset()

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    (arm_line,) = ax.plot([], [], "-o", linewidth=5, markersize=8, color="C0", label="arm")
    (path_line,) = ax.plot([], [], "-", linewidth=2.2, color="C2", alpha=0.85, label="end-effector path")
    target = env.env.config.target
    ax.scatter([target[0]], [target[1]], marker="x", s=120, color="C3", label="target")
    ax.set_title("Residual PPO Reaching Rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.25, 1.85)
    ax.set_ylim(-0.55, 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    frames: list[np.ndarray] = []
    ee_path_x: list[float] = []
    ee_path_y: list[float] = []
    render_every = 2

    for step in range(env.env.config.max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)

        if step % render_every == 0 or terminated or truncated:
            joints = joint_positions_2link(env.env.q, env.env.config.link_lengths)
            xs = [point[0] for point in joints]
            ys = [point[1] for point in joints]
            ee_path_x.append(xs[-1])
            ee_path_y.append(ys[-1])
            arm_line.set_data(xs, ys)
            path_line.set_data(ee_path_x, ee_path_y)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())

        if terminated or truncated:
            print(f"terminated={terminated}, truncated={truncated}, steps={env.env.steps}, final_error={info['error']:.6f}")
            break

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=30, macro_block_size=1)
    print(f"Rendered {len(frames)} frames at 30 fps")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
