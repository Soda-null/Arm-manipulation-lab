"""Stage 05 obstacle-avoidance video recording.

This script records the planar potential-field obstacle-avoidance rollout as an
MP4 video using a headless matplotlib renderer.
"""

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

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control
from common.kinematics import forward_kinematics_2link, jacobian_2link, joint_positions_2link
from common.potential_fields import attractive_force, repulsive_force, tangential_force


def main() -> None:
    """Render the obstacle-avoidance rollout to an MP4 file."""
    print("Stage 05: record obstacle-avoidance video")

    link_lengths = (1.0, 0.7)
    target = (1.0, 0.55)
    obstacle_center = (1.5, 0.05)
    obstacle_radius = 0.18
    influence_distance = 0.34
    q = [0.0, 0.0]
    dq = [0.0, 0.0]
    dt = 0.01
    steps = 1000
    fps = 30
    render_every = 3

    output_path = PROJECT_ROOT / "results/videos/obstacle_avoidance_potential_field.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    (arm_line,) = ax.plot([], [], "-o", linewidth=5, markersize=8, color="C0", label="arm")
    (path_line,) = ax.plot([], [], "-", linewidth=2.2, color="C2", alpha=0.85, label="end-effector path")
    obstacle = plt.Circle(obstacle_center, obstacle_radius, color="C3", alpha=0.28, label="obstacle")
    influence = plt.Circle(obstacle_center, influence_distance + obstacle_radius, fill=False, color="C3", alpha=0.16, linestyle="--")
    ax.add_patch(obstacle)
    ax.add_patch(influence)
    ax.scatter([target[0]], [target[1]], marker="x", s=120, color="C3", label="target")
    ax.set_title("Potential-Field Obstacle Avoidance")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.2, 1.8)
    ax.set_ylim(-0.45, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    ee_path_x: list[float] = []
    ee_path_y: list[float] = []
    frames: list[np.ndarray] = []

    for step in range(steps):
        position = forward_kinematics_2link(q, link_lengths)
        jacobian = jacobian_2link(q, link_lengths)
        velocity = (
            jacobian[0][0] * dq[0] + jacobian[0][1] * dq[1],
            jacobian[1][0] * dq[0] + jacobian[1][1] * dq[1],
        )
        attraction = attractive_force(position, target, velocity, kp=20.0, kd=5.5)
        repulsion = repulsive_force(position, obstacle_center, obstacle_radius, influence_distance, gain=0.015)
        tangent = tangential_force(position, target, obstacle_center, obstacle_radius, influence_distance, gain=1.2)
        force = (attraction[0] + repulsion[0] + tangent[0], attraction[1] + repulsion[1] + tangent[1])
        torque = jacobian_transpose_control(jacobian, force)
        torque = [max(-25.0, min(25.0, value)) for value in torque]
        q, dq = integrate_joint_dynamics(q, dq, torque, dt, damping=0.45)

        if step % render_every == 0:
            joints = joint_positions_2link(q, link_lengths)
            xs = [point[0] for point in joints]
            ys = [point[1] for point in joints]
            ee_path_x.append(xs[-1])
            ee_path_y.append(ys[-1])

            arm_line.set_data(xs, ys)
            path_line.set_data(ee_path_x, ee_path_y)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)

    print(f"Rendered {len(frames)} frames at {fps} fps")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
