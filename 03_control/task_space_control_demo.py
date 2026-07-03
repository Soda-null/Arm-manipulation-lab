"""Stage 03 task-space control demo.

This script simulates task-space PD control mapped through J^T for a planar arm.
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

import matplotlib.pyplot as plt

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control, task_space_pd
from common.kinematics import forward_kinematics_2link, jacobian_2link


def main() -> None:
    """Run a task-space reaching demo and save end-effector trajectory."""
    print("Stage 03: task-space control demo")

    link_lengths = (1.0, 0.7)
    q = [0.2, 0.2]
    dq = [0.0, 0.0]
    target = (1.0, 0.5)
    dt = 0.01
    steps = 700
    positions: list[tuple[float, float]] = []
    errors: list[float] = []

    for _ in range(steps):
        position = forward_kinematics_2link(q, link_lengths)
        jacobian = jacobian_2link(q, link_lengths)
        velocity = (
            jacobian[0][0] * dq[0] + jacobian[0][1] * dq[1],
            jacobian[1][0] * dq[0] + jacobian[1][1] * dq[1],
        )
        force = task_space_pd(position, target, velocity, kp=18.0, kd=4.0)
        torque = jacobian_transpose_control(jacobian, force)
        q, dq = integrate_joint_dynamics(q, dq, torque, dt, damping=0.25)
        positions.append(position)
        errors.append(((target[0] - position[0]) ** 2 + (target[1] - position[1]) ** 2) ** 0.5)

    output_path = PROJECT_ROOT / "results/plots/task_space_reaching.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [point[0] for point in positions]
    ys = [point[1] for point in positions]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, ys, label="end-effector path")
    axes[0].scatter([target[0]], [target[1]], marker="x", s=100, label="target")
    axes[0].set_title("Task-Space Reaching Path")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot([index * dt for index in range(steps)], errors, color="C3")
    axes[1].set_title("End-Effector Error")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Final task-space error = {errors[-1]:.6f}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
