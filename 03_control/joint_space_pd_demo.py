"""Stage 03 joint-space PD demo.

This script simulates a simple joint-space PD controller for a planar arm.
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

from common.controllers import integrate_joint_dynamics, joint_space_pd


def main() -> None:
    """Run a joint-space PD tracking demo and save an error plot."""
    print("Stage 03: joint-space PD demo")

    q = [0.0, 0.0]
    dq = [0.0, 0.0]
    q_desired = [0.8, -0.6]
    dt = 0.01
    steps = 500
    errors: list[float] = []
    q_history: list[list[float]] = []

    for _ in range(steps):
        torque = joint_space_pd(q, q_desired, dq, kp=35.0, kd=8.0)
        q, dq = integrate_joint_dynamics(q, dq, torque, dt, damping=0.2)
        q_history.append(q)
        errors.append(sum((q_desired_i - q_i) ** 2 for q_desired_i, q_i in zip(q_desired, q)) ** 0.5)

    output_path = PROJECT_ROOT / "results/plots/joint_pd_reaching.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = [index * dt for index in range(steps)]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(time, [values[0] for values in q_history], label="q1")
    axes[0].plot(time, [values[1] for values in q_history], label="q2")
    axes[0].axhline(q_desired[0], linestyle="--", color="C0", alpha=0.6)
    axes[0].axhline(q_desired[1], linestyle="--", color="C1", alpha=0.6)
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].set_title("Joint-Space PD Tracking")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(time, errors, color="C3")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("joint error norm")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Final joint error = {errors[-1]:.6f}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
