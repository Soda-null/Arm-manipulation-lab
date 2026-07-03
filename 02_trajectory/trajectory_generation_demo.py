"""Stage 02 trajectory generation demo.

This script demonstrates simple 2D trajectory generation and saves a path plot.
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

from common.trajectories import cubic_trajectory, linear_trajectory, minimum_jerk_trajectory, quintic_trajectory


def main() -> None:
    """Generate and plot basic point-to-point trajectories."""
    print("Stage 02: trajectory generation demo")

    start = (0.2, 0.2)
    goal = (1.2, 0.8)
    num_steps = 100
    trajectories = {
        "linear": linear_trajectory(start, goal, num_steps),
        "cubic": cubic_trajectory(start, goal, num_steps),
        "quintic": quintic_trajectory(start, goal, num_steps),
        "minimum jerk": minimum_jerk_trajectory(start, goal, num_steps),
    }

    output_path = PROJECT_ROOT / "results/plots/trajectory_generation_demo.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, points in trajectories.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, label=name)

    ax.scatter([start[0]], [start[1]], marker="o", s=80, label="start")
    ax.scatter([goal[0]], [goal[1]], marker="x", s=100, label="goal")
    ax.set_title("Point-to-Point Trajectory Generation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
