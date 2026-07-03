"""Stage 02 trajectory profile comparison.

This script compares position and velocity profiles for common trajectory
time-scaling methods and writes a small metrics table.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from common.metrics import control_effort, path_length, smoothness
from common.trajectories import (
    cubic_trajectory,
    linear_trajectory,
    minimum_jerk_trajectory,
    quintic_trajectory,
    trajectory_velocities,
)


def main() -> None:
    """Generate trajectory profile plots and metrics."""
    print("Stage 02: compare trajectory profiles")

    start = (0.0,)
    goal = (1.0,)
    num_steps = 100
    trajectories = {
        "linear": linear_trajectory(start, goal, num_steps),
        "cubic": cubic_trajectory(start, goal, num_steps),
        "quintic": quintic_trajectory(start, goal, num_steps),
        "minimum_jerk": minimum_jerk_trajectory(start, goal, num_steps),
    }

    plot_path = PROJECT_ROOT / "results/plots/trajectory_profiles.png"
    table_path = PROJECT_ROOT / "results/tables/trajectory_metrics.csv"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    rows: list[dict[str, str]] = []

    for name, points in trajectories.items():
        positions = [point[0] for point in points]
        velocities = [velocity[0] for velocity in trajectory_velocities(points)]
        time_positions = [index / (num_steps - 1) for index in range(num_steps)]
        time_velocities = [index / (num_steps - 2) for index in range(num_steps - 1)]

        axes[0].plot(time_positions, positions, label=name)
        axes[1].plot(time_velocities, velocities, label=name)

        rows.append(
            {
                "method": name,
                "path_length": f"{path_length(points):.6f}",
                "smoothness": f"{smoothness(points):.6f}",
                "velocity_effort": f"{control_effort(trajectory_velocities(points)):.6f}",
            }
        )

    axes[0].set_title("Trajectory Position Profiles")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Trajectory Velocity Profiles")
    axes[1].set_xlabel("normalized time")
    axes[1].set_ylabel("velocity")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "path_length", "smoothness", "velocity_effort"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved plot to {plot_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
