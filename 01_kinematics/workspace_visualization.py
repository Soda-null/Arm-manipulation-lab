"""Stage 01 workspace visualization.

This script samples joint configurations and plots the reachable workspace for a
planar 2-link arm.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from common.kinematics import forward_kinematics_2link, manipulability_2link


def main() -> None:
    """Generate a reachable workspace plot with near-singular samples highlighted."""
    print("Stage 01: workspace visualization")

    link_lengths = (1.0, 0.7)
    q1_values = np.linspace(-math.pi, math.pi, 181)
    q2_values = np.linspace(-math.pi, math.pi, 181)
    workspace_points: list[tuple[float, float]] = []
    singular_points: list[tuple[float, float]] = []

    for q1 in q1_values:
        for q2 in q2_values:
            point = forward_kinematics_2link((q1, q2), link_lengths)
            workspace_points.append(point)
            if manipulability_2link((q1, q2), link_lengths) < 0.03:
                singular_points.append(point)

    output_path = PROJECT_ROOT / "results/plots/workspace_2link.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wx = [point[0] for point in workspace_points]
    wy = [point[1] for point in workspace_points]
    sx = [point[0] for point in singular_points]
    sy = [point[1] for point in singular_points]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(wx, wy, s=1, alpha=0.18, label="reachable samples")
    ax.scatter(sx, sy, s=2, alpha=0.35, label="near-singular samples")
    ax.set_title("2-Link Planar Arm Workspace")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Sampled {len(workspace_points)} configurations")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
