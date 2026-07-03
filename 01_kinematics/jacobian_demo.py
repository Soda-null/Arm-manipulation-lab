"""Stage 01 Jacobian demo.

This script visualizes planar 2-link manipulability over joint space.
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

from common.kinematics import manipulability_2link


def main() -> None:
    """Generate a manipulability heatmap for a 2-link planar arm."""
    print("Stage 01: Jacobian demo")

    link_lengths = (1.0, 0.7)
    q1_values = np.linspace(-math.pi, math.pi, 181)
    q2_values = np.linspace(-math.pi, math.pi, 181)
    values = np.zeros((len(q2_values), len(q1_values)))

    for row, q2 in enumerate(q2_values):
        for col, q1 in enumerate(q1_values):
            values[row, col] = manipulability_2link((q1, q2), link_lengths)

    output_path = PROJECT_ROOT / "results/plots/manipulability_map.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        values,
        origin="lower",
        extent=[-math.pi, math.pi, -math.pi, math.pi],
        aspect="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="manipulability |det(J)|")
    ax.set_title("2-Link Planar Arm Manipulability")
    ax.set_xlabel("q1 [rad]")
    ax.set_ylabel("q2 [rad]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
