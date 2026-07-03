"""Stage 03 Jacobian transpose control demo.

This script visualizes how a task-space force maps to joint torques through
the Jacobian transpose.
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

from common.controllers import jacobian_transpose_control
from common.kinematics import forward_kinematics_2link, jacobian_2link, joint_positions_2link


def main() -> None:
    """Compute and visualize J^T F for one planar arm configuration."""
    print("Stage 03: Jacobian transpose control demo")

    link_lengths = (1.0, 0.7)
    q = (0.5, -0.8)
    target = (1.1, 0.4)
    ee = forward_kinematics_2link(q, link_lengths)
    force = (target[0] - ee[0], target[1] - ee[1])
    torque = jacobian_transpose_control(jacobian_2link(q, link_lengths), force)
    joints = joint_positions_2link(q, link_lengths)

    output_path = PROJECT_ROOT / "results/plots/jacobian_transpose_demo.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [point[0] for point in joints]
    ys = [point[1] for point in joints]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "-o", linewidth=4, markersize=8, label="2-link arm")
    ax.scatter([target[0]], [target[1]], marker="x", s=120, label="target")
    ax.arrow(ee[0], ee[1], force[0], force[1], head_width=0.04, length_includes_head=True, color="C3", label="task force")
    ax.set_title("Jacobian Transpose Mapping")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Task force = ({force[0]:.4f}, {force[1]:.4f})")
    print(f"Joint torque = ({torque[0]:.4f}, {torque[1]:.4f})")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
