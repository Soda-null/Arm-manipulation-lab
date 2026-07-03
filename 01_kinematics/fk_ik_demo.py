"""Stage 01 FK/IK demo.

This script demonstrates forward and inverse kinematics for a planar 2-link arm
and saves a simple target-reaching plot.
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

from common.kinematics import forward_kinematics_2link, inverse_kinematics_2link, joint_positions_2link


def main() -> None:
    """Run a small FK/IK smoke demo and save the resulting plot."""
    print("Stage 01: FK/IK demo")

    link_lengths = (1.0, 0.7)
    target = (1.0, 0.5)
    q = inverse_kinematics_2link(target, link_lengths)
    end_effector = forward_kinematics_2link(q, link_lengths)
    joints = joint_positions_2link(q, link_lengths)

    output_path = PROJECT_ROOT / "results/plots/fk_ik_demo.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [point[0] for point in joints]
    ys = [point[1] for point in joints]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "-o", linewidth=4, markersize=8, label="2-link arm")
    ax.scatter([target[0]], [target[1]], marker="x", s=120, label="target")
    ax.scatter([end_effector[0]], [end_effector[1]], s=60, label="FK end effector")
    ax.set_title("2-Link Planar Arm FK/IK Demo")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"IK solution q = ({q[0]:.4f}, {q[1]:.4f}) rad")
    print(f"FK end effector = ({end_effector[0]:.4f}, {end_effector[1]:.4f})")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
