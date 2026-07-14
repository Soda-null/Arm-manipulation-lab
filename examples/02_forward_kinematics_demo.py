from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.visualization.plot_arm_3d import plot_arm_3d


def main() -> None:
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    configs = [
        np.deg2rad([0.0, 20.0, -45.0]),
        np.deg2rad([35.0, 35.0, -65.0]),
        np.deg2rad([-45.0, 10.0, -30.0]),
    ]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx, q in enumerate(configs, start=1):
        T = forward_kinematics(robot, q)
        print(f"Config {idx}: end-effector position = {T[:3, 3]}")
        plot_arm_3d(ax, robot, q, label=f"q{idx}")

    ax.set_title("Forward kinematics demo")
    fig.tight_layout()
    out = results_dir / "fk_demo.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
