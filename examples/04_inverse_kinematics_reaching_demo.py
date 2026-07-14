from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.visualization.plot_arm_3d import plot_arm_3d


def main() -> None:
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 15.0, -25.0])
    target = np.array([1.25, 0.35, 0.75])

    result = damped_least_squares_ik(
        robot,
        target_position=target,
        q_init=q_init,
        max_iters=120,
        damping=0.03,
        step_size=0.65,
    )
    final_position = forward_kinematics(robot, result.q)[:3, 3]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.errors, linewidth=2)
    ax.set_title("Damped least-squares IK convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Position error [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = results_dir / "ik_convergence.png"
    fig.savefig(out, dpi=160)

    pose_fig = plt.figure(figsize=(7, 6))
    pose_ax = pose_fig.add_subplot(111, projection="3d")
    plot_arm_3d(pose_ax, robot, result.q, label="IK solution")
    pose_ax.scatter(target[0], target[1], target[2], s=80, marker="*", color="red", label="target")
    pose_ax.legend()
    pose_ax.set_title("Final IK pose")
    pose_fig.tight_layout()
    pose_out = results_dir / "ik_final_pose.png"
    pose_fig.savefig(pose_out, dpi=160)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Final q [deg]: {np.rad2deg(result.q)}")
    print(f"Target: {target}")
    print(f"Final position: {final_position}")
    print(f"Final error: {np.linalg.norm(target - final_position):.6f} m")
    print(f"Saved {out}")
    print(f"Saved {pose_out}")


if __name__ == "__main__":
    main()
