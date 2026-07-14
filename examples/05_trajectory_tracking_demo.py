from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.control.resolved_rate_control import track_task_space_trajectory
from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.planning.task_space_trajectory import circular_target_trajectory
from arm_lab.visualization.plot_frames import set_axes_equal


def main() -> None:
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_seed = np.deg2rad([0.0, 20.0, -55.0])
    trajectory = circular_target_trajectory(radius=0.22, height=0.72, steps=160)
    start_result = damped_least_squares_ik(
        robot=robot,
        target_position=trajectory[0],
        q_init=q_seed,
        max_iters=100,
        damping=0.03,
        step_size=0.65,
    )
    result = track_task_space_trajectory(
        robot=robot,
        target_positions=trajectory,
        q_init=start_result.q,
        gain=0.9,
        damping=0.035,
        max_step_norm=0.07,
    )

    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(
        result.target_positions[:, 0],
        result.target_positions[:, 1],
        result.target_positions[:, 2],
        "--",
        linewidth=2,
        label="target path",
    )
    ax.plot(
        result.end_effector_positions[:, 0],
        result.end_effector_positions[:, 1],
        result.end_effector_positions[:, 2],
        linewidth=2,
        label="end-effector path",
    )
    ax.scatter(*result.target_positions[0], marker="o", s=45, label="start")
    ax.scatter(*result.target_positions[-1], marker="x", s=60, label="finish")
    ax.set_title("Resolved-rate task-space tracking")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    set_axes_equal(ax)

    err_ax = fig.add_subplot(122)
    err_ax.plot(result.errors, linewidth=2)
    err_ax.set_title("Tracking error")
    err_ax.set_xlabel("trajectory step")
    err_ax.set_ylabel("position error [m]")
    err_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = results_dir / "trajectory_tracking_demo.png"
    fig.savefig(out, dpi=160)
    print(f"Mean tracking error: {np.mean(result.errors):.5f} m")
    print(f"Max tracking error: {np.max(result.errors):.5f} m")
    print(f"Final tracking error: {result.errors[-1]:.5f} m")
    print(f"Initial target alignment converged: {start_result.converged}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
