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
from arm_lab.simulation.mujoco_arm import (
    check_mujoco_available,
    run_mujoco_trajectory_tracking,
    save_mujoco_trajectory_animation,
)
from arm_lab.visualization.plot_frames import set_axes_equal


def main() -> None:
    if not check_mujoco_available():
        print("MuJoCo is not installed; skipping trajectory tracking.")
        return

    figures_dir = Path("results/figures")
    gifs_dir = Path("results/gifs")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_seed = np.deg2rad([0.0, 20.0, -55.0])
    target_path = circular_target_trajectory(radius=0.18, height=0.72, steps=90)
    start_result = damped_least_squares_ik(
        robot=robot,
        target_position=target_path[0],
        q_init=q_seed,
        max_iters=100,
        damping=0.03,
        step_size=0.65,
    )
    kinematic_tracking = track_task_space_trajectory(
        robot=robot,
        target_positions=target_path,
        q_init=start_result.q,
        gain=0.9,
        damping=0.035,
        max_step_norm=0.055,
    )
    mujoco_tracking = run_mujoco_trajectory_tracking(
        q_init=start_result.q,
        q_des_history=kinematic_tracking.q_history,
        target_positions=target_path,
        hold_steps=12,
    )

    table = np.column_stack(
        [
            mujoco_tracking.time,
            target_path,
            kinematic_tracking.end_effector_positions,
            mujoco_tracking.end_effector_positions,
            kinematic_tracking.errors,
            mujoco_tracking.position_errors,
        ]
    )
    table_path = tables_dir / "mujoco_trajectory_tracking.csv"
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header=(
            "time_s,target_x,target_y,target_z,"
            "kinematic_x,kinematic_y,kinematic_z,"
            "mujoco_x,mujoco_y,mujoco_z,"
            "kinematic_error_m,mujoco_error_m"
        ),
        comments="",
    )

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], "--", linewidth=2, label="target")
    ax.plot(
        kinematic_tracking.end_effector_positions[:, 0],
        kinematic_tracking.end_effector_positions[:, 1],
        kinematic_tracking.end_effector_positions[:, 2],
        linewidth=2,
        label="kinematic path",
    )
    ax.plot(
        mujoco_tracking.end_effector_positions[:, 0],
        mujoco_tracking.end_effector_positions[:, 1],
        mujoco_tracking.end_effector_positions[:, 2],
        linewidth=2,
        label="MuJoCo path",
    )
    ax.set_title("Kinematic vs MuJoCo trajectory tracking")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    set_axes_equal(ax)

    err_ax = fig.add_subplot(122)
    err_ax.plot(kinematic_tracking.errors, label="kinematic")
    err_ax.plot(mujoco_tracking.position_errors, label="MuJoCo")
    err_ax.set_title("Tracking error comparison")
    err_ax.set_xlabel("trajectory step")
    err_ax.set_ylabel("position error [m]")
    err_ax.grid(True, alpha=0.3)
    err_ax.legend()

    fig.tight_layout()
    figure_path = figures_dir / "mujoco_trajectory_tracking_demo.png"
    fig.savefig(figure_path, dpi=160)

    gif_path = save_mujoco_trajectory_animation(
        result=mujoco_tracking,
        output_path=gifs_dir / "mujoco_trajectory_tracking_demo.gif",
        every=3,
        fps=12,
    )

    print(f"Initial target alignment converged: {start_result.converged}")
    print(f"Kinematic mean error: {np.mean(kinematic_tracking.errors):.5f} m")
    print(f"Kinematic max error: {np.max(kinematic_tracking.errors):.5f} m")
    print(f"MuJoCo mean error: {np.mean(mujoco_tracking.position_errors):.5f} m")
    print(f"MuJoCo max error: {np.max(mujoco_tracking.position_errors):.5f} m")
    print(f"Saved {figure_path}")
    print(f"Saved {gif_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
