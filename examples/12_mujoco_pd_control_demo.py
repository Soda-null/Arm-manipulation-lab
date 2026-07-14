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
    run_mujoco_pd_trajectory_tracking,
    run_mujoco_trajectory_tracking,
)
from arm_lab.visualization.plot_frames import set_axes_equal


def finite_difference_joint_velocity(q_history: np.ndarray, sample_dt: float) -> np.ndarray:
    """Estimate desired joint velocity from sampled desired joint angles."""

    qd = np.zeros_like(q_history)
    qd[1:-1] = (q_history[2:] - q_history[:-2]) / (2.0 * sample_dt)
    qd[0] = (q_history[1] - q_history[0]) / sample_dt
    qd[-1] = (q_history[-1] - q_history[-2]) / sample_dt
    return qd


def main() -> None:
    if not check_mujoco_available():
        print("MuJoCo is not installed; skipping PD control demo.")
        return

    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_seed = np.deg2rad([0.0, 20.0, -55.0])
    target_path = circular_target_trajectory(radius=0.16, height=0.72, steps=80)
    hold_steps = 18
    physics_dt = 0.002
    sample_dt = hold_steps * physics_dt

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
        max_step_norm=0.05,
    )
    q_des_history = kinematic_tracking.q_history
    qd_des_history = finite_difference_joint_velocity(q_des_history, sample_dt)

    position_servo = run_mujoco_trajectory_tracking(
        q_init=start_result.q,
        q_des_history=q_des_history,
        target_positions=target_path,
        hold_steps=hold_steps,
        actuator_kp=7600.0,
        joint_damping=1.0,
    )
    pd_tracking = run_mujoco_pd_trajectory_tracking(
        q_init=start_result.q,
        q_des_history=q_des_history,
        qd_des_history=qd_des_history,
        target_positions=target_path,
        hold_steps=hold_steps,
        kp=4200.0,
        kd=300.0,
        torque_limit=220.0,
    )

    table = np.column_stack(
        [
            position_servo.time,
            target_path,
            position_servo.end_effector_positions,
            pd_tracking.end_effector_positions,
            position_servo.position_errors,
            pd_tracking.position_errors,
            pd_tracking.torque_history,
        ]
    )
    table_path = tables_dir / "mujoco_pd_control_comparison.csv"
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header=(
            "time_s,target_x,target_y,target_z,"
            "position_x,position_y,position_z,"
            "pd_x,pd_y,pd_z,"
            "position_error_m,pd_error_m,"
            "tau_base,tau_shoulder,tau_elbow"
        ),
        comments="",
    )

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], "--", linewidth=2, label="target")
    ax.plot(
        position_servo.end_effector_positions[:, 0],
        position_servo.end_effector_positions[:, 1],
        position_servo.end_effector_positions[:, 2],
        linewidth=2,
        label="position actuator",
    )
    ax.plot(
        pd_tracking.end_effector_positions[:, 0],
        pd_tracking.end_effector_positions[:, 1],
        pd_tracking.end_effector_positions[:, 2],
        linewidth=2,
        label="custom PD torque",
    )
    ax.set_title("MuJoCo control comparison")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    set_axes_equal(ax)

    err_ax = fig.add_subplot(122)
    err_ax.plot(position_servo.position_errors, label="position actuator")
    err_ax.plot(pd_tracking.position_errors, label="custom PD torque")
    err_ax.set_title("Tracking error")
    err_ax.set_xlabel("trajectory step")
    err_ax.set_ylabel("position error [m]")
    err_ax.grid(True, alpha=0.3)
    err_ax.legend()

    fig.tight_layout()
    figure_path = figures_dir / "mujoco_pd_control_comparison.png"
    fig.savefig(figure_path, dpi=160)

    print(f"Initial target alignment converged: {start_result.converged}")
    print(f"Position actuator mean error: {np.mean(position_servo.position_errors):.5f} m")
    print(f"Position actuator max error: {np.max(position_servo.position_errors):.5f} m")
    print(f"Custom PD mean error: {np.mean(pd_tracking.position_errors):.5f} m")
    print(f"Custom PD max error: {np.max(pd_tracking.position_errors):.5f} m")
    print(f"Max abs torque: {np.max(np.abs(pd_tracking.torque_history)):.2f} Nm")
    print(f"Saved {figure_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
