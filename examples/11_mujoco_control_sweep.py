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
from arm_lab.simulation.mujoco_arm import check_mujoco_available, run_mujoco_trajectory_tracking


def main() -> None:
    if not check_mujoco_available():
        print("MuJoCo is not installed; skipping control sweep.")
        return

    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_seed = np.deg2rad([0.0, 20.0, -55.0])
    target_path = circular_target_trajectory(radius=0.18, height=0.72, steps=70)
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

    actuator_kps = [1800.0, 4200.0, 7600.0]
    joint_dampings = [1.0, 2.0]
    hold_steps_values = [6, 12, 24]

    rows: list[list[float]] = []
    config_id = 0
    for joint_damping in joint_dampings:
        for actuator_kp in actuator_kps:
            for hold_steps in hold_steps_values:
                result = run_mujoco_trajectory_tracking(
                    q_init=start_result.q,
                    q_des_history=kinematic_tracking.q_history,
                    target_positions=target_path,
                    hold_steps=hold_steps,
                    actuator_kp=actuator_kp,
                    joint_damping=joint_damping,
                )
                rows.append(
                    [
                        config_id,
                        actuator_kp,
                        joint_damping,
                        hold_steps,
                        float(result.time[-1]),
                        float(np.mean(result.position_errors)),
                        float(np.max(result.position_errors)),
                        float(result.position_errors[-1]),
                    ]
                )
                config_id += 1

    table = np.array(rows)
    table_path = tables_dir / "mujoco_control_sweep.csv"
    np.savetxt(
        table_path,
        table,
        delimiter=",",
        header="config_id,actuator_kp,joint_damping,hold_steps,duration_s,mean_error_m,max_error_m,final_error_m",
        comments="",
    )

    best_mean = table[np.argmin(table[:, 5])]
    best_max = table[np.argmin(table[:, 6])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for joint_damping in joint_dampings:
        damping_rows = table[table[:, 2] == joint_damping]
        for actuator_kp in actuator_kps:
            subset = damping_rows[damping_rows[:, 1] == actuator_kp]
            subset = subset[np.argsort(subset[:, 3])]
            axes[0].plot(
                subset[:, 3],
                subset[:, 5],
                marker="o",
                label=f"kp={actuator_kp:.0f}, damping={joint_damping:.0f}",
            )
    axes[0].set_title("Mean tracking error")
    axes[0].set_xlabel("hold_steps")
    axes[0].set_ylabel("mean error [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    scatter = axes[1].scatter(table[:, 5], table[:, 6], c=table[:, 3], s=75, cmap="viridis")
    axes[1].scatter(best_mean[5], best_mean[6], marker="*", s=170, color="red", label="best mean")
    axes[1].set_title("Mean vs max error")
    axes[1].set_xlabel("mean error [m]")
    axes[1].set_ylabel("max error [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label("hold_steps")

    fig.tight_layout()
    figure_path = figures_dir / "mujoco_control_sweep.png"
    fig.savefig(figure_path, dpi=160)

    print(f"Initial target alignment converged: {start_result.converged}")
    print(
        "Best mean error config: "
        f"kp={best_mean[1]:.0f}, damping={best_mean[2]:.0f}, hold_steps={best_mean[3]:.0f}, "
        f"mean={best_mean[5]:.5f} m, max={best_mean[6]:.5f} m"
    )
    print(
        "Best max error config: "
        f"kp={best_max[1]:.0f}, damping={best_max[2]:.0f}, hold_steps={best_max[3]:.0f}, "
        f"mean={best_max[5]:.5f} m, max={best_max[6]:.5f} m"
    )
    print(f"Saved {table_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
