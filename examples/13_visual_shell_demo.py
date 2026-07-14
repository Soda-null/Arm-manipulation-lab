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
from arm_lab.visualization.robot_shell import plot_robot_shell, save_shell_tracking_animation


def main() -> None:
    figures_dir = Path("results/figures")
    gifs_dir = Path("results/gifs")
    figures_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

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
    tracking = track_task_space_trajectory(
        robot=robot,
        target_positions=target_path,
        q_init=start_result.q,
        gain=0.9,
        damping=0.035,
        max_step_norm=0.055,
    )

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], "--", color="0.55", label="target path")
    ax.plot(
        tracking.end_effector_positions[:, 0],
        tracking.end_effector_positions[:, 1],
        tracking.end_effector_positions[:, 2],
        color="#e56b2f",
        linewidth=2,
        label="tracked path",
    )
    plot_robot_shell(ax, robot, tracking.q_history[len(tracking.q_history) // 3], label="arm shell")
    ax.set_title("3D arm visual shell")
    ax.legend(loc="upper right")
    figure_path = figures_dir / "robot_shell_demo.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)

    gif_path = save_shell_tracking_animation(
        robot=robot,
        tracking=tracking,
        output_path=gifs_dir / "robot_shell_tracking_demo.gif",
        every=4,
        fps=12,
    )

    print(f"Initial target alignment converged: {start_result.converged}")
    print(f"Mean tracking error: {np.mean(tracking.errors):.5f} m")
    print(f"Saved {figure_path}")
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
