from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from arm_lab.control.resolved_rate_control import track_task_space_trajectory
from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.planning.task_space_trajectory import circular_target_trajectory
from arm_lab.visualization.animate_trajectory import save_tracking_animation


def main() -> None:
    output_dir = Path("results/gifs")
    output_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_seed = np.deg2rad([0.0, 20.0, -55.0])
    trajectory = circular_target_trajectory(radius=0.22, height=0.72, steps=120)
    start_result = damped_least_squares_ik(
        robot=robot,
        target_position=trajectory[0],
        q_init=q_seed,
        max_iters=100,
        damping=0.03,
        step_size=0.65,
    )
    tracking = track_task_space_trajectory(
        robot=robot,
        target_positions=trajectory,
        q_init=start_result.q,
        gain=0.9,
        damping=0.035,
        max_step_norm=0.07,
    )
    output = save_tracking_animation(
        robot=robot,
        tracking=tracking,
        output_path=output_dir / "trajectory_tracking_demo.gif",
        every=3,
        fps=12,
    )

    print(f"Initial target alignment converged: {start_result.converged}")
    print(f"Mean tracking error: {np.mean(tracking.errors):.5f} m")
    print(f"Max tracking error: {np.max(tracking.errors):.5f} m")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
