"""Animation helpers for tracked manipulator trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from arm_lab.control.resolved_rate_control import TrajectoryTrackingResult
from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
from arm_lab.kinematics.robot_model import SerialArm
from arm_lab.visualization.plot_frames import set_axes_equal


def save_tracking_animation(
    robot: SerialArm,
    tracking: TrajectoryTrackingResult,
    output_path: str | Path,
    every: int = 3,
    fps: int = 12,
) -> Path:
    """Save a GIF showing the arm tracking a Cartesian target trajectory."""

    if every < 1:
        raise ValueError("every must be at least 1.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = np.arange(0, len(tracking.q_history), every)
    if frame_indices[-1] != len(tracking.q_history) - 1:
        frame_indices = np.append(frame_indices, len(tracking.q_history) - 1)

    all_points = np.vstack([tracking.target_positions, tracking.end_effector_positions])
    mins = all_points.min(axis=0) - 0.2
    maxs = all_points.max(axis=0) + 0.2
    mins[2] = min(0.0, mins[2])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_number: int):
        idx = int(frame_indices[frame_number])
        q = tracking.q_history[idx]
        fk = forward_kinematics_with_frames(robot, q)
        arm_points = fk.link_points
        target = tracking.target_positions[idx]
        ee_path = tracking.end_effector_positions[: idx + 1]

        ax.clear()
        ax.plot(
            tracking.target_positions[:, 0],
            tracking.target_positions[:, 1],
            tracking.target_positions[:, 2],
            "--",
            color="0.55",
            linewidth=1.5,
            label="target path",
        )
        ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="#ff7f0e", linewidth=2.2, label="tracked path")
        ax.plot(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2], "-o", color="#1f77b4", linewidth=4, label="arm")
        ax.scatter(*target, marker="*", s=110, color="red", label="current target")
        ax.scatter(*arm_points[-1], marker="x", s=70, color="black", label="end effector")
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"Task-space tracking | step {idx:03d} | error {tracking.errors[idx]:.4f} m")
        ax.legend(loc="upper right")
        set_axes_equal(ax)
        return ax.lines + ax.collections

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output
