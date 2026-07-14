"""Visual shell rendering for the simple 3D arm."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from arm_lab.control.resolved_rate_control import TrajectoryTrackingResult
from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
from arm_lab.kinematics.robot_model import SerialArm
from arm_lab.visualization.plot_frames import set_axes_equal


SHELL_COLORS = {
    "base": "#263241",
    "joint": "#1aa39a",
    "link": "#d6a83f",
    "gripper": "#f2f4f5",
    "gripper_bar": "#5f6873",
    "path": "#e56b2f",
    "target": "#d62728",
}


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    return axis, u, v


def _plot_cylinder_between(ax, start: np.ndarray, end: np.ndarray, radius: float, color: str, alpha: float = 1.0) -> None:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return
    axis, u, v = _orthonormal_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    s = np.linspace(0.0, length, 10)
    theta_grid, s_grid = np.meshgrid(theta, s)
    center = start + s_grid[..., None] * axis
    circle = radius * (np.cos(theta_grid)[..., None] * u + np.sin(theta_grid)[..., None] * v)
    surface = center + circle
    ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], color=color, alpha=alpha, linewidth=0, shade=True)


def _plot_sphere(ax, center: np.ndarray, radius: float, color: str, alpha: float = 1.0) -> None:
    phi = np.linspace(0.0, np.pi, 16)
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = center[1] + radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = center[2] + radius * np.cos(phi_grid)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def _plot_gripper(ax, wrist: np.ndarray, tip: np.ndarray) -> None:
    direction = tip - wrist
    if np.linalg.norm(direction) < 1e-9:
        return
    axis, u, _v = _orthonormal_basis(direction)
    palm = tip - 0.08 * axis
    left_base = palm + 0.055 * u
    right_base = palm - 0.055 * u
    finger_tip_left = left_base + 0.16 * axis
    finger_tip_right = right_base + 0.16 * axis
    _plot_cylinder_between(ax, left_base, finger_tip_left, radius=0.014, color=SHELL_COLORS["gripper"])
    _plot_cylinder_between(ax, right_base, finger_tip_right, radius=0.014, color=SHELL_COLORS["gripper"])
    _plot_cylinder_between(ax, left_base, right_base, radius=0.018, color=SHELL_COLORS["gripper_bar"])


def plot_robot_shell(ax, robot: SerialArm, q: np.ndarray, label: str | None = "robot shell") -> np.ndarray:
    """Render a simple mechanical-looking shell around the current arm pose."""

    fk = forward_kinematics_with_frames(robot, q)
    points = fk.link_points

    for point in points[1:]:
        _plot_sphere(ax, point, radius=0.075, color=SHELL_COLORS["joint"])
    for start, end, radius in zip(points[1:-1], points[2:], [0.048, 0.04]):
        _plot_cylinder_between(ax, start, end, radius=radius, color=SHELL_COLORS["link"])
    _plot_gripper(ax, points[-2], points[-1])

    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#202020", linewidth=1.2, alpha=0.5, label=label)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], marker="x", s=70, color="black")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    set_axes_equal(ax)
    return points


def save_shell_tracking_animation(
    robot: SerialArm,
    tracking: TrajectoryTrackingResult,
    output_path: str | Path,
    every: int = 4,
    fps: int = 12,
) -> Path:
    """Save a GIF of the shell-styled arm following a tracked path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = np.arange(0, len(tracking.q_history), every)
    if frame_indices[-1] != len(tracking.q_history) - 1:
        frame_indices = np.append(frame_indices, len(tracking.q_history) - 1)

    arm_points_over_time = [
        forward_kinematics_with_frames(robot, tracking.q_history[idx]).link_points
        for idx in frame_indices
    ]
    all_points = np.vstack([tracking.target_positions, tracking.end_effector_positions, *arm_points_over_time])
    mins = all_points.min(axis=0) - 0.25
    maxs = all_points.max(axis=0) + 0.25
    mins[2] = min(0.0, mins[2])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_number: int):
        idx = int(frame_indices[frame_number])
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
        ax.plot(
            tracking.end_effector_positions[: idx + 1, 0],
            tracking.end_effector_positions[: idx + 1, 1],
            tracking.end_effector_positions[: idx + 1, 2],
            color=SHELL_COLORS["path"],
            linewidth=2,
            label="tracked path",
        )
        plot_robot_shell(ax, robot, tracking.q_history[idx], label="arm shell")
        ax.scatter(*tracking.target_positions[idx], marker="*", s=120, color=SHELL_COLORS["target"], label="target")
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_title(f"Visual shell tracking | step {idx:03d} | error {tracking.errors[idx]:.4f} m")
        ax.legend(loc="upper right")
        return ax.lines + ax.collections

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output
