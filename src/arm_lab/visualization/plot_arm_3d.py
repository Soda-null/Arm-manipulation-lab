"""3D plotting helpers for the simple serial arm."""

from __future__ import annotations

import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
from arm_lab.kinematics.robot_model import SerialArm
from arm_lab.visualization.plot_frames import set_axes_equal


def plot_arm_3d(ax, robot: SerialArm, q: np.ndarray, label: str = "arm") -> np.ndarray:
    """Plot the arm links and joints for a given configuration."""

    fk = forward_kinematics_with_frames(robot, q)
    points = fk.link_points
    ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", linewidth=3, label=label)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=60, marker="x", color="black")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper right")
    set_axes_equal(ax)
    return points
