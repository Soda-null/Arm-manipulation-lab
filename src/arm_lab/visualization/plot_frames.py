"""Plot coordinate frames in 3D."""

from __future__ import annotations

import numpy as np


def plot_frame(ax, T: np.ndarray, name: str = "", axis_length: float = 0.25) -> None:
    """Draw a coordinate frame represented by a homogeneous transform."""

    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ["r", "g", "b"]
    for i, color in enumerate(colors):
        direction = R[:, i] * axis_length
        ax.quiver(*origin, *direction, color=color, linewidth=1.5)
    if name:
        ax.text(*origin, name)


def set_axes_equal(ax) -> None:
    """Set equal scale for 3D Matplotlib axes."""

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])
