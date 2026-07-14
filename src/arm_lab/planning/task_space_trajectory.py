"""Task-space trajectory helpers."""

from __future__ import annotations

import numpy as np


def circular_target_trajectory(radius: float = 0.25, height: float = 0.6, steps: int = 100) -> np.ndarray:
    """Create a simple circular end-effector target path in 3D."""

    theta = np.linspace(0.0, 2.0 * np.pi, steps)
    x = 1.0 + radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, height)
    return np.column_stack([x, y, z])
