"""Joint-space trajectory generation."""

from __future__ import annotations

import numpy as np


def linear_joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray, steps: int) -> np.ndarray:
    """Interpolate linearly between two joint configurations."""

    if steps < 2:
        raise ValueError("steps must be at least 2.")
    return np.linspace(np.asarray(q_start, dtype=float), np.asarray(q_goal, dtype=float), steps)
