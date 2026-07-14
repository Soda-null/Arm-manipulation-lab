"""Task-space PD control placeholder."""

from __future__ import annotations

import numpy as np


def task_space_pd(x: np.ndarray, xd: np.ndarray, x_des: np.ndarray, xd_des: np.ndarray, kp: float, kd: float) -> np.ndarray:
    """Compute a Cartesian PD command for a point end effector."""

    return kp * (np.asarray(x_des) - np.asarray(x)) + kd * (np.asarray(xd_des) - np.asarray(xd))
