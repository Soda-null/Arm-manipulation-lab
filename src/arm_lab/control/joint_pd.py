"""Joint-space PD control."""

from __future__ import annotations

import numpy as np


def joint_pd(q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, qd_des: np.ndarray, kp: float, kd: float) -> np.ndarray:
    """Compute a simple joint-space PD command."""

    return kp * (np.asarray(q_des) - np.asarray(q)) + kd * (np.asarray(qd_des) - np.asarray(qd))
