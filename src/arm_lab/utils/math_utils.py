"""Small numerical helpers."""

from __future__ import annotations

import numpy as np


def is_rotation_matrix(R: np.ndarray, atol: float = 1e-8) -> bool:
    """Check whether a matrix is approximately a valid 3D rotation."""

    R = np.asarray(R, dtype=float)
    return R.shape == (3, 3) and np.allclose(R.T @ R, np.eye(3), atol=atol) and np.isclose(np.linalg.det(R), 1.0, atol=atol)
