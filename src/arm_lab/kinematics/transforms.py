"""Rigid-body transform helpers for 3D robot kinematics."""

from __future__ import annotations

import numpy as np


def rot_x(theta: float) -> np.ndarray:
    """Return a 3D rotation matrix for a rotation about the x-axis."""

    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def rot_y(theta: float) -> np.ndarray:
    """Return a 3D rotation matrix for a rotation about the y-axis."""

    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rot_z(theta: float) -> np.ndarray:
    """Return a 3D rotation matrix for a rotation about the z-axis."""

    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix used for cross products."""

    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Return a rotation matrix for a revolute joint axis and angle."""

    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Joint axis must be nonzero.")
    unit_axis = axis / norm
    K = skew(unit_axis)
    eye = np.eye(3)
    return eye + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transform from rotation and translation."""

    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float).reshape(3)
    if R.shape != (3, 3):
        raise ValueError("R must have shape (3, 3).")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def translation(p: np.ndarray) -> np.ndarray:
    """Create a pure translation transform."""

    return make_transform(np.eye(3), np.asarray(p, dtype=float))


def rotation_transform(R: np.ndarray) -> np.ndarray:
    """Create a pure rotation transform."""

    return make_transform(R, np.zeros(3))


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must have shape (4, 4).")
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def compose_transforms(*Ts: np.ndarray) -> np.ndarray:
    """Compose transforms in left-to-right order."""

    result = np.eye(4)
    for T in Ts:
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError("Every transform must have shape (4, 4).")
        result = result @ T
    return result
