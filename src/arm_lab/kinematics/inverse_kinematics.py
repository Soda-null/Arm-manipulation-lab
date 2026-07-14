"""Damped least-squares inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.jacobian import compute_geometric_jacobian
from arm_lab.kinematics.robot_model import SerialArm


@dataclass(frozen=True)
class IKResult:
    """Result from a position-only inverse-kinematics solve."""

    q: np.ndarray
    errors: np.ndarray
    converged: bool
    iterations: int


def damped_least_squares_ik(
    robot: SerialArm,
    target_position: np.ndarray,
    q_init: np.ndarray,
    max_iters: int = 100,
    damping: float = 1e-2,
    step_size: float = 0.7,
    tolerance: float = 1e-4,
) -> IKResult:
    """Use DLS IK to move the end effector toward a target position.

    This solves only the translational part of IK. Damping keeps the update
    stable near singular configurations by solving with J J^T + lambda^2 I.
    """

    target_position = np.asarray(target_position, dtype=float).reshape(3)
    q = np.asarray(q_init, dtype=float).copy()
    errors: list[float] = []

    for iteration in range(max_iters):
        current_position = forward_kinematics(robot, q)[:3, 3]
        error = target_position - current_position
        error_norm = float(np.linalg.norm(error))
        errors.append(error_norm)
        if error_norm < tolerance:
            return IKResult(q=q, errors=np.array(errors), converged=True, iterations=iteration)

        J_pos = compute_geometric_jacobian(robot, q)[:3, :]
        lhs = J_pos @ J_pos.T + (damping**2) * np.eye(3)
        dq = J_pos.T @ np.linalg.solve(lhs, error)
        q = q + step_size * dq

    return IKResult(q=q, errors=np.array(errors), converged=False, iterations=max_iters)
