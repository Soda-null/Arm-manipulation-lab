"""Resolved-rate control for small task-space motions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.jacobian import compute_geometric_jacobian
from arm_lab.kinematics.robot_model import SerialArm


@dataclass(frozen=True)
class TrajectoryTrackingResult:
    """Recorded states from a task-space resolved-rate tracking run."""

    q_history: np.ndarray
    target_positions: np.ndarray
    end_effector_positions: np.ndarray
    errors: np.ndarray


def resolved_rate_step(J_pos: np.ndarray, position_error: np.ndarray, damping: float = 1e-2) -> np.ndarray:
    """Return a damped least-squares joint update for a Cartesian error."""

    J_pos = np.asarray(J_pos, dtype=float)
    error = np.asarray(position_error, dtype=float)
    lhs = J_pos @ J_pos.T + (damping**2) * np.eye(J_pos.shape[0])
    return J_pos.T @ np.linalg.solve(lhs, error)


def track_task_space_trajectory(
    robot: SerialArm,
    target_positions: np.ndarray,
    q_init: np.ndarray,
    gain: float = 0.85,
    damping: float = 0.03,
    max_step_norm: float = 0.08,
) -> TrajectoryTrackingResult:
    """Track a Cartesian target path with resolved-rate Jacobian updates.

    This is a kinematic controller, not a physics simulation. At each sample it
    measures the current end-effector position, computes a Cartesian error, and
    applies a damped least-squares joint update.
    """

    targets = np.asarray(target_positions, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("target_positions must have shape (N, 3).")

    q = np.asarray(q_init, dtype=float).copy()
    if q.shape != (robot.dof,):
        raise ValueError(f"Expected q_init with shape ({robot.dof},), got {q.shape}.")

    q_history: list[np.ndarray] = []
    ee_history: list[np.ndarray] = []
    errors: list[float] = []

    for target in targets:
        current_position = forward_kinematics(robot, q)[:3, 3]
        position_error = target - current_position
        J_pos = compute_geometric_jacobian(robot, q)[:3, :]
        dq = gain * resolved_rate_step(J_pos, position_error, damping=damping)

        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step_norm:
            dq *= max_step_norm / step_norm

        q = q + dq
        updated_position = forward_kinematics(robot, q)[:3, 3]

        q_history.append(q.copy())
        ee_history.append(updated_position)
        errors.append(float(np.linalg.norm(target - updated_position)))

    return TrajectoryTrackingResult(
        q_history=np.vstack(q_history),
        target_positions=targets,
        end_effector_positions=np.vstack(ee_history),
        errors=np.array(errors),
    )
