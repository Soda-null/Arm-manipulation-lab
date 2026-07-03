"""Classical robot arm controller skeletons.

This module will contain joint-space PD, task-space control, Jacobian transpose,
and residual-controller utility functions.
"""

from __future__ import annotations

from typing import Sequence


def joint_space_pd(
    q: Sequence[float],
    q_desired: Sequence[float],
    dq: Sequence[float],
    kp: float,
    kd: float,
) -> list[float]:
    """Compute joint-space PD control effort."""
    q_values = _as_float_tuple(q, "q")
    q_desired_values = _as_float_tuple(q_desired, "q_desired")
    dq_values = _as_float_tuple(dq, "dq")
    _validate_same_dimension(q_values, q_desired_values)
    _validate_same_dimension(q_values, dq_values)

    return [kp * (q_desired_i - q_i) - kd * dq_i for q_i, q_desired_i, dq_i in zip(q_values, q_desired_values, dq_values)]


def task_space_pd(
    position: Sequence[float],
    target: Sequence[float],
    velocity: Sequence[float],
    kp: float,
    kd: float,
) -> list[float]:
    """Compute task-space PD force."""
    position_values = _as_float_tuple(position, "position")
    target_values = _as_float_tuple(target, "target")
    velocity_values = _as_float_tuple(velocity, "velocity")
    _validate_same_dimension(position_values, target_values)
    _validate_same_dimension(position_values, velocity_values)

    return [
        kp * (target_i - position_i) - kd * velocity_i
        for position_i, target_i, velocity_i in zip(position_values, target_values, velocity_values)
    ]


def jacobian_transpose_control(jacobian: Sequence[Sequence[float]], task_force: Sequence[float]) -> list[float]:
    """Map a task-space force to joint torques using J^T F."""
    task_force_values = _as_float_tuple(task_force, "task_force")
    rows = [tuple(float(value) for value in row) for row in jacobian]
    if not rows:
        raise ValueError("jacobian must not be empty.")
    if len(rows) != len(task_force_values):
        raise ValueError(f"jacobian row count must match task_force length, got {len(rows)} and {len(task_force_values)}.")

    num_columns = len(rows[0])
    if num_columns == 0:
        raise ValueError("jacobian must have at least one column.")
    for row in rows:
        if len(row) != num_columns:
            raise ValueError("jacobian rows must all have the same length.")

    return [sum(rows[row_index][col_index] * task_force_values[row_index] for row_index in range(len(rows))) for col_index in range(num_columns)]


def integrate_joint_dynamics(
    q: Sequence[float],
    dq: Sequence[float],
    torque: Sequence[float],
    dt: float,
    damping: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Integrate a simple unit-inertia joint model for controller demos."""
    q_values = _as_float_tuple(q, "q")
    dq_values = _as_float_tuple(dq, "dq")
    torque_values = _as_float_tuple(torque, "torque")
    _validate_same_dimension(q_values, dq_values)
    _validate_same_dimension(q_values, torque_values)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    next_dq = [dq_i + (torque_i - damping * dq_i) * dt for dq_i, torque_i in zip(dq_values, torque_values)]
    next_q = [q_i + dq_i * dt for q_i, dq_i in zip(q_values, next_dq)]
    return next_q, next_dq


def _as_float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Convert a non-empty numeric sequence to a tuple of floats."""
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return tuple(float(value) for value in values)


def _validate_same_dimension(a: Sequence[float], b: Sequence[float]) -> None:
    """Validate that two vectors have matching dimensions."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match, got {len(a)} and {len(b)}.")
