"""Trajectory generation utilities.

This module will provide linear, cubic, quintic, and minimum-jerk trajectory
helpers for later stage demos.
"""

from __future__ import annotations

from typing import Sequence


def linear_trajectory(start: Sequence[float], goal: Sequence[float], num_steps: int) -> list[tuple[float, ...]]:
    """Generate a linear interpolation trajectory."""
    return _interpolate(start, goal, num_steps, lambda tau: tau)


def cubic_trajectory(start: Sequence[float], goal: Sequence[float], num_steps: int) -> list[tuple[float, ...]]:
    """Generate a cubic trajectory."""
    return _interpolate(start, goal, num_steps, lambda tau: 3.0 * tau**2 - 2.0 * tau**3)


def quintic_trajectory(start: Sequence[float], goal: Sequence[float], num_steps: int) -> list[tuple[float, ...]]:
    """Generate a quintic trajectory."""
    return _interpolate(start, goal, num_steps, lambda tau: 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)


def minimum_jerk_trajectory(start: Sequence[float], goal: Sequence[float], num_steps: int) -> list[tuple[float, ...]]:
    """Generate a minimum-jerk trajectory."""
    return quintic_trajectory(start, goal, num_steps)


def trajectory_velocities(points: Sequence[Sequence[float]], dt: float = 1.0) -> list[tuple[float, ...]]:
    """Estimate trajectory velocities with first differences."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if len(points) < 2:
        return []

    return [
        tuple((float(curr) - float(prev)) / dt for prev, curr in zip(points[index - 1], points[index]))
        for index in range(1, len(points))
    ]


def _interpolate(
    start: Sequence[float],
    goal: Sequence[float],
    num_steps: int,
    time_scaling,
) -> list[tuple[float, ...]]:
    """Interpolate between two vectors using a scalar time-scaling function."""
    start_values = _as_float_tuple(start, "start")
    goal_values = _as_float_tuple(goal, "goal")
    if len(start_values) != len(goal_values):
        raise ValueError(f"start and goal must have the same dimension, got {len(start_values)} and {len(goal_values)}.")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    trajectory: list[tuple[float, ...]] = []
    for step in range(num_steps):
        tau = step / (num_steps - 1)
        scale = time_scaling(tau)
        point = tuple(start_i + scale * (goal_i - start_i) for start_i, goal_i in zip(start_values, goal_values))
        trajectory.append(point)
    return trajectory


def _as_float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Convert a numeric sequence to a tuple of floats."""
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return tuple(float(value) for value in values)
