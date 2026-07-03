"""Evaluation metric skeletons for reaching and control experiments."""

from __future__ import annotations

import math
from typing import Sequence


def final_position_error(position: Sequence[float], target: Sequence[float]) -> float:
    """Compute final end-effector position error."""
    return _euclidean_distance(position, target)


def path_length(points: Sequence[Sequence[float]]) -> float:
    """Compute path length for a sequence of positions."""
    if len(points) < 2:
        return 0.0
    return sum(_euclidean_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def smoothness(actions: Sequence[Sequence[float]]) -> float:
    """Compute a smoothness proxy from squared second differences."""
    if len(actions) < 3:
        return 0.0

    total = 0.0
    for index in range(2, len(actions)):
        previous = _as_float_tuple(actions[index - 2], "actions")
        current = _as_float_tuple(actions[index - 1], "actions")
        next_value = _as_float_tuple(actions[index], "actions")
        _validate_same_dimension(previous, current)
        _validate_same_dimension(current, next_value)
        total += sum((next_i - 2.0 * current_i + previous_i) ** 2 for previous_i, current_i, next_i in zip(previous, current, next_value))
    return total


def control_effort(actions: Sequence[Sequence[float]]) -> float:
    """Compute cumulative control effort."""
    return sum(sum(float(value) ** 2 for value in action) for action in actions)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance between equal-length vectors."""
    a_values = _as_float_tuple(a, "a")
    b_values = _as_float_tuple(b, "b")
    _validate_same_dimension(a_values, b_values)
    return math.sqrt(sum((a_i - b_i) ** 2 for a_i, b_i in zip(a_values, b_values)))


def _as_float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Convert a sequence to floats and reject empty vectors."""
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return tuple(float(value) for value in values)


def _validate_same_dimension(a: Sequence[float], b: Sequence[float]) -> None:
    """Validate that two vectors have matching dimensions."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions must match, got {len(a)} and {len(b)}.")
