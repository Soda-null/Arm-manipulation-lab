"""Potential-field helpers for planar obstacle avoidance."""

from __future__ import annotations

import math
from typing import Sequence


def attractive_force(
    position: Sequence[float],
    target: Sequence[float],
    velocity: Sequence[float],
    kp: float,
    kd: float,
) -> tuple[float, float]:
    """Compute an attractive task-space PD force toward the target."""
    px, py = _expect_2d(position, "position")
    tx, ty = _expect_2d(target, "target")
    vx, vy = _expect_2d(velocity, "velocity")
    return kp * (tx - px) - kd * vx, kp * (ty - py) - kd * vy


def repulsive_force(
    position: Sequence[float],
    obstacle_center: Sequence[float],
    obstacle_radius: float,
    influence_distance: float,
    gain: float,
) -> tuple[float, float]:
    """Compute repulsive force from a circular obstacle."""
    px, py = _expect_2d(position, "position")
    ox, oy = _expect_2d(obstacle_center, "obstacle_center")
    if obstacle_radius <= 0.0:
        raise ValueError("obstacle_radius must be positive.")
    if influence_distance <= obstacle_radius:
        raise ValueError("influence_distance must be larger than obstacle_radius.")

    dx = px - ox
    dy = py - oy
    center_distance = max(math.sqrt(dx * dx + dy * dy), 1e-9)
    clearance = center_distance - obstacle_radius
    if clearance >= influence_distance:
        return 0.0, 0.0

    direction = (dx / center_distance, dy / center_distance)
    magnitude = gain * (1.0 / max(clearance, 1e-6) - 1.0 / influence_distance) / max(clearance, 1e-6) ** 2
    return magnitude * direction[0], magnitude * direction[1]


def tangential_force(
    position: Sequence[float],
    target: Sequence[float],
    obstacle_center: Sequence[float],
    obstacle_radius: float,
    influence_distance: float,
    gain: float,
) -> tuple[float, float]:
    """Compute a small tangential force around an obstacle to reduce local minima."""
    px, py = _expect_2d(position, "position")
    tx, ty = _expect_2d(target, "target")
    ox, oy = _expect_2d(obstacle_center, "obstacle_center")

    dx = px - ox
    dy = py - oy
    center_distance = max(math.sqrt(dx * dx + dy * dy), 1e-9)
    clearance = center_distance - obstacle_radius
    if clearance >= influence_distance:
        return 0.0, 0.0

    radial = (dx / center_distance, dy / center_distance)
    tangent_ccw = (-radial[1], radial[0])
    to_target = (tx - px, ty - py)
    if tangent_ccw[0] * to_target[0] + tangent_ccw[1] * to_target[1] < 0.0:
        tangent_ccw = (-tangent_ccw[0], -tangent_ccw[1])

    scale = gain * (1.0 - max(clearance, 0.0) / influence_distance)
    return scale * tangent_ccw[0], scale * tangent_ccw[1]


def obstacle_collision(position: Sequence[float], obstacle_center: Sequence[float], obstacle_radius: float) -> bool:
    """Return whether a point is inside a circular obstacle."""
    px, py = _expect_2d(position, "position")
    ox, oy = _expect_2d(obstacle_center, "obstacle_center")
    return math.sqrt((px - ox) ** 2 + (py - oy) ** 2) <= obstacle_radius


def _expect_2d(values: Sequence[float], name: str) -> tuple[float, float]:
    """Validate and unpack a 2D vector."""
    if len(values) != 2:
        raise ValueError(f"{name} must be 2D, got length {len(values)}.")
    return float(values[0]), float(values[1])
