"""Kinematics utilities for planar robot arms.

This module will contain forward kinematics, inverse kinematics, Jacobian,
workspace, and manipulability helpers for 2-link and 3-link planar arms.
"""

from __future__ import annotations

import math
from typing import Sequence


def forward_kinematics_2link(q: Sequence[float], link_lengths: Sequence[float]) -> tuple[float, float]:
    """Compute planar 2-link forward kinematics."""
    q1, q2 = _expect_length(q, 2, "q")
    l1, l2 = _expect_length(link_lengths, 2, "link_lengths")

    x = l1 * math.cos(q1) + l2 * math.cos(q1 + q2)
    y = l1 * math.sin(q1) + l2 * math.sin(q1 + q2)
    return x, y


def inverse_kinematics_2link(target: Sequence[float], link_lengths: Sequence[float]) -> tuple[float, float]:
    """Compute a planar 2-link inverse kinematics solution."""
    x, y = _expect_length(target, 2, "target")
    l1, l2 = _expect_length(link_lengths, 2, "link_lengths")

    radius_sq = x * x + y * y
    cos_q2 = (radius_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cos_q2 < -1.0 - 1e-9 or cos_q2 > 1.0 + 1e-9:
        raise ValueError(f"Target {target} is outside the reachable workspace.")

    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.acos(cos_q2)
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    return q1, q2


def jacobian_2link(q: Sequence[float], link_lengths: Sequence[float]) -> list[list[float]]:
    """Compute the planar 2-link end-effector Jacobian."""
    q1, q2 = _expect_length(q, 2, "q")
    l1, l2 = _expect_length(link_lengths, 2, "link_lengths")
    q12 = q1 + q2

    return [
        [-l1 * math.sin(q1) - l2 * math.sin(q12), -l2 * math.sin(q12)],
        [l1 * math.cos(q1) + l2 * math.cos(q12), l2 * math.cos(q12)],
    ]


def joint_positions_2link(q: Sequence[float], link_lengths: Sequence[float]) -> list[tuple[float, float]]:
    """Return base, elbow, and end-effector positions for a planar 2-link arm."""
    q1, q2 = _expect_length(q, 2, "q")
    l1, l2 = _expect_length(link_lengths, 2, "link_lengths")

    elbow = (l1 * math.cos(q1), l1 * math.sin(q1))
    ee = forward_kinematics_2link((q1, q2), (l1, l2))
    return [(0.0, 0.0), elbow, ee]


def manipulability_2link(q: Sequence[float], link_lengths: Sequence[float]) -> float:
    """Compute Yoshikawa manipulability for a planar 2-link arm."""
    jacobian = jacobian_2link(q, link_lengths)
    determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    return abs(determinant)


def forward_kinematics_3link(q: Sequence[float], link_lengths: Sequence[float]) -> tuple[float, float]:
    """Compute planar 3-link forward kinematics."""
    raise NotImplementedError("Stage 01 extension pending.")


def _expect_length(values: Sequence[float], expected_length: int, name: str) -> tuple[float, ...]:
    """Validate a short numeric sequence and return it as floats."""
    if len(values) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(values)}.")
    return tuple(float(value) for value in values)
