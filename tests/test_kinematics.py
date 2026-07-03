"""Tests for Stage 01 planar arm kinematics."""

from __future__ import annotations

import math
import unittest

from common.kinematics import (
    forward_kinematics_2link,
    inverse_kinematics_2link,
    jacobian_2link,
    joint_positions_2link,
    manipulability_2link,
)


class TestPlanar2LinkKinematics(unittest.TestCase):
    """Numerical checks for 2-link planar arm kinematics."""

    def setUp(self) -> None:
        self.link_lengths = (1.0, 0.7)

    def assertPointAlmostEqual(
        self,
        actual: tuple[float, float],
        expected: tuple[float, float],
        places: int = 7,
    ) -> None:
        """Assert that two 2D points are almost equal."""
        self.assertAlmostEqual(actual[0], expected[0], places=places)
        self.assertAlmostEqual(actual[1], expected[1], places=places)

    def test_forward_kinematics_zero_pose(self) -> None:
        """The zero pose should place both links along the positive x-axis."""
        point = forward_kinematics_2link((0.0, 0.0), self.link_lengths)

        self.assertPointAlmostEqual(point, (1.7, 0.0))

    def test_forward_kinematics_right_angle_pose(self) -> None:
        """A shoulder angle of pi/2 should rotate both links upward."""
        point = forward_kinematics_2link((math.pi / 2.0, 0.0), self.link_lengths)

        self.assertPointAlmostEqual(point, (0.0, 1.7))

    def test_inverse_kinematics_round_trips_through_forward_kinematics(self) -> None:
        """IK followed by FK should recover a reachable target."""
        target = (1.0, 0.5)
        q = inverse_kinematics_2link(target, self.link_lengths)
        recovered = forward_kinematics_2link(q, self.link_lengths)

        self.assertPointAlmostEqual(recovered, target)

    def test_inverse_kinematics_rejects_unreachable_target(self) -> None:
        """Targets beyond the arm reach should raise ValueError."""
        with self.assertRaises(ValueError):
            inverse_kinematics_2link((2.0, 0.0), self.link_lengths)

    def test_jacobian_matches_finite_difference(self) -> None:
        """The analytic Jacobian should match a finite-difference estimate."""
        q = (0.4, -0.8)
        eps = 1e-6
        jacobian = jacobian_2link(q, self.link_lengths)
        base = forward_kinematics_2link(q, self.link_lengths)

        for joint_index in range(2):
            q_perturbed = list(q)
            q_perturbed[joint_index] += eps
            perturbed = forward_kinematics_2link(q_perturbed, self.link_lengths)
            finite_difference = (
                (perturbed[0] - base[0]) / eps,
                (perturbed[1] - base[1]) / eps,
            )

            self.assertAlmostEqual(jacobian[0][joint_index], finite_difference[0], places=5)
            self.assertAlmostEqual(jacobian[1][joint_index], finite_difference[1], places=5)

    def test_joint_positions_include_base_elbow_and_end_effector(self) -> None:
        """Joint-position helper should return base, elbow, and end effector."""
        positions = joint_positions_2link((0.0, 0.0), self.link_lengths)

        self.assertEqual(len(positions), 3)
        self.assertPointAlmostEqual(positions[0], (0.0, 0.0))
        self.assertPointAlmostEqual(positions[1], (1.0, 0.0))
        self.assertPointAlmostEqual(positions[2], (1.7, 0.0))

    def test_manipulability_zero_at_straight_singularity(self) -> None:
        """A fully extended 2-link arm should have zero manipulability."""
        value = manipulability_2link((0.0, 0.0), self.link_lengths)

        self.assertAlmostEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
