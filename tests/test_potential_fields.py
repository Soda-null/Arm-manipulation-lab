"""Tests for Stage 05 potential-field helpers."""

from __future__ import annotations

import unittest

from common.potential_fields import attractive_force, obstacle_collision, repulsive_force, tangential_force


class TestPotentialFields(unittest.TestCase):
    """Numerical checks for circular-obstacle potential fields."""

    def test_attractive_force_points_toward_target(self) -> None:
        """Attractive force should point from current position to target."""
        force = attractive_force(position=(0.0, 0.0), target=(1.0, 0.5), velocity=(0.0, 0.0), kp=2.0, kd=1.0)

        self.assertEqual(force, (2.0, 1.0))

    def test_repulsive_force_is_zero_outside_influence(self) -> None:
        """Obstacle repulsion should vanish outside the influence distance."""
        force = repulsive_force(
            position=(2.0, 0.0),
            obstacle_center=(0.0, 0.0),
            obstacle_radius=0.2,
            influence_distance=0.5,
            gain=1.0,
        )

        self.assertEqual(force, (0.0, 0.0))

    def test_repulsive_force_pushes_away_from_obstacle(self) -> None:
        """Obstacle repulsion should point away from the obstacle center."""
        force = repulsive_force(
            position=(0.4, 0.0),
            obstacle_center=(0.0, 0.0),
            obstacle_radius=0.2,
            influence_distance=0.5,
            gain=0.1,
        )

        self.assertGreater(force[0], 0.0)
        self.assertAlmostEqual(force[1], 0.0)

    def test_tangential_force_is_zero_outside_influence(self) -> None:
        """Tangential force should vanish outside the obstacle influence region."""
        force = tangential_force(
            position=(2.0, 0.0),
            target=(1.0, 1.0),
            obstacle_center=(0.0, 0.0),
            obstacle_radius=0.2,
            influence_distance=0.5,
            gain=1.0,
        )

        self.assertEqual(force, (0.0, 0.0))

    def test_obstacle_collision(self) -> None:
        """Collision helper should identify points inside a circular obstacle."""
        self.assertTrue(obstacle_collision((0.1, 0.0), (0.0, 0.0), 0.2))
        self.assertFalse(obstacle_collision((0.3, 0.0), (0.0, 0.0), 0.2))


if __name__ == "__main__":
    unittest.main()
