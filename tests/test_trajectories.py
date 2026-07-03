"""Tests for Stage 02 trajectory generation and metrics."""

from __future__ import annotations

import unittest

from common.metrics import control_effort, path_length, smoothness
from common.trajectories import (
    cubic_trajectory,
    linear_trajectory,
    minimum_jerk_trajectory,
    quintic_trajectory,
    trajectory_velocities,
)


class TestTrajectoryGeneration(unittest.TestCase):
    """Numerical checks for point-to-point trajectory helpers."""

    def setUp(self) -> None:
        self.start = (0.0, 0.0)
        self.goal = (1.0, 0.5)
        self.num_steps = 11

    def test_all_trajectories_start_and_end_correctly(self) -> None:
        """Every trajectory method should preserve endpoints."""
        for generator in [linear_trajectory, cubic_trajectory, quintic_trajectory, minimum_jerk_trajectory]:
            with self.subTest(generator=generator.__name__):
                trajectory = generator(self.start, self.goal, self.num_steps)

                self.assertEqual(len(trajectory), self.num_steps)
                self.assertEqual(trajectory[0], self.start)
                self.assertEqual(trajectory[-1], self.goal)

    def test_linear_trajectory_has_constant_velocity(self) -> None:
        """Linear interpolation should produce equal finite differences."""
        trajectory = linear_trajectory((0.0,), (1.0,), 6)
        velocities = trajectory_velocities(trajectory)

        for velocity in velocities:
            self.assertAlmostEqual(velocity[0], 0.2)

    def test_quintic_trajectory_has_small_endpoint_steps(self) -> None:
        """Quintic time scaling should start and stop more gently than linear."""
        linear = linear_trajectory((0.0,), (1.0,), 21)
        quintic = quintic_trajectory((0.0,), (1.0,), 21)

        linear_start_step = linear[1][0] - linear[0][0]
        quintic_start_step = quintic[1][0] - quintic[0][0]

        self.assertLess(quintic_start_step, linear_start_step)

    def test_invalid_num_steps_is_rejected(self) -> None:
        """A trajectory needs at least start and goal samples."""
        with self.assertRaises(ValueError):
            linear_trajectory(self.start, self.goal, 1)

    def test_dimension_mismatch_is_rejected(self) -> None:
        """Start and goal must have matching dimensions."""
        with self.assertRaises(ValueError):
            cubic_trajectory((0.0,), (1.0, 2.0), 10)

    def test_basic_metrics(self) -> None:
        """Path length, smoothness, and effort should return expected simple values."""
        points = [(0.0,), (0.5,), (1.0,)]

        self.assertAlmostEqual(path_length(points), 1.0)
        self.assertAlmostEqual(smoothness(points), 0.0)
        self.assertAlmostEqual(control_effort(points), 1.25)


if __name__ == "__main__":
    unittest.main()
