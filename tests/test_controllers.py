"""Tests for Stage 03 controller helpers."""

from __future__ import annotations

import unittest

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control, joint_space_pd, task_space_pd


class TestControllers(unittest.TestCase):
    """Numerical checks for classical controller helper functions."""

    def test_joint_space_pd(self) -> None:
        """Joint PD should compute kp position error minus kd velocity."""
        torque = joint_space_pd(q=(0.0, 1.0), q_desired=(1.0, 0.5), dq=(0.1, -0.2), kp=10.0, kd=2.0)

        self.assertAlmostEqual(torque[0], 9.8)
        self.assertAlmostEqual(torque[1], -4.6)

    def test_task_space_pd(self) -> None:
        """Task-space PD should compute force from Cartesian error and velocity."""
        force = task_space_pd(position=(0.2, 0.4), target=(1.0, 0.5), velocity=(0.1, -0.2), kp=5.0, kd=1.0)

        self.assertAlmostEqual(force[0], 3.9)
        self.assertAlmostEqual(force[1], 0.7)

    def test_jacobian_transpose_control(self) -> None:
        """J^T F should map task-space force into joint torques."""
        jacobian = [[1.0, 2.0], [3.0, 4.0]]
        torque = jacobian_transpose_control(jacobian, task_force=(5.0, 6.0))

        self.assertEqual(torque, [23.0, 34.0])

    def test_integrate_joint_dynamics(self) -> None:
        """The simple unit-inertia integrator should advance dq and q."""
        q, dq = integrate_joint_dynamics(q=(0.0,), dq=(0.0,), torque=(2.0,), dt=0.1)

        self.assertAlmostEqual(dq[0], 0.2)
        self.assertAlmostEqual(q[0], 0.02)

    def test_dimension_mismatch_is_rejected(self) -> None:
        """Controller helpers should reject inconsistent vector dimensions."""
        with self.assertRaises(ValueError):
            joint_space_pd(q=(0.0,), q_desired=(1.0, 2.0), dq=(0.0,), kp=1.0, kd=1.0)


if __name__ == "__main__":
    unittest.main()
