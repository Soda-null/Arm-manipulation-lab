"""Tests for the Gymnasium reaching adapter."""

from __future__ import annotations

import unittest

import numpy as np

from common.gym_reaching_env import GymnasiumPlanarReachingEnv


class TestGymnasiumPlanarReachingEnv(unittest.TestCase):
    """Smoke tests for the Gymnasium adapter."""

    def test_reset_and_step_vanilla(self) -> None:
        """Vanilla Gymnasium env should expose standard reset/step outputs."""
        env = GymnasiumPlanarReachingEnv(action_mode="vanilla")
        observation, info = env.reset()
        next_observation, reward, terminated, truncated, step_info = env.step(np.zeros(2, dtype=np.float32))

        self.assertEqual(observation.shape, (8,))
        self.assertEqual(next_observation.shape, (8,))
        self.assertIsInstance(info, dict)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("error", step_info)

    def test_reset_and_step_residual(self) -> None:
        """Residual Gymnasium env should run one controller-prior step."""
        env = GymnasiumPlanarReachingEnv(action_mode="residual")
        observation, _ = env.reset()
        next_observation, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

        self.assertEqual(observation.shape, (8,))
        self.assertEqual(next_observation.shape, (8,))
        self.assertIn("prior_torque_norm", info)


if __name__ == "__main__":
    unittest.main()
