"""Tests for the Stage 06 lightweight reaching environment."""

from __future__ import annotations

import unittest

from common.reaching_env import PlanarReachingEnv, ReachingEnvConfig


class TestPlanarReachingEnv(unittest.TestCase):
    """Smoke tests for vanilla and residual action modes."""

    def test_reset_observation_shape(self) -> None:
        """Reset should return the compact eight-value observation."""
        env = PlanarReachingEnv()
        observation = env.reset()

        self.assertEqual(len(observation), 8)

    def test_vanilla_zero_torque_does_not_succeed_immediately(self) -> None:
        """Zero torque baseline should not magically solve the reaching task."""
        env = PlanarReachingEnv(action_mode="vanilla")
        env.reset()
        _, _, terminated, _, info = env.step((0.0, 0.0))

        self.assertFalse(terminated)
        self.assertGreater(info["error"], env.config.success_threshold)

    def test_residual_zero_action_uses_controller_prior(self) -> None:
        """Residual mode with zero residual should still use the controller prior."""
        config = ReachingEnvConfig(max_steps=200)
        env = PlanarReachingEnv(config=config, action_mode="residual")
        env.reset()
        initial_error = env.current_error()
        final_error = initial_error

        for _ in range(config.max_steps):
            _, _, terminated, truncated, info = env.step((0.0, 0.0))
            final_error = info["error"]
            if terminated or truncated:
                break

        self.assertLess(final_error, initial_error)

    def test_invalid_action_mode_is_rejected(self) -> None:
        """Only vanilla and residual action modes should be accepted."""
        with self.assertRaises(ValueError):
            PlanarReachingEnv(action_mode="bad")


if __name__ == "__main__":
    unittest.main()
