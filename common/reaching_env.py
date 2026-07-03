"""Lightweight planar reaching environment for residual-control experiments.

This module intentionally avoids a hard dependency on Gymnasium so that the
project can test the residual-control interface before full PPO training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control, task_space_pd
from common.kinematics import forward_kinematics_2link, jacobian_2link
from common.metrics import final_position_error


@dataclass(frozen=True)
class ReachingEnvConfig:
    """Configuration for the lightweight planar reaching environment."""

    link_lengths: tuple[float, float] = (1.0, 0.7)
    target: tuple[float, float] = (1.0, 0.5)
    initial_q: tuple[float, float] = (0.0, 0.0)
    initial_dq: tuple[float, float] = (0.0, 0.0)
    dt: float = 0.01
    max_steps: int = 500
    success_threshold: float = 0.05
    torque_limit: float = 25.0
    residual_scale: float = 4.0
    joint_damping: float = 0.35
    task_kp: float = 18.0
    task_kd: float = 4.0
    effort_penalty: float = 0.001
    success_bonus: float = 5.0


class PlanarReachingEnv:
    """Small deterministic 2-link reaching environment.

    Action mode:
    * ``vanilla``: action is direct joint torque.
    * ``residual``: action is added to a task-space controller prior.
    """

    def __init__(self, config: ReachingEnvConfig | None = None, action_mode: str = "vanilla") -> None:
        if action_mode not in {"vanilla", "residual"}:
            raise ValueError("action_mode must be 'vanilla' or 'residual'.")
        self.config = config or ReachingEnvConfig()
        self.action_mode = action_mode
        self.q = list(self.config.initial_q)
        self.dq = list(self.config.initial_dq)
        self.steps = 0

    def reset(self) -> list[float]:
        """Reset the environment and return the initial observation."""
        self.q = list(self.config.initial_q)
        self.dq = list(self.config.initial_dq)
        self.steps = 0
        return self._observation()

    def step(self, action: Sequence[float]) -> tuple[list[float], float, bool, bool, dict[str, float]]:
        """Advance the environment by one step."""
        if len(action) != 2:
            raise ValueError(f"action must have length 2, got {len(action)}.")

        if self.action_mode == "vanilla":
            torque = [float(action[0]), float(action[1])]
        else:
            prior = self.controller_prior()
            torque = [
                prior[0] + self.config.residual_scale * float(action[0]),
                prior[1] + self.config.residual_scale * float(action[1]),
            ]
        torque = [self._clip(value, -self.config.torque_limit, self.config.torque_limit) for value in torque]

        self.q, self.dq = integrate_joint_dynamics(
            self.q,
            self.dq,
            torque,
            self.config.dt,
            damping=self.config.joint_damping,
        )
        self.steps += 1

        error = self.current_error()
        effort = sum(value * value for value in torque)
        reward = -error - self.config.effort_penalty * effort
        terminated = error <= self.config.success_threshold
        if terminated:
            reward += self.config.success_bonus
        truncated = self.steps >= self.config.max_steps
        info = {
            "error": error,
            "effort": effort,
            "success": float(terminated),
            "prior_torque_norm": sum(value * value for value in self.controller_prior()) ** 0.5,
        }
        return self._observation(), reward, terminated, truncated, info

    def controller_prior(self) -> list[float]:
        """Compute a task-space PD plus Jacobian-transpose controller prior."""
        position = forward_kinematics_2link(self.q, self.config.link_lengths)
        jacobian = jacobian_2link(self.q, self.config.link_lengths)
        velocity = (
            jacobian[0][0] * self.dq[0] + jacobian[0][1] * self.dq[1],
            jacobian[1][0] * self.dq[0] + jacobian[1][1] * self.dq[1],
        )
        force = task_space_pd(position, self.config.target, velocity, self.config.task_kp, self.config.task_kd)
        return jacobian_transpose_control(jacobian, force)

    def current_error(self) -> float:
        """Return current end-effector distance to target."""
        position = forward_kinematics_2link(self.q, self.config.link_lengths)
        return final_position_error(position, self.config.target)

    def _observation(self) -> list[float]:
        """Return a compact observation vector."""
        ee_x, ee_y = forward_kinematics_2link(self.q, self.config.link_lengths)
        return [
            self.q[0],
            self.q[1],
            self.dq[0],
            self.dq[1],
            ee_x,
            ee_y,
            self.config.target[0],
            self.config.target[1],
        ]

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        """Clip a scalar value."""
        return max(lower, min(upper, value))
