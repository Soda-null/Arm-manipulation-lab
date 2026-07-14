"""Forward kinematics for the staged spatial manipulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arm_lab.kinematics.robot_model import SerialArm
from arm_lab.kinematics.transforms import axis_angle, rotation_transform, translation


@dataclass(frozen=True)
class FKResult:
    """Forward-kinematics result with intermediate joint data."""

    end_effector: np.ndarray
    joint_positions: np.ndarray
    joint_axes_world: np.ndarray
    link_points: np.ndarray


def forward_kinematics_with_frames(robot: SerialArm, q: np.ndarray) -> FKResult:
    """Compute end-effector pose plus joint positions and axes.

    The implementation is intentionally explicit: each joint first moves to
    its local origin, records the world-space revolute axis, applies the joint
    rotation, and then advances along the link vector.
    """

    q = np.asarray(q, dtype=float)
    if q.shape != (robot.dof,):
        raise ValueError(f"Expected q with shape ({robot.dof},), got {q.shape}.")

    T = np.eye(4)
    joint_positions: list[np.ndarray] = []
    joint_axes_world: list[np.ndarray] = []
    link_points = [T[:3, 3].copy()]

    for joint, theta in zip(robot.joints, q):
        T = T @ translation(joint.origin)
        joint_positions.append(T[:3, 3].copy())

        axis_world = T[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
        joint_axes_world.append(axis_world)

        T = T @ rotation_transform(axis_angle(joint.axis, theta))
        T = T @ translation(joint.link_vector)
        link_points.append(T[:3, 3].copy())

    return FKResult(
        end_effector=T,
        joint_positions=np.vstack(joint_positions),
        joint_axes_world=np.vstack(joint_axes_world),
        link_points=np.vstack(link_points),
    )


def forward_kinematics(robot: SerialArm, q: np.ndarray) -> np.ndarray:
    """Return the 4x4 end-effector transform for joint configuration q."""

    return forward_kinematics_with_frames(robot, q).end_effector
