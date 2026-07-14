"""Simple serial manipulator model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RevoluteJoint:
    """A revolute joint followed by a fixed link displacement.

    The axis and link vector are expressed in the current local frame. The
    origin shifts from the previous link endpoint to this joint before the
    revolute motion is applied.
    """

    axis: np.ndarray
    origin: np.ndarray
    link_vector: np.ndarray
    name: str = "joint"


@dataclass(frozen=True)
class SerialArm:
    """A minimal serial manipulator made from revolute joints."""

    joints: list[RevoluteJoint]
    name: str = "simple_spatial_arm"

    @property
    def dof(self) -> int:
        """Return the number of actuated revolute joints."""

        return len(self.joints)


def create_default_arm(include_wrist: bool = False) -> SerialArm:
    """Create a small spatial arm with base yaw, shoulder pitch, and elbow pitch."""

    joints = [
        RevoluteJoint(
            axis=np.array([0.0, 0.0, 1.0]),
            origin=np.array([0.0, 0.0, 0.25]),
            link_vector=np.array([0.0, 0.0, 0.35]),
            name="base_yaw",
        ),
        RevoluteJoint(
            axis=np.array([0.0, 1.0, 0.0]),
            origin=np.zeros(3),
            link_vector=np.array([1.0, 0.0, 0.0]),
            name="shoulder_pitch",
        ),
        RevoluteJoint(
            axis=np.array([0.0, 1.0, 0.0]),
            origin=np.zeros(3),
            link_vector=np.array([0.8, 0.0, 0.0]),
            name="elbow_pitch",
        ),
    ]
    if include_wrist:
        joints.append(
            RevoluteJoint(
                axis=np.array([0.0, 1.0, 0.0]),
                origin=np.zeros(3),
                link_vector=np.array([0.35, 0.0, 0.0]),
                name="wrist_pitch_placeholder",
            )
        )
    return SerialArm(joints=joints)
