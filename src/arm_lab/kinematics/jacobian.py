"""Jacobian computations for simple serial arms."""

from __future__ import annotations

import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
from arm_lab.kinematics.robot_model import SerialArm


def compute_geometric_jacobian(robot: SerialArm, q: np.ndarray) -> np.ndarray:
    """Compute a 6xN geometric Jacobian for revolute joints.

    The top three rows map joint rates to end-effector linear velocity. The
    bottom three rows map joint rates to angular velocity.
    """

    fk = forward_kinematics_with_frames(robot, q)
    p_ee = fk.end_effector[:3, 3]
    J = np.zeros((6, robot.dof))

    for i in range(robot.dof):
        axis = fk.joint_axes_world[i]
        p_joint = fk.joint_positions[i]
        J[:3, i] = np.cross(axis, p_ee - p_joint)
        J[3:, i] = axis

    return J
