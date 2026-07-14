import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.robot_model import create_default_arm


def test_fk_returns_valid_transform():
    robot = create_default_arm()
    T = forward_kinematics(robot, np.deg2rad([10.0, 20.0, -30.0]))
    assert T.shape == (4, 4)
    assert np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]))
