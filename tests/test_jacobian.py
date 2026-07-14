import numpy as np

from arm_lab.kinematics.jacobian import compute_geometric_jacobian
from arm_lab.kinematics.robot_model import create_default_arm


def test_jacobian_has_expected_dimensions():
    robot = create_default_arm()
    J = compute_geometric_jacobian(robot, np.deg2rad([5.0, 25.0, -45.0]))
    assert J.shape == (6, robot.dof)
