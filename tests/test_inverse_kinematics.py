import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm


def test_ik_reduces_position_error_for_reachable_target():
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 10.0, -15.0])
    target = np.array([1.2, 0.25, 0.75])
    initial_error = np.linalg.norm(target - forward_kinematics(robot, q_init)[:3, 3])

    result = damped_least_squares_ik(robot, target, q_init, max_iters=100, damping=0.03, step_size=0.6)
    final_error = np.linalg.norm(target - forward_kinematics(robot, result.q)[:3, 3])

    assert final_error < initial_error
    assert result.errors[-1] < result.errors[0]
