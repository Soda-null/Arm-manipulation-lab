import numpy as np

from arm_lab.control.resolved_rate_control import track_task_space_trajectory
from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.planning.task_space_trajectory import circular_target_trajectory


def test_resolved_rate_tracking_records_expected_shapes():
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 20.0, -55.0])
    targets = circular_target_trajectory(radius=0.08, height=0.72, steps=12)

    result = track_task_space_trajectory(robot, targets, q_init, max_step_norm=0.08)

    assert result.q_history.shape == (12, robot.dof)
    assert result.end_effector_positions.shape == (12, 3)
    assert result.errors.shape == (12,)
    assert np.all(np.isfinite(result.errors))


def test_resolved_rate_tracking_moves_toward_first_target():
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 20.0, -55.0])
    target = np.array([[1.18, 0.1, 0.72]])
    initial_error = np.linalg.norm(target[0] - forward_kinematics(robot, q_init)[:3, 3])

    result = track_task_space_trajectory(robot, target, q_init, gain=0.9, max_step_norm=0.08)

    assert result.errors[0] < initial_error
