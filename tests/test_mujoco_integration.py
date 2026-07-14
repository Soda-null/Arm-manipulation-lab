import numpy as np
import pytest

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.simulation.mujoco_arm import (
    run_mujoco_pd_trajectory_tracking,
    run_mujoco_position_reaching,
    run_mujoco_trajectory_tracking,
    simple_arm_motor_xml_path,
    simple_arm_xml_path,
    validate_mujoco_fk_alignment,
)

mujoco = pytest.importorskip("mujoco")


def test_mujoco_model_matches_numpy_fk_at_ik_solution():
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 15.0, -25.0])
    target = np.array([1.25, 0.35, 0.75])
    ik_result = damped_least_squares_ik(robot, target, q_init)

    model = mujoco.MjModel.from_xml_path(str(simple_arm_xml_path()))
    data = mujoco.MjData(model)
    data.qpos[:] = ik_result.q
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    numpy_position = forward_kinematics(robot, ik_result.q)[:3, 3]
    assert np.allclose(data.site_xpos[site_id], numpy_position, atol=1e-8)


def test_mujoco_model_matches_numpy_fk_over_sampled_workspace():
    q_samples = np.deg2rad(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [30.0, 25.0, -45.0],
                [-60.0, 55.0, -90.0],
                [120.0, -20.0, 20.0],
            ]
        )
    )

    result = validate_mujoco_fk_alignment(q_samples)

    assert result.position_errors.shape == (4,)
    assert np.max(result.position_errors) < 1e-8


def test_mujoco_position_reaching_reduces_error():
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 15.0, -25.0])
    target = np.array([1.25, 0.35, 0.75])
    ik_result = damped_least_squares_ik(robot, target, q_init)

    result = run_mujoco_position_reaching(q_init, ik_result.q, target, duration=1.6)

    assert result.position_errors[-1] < 0.05
    assert result.position_errors[-1] < result.position_errors[0]


def test_mujoco_trajectory_tracking_records_expected_shapes():
    q_init = np.deg2rad([0.0, 20.0, -55.0])
    q_des_history = np.tile(q_init, (5, 1))
    target_positions = np.tile(forward_kinematics(create_default_arm(), q_init)[:3, 3], (5, 1))

    result = run_mujoco_trajectory_tracking(q_init, q_des_history, target_positions, hold_steps=2)

    assert result.qpos_history.shape == (5, 3)
    assert result.q_des_history.shape == (5, 3)
    assert result.target_positions.shape == (5, 3)
    assert result.end_effector_positions.shape == (5, 3)
    assert result.position_errors.shape == (5,)
    assert np.all(np.isfinite(result.position_errors))


def test_mujoco_trajectory_tracking_accepts_control_parameters():
    q_init = np.deg2rad([0.0, 20.0, -55.0])
    q_des_history = np.tile(q_init, (4, 1))
    target_positions = np.tile(forward_kinematics(create_default_arm(), q_init)[:3, 3], (4, 1))

    result = run_mujoco_trajectory_tracking(
        q_init,
        q_des_history,
        target_positions,
        hold_steps=2,
        actuator_kp=3200.0,
        joint_damping=1.5,
    )

    assert result.position_errors[-1] < 0.05


def test_mujoco_motor_model_matches_numpy_fk_at_static_q():
    q = np.deg2rad([20.0, 30.0, -70.0])
    model = mujoco.MjModel.from_xml_path(str(simple_arm_motor_xml_path()))
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    numpy_position = forward_kinematics(create_default_arm(), q)[:3, 3]
    assert np.allclose(data.site_xpos[site_id], numpy_position, atol=1e-8)


def test_mujoco_pd_tracking_records_expected_shapes():
    q_init = np.deg2rad([0.0, 20.0, -55.0])
    q_des_history = np.tile(q_init, (5, 1))
    qd_des_history = np.zeros_like(q_des_history)
    target_positions = np.tile(forward_kinematics(create_default_arm(), q_init)[:3, 3], (5, 1))

    result = run_mujoco_pd_trajectory_tracking(
        q_init,
        q_des_history,
        target_positions,
        qd_des_history=qd_des_history,
        hold_steps=2,
        kp=120.0,
        kd=24.0,
    )

    assert result.qpos_history.shape == (5, 3)
    assert result.qvel_history.shape == (5, 3)
    assert result.torque_history.shape == (5, 3)
    assert result.position_errors.shape == (5,)
    assert np.all(np.isfinite(result.torque_history))
