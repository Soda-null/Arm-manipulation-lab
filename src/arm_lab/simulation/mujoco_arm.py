"""MuJoCo integration helpers for the simple spatial arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def check_mujoco_available() -> bool:
    """Return True if the optional mujoco Python package can be imported."""

    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class MuJoCoReachResult:
    """Recorded state from a MuJoCo joint-position reaching simulation."""

    time: np.ndarray
    qpos_history: np.ndarray
    q_des: np.ndarray
    end_effector_positions: np.ndarray
    target_position: np.ndarray
    position_errors: np.ndarray


@dataclass(frozen=True)
class MuJoCoFKValidationResult:
    """Comparison between NumPy FK and MuJoCo site positions over sampled q."""

    q_samples: np.ndarray
    numpy_positions: np.ndarray
    mujoco_positions: np.ndarray
    position_errors: np.ndarray


@dataclass(frozen=True)
class MuJoCoTrajectoryTrackingResult:
    """Recorded state from MuJoCo following a joint target sequence."""

    time: np.ndarray
    qpos_history: np.ndarray
    q_des_history: np.ndarray
    target_positions: np.ndarray
    end_effector_positions: np.ndarray
    position_errors: np.ndarray


@dataclass(frozen=True)
class MuJoCoPDTrackingResult:
    """Recorded state from custom joint-space PD torque control in MuJoCo."""

    time: np.ndarray
    qpos_history: np.ndarray
    qvel_history: np.ndarray
    q_des_history: np.ndarray
    qd_des_history: np.ndarray
    torque_history: np.ndarray
    target_positions: np.ndarray
    end_effector_positions: np.ndarray
    position_errors: np.ndarray


def simple_arm_xml_path() -> Path:
    """Return the bundled MJCF file for the simple spatial arm."""

    return Path(__file__).with_name("simple_arm_scene.xml")


def simple_arm_motor_xml_path() -> Path:
    """Return the bundled MJCF file with motor actuators for torque control."""

    return Path(__file__).with_name("simple_arm_motor_scene.xml")


def validate_mujoco_fk_alignment(q_samples: np.ndarray) -> MuJoCoFKValidationResult:
    """Compare NumPy FK and MuJoCo end-effector site positions.

    This checks whether the analytical kinematic model and the MJCF model agree
    on joint axes, link lengths, and frame placement across many configurations.
    """

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed.") from exc

    from arm_lab.kinematics.forward_kinematics import forward_kinematics
    from arm_lab.kinematics.robot_model import create_default_arm

    q_samples = np.asarray(q_samples, dtype=float)
    robot = create_default_arm()
    if q_samples.ndim != 2 or q_samples.shape[1] != robot.dof:
        raise ValueError(f"q_samples must have shape (N, {robot.dof}).")

    model = mujoco.MjModel.from_xml_path(str(simple_arm_xml_path()))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    numpy_positions: list[np.ndarray] = []
    mujoco_positions: list[np.ndarray] = []
    errors: list[float] = []

    for q in q_samples:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        numpy_position = forward_kinematics(robot, q)[:3, 3]
        mujoco_position = data.site_xpos[site_id].copy()
        numpy_positions.append(numpy_position)
        mujoco_positions.append(mujoco_position)
        errors.append(float(np.linalg.norm(numpy_position - mujoco_position)))

    return MuJoCoFKValidationResult(
        q_samples=q_samples.copy(),
        numpy_positions=np.vstack(numpy_positions),
        mujoco_positions=np.vstack(mujoco_positions),
        position_errors=np.array(errors),
    )


def run_mujoco_position_reaching(
    q_init: np.ndarray,
    q_des: np.ndarray,
    target_position: np.ndarray,
    duration: float = 2.5,
) -> MuJoCoReachResult:
    """Simulate MuJoCo position actuators moving toward an IK target.

    The controller is deliberately simple: each position actuator receives the
    IK-computed desired joint angle. MuJoCo handles inertia, damping, gravity,
    and integration, so the recorded error is the physical execution error.
    """

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed.") from exc

    model = mujoco.MjModel.from_xml_path(str(simple_arm_xml_path()))
    data = mujoco.MjData(model)
    q_init = np.asarray(q_init, dtype=float)
    q_des = np.asarray(q_des, dtype=float)
    target_position = np.asarray(target_position, dtype=float)

    if q_init.shape != (model.nq,) or q_des.shape != (model.nu,):
        raise ValueError(f"Expected q_init shape ({model.nq},) and q_des shape ({model.nu},).")

    data.qpos[:] = q_init
    data.ctrl[:] = q_des

    end_effector_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    model.site_pos[target_site_id] = target_position

    steps = int(duration / model.opt.timestep)
    times: list[float] = []
    qpos_history: list[np.ndarray] = []
    ee_history: list[np.ndarray] = []
    errors: list[float] = []

    for _ in range(steps):
        data.ctrl[:] = q_des
        mujoco.mj_step(model, data)
        ee_position = data.site_xpos[end_effector_site_id].copy()
        times.append(float(data.time))
        qpos_history.append(data.qpos.copy())
        ee_history.append(ee_position)
        errors.append(float(np.linalg.norm(target_position - ee_position)))

    return MuJoCoReachResult(
        time=np.array(times),
        qpos_history=np.vstack(qpos_history),
        q_des=q_des.copy(),
        end_effector_positions=np.vstack(ee_history),
        target_position=target_position.copy(),
        position_errors=np.array(errors),
    )


def run_mujoco_trajectory_tracking(
    q_init: np.ndarray,
    q_des_history: np.ndarray,
    target_positions: np.ndarray,
    hold_steps: int = 10,
    actuator_kp: float | None = None,
    joint_damping: float | None = None,
) -> MuJoCoTrajectoryTrackingResult:
    """Track a sequence of desired joint angles in MuJoCo.

    Each target sample is held for `hold_steps` physics steps. This models a
    simple joint-space position servo following a trajectory generated by the
    analytical kinematics code.
    """

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed.") from exc

    model = mujoco.MjModel.from_xml_path(str(simple_arm_xml_path()))
    if actuator_kp is not None:
        _set_position_actuator_kp(model, actuator_kp)
    if joint_damping is not None:
        model.dof_damping[:] = joint_damping
    data = mujoco.MjData(model)
    q_init = np.asarray(q_init, dtype=float)
    q_des_history = np.asarray(q_des_history, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)

    if hold_steps < 1:
        raise ValueError("hold_steps must be at least 1.")
    if q_init.shape != (model.nq,):
        raise ValueError(f"Expected q_init shape ({model.nq},), got {q_init.shape}.")
    if q_des_history.ndim != 2 or q_des_history.shape[1] != model.nu:
        raise ValueError(f"Expected q_des_history shape (N, {model.nu}).")
    if target_positions.shape != (q_des_history.shape[0], 3):
        raise ValueError("target_positions must have shape (N, 3) matching q_des_history.")

    data.qpos[:] = q_init
    data.ctrl[:] = q_des_history[0]
    end_effector_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")

    times: list[float] = []
    qpos_samples: list[np.ndarray] = []
    q_des_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    ee_samples: list[np.ndarray] = []
    errors: list[float] = []

    for q_des, target in zip(q_des_history, target_positions):
        model.site_pos[target_site_id] = target
        for _ in range(hold_steps):
            data.ctrl[:] = q_des
            mujoco.mj_step(model, data)

        ee_position = data.site_xpos[end_effector_site_id].copy()
        times.append(float(data.time))
        qpos_samples.append(data.qpos.copy())
        q_des_samples.append(q_des.copy())
        target_samples.append(target.copy())
        ee_samples.append(ee_position)
        errors.append(float(np.linalg.norm(target - ee_position)))

    return MuJoCoTrajectoryTrackingResult(
        time=np.array(times),
        qpos_history=np.vstack(qpos_samples),
        q_des_history=np.vstack(q_des_samples),
        target_positions=np.vstack(target_samples),
        end_effector_positions=np.vstack(ee_samples),
        position_errors=np.array(errors),
    )


def run_mujoco_pd_trajectory_tracking(
    q_init: np.ndarray,
    q_des_history: np.ndarray,
    target_positions: np.ndarray,
    qd_des_history: np.ndarray | None = None,
    hold_steps: int = 12,
    kp: float = 220.0,
    kd: float = 38.0,
    torque_limit: float = 220.0,
) -> MuJoCoPDTrackingResult:
    """Track joint targets with a custom joint-space PD torque controller.

    This uses motor actuators and explicitly computes torque commands:
    tau = kp * (q_des - q) + kd * (qd_des - qd).
    """

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is not installed.") from exc

    model = mujoco.MjModel.from_xml_path(str(simple_arm_motor_xml_path()))
    data = mujoco.MjData(model)
    q_init = np.asarray(q_init, dtype=float)
    q_des_history = np.asarray(q_des_history, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    if qd_des_history is None:
        qd_des_history = np.zeros_like(q_des_history)
    else:
        qd_des_history = np.asarray(qd_des_history, dtype=float)

    if hold_steps < 1:
        raise ValueError("hold_steps must be at least 1.")
    if q_init.shape != (model.nq,):
        raise ValueError(f"Expected q_init shape ({model.nq},), got {q_init.shape}.")
    if q_des_history.ndim != 2 or q_des_history.shape[1] != model.nu:
        raise ValueError(f"Expected q_des_history shape (N, {model.nu}).")
    if qd_des_history.shape != q_des_history.shape:
        raise ValueError("qd_des_history must match q_des_history shape.")
    if target_positions.shape != (q_des_history.shape[0], 3):
        raise ValueError("target_positions must have shape (N, 3) matching q_des_history.")

    data.qpos[:] = q_init
    end_effector_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")

    times: list[float] = []
    qpos_samples: list[np.ndarray] = []
    qvel_samples: list[np.ndarray] = []
    q_des_samples: list[np.ndarray] = []
    qd_des_samples: list[np.ndarray] = []
    torque_samples: list[np.ndarray] = []
    target_samples: list[np.ndarray] = []
    ee_samples: list[np.ndarray] = []
    errors: list[float] = []

    for q_des, qd_des, target in zip(q_des_history, qd_des_history, target_positions):
        model.site_pos[target_site_id] = target
        tau = np.zeros(model.nu)
        for _ in range(hold_steps):
            q_error = q_des - data.qpos
            qd_error = qd_des - data.qvel
            tau = kp * q_error + kd * qd_error
            tau = np.clip(tau, -torque_limit, torque_limit)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        ee_position = data.site_xpos[end_effector_site_id].copy()
        times.append(float(data.time))
        qpos_samples.append(data.qpos.copy())
        qvel_samples.append(data.qvel.copy())
        q_des_samples.append(q_des.copy())
        qd_des_samples.append(qd_des.copy())
        torque_samples.append(tau.copy())
        target_samples.append(target.copy())
        ee_samples.append(ee_position)
        errors.append(float(np.linalg.norm(target - ee_position)))

    return MuJoCoPDTrackingResult(
        time=np.array(times),
        qpos_history=np.vstack(qpos_samples),
        qvel_history=np.vstack(qvel_samples),
        q_des_history=np.vstack(q_des_samples),
        qd_des_history=np.vstack(qd_des_samples),
        torque_history=np.vstack(torque_samples),
        target_positions=np.vstack(target_samples),
        end_effector_positions=np.vstack(ee_samples),
        position_errors=np.array(errors),
    )


def _set_position_actuator_kp(model, actuator_kp: float) -> None:
    """Update MuJoCo position actuator stiffness while preserving damping ratio roughly."""

    if actuator_kp <= 0:
        raise ValueError("actuator_kp must be positive.")

    old_kp = float(model.actuator_gainprm[0, 0])
    if old_kp <= 0:
        raise ValueError("Current model actuator kp must be positive.")
    velocity_damping_scale = np.sqrt(actuator_kp / old_kp)

    model.actuator_gainprm[:, 0] = actuator_kp
    model.actuator_biasprm[:, 1] = -actuator_kp
    model.actuator_biasprm[:, 2] *= velocity_damping_scale


def save_mujoco_reach_animation(
    result: MuJoCoReachResult,
    output_path: str | Path,
    every: int = 12,
    fps: int = 12,
) -> Path:
    """Save a GIF from recorded MuJoCo qpos states using Matplotlib.

    This avoids requiring an OpenGL context while still animating the actual
    joint trajectory produced by MuJoCo's physics simulation.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
    from arm_lab.kinematics.robot_model import create_default_arm
    from arm_lab.visualization.plot_frames import set_axes_equal

    if every < 1:
        raise ValueError("every must be at least 1.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = create_default_arm()

    frame_indices = np.arange(0, len(result.qpos_history), every)
    if frame_indices[-1] != len(result.qpos_history) - 1:
        frame_indices = np.append(frame_indices, len(result.qpos_history) - 1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    all_points = np.vstack([result.end_effector_positions, result.target_position.reshape(1, 3)])
    mins = all_points.min(axis=0) - 0.25
    maxs = all_points.max(axis=0) + 0.25
    mins[2] = min(0.0, mins[2])

    def draw_frame(frame_number: int):
        idx = int(frame_indices[frame_number])
        q = result.qpos_history[idx]
        fk = forward_kinematics_with_frames(robot, q)
        arm_points = fk.link_points
        ee_path = result.end_effector_positions[: idx + 1]

        ax.clear()
        ax.plot(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2], "-o", linewidth=4, label="MuJoCo arm")
        ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="#ff7f0e", linewidth=2, label="end-effector path")
        ax.scatter(*result.target_position, marker="*", s=120, color="red", label="target")
        ax.scatter(*result.end_effector_positions[idx], marker="x", s=70, color="black", label="end effector")
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"MuJoCo reaching | t {result.time[idx]:.2f}s | error {result.position_errors[idx]:.3f} m")
        ax.legend(loc="upper right")
        set_axes_equal(ax)
        return ax.lines + ax.collections

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output


def save_mujoco_trajectory_animation(
    result: MuJoCoTrajectoryTrackingResult,
    output_path: str | Path,
    every: int = 3,
    fps: int = 12,
) -> Path:
    """Save a GIF of MuJoCo trajectory tracking from recorded qpos states."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from arm_lab.kinematics.forward_kinematics import forward_kinematics_with_frames
    from arm_lab.kinematics.robot_model import create_default_arm
    from arm_lab.visualization.plot_frames import set_axes_equal

    if every < 1:
        raise ValueError("every must be at least 1.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    robot = create_default_arm()
    frame_indices = np.arange(0, len(result.qpos_history), every)
    if frame_indices[-1] != len(result.qpos_history) - 1:
        frame_indices = np.append(frame_indices, len(result.qpos_history) - 1)

    all_points = np.vstack([result.target_positions, result.end_effector_positions])
    mins = all_points.min(axis=0) - 0.25
    maxs = all_points.max(axis=0) + 0.25
    mins[2] = min(0.0, mins[2])

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(frame_number: int):
        idx = int(frame_indices[frame_number])
        fk = forward_kinematics_with_frames(robot, result.qpos_history[idx])
        arm_points = fk.link_points
        ee_path = result.end_effector_positions[: idx + 1]

        ax.clear()
        ax.plot(
            result.target_positions[:, 0],
            result.target_positions[:, 1],
            result.target_positions[:, 2],
            "--",
            color="0.55",
            linewidth=1.5,
            label="target path",
        )
        ax.plot(ee_path[:, 0], ee_path[:, 1], ee_path[:, 2], color="#ff7f0e", linewidth=2, label="MuJoCo path")
        ax.plot(arm_points[:, 0], arm_points[:, 1], arm_points[:, 2], "-o", linewidth=4, label="MuJoCo arm")
        ax.scatter(*result.target_positions[idx], marker="*", s=110, color="red", label="current target")
        ax.scatter(*result.end_effector_positions[idx], marker="x", s=70, color="black", label="end effector")
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(f"MuJoCo trajectory tracking | step {idx:03d} | error {result.position_errors[idx]:.3f} m")
        ax.legend(loc="upper right")
        set_axes_equal(ax)
        return ax.lines + ax.collections

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output
