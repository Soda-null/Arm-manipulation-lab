from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.kinematics.inverse_kinematics import damped_least_squares_ik
from arm_lab.kinematics.robot_model import create_default_arm
from arm_lab.simulation.mujoco_arm import (
    check_mujoco_available,
    run_mujoco_position_reaching,
    save_mujoco_reach_animation,
)


def main() -> None:
    if not check_mujoco_available():
        print("MuJoCo is not installed. This is expected for the scaffold stage.")
        print("Future work: load src/arm_lab/simulation/simple_arm_scene.xml and compare it with the NumPy model.")
        return

    figures_dir = Path("results/figures")
    gifs_dir = Path("results/gifs")
    figures_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 15.0, -25.0])
    target_position = np.array([1.25, 0.35, 0.75])
    ik_result = damped_least_squares_ik(
        robot=robot,
        target_position=target_position,
        q_init=q_init,
        max_iters=120,
        damping=0.03,
        step_size=0.65,
    )

    sim_result = run_mujoco_position_reaching(
        q_init=q_init,
        q_des=ik_result.q,
        target_position=target_position,
        duration=2.5,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(sim_result.time, sim_result.position_errors, linewidth=2)
    axes[0].set_title("MuJoCo reaching error")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("end-effector error [m]")
    axes[0].grid(True, alpha=0.3)

    q_deg = np.rad2deg(sim_result.qpos_history)
    q_des_deg = np.rad2deg(sim_result.q_des)
    labels = ["base yaw", "shoulder", "elbow"]
    for idx, label in enumerate(labels):
        axes[1].plot(sim_result.time, q_deg[:, idx], label=label)
        axes[1].axhline(q_des_deg[idx], linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Joint position actuator tracking")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("joint angle [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = figures_dir / "mujoco_reaching_demo.png"
    fig.savefig(out, dpi=160)
    gif_out = save_mujoco_reach_animation(
        result=sim_result,
        output_path=gifs_dir / "mujoco_reaching_demo.gif",
        every=14,
        fps=12,
    )

    print(f"IK converged: {ik_result.converged}")
    print(f"IK q_des [deg]: {np.rad2deg(ik_result.q)}")
    print(f"Final MuJoCo q [deg]: {np.rad2deg(sim_result.qpos_history[-1])}")
    print(f"Initial MuJoCo error: {sim_result.position_errors[0]:.5f} m")
    print(f"Final MuJoCo error: {sim_result.position_errors[-1]:.5f} m")
    print(f"Saved {out}")
    print(f"Saved {gif_out}")


if __name__ == "__main__":
    main()
