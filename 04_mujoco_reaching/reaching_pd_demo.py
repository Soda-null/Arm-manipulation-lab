"""Stage 04 MuJoCo reaching PD demo.

This script runs a fixed-target reaching demo for the planar arm MuJoCo model
using an IK target and joint-space PD torque control.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import mujoco

from common.controllers import joint_space_pd
from common.kinematics import inverse_kinematics_2link

XML_PATH = PROJECT_ROOT / "04_mujoco_reaching/xml/planar_2link_arm.xml"


def main() -> None:
    """Run a fixed-target MuJoCo reaching demo and save metrics."""
    print("Stage 04: MuJoCo reaching PD demo")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    link_lengths = (1.0, 0.7)
    target_xy = (1.0, 0.5)
    q_desired = inverse_kinematics_2link(target_xy, link_lengths)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")

    dt = model.opt.timestep
    steps = 2000
    errors: list[float] = []
    q_history: list[tuple[float, float]] = []
    ee_history: list[tuple[float, float]] = []
    torque_history: list[tuple[float, float]] = []

    for _ in range(steps):
        q = (float(data.qpos[0]), float(data.qpos[1]))
        dq = (float(data.qvel[0]), float(data.qvel[1]))
        torque = joint_space_pd(q, q_desired, dq, kp=28.0, kd=4.5)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)

        ee_position = data.site_xpos[ee_site_id]
        ee_xy = (float(ee_position[0]), float(ee_position[1]))
        error = ((target_xy[0] - ee_xy[0]) ** 2 + (target_xy[1] - ee_xy[1]) ** 2) ** 0.5
        errors.append(error)
        q_history.append((float(data.qpos[0]), float(data.qpos[1])))
        ee_history.append(ee_xy)
        torque_history.append((float(torque[0]), float(torque[1])))

    plot_path = PROJECT_ROOT / "results/plots/mujoco_reaching_pd.png"
    table_path = PROJECT_ROOT / "results/tables/mujoco_reaching_metrics.csv"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    time = [index * dt for index in range(steps)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([point[0] for point in ee_history], [point[1] for point in ee_history], label="end-effector path")
    axes[0].scatter([target_xy[0]], [target_xy[1]], marker="x", s=100, label="target")
    axes[0].set_title("MuJoCo Joint-PD Reaching")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(time, errors, color="C3")
    axes[1].set_title("End-Effector Error")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "target_x", "target_y", "final_error", "steps", "dt"])
        writer.writeheader()
        writer.writerow(
            {
                "method": "joint_pd_ik_target",
                "target_x": f"{target_xy[0]:.6f}",
                "target_y": f"{target_xy[1]:.6f}",
                "final_error": f"{errors[-1]:.6f}",
                "steps": str(steps),
                "dt": f"{dt:.6f}",
            }
        )

    print(f"IK target q = ({q_desired[0]:.4f}, {q_desired[1]:.4f}) rad")
    print(f"Final MuJoCo end-effector error = {errors[-1]:.6f}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved table to {table_path}")


if __name__ == "__main__":
    main()
