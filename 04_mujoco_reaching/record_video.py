"""Stage 04 MuJoCo video recording.

This script records a fixed-target reaching rollout from the planar arm MuJoCo
model and saves it as an MP4 video. It uses MuJoCo for simulation and a
matplotlib 2D renderer for robust headless video generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from common.controllers import joint_space_pd
from common.kinematics import inverse_kinematics_2link, joint_positions_2link

XML_PATH = PROJECT_ROOT / "04_mujoco_reaching/xml/planar_2link_arm.xml"


def main() -> None:
    """Render a MuJoCo reaching rollout to an MP4 file."""
    print("Stage 04: record MuJoCo reaching video")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    target_xy = (1.0, 0.5)
    link_lengths = (1.0, 0.7)
    q_desired = inverse_kinematics_2link(target_xy, (1.0, 0.7))
    output_path = PROJECT_ROOT / "results/videos/mujoco_reaching_pd.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = 30
    duration_seconds = 4.0
    steps = int(duration_seconds / model.opt.timestep)
    render_every = max(1, int(1.0 / (fps * model.opt.timestep)))
    frames: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    (arm_line,) = ax.plot([], [], "-o", linewidth=5, markersize=8, color="C0", label="arm")
    (path_line,) = ax.plot([], [], "-", linewidth=2, color="C2", alpha=0.8, label="end-effector path")
    ax.scatter([target_xy[0]], [target_xy[1]], marker="x", s=120, color="C3", label="target")
    ax.set_title("MuJoCo Joint-PD Reaching")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.3, 1.8)
    ax.set_ylim(-0.4, 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    ee_path_x: list[float] = []
    ee_path_y: list[float] = []

    for step in range(steps):
        q = (float(data.qpos[0]), float(data.qpos[1]))
        dq = (float(data.qvel[0]), float(data.qvel[1]))
        data.ctrl[:] = joint_space_pd(q, q_desired, dq, kp=28.0, kd=4.5)
        mujoco.mj_step(model, data)

        if step % render_every == 0:
            joints = joint_positions_2link((float(data.qpos[0]), float(data.qpos[1])), link_lengths)
            xs = [point[0] for point in joints]
            ys = [point[1] for point in joints]
            ee_path_x.append(xs[-1])
            ee_path_y.append(ys[-1])

            arm_line.set_data(xs, ys)
            path_line.set_data(ee_path_x, ee_path_y)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)

    print(f"Rendered {len(frames)} frames at {fps} fps")
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    main()
