from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

if "--save-only" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from arm_lab.kinematics.forward_kinematics import forward_kinematics, forward_kinematics_with_frames
from arm_lab.kinematics.inverse_kinematics import IKResult, damped_least_squares_ik
from arm_lab.kinematics.robot_model import SerialArm, create_default_arm
from arm_lab.visualization.plot_frames import set_axes_equal


class InteractiveIKViewer:
    """Small Matplotlib viewer for target-driven 3D IK experiments."""

    def __init__(self, robot: SerialArm, q_init: np.ndarray, target: np.ndarray) -> None:
        self.robot = robot
        self.q_init = q_init.copy()
        self.q = q_init.copy()
        self.target = target.copy()
        self.last_result: IKResult | None = None

        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(121, projection="3d")
        self.error_ax = self.fig.add_subplot(222)
        self.info_ax = self.fig.add_subplot(224)
        self.fig.subplots_adjust(left=0.08, bottom=0.25, right=0.96, top=0.92, wspace=0.32)

        slider_color = "0.9"
        self.slider_axes = {
            "x": self.fig.add_axes([0.14, 0.15, 0.56, 0.03], facecolor=slider_color),
            "y": self.fig.add_axes([0.14, 0.10, 0.56, 0.03], facecolor=slider_color),
            "z": self.fig.add_axes([0.14, 0.05, 0.56, 0.03], facecolor=slider_color),
        }
        self.sliders = {
            "x": Slider(self.slider_axes["x"], "target x", -1.8, 1.8, valinit=target[0], valstep=0.01),
            "y": Slider(self.slider_axes["y"], "target y", -1.8, 1.8, valinit=target[1], valstep=0.01),
            "z": Slider(self.slider_axes["z"], "target z", 0.05, 1.8, valinit=target[2], valstep=0.01),
        }

        solve_ax = self.fig.add_axes([0.82, 0.12, 0.12, 0.045])
        reset_ax = self.fig.add_axes([0.82, 0.055, 0.12, 0.045])
        self.solve_button = Button(solve_ax, "Solve IK")
        self.reset_button = Button(reset_ax, "Reset")

        for slider in self.sliders.values():
            slider.on_changed(self.on_target_changed)
        self.solve_button.on_clicked(self.solve)
        self.reset_button.on_clicked(self.reset)

    def on_target_changed(self, _value: float) -> None:
        """Update target marker when sliders move."""

        self.target = np.array([self.sliders["x"].val, self.sliders["y"].val, self.sliders["z"].val])
        self.draw()

    def solve(self, _event=None) -> None:
        """Run damped least-squares IK from the current joint configuration."""

        self.last_result = damped_least_squares_ik(
            self.robot,
            target_position=self.target,
            q_init=self.q,
            max_iters=140,
            damping=0.03,
            step_size=0.65,
            tolerance=1e-4,
        )
        self.q = self.last_result.q
        self.draw()

    def reset(self, _event=None) -> None:
        """Reset the arm to the initial configuration."""

        self.q = self.q_init.copy()
        self.last_result = None
        self.draw()

    def draw(self) -> None:
        """Redraw the arm, target, convergence plot, and numerical readout."""

        fk = forward_kinematics_with_frames(self.robot, self.q)
        points = fk.link_points
        current_position = forward_kinematics(self.robot, self.q)[:3, 3]
        error = float(np.linalg.norm(self.target - current_position))

        self.ax.clear()
        self.ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", linewidth=3, label="arm")
        self.ax.scatter(*self.target, marker="*", s=110, color="red", label="target")
        self.ax.scatter(*current_position, marker="x", s=70, color="black", label="end effector")
        self.ax.set_title("Interactive 3D IK viewer")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")
        self.ax.set_xlim(-1.9, 1.9)
        self.ax.set_ylim(-1.9, 1.9)
        self.ax.set_zlim(0.0, 2.0)
        self.ax.legend(loc="upper right")
        set_axes_equal(self.ax)

        self.error_ax.clear()
        if self.last_result is not None:
            self.error_ax.plot(self.last_result.errors, linewidth=2)
            self.error_ax.set_title("IK error convergence")
            self.error_ax.set_xlabel("iteration")
            self.error_ax.set_ylabel("error [m]")
            self.error_ax.grid(True, alpha=0.3)
        else:
            self.error_ax.text(0.5, 0.5, "Move target, then Solve IK", ha="center", va="center")
            self.error_ax.set_axis_off()

        self.info_ax.clear()
        self.info_ax.set_axis_off()
        q_deg = np.rad2deg(self.q)
        converged = self.last_result.converged if self.last_result is not None else False
        iterations = self.last_result.iterations if self.last_result is not None else 0
        info = [
            f"target: [{self.target[0]: .2f}, {self.target[1]: .2f}, {self.target[2]: .2f}] m",
            f"end effector: [{current_position[0]: .2f}, {current_position[1]: .2f}, {current_position[2]: .2f}] m",
            f"error: {error:.5f} m",
            f"converged: {converged}",
            f"iterations: {iterations}",
            f"q deg: [{q_deg[0]: .1f}, {q_deg[1]: .1f}, {q_deg[2]: .1f}]",
        ]
        self.info_ax.text(0.0, 0.95, "\n".join(info), va="top", family="monospace", fontsize=10)

        self.fig.canvas.draw_idle()

    def save(self, path: Path) -> None:
        """Save the current viewer state as a PNG."""

        self.draw()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 3D IK viewer for the default spatial arm.")
    parser.add_argument("--target", nargs=3, type=float, default=[1.25, 0.35, 0.75], metavar=("X", "Y", "Z"))
    parser.add_argument("--save-only", action="store_true", help="Solve once and save a figure without opening a GUI.")
    parser.add_argument("--output", default="results/figures/interactive_ik_viewer.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = create_default_arm()
    q_init = np.deg2rad([0.0, 15.0, -25.0])
    viewer = InteractiveIKViewer(robot, q_init=q_init, target=np.array(args.target, dtype=float))

    if args.save_only:
        viewer.solve()
        viewer.save(Path(args.output))
        print(f"Saved {args.output}")
        return

    viewer.draw()
    plt.show()


if __name__ == "__main__":
    main()
