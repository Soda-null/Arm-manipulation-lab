from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.kinematics.transforms import compose_transforms, make_transform, rot_x, rot_y, rot_z
from arm_lab.visualization.plot_frames import plot_frame, set_axes_equal


def main() -> None:
    results_dir = Path("results/figures")
    results_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        ("world", np.eye(4)),
        ("A", make_transform(rot_z(np.deg2rad(35)), np.array([0.7, 0.1, 0.2]))),
        (
            "B",
            compose_transforms(
                make_transform(rot_z(np.deg2rad(35)), np.array([0.7, 0.1, 0.2])),
                make_transform(rot_y(np.deg2rad(-30)) @ rot_x(np.deg2rad(20)), np.array([0.45, 0.2, 0.35])),
            ),
        ),
    ]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name, T in frames:
        plot_frame(ax, T, name=name, axis_length=0.25)

    ax.set_title("3D coordinate frame transform demo")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(-0.2, 1.5)
    ax.set_ylim(-0.6, 1.0)
    ax.set_zlim(0.0, 1.2)
    set_axes_equal(ax)
    fig.tight_layout()
    out = results_dir / "transform_demo.png"
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
