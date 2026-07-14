from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_lab.simulation.mujoco_arm import check_mujoco_available, validate_mujoco_fk_alignment


def main() -> None:
    if not check_mujoco_available():
        print("MuJoCo is not installed; skipping model validation.")
        return

    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    q_samples = np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, size=80),
            rng.uniform(np.deg2rad(-45.0), np.deg2rad(85.0), size=80),
            rng.uniform(np.deg2rad(-125.0), np.deg2rad(35.0), size=80),
        ]
    )
    validation = validate_mujoco_fk_alignment(q_samples)

    table = np.column_stack(
        [
            validation.q_samples,
            validation.numpy_positions,
            validation.mujoco_positions,
            validation.position_errors,
        ]
    )
    table_path = tables_dir / "mujoco_fk_validation.csv"
    header = (
        "q_base_rad,q_shoulder_rad,q_elbow_rad,"
        "numpy_x,numpy_y,numpy_z,"
        "mujoco_x,mujoco_y,mujoco_z,"
        "position_error_m"
    )
    np.savetxt(table_path, table, delimiter=",", header=header, comments="")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(validation.position_errors, marker="o", linewidth=1.5, markersize=3)
    axes[0].set_title("MuJoCo vs NumPy FK error")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("position error [m]")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(validation.position_errors, bins=16)
    axes[1].set_title("Error distribution")
    axes[1].set_xlabel("position error [m]")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figure_path = figures_dir / "mujoco_fk_validation.png"
    fig.savefig(figure_path, dpi=160)

    print(f"Samples: {len(validation.position_errors)}")
    print(f"Mean FK mismatch: {np.mean(validation.position_errors):.6e} m")
    print(f"Max FK mismatch: {np.max(validation.position_errors):.6e} m")
    print(f"Saved {table_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
