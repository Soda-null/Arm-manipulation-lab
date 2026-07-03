"""Compare Stage 06 vanilla PPO and residual PPO smoke-training results."""

from __future__ import annotations

import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt


def _read_single_row(path: Path) -> dict[str, str]:
    """Read a single-row CSV metrics file."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, got {len(rows)}.")
    return rows[0]


def main() -> None:
    """Write a combined PPO comparison table and plot."""
    print("Stage 06: compare PPO results")
    rows = [
        _read_single_row(PROJECT_ROOT / "results/tables/vanilla_ppo_metrics.csv"),
        _read_single_row(PROJECT_ROOT / "results/tables/residual_ppo_metrics.csv"),
    ]

    table_path = PROJECT_ROOT / "results/tables/ppo_training_comparison.csv"
    plot_path = PROJECT_ROOT / "results/plots/ppo_training_comparison.png"
    fieldnames = ["method", "total_timesteps", "success_rate", "final_error", "steps", "return", "control_effort"]
    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = [row["method"] for row in rows]
    axes[0].bar(labels, [float(row["success_rate"]) for row in rows], color=["C1", "C2"])
    axes[0].set_title("PPO Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, [float(row["final_error"]) for row in rows], color=["C1", "C2"])
    axes[1].set_title("PPO Final Error")
    axes[1].grid(True, axis="y", alpha=0.3)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved table to {table_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
