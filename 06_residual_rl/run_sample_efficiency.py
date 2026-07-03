"""Run a small sample-efficiency study for vanilla PPO and residual PPO.

This is still an early-stage experiment: it uses a small grid of seeds and
timesteps so results can be regenerated quickly during development.
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

from common.ppo_utils import evaluate_model, train_ppo


def _mean(values: list[float]) -> float:
    """Compute a simple mean."""
    return sum(values) / len(values)


def main() -> None:
    """Train and compare PPO variants across a small seed/timestep grid."""
    print("Stage 06: sample-efficiency study")

    timesteps_grid = [1024, 2048, 4096]
    seeds = [3, 7, 11]
    action_modes = [
        ("vanilla_ppo", "vanilla"),
        ("residual_ppo", "residual"),
    ]

    detail_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    policies_dir = PROJECT_ROOT / "results/policies/sample_efficiency"
    policies_dir.mkdir(parents=True, exist_ok=True)

    for method, action_mode in action_modes:
        for total_timesteps in timesteps_grid:
            metrics_for_group: list[dict[str, float]] = []
            for seed in seeds:
                model_path = policies_dir / f"{method}_{total_timesteps}_seed{seed}.zip"
                print(f"Training {method}: timesteps={total_timesteps}, seed={seed}")
                model = train_ppo(action_mode=action_mode, total_timesteps=total_timesteps, seed=seed, model_path=model_path)
                metrics = evaluate_model(model, action_mode=action_mode, episodes=3)
                metrics_for_group.append(metrics)
                detail_rows.append(
                    {
                        "method": method,
                        "action_mode": action_mode,
                        "total_timesteps": str(total_timesteps),
                        "seed": str(seed),
                        "success_rate": f"{metrics['success_rate']:.6f}",
                        "final_error": f"{metrics['final_error']:.6f}",
                        "steps": f"{metrics['steps']:.6f}",
                        "return": f"{metrics['return']:.6f}",
                        "control_effort": f"{metrics['control_effort']:.6f}",
                    }
                )

            summary_rows.append(
                {
                    "method": method,
                    "action_mode": action_mode,
                    "total_timesteps": str(total_timesteps),
                    "mean_success_rate": f"{_mean([m['success_rate'] for m in metrics_for_group]):.6f}",
                    "mean_final_error": f"{_mean([m['final_error'] for m in metrics_for_group]):.6f}",
                    "mean_steps": f"{_mean([m['steps'] for m in metrics_for_group]):.6f}",
                    "mean_return": f"{_mean([m['return'] for m in metrics_for_group]):.6f}",
                    "mean_control_effort": f"{_mean([m['control_effort'] for m in metrics_for_group]):.6f}",
                }
            )

    detail_path = PROJECT_ROOT / "results/tables/ppo_sample_efficiency_detail.csv"
    summary_path = PROJECT_ROOT / "results/tables/ppo_sample_efficiency_summary.csv"
    plot_path = PROJECT_ROOT / "results/plots/ppo_sample_efficiency.png"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    detail_fields = [
        "method",
        "action_mode",
        "total_timesteps",
        "seed",
        "success_rate",
        "final_error",
        "steps",
        "return",
        "control_effort",
    ]
    summary_fields = [
        "method",
        "action_mode",
        "total_timesteps",
        "mean_success_rate",
        "mean_final_error",
        "mean_steps",
        "mean_return",
        "mean_control_effort",
    ]

    with detail_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, _ in action_modes:
        rows = [row for row in summary_rows if row["method"] == method]
        xs = [int(row["total_timesteps"]) for row in rows]
        axes[0].plot(xs, [float(row["mean_success_rate"]) for row in rows], "-o", label=method)
        axes[1].plot(xs, [float(row["mean_final_error"]) for row in rows], "-o", label=method)

    axes[0].set_title("Sample Efficiency: Success Rate")
    axes[0].set_xlabel("training timesteps")
    axes[0].set_ylabel("mean success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Sample Efficiency: Final Error")
    axes[1].set_xlabel("training timesteps")
    axes[1].set_ylabel("mean final error")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved detail table to {detail_path}")
    print(f"Saved summary table to {summary_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
