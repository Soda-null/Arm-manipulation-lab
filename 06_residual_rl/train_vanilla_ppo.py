"""Stage 06 vanilla PPO training entry point."""

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

from common.ppo_utils import evaluate_model, train_ppo


def main() -> None:
    """Train and evaluate a small vanilla PPO baseline."""
    print("Stage 06: train vanilla PPO")
    total_timesteps = 4096
    model_path = PROJECT_ROOT / "results/policies/vanilla_ppo_reaching.zip"
    table_path = PROJECT_ROOT / "results/tables/vanilla_ppo_metrics.csv"

    model = train_ppo(action_mode="vanilla", total_timesteps=total_timesteps, seed=7, model_path=model_path)
    metrics = evaluate_model(model, action_mode="vanilla", episodes=5)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "total_timesteps", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"method": "vanilla_ppo", "total_timesteps": total_timesteps, **metrics})

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {table_path}")
    print(f"success_rate={metrics['success_rate']:.3f}, final_error={metrics['final_error']:.6f}, steps={metrics['steps']:.1f}")


if __name__ == "__main__":
    main()
