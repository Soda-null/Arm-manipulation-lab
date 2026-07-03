"""Stage 06 policy evaluation.

This script evaluates lightweight reaching policies before full PPO training:
direct zero torque, controller prior, and a small scripted residual correction.
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

from common.reaching_env import PlanarReachingEnv


def evaluate_policy(name: str, action_mode: str, residual_gain: float = 0.0) -> dict[str, float | int | str]:
    """Evaluate one deterministic policy variant."""
    env = PlanarReachingEnv(action_mode=action_mode)
    observation = env.reset()
    errors: list[float] = [env.current_error()]
    rewards: list[float] = []
    total_effort = 0.0
    success = 0

    while True:
        if action_mode == "vanilla":
            action = [0.0, 0.0]
        else:
            ee_x, ee_y, target_x, target_y = observation[4], observation[5], observation[6], observation[7]
            action = [residual_gain * (target_x - ee_x), residual_gain * (target_y - ee_y)]

        observation, reward, terminated, truncated, info = env.step(action)
        errors.append(float(info["error"]))
        rewards.append(float(reward))
        total_effort += float(info["effort"])
        if terminated:
            success = 1
        if terminated or truncated:
            break

    return {
        "method": name,
        "success": success,
        "steps": env.steps,
        "final_error": errors[-1],
        "return": sum(rewards),
        "control_effort": total_effort,
    }


def main() -> None:
    """Evaluate the current residual-control interface and save results."""
    print("Stage 06: evaluate policies")
    rows = [
        evaluate_policy("vanilla_zero_torque", action_mode="vanilla"),
        evaluate_policy("controller_prior_zero_residual", action_mode="residual", residual_gain=0.0),
        evaluate_policy("controller_prior_scripted_residual", action_mode="residual", residual_gain=0.25),
    ]

    table_path = PROJECT_ROOT / "results/tables/residual_rl_interface_benchmark.csv"
    plot_path = PROJECT_ROOT / "results/plots/residual_rl_interface_benchmark.png"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "success", "steps", "final_error", "return", "control_effort"]
    with table_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "success": row["success"],
                    "steps": row["steps"],
                    "final_error": f"{row['final_error']:.6f}",
                    "return": f"{row['return']:.6f}",
                    "control_effort": f"{row['control_effort']:.6f}",
                }
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(row["method"]) for row in rows], [float(row["final_error"]) for row in rows])
    ax.set_title("Residual-Control Interface Benchmark")
    ax.set_ylabel("final error")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    for row in rows:
        print(f"{row['method']}: success={row['success']}, steps={row['steps']}, final_error={row['final_error']:.6f}")
    print(f"Saved table to {table_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
