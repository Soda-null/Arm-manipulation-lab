"""Stage 05 obstacle-reaching benchmark.

This script compares a direct task-space controller against a potential-field
controller for a simple circular-obstacle reaching task.
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

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control
from common.kinematics import forward_kinematics_2link, jacobian_2link
from common.metrics import control_effort, final_position_error, path_length, smoothness
from common.potential_fields import attractive_force, obstacle_collision, repulsive_force, tangential_force


def run_rollout(use_obstacle_avoidance: bool) -> dict[str, float | int | str]:
    """Run one controller variant and return benchmark metrics."""
    link_lengths = (1.0, 0.7)
    target = (1.0, 0.55)
    obstacle_center = (1.5, 0.05)
    obstacle_radius = 0.18
    influence_distance = 0.34
    q = [0.0, 0.0]
    dq = [0.0, 0.0]
    dt = 0.01
    steps = 1000

    positions: list[tuple[float, float]] = []
    torques: list[tuple[float, float]] = []
    collision_count = 0

    for _ in range(steps):
        position = forward_kinematics_2link(q, link_lengths)
        jacobian = jacobian_2link(q, link_lengths)
        velocity = (
            jacobian[0][0] * dq[0] + jacobian[0][1] * dq[1],
            jacobian[1][0] * dq[0] + jacobian[1][1] * dq[1],
        )
        attraction = attractive_force(position, target, velocity, kp=20.0, kd=5.5)
        if use_obstacle_avoidance:
            repulsion = repulsive_force(position, obstacle_center, obstacle_radius, influence_distance, gain=0.015)
            tangent = tangential_force(position, target, obstacle_center, obstacle_radius, influence_distance, gain=1.2)
        else:
            repulsion = (0.0, 0.0)
            tangent = (0.0, 0.0)
        force = (attraction[0] + repulsion[0] + tangent[0], attraction[1] + repulsion[1] + tangent[1])
        torque = jacobian_transpose_control(jacobian, force)
        torque = [max(-25.0, min(25.0, value)) for value in torque]
        q, dq = integrate_joint_dynamics(q, dq, torque, dt, damping=0.45)

        positions.append(position)
        torques.append((float(torque[0]), float(torque[1])))
        if obstacle_collision(position, obstacle_center, obstacle_radius):
            collision_count += 1

    error = final_position_error(positions[-1], target)
    minimum_clearance = min(
        ((position[0] - obstacle_center[0]) ** 2 + (position[1] - obstacle_center[1]) ** 2) ** 0.5 - obstacle_radius
        for position in positions
    )
    return {
        "method": "potential_field_jt" if use_obstacle_avoidance else "task_space_pd_direct",
        "success": int(error <= 0.05 and collision_count == 0),
        "final_error": error,
        "collision_count": collision_count,
        "minimum_clearance": minimum_clearance,
        "path_length": path_length(positions),
        "smoothness": smoothness(positions),
        "control_effort": control_effort(torques),
    }


def main() -> None:
    """Run the obstacle-reaching benchmark and save a CSV table."""
    print("Stage 05: benchmark obstacle reaching")
    output_path = PROJECT_ROOT / "results/tables/obstacle_reaching_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [run_rollout(use_obstacle_avoidance=False), run_rollout(use_obstacle_avoidance=True)]
    fieldnames = [
        "method",
        "success",
        "final_error",
        "collision_count",
        "minimum_clearance",
        "path_length",
        "smoothness",
        "control_effort",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "success": row["success"],
                    "final_error": f"{row['final_error']:.6f}",
                    "collision_count": row["collision_count"],
                    "minimum_clearance": f"{row['minimum_clearance']:.6f}",
                    "path_length": f"{row['path_length']:.6f}",
                    "smoothness": f"{row['smoothness']:.6f}",
                    "control_effort": f"{row['control_effort']:.6f}",
                }
            )

    for row in rows:
        print(
            f"{row['method']}: success={row['success']}, "
            f"final_error={row['final_error']:.6f}, collisions={row['collision_count']}, "
            f"minimum_clearance={row['minimum_clearance']:.6f}"
        )
    print(f"Saved table to {output_path}")


if __name__ == "__main__":
    main()
