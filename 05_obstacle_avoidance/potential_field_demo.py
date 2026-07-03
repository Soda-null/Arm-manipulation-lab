"""Stage 05 potential-field obstacle avoidance demo.

This script demonstrates target attraction, obstacle repulsion, and Jacobian
transpose control for planar arm obstacle avoidance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from common.controllers import integrate_joint_dynamics, jacobian_transpose_control
from common.kinematics import forward_kinematics_2link, jacobian_2link, joint_positions_2link
from common.potential_fields import attractive_force, obstacle_collision, repulsive_force, tangential_force


def run_potential_field_rollout() -> dict[str, object]:
    """Run a planar potential-field rollout and return trajectory data."""
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
    arm_snapshots: list[list[tuple[float, float]]] = []
    collision_count = 0

    for step in range(steps):
        position = forward_kinematics_2link(q, link_lengths)
        jacobian = jacobian_2link(q, link_lengths)
        velocity = (
            jacobian[0][0] * dq[0] + jacobian[0][1] * dq[1],
            jacobian[1][0] * dq[0] + jacobian[1][1] * dq[1],
        )
        attraction = attractive_force(position, target, velocity, kp=20.0, kd=5.5)
        repulsion = repulsive_force(position, obstacle_center, obstacle_radius, influence_distance, gain=0.015)
        tangent = tangential_force(position, target, obstacle_center, obstacle_radius, influence_distance, gain=1.2)
        force = (attraction[0] + repulsion[0] + tangent[0], attraction[1] + repulsion[1] + tangent[1])
        torque = jacobian_transpose_control(jacobian, force)
        torque = [max(-25.0, min(25.0, value)) for value in torque]
        q, dq = integrate_joint_dynamics(q, dq, torque, dt, damping=0.45)

        positions.append(position)
        if step % 100 == 0:
            arm_snapshots.append(joint_positions_2link(q, link_lengths))
        if obstacle_collision(position, obstacle_center, obstacle_radius):
            collision_count += 1

    final_error = ((target[0] - positions[-1][0]) ** 2 + (target[1] - positions[-1][1]) ** 2) ** 0.5
    return {
        "positions": positions,
        "arm_snapshots": arm_snapshots,
        "target": target,
        "obstacle_center": obstacle_center,
        "obstacle_radius": obstacle_radius,
        "collision_count": collision_count,
        "final_error": final_error,
    }


def main() -> None:
    """Run and plot a potential-field obstacle avoidance demo."""
    print("Stage 05: potential-field obstacle avoidance demo")
    result = run_potential_field_rollout()
    positions = result["positions"]
    target = result["target"]
    obstacle_center = result["obstacle_center"]
    obstacle_radius = result["obstacle_radius"]

    output_path = PROJECT_ROOT / "results/plots/obstacle_avoidance_trajectory.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([point[0] for point in positions], [point[1] for point in positions], linewidth=2.5, label="end-effector path")
    for snapshot in result["arm_snapshots"]:
        ax.plot([point[0] for point in snapshot], [point[1] for point in snapshot], "-o", alpha=0.22, color="C0")
    obstacle = plt.Circle(obstacle_center, obstacle_radius, color="C3", alpha=0.28, label="obstacle")
    ax.add_patch(obstacle)
    ax.scatter([target[0]], [target[1]], marker="x", s=120, color="C3", label="target")
    ax.set_title("Potential-Field Obstacle Avoidance")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.2, 1.8)
    ax.set_ylim(-0.5, 1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Final error = {result['final_error']:.6f}")
    print(f"Collision count = {result['collision_count']}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
