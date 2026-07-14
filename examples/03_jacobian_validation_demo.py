from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arm_lab.kinematics.forward_kinematics import forward_kinematics
from arm_lab.kinematics.jacobian import compute_geometric_jacobian
from arm_lab.kinematics.robot_model import create_default_arm


def main() -> None:
    robot = create_default_arm()
    q = np.deg2rad([25.0, 30.0, -55.0])
    q_dot = np.array([0.15, -0.08, 0.12])
    dt = 1e-6

    J = compute_geometric_jacobian(robot, q)
    v_pred = J[:3, :] @ q_dot

    p0 = forward_kinematics(robot, q)[:3, 3]
    p1 = forward_kinematics(robot, q + q_dot * dt)[:3, 3]
    v_fd = (p1 - p0) / dt
    error = np.linalg.norm(v_pred - v_fd)

    print("Jacobian validation")
    print(f"Predicted velocity:       {v_pred}")
    print(f"Finite-difference velocity: {v_fd}")
    print(f"L2 error: {error:.6e}")


if __name__ == "__main__":
    main()
