# 00 Project Overview

This lab is a staged 3D manipulator project. The first goal is to make the math
visible: coordinate frames, homogeneous transforms, forward kinematics,
Jacobians, and inverse kinematics.

The current implementation is intentionally small. It gives future work a
clean place to land without pretending that planning, control, MuJoCo, and RL
are already complete.

## Stages

1. Coordinate frames and transforms.
2. Forward kinematics for a serial arm.
3. Geometric Jacobian validation.
4. Damped least-squares inverse kinematics.
5. Trajectory generation.
6. Joint-space and task-space control.
7. MuJoCo model matching.
8. Future extensions.
