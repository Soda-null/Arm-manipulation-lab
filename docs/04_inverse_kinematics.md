# 04 Inverse Kinematics

Inverse kinematics searches for joint angles that move the end effector toward
a target. This project starts with position-only damped least-squares IK:

```text
dq = J^T (J J^T + lambda^2 I)^-1 error
```

Damping reduces unstable updates near singular configurations. The current
solver is a learning tool, not a production IK library. It should be tested
against reachable and unreachable targets.
