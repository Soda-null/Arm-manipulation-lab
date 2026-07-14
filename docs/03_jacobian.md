# 03 Jacobian

The geometric Jacobian maps joint velocities to end-effector spatial velocity:

```text
[ v ] = J(q) q_dot
[ w ]
```

For a revolute joint, the translational part is:

```text
Jv_i = axis_i x (p_ee - p_joint_i)
```

The angular part is the joint axis in world coordinates. The validation demo
compares the Jacobian-predicted end-effector velocity with a finite-difference
estimate.
