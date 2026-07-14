# 05 Trajectory Generation

Trajectory generation compares joint-space paths with task-space paths. The
current code includes a linear joint interpolation helper, a circular
task-space target generator, and a resolved-rate tracking demo.

The first task-space tracking loop is kinematic: it uses the current
end-effector error and a damped least-squares Jacobian update to move the joint
configuration toward each target sample.

Future work should add timing laws, explicit velocity limits, and comparisons
between joint-space interpolation, repeated IK, and resolved-rate tracking.
