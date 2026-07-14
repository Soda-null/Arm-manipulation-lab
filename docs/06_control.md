# 06 Control

Control will connect desired motion to joint commands. The current project has
small PD helpers and a resolved-rate update. These are placeholders for later
closed-loop simulations.

Near-term control tasks:

- Joint-space PD tracking.
- Task-space PD with Jacobian transpose or pseudo-inverse mapping.
- Resolved-rate control with singularity handling.

The first custom MuJoCo controller is a joint-space PD torque controller:

```text
tau = Kp(q_des - q) + Kd(qd_des - qd)
```

It intentionally omits integral control, gravity compensation, and inverse
dynamics so it can serve as a minimal baseline against MuJoCo's built-in
position actuators.
