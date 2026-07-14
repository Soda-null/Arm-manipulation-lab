# 07 MuJoCo Integration

MuJoCo integration begins with a minimal 3-DoF reaching task. The repository
includes an MJCF model under `src/arm_lab/simulation/simple_arm_scene.xml`.
It matches the current kinematic model closely enough for early experiments:
base yaw, shoulder pitch, elbow pitch, simple capsule links, an end-effector
site, a target site, and joint position actuators.

The main risk is model mismatch: the analytical NumPy arm and the MuJoCo XML
arm must agree on joint axes, link lengths, and frame conventions.

The first integration pass keeps the existing IK algorithm unchanged:

```text
target position -> DLS IK q_des -> MuJoCo position actuators -> physical error
```

Future MuJoCo work should validate frame alignment more strictly, add video
rendering, compare joint-space PD settings, and then introduce contact tasks.

The current GIF export animates the MuJoCo-simulated joint states with
Matplotlib. This keeps the demo reproducible in headless environments where
MuJoCo's OpenGL renderer may not have access to a macOS CoreGraphics context.

The validation demo samples joint configurations and compares the NumPy FK
end-effector position with the MuJoCo `end_effector` site position. This should
stay near numerical precision. If it grows, the MJCF and analytical model have
drifted apart.

The MuJoCo trajectory-tracking demo uses the kinematic tracker to produce a
sequence of desired joint angles, then asks MuJoCo position actuators to follow
that sequence. Its error is expected to be larger than the pure kinematic
tracker because the simulated arm has inertia, damping, gravity, and actuator
response.

The control sweep varies actuator stiffness, joint damping, and trajectory
sample hold time. This turns the observed tracking gap into a measurable
control question rather than a one-off tuning guess.
