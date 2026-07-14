# 02 Forward Kinematics

Forward kinematics maps joint angles to an end-effector pose. The current
model is a simple serial arm:

- Joint 1: base yaw around z.
- Joint 2: shoulder pitch around y.
- Joint 3: elbow pitch around y.
- Joint 4: optional wrist pitch placeholder.

Each joint stores a local axis, a local origin offset, and a link vector. The
implementation walks from base to tip, applying origin offsets, joint
rotations, and link translations.
