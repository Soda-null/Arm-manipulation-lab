# 01 Coordinate Frames

A coordinate frame defines a local origin and three orthonormal axes. In this
project, a pose is represented by a homogeneous transform:

```text
T = [ R  p ]
    [ 0  1 ]
```

`R` is a 3x3 rotation matrix and `p` is a 3D position vector. The helper
functions in `src/arm_lab/kinematics/transforms.py` provide rotations about
the x, y, and z axes, transform construction, inversion, and composition.

The first validation target is simple: rotations should remain orthonormal,
and a transform multiplied by its inverse should be close to identity.
