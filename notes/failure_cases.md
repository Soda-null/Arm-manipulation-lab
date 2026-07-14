# Failure Cases

Likely issues to study as the lab grows:

- Singularities: the Jacobian can lose rank and produce unstable updates.
- Unreachable targets: IK may stall when the target is outside the workspace.
- Poor IK convergence: damping, step size, and initial configuration matter.
- Numerical instability: finite differences and pseudo-inverses can amplify noise.
- Analytical/MuJoCo mismatch: frame conventions, joint axes, and link lengths can diverge.
- Trajectory tracking error: feasible points are not automatically easy to follow.
- Collision handling limitations: the current scaffold has no obstacle model.
