# Stage 01: Kinematics Foundations

## Goal

Implement FK, IK, Jacobian, workspace visualization, and manipulability analysis for planar arms.

## Planned Scripts

* `01_kinematics/fk_ik_demo.py`
* `01_kinematics/jacobian_demo.py`
* `01_kinematics/workspace_visualization.py`

## Expected Outputs

* `results/plots/fk_ik_demo.png`
* `results/plots/workspace_2link.png`
* `results/plots/manipulability_map.png`

## Status

Initial implementation complete.

## Notes

Generated first-pass Stage 01 plots:

* `results/plots/fk_ik_demo.png`
* `results/plots/workspace_2link.png`
* `results/plots/manipulability_map.png`

Added kinematics tests covering FK reference poses, IK round-trip accuracy, unreachable targets, Jacobian finite differences, joint-position helpers, and singularity manipulability.
