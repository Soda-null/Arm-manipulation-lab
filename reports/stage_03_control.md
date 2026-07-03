# Stage 03: Classical Control

## Goal

Implement joint-space PD, task-space control, and Jacobian transpose control for reaching.

## Planned Scripts

* `03_control/joint_space_pd_demo.py`
* `03_control/task_space_control_demo.py`
* `03_control/jacobian_transpose_demo.py`

## Expected Outputs

* reaching plots
* error curves
* control effort comparison

## Status

Initial implementation complete.

## Notes

Generated first-pass Stage 03 outputs:

* `results/plots/joint_pd_reaching.png`
* `results/plots/task_space_reaching.png`
* `results/plots/jacobian_transpose_demo.png`

Added controller tests covering joint-space PD, task-space PD, Jacobian transpose mapping, simple joint integration, and dimension validation.
