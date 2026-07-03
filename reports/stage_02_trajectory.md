# Stage 02: Trajectory Generation

## Goal

Implement and compare linear, cubic, quintic, and minimum-jerk trajectories.

## Planned Scripts

* `02_trajectory/trajectory_generation_demo.py`
* `02_trajectory/compare_trajectory_profiles.py`

## Expected Outputs

* trajectory comparison plots
* smoothness metric table

## Status

Initial implementation complete.

## Notes

Generated first-pass Stage 02 outputs:

* `results/plots/trajectory_generation_demo.png`
* `results/plots/trajectory_profiles.png`
* `results/tables/trajectory_metrics.csv`

Added trajectory tests covering endpoint preservation, constant linear velocity, gentle quintic endpoints, invalid inputs, and basic metrics.
