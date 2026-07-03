# Stage 05: Obstacle Avoidance

## Goal

Add obstacle-reaching tasks, implement a potential-field baseline, and benchmark success and collision metrics.

## Planned Scripts

* `05_obstacle_avoidance/potential_field_demo.py`
* `05_obstacle_avoidance/benchmark_obstacle_reaching.py`

## Expected Outputs

* trajectory plots with obstacle
* collision/success table

## Status

Initial implementation complete.

## Notes

Generated first-pass Stage 05 outputs:

* `results/plots/obstacle_avoidance_trajectory.png`
* `results/tables/obstacle_reaching_benchmark.csv`

The benchmark currently compares a direct task-space PD baseline against a potential-field plus Jacobian-transpose controller. Both reach the target in the first scenario; the potential-field controller keeps slightly more clearance from the obstacle. This should be treated as an early baseline, not a final claim.
