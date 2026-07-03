# Roadmap

## Stage 01: Kinematics Foundations

Goal:

* implement FK, IK, Jacobian for 2-link and possibly 3-link planar arms
* visualize workspace and singularities

Expected outputs:

* `results/plots/workspace_2link.png`
* `results/plots/fk_ik_demo.png`
* `results/plots/manipulability_map.png`

Exit criteria:

* given target point, IK can solve reachable targets
* FK verifies end-effector position
* workspace plot generated

## Stage 02: Trajectory Generation

Goal:

* implement linear, cubic, quintic, minimum-jerk trajectories
* compare path smoothness and velocity profiles

Expected outputs:

* trajectory comparison plots
* smoothness metric table

Exit criteria:

* trajectory scripts generate plots and basic metrics

## Stage 03: Classical Control

Goal:

* implement joint-space PD control
* implement task-space control
* implement Jacobian transpose / pseudo-inverse control

Expected outputs:

* reaching plots
* error curves
* control effort comparison

Exit criteria:

* controllers can reach static targets in simple simulation

## Stage 04: MuJoCo Reaching

Goal:

* create/load simple planar arm XML
* run joint-space PD reaching
* record reaching video

Expected outputs:

* MuJoCo reaching video
* reaching metrics table

Exit criteria:

* arm reaches a fixed target in MuJoCo

## Stage 05: Obstacle Avoidance

Goal:

* add obstacle-reaching task
* implement potential-field baseline
* benchmark success and collision metrics

Expected outputs:

* trajectory plots with obstacle
* collision/success table

Exit criteria:

* controller can reach around simple obstacle in selected cases

## Stage 06: Residual RL

Goal:

* compare vanilla PPO against controller-prior residual PPO
* evaluate sample efficiency, final error, smoothness, control effort

Expected outputs:

* training curves
* benchmark table
* demo video

Exit criteria:

* residual policy improves or clearly analyzes failure relative to baseline

## Final Deliverables

* README
* technical report
* result plots
* benchmark tables
* short demo videos
* reproducibility commands
