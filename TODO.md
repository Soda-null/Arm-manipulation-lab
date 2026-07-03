# TODO

## Stage 01: Kinematics

* [x] Implement forward_kinematics_2link
* [x] Implement inverse_kinematics_2link
* [x] Implement jacobian_2link
* [x] Implement workspace visualization
* [x] Add singularity/manipulability plot
* [x] Add Stage 01 kinematics tests

## Stage 02: Trajectory

* [x] Implement linear interpolation
* [x] Implement cubic trajectory
* [x] Implement quintic trajectory
* [x] Implement minimum-jerk trajectory
* [x] Compare smoothness and path length
* [x] Add Stage 02 trajectory tests

## Stage 03: Control

* [x] Implement joint-space PD controller
* [x] Implement task-space controller
* [x] Implement Jacobian transpose controller
* [x] Compare final error and control effort
* [x] Add Stage 03 controller tests

## Stage 04: MuJoCo Reaching

* [x] Create planar 2-link arm XML
* [x] Load model with MuJoCo
* [x] Implement reaching PD demo
* [x] Record reaching video
* [x] Add Stage 04 MuJoCo model tests

## Stage 05: Obstacle Avoidance

* [x] Add obstacle geometry
* [x] Implement potential field controller
* [x] Add collision metric
* [x] Benchmark reaching with obstacles
* [x] Add Stage 05 potential-field tests

## Stage 06: Residual RL

* [x] Define lightweight reaching environment wrapper
* [x] Implement residual action interface
* [x] Add Stage 06 interface benchmark
* [x] Add Stage 06 environment tests
* [x] Train vanilla PPO smoke baseline
* [x] Train residual PPO smoke baseline
* [x] Compare first PPO smoke-training results
* [x] Compare small multi-seed sample efficiency
* [ ] Generate final report
