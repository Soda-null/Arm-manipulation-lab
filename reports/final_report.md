# MuJoCo Arm Manipulation Lab

## Abstract

This project is a robot arm manipulation mini-lab that connects classical robot modeling, trajectory generation, control, MuJoCo simulation, obstacle avoidance, and residual reinforcement learning. The central question is whether classical kinematics and task-space controllers can provide useful priors for learning-based reaching.

The current implementation uses a planar 2-link arm as a controlled testbed. It includes forward and inverse kinematics, Jacobian analysis, smooth trajectory generation, joint-space and task-space control, MuJoCo reaching, potential-field obstacle avoidance, and a small PPO comparison between direct torque learning and residual learning on top of a controller prior.

The results are early-stage development benchmarks, not final research claims. Still, they show a clear and useful pattern: in this simple reaching setup, residual PPO is easier to train than vanilla PPO because it does not need to discover the entire control behavior from scratch.

## Project Motivation

Robot learning can be difficult when a policy must learn low-level control entirely from trial and error. For a robot arm, much of the structure is already known: kinematics, Jacobians, task-space error feedback, and simple control laws can provide useful behavior before learning begins.

This project studies a practical hybrid idea:

```text
final_action = controller_prior + residual_action
```

The controller prior provides a stable classical baseline. The learned residual policy only needs to correct what the controller does not handle well. This mirrors a broader robotics pattern: use analytical structure where it is reliable, and use learning for the parts that are hard to model or tune.

## Core Question

Can classical kinematics and task-space control provide useful priors for learning-based robot arm reaching and obstacle avoidance?

## System Overview

```mermaid
flowchart LR
    A["Planar arm kinematics"] --> B["FK / IK / Jacobian"]
    B --> C["Trajectory generation"]
    C --> D["Classical control"]
    D --> E["MuJoCo reaching"]
    E --> F["Obstacle avoidance"]
    F --> G["Residual PPO"]
    G --> H["Benchmark and report"]
```

## Implemented Stages

| Stage | Component | Main outputs |
| --- | --- | --- |
| 01 | Kinematics | FK, IK, Jacobian, workspace, manipulability plots |
| 02 | Trajectory generation | Linear, cubic, quintic, minimum-jerk trajectory profiles |
| 03 | Classical control | Joint-space PD, task-space PD, Jacobian transpose control |
| 04 | MuJoCo reaching | 2-link XML model, reaching metrics, reaching video |
| 05 | Obstacle avoidance | Potential-field controller, benchmark table, avoidance video |
| 06 | Residual RL | Gymnasium adapter, PPO smoke training, sample-efficiency study |

## Methods

### Kinematics

The first stage implements planar 2-link kinematics:

```text
q -> x
x -> q
q_dot -> x_dot through J(q)
```

Forward kinematics maps joint angles to end-effector position. Inverse kinematics solves reachable target points. The Jacobian maps joint velocities into task-space velocities and supports Jacobian-based control.

Main outputs:

* `results/plots/fk_ik_demo.png`
* `results/plots/workspace_2link.png`
* `results/plots/manipulability_map.png`

### Trajectory Generation

The trajectory module generates point-to-point reference trajectories using scalar time-scaling:

```text
p(t) = start + s(t) * (goal - start)
```

Implemented time scalings:

* linear
* cubic
* quintic
* minimum jerk

This stage does not solve obstacle avoidance or inverse kinematics. It defines smooth intermediate reference points that later controllers can track.

Main outputs:

* `results/plots/trajectory_generation_demo.png`
* `results/plots/trajectory_profiles.png`
* `results/tables/trajectory_metrics.csv`

### Classical Control

The project implements three classical control components:

Joint-space PD:

```text
tau = Kp * (q_target - q) - Kd * q_dot
```

Task-space PD:

```text
F = Kp * (x_target - x) - Kd * x_dot
```

Jacobian transpose mapping:

```text
tau = J(q)^T F
```

Together, task-space PD and Jacobian transpose control form an important baseline:

```text
target error -> task-space force -> joint torque
```

Main outputs:

* `results/plots/joint_pd_reaching.png`
* `results/plots/task_space_reaching.png`
* `results/plots/jacobian_transpose_demo.png`

### MuJoCo Reaching

The MuJoCo stage creates a simple planar 2-link arm XML model and runs fixed-target reaching with IK plus joint-space PD control.

The current MuJoCo reaching result:

| Method | Target | Final error | Steps | dt |
| --- | --- | --- | --- | --- |
| joint PD with IK target | `(1.0, 0.5)` | `0.000023` | `2000` | `0.002` |

Main outputs:

* `results/plots/mujoco_reaching_pd.png`
* `results/tables/mujoco_reaching_metrics.csv`
* `results/videos/mujoco_reaching_pd.mp4`

### Obstacle Avoidance

Obstacle avoidance uses a circular obstacle and potential-field control:

```text
F = F_attractive + F_repulsive + F_tangential
tau = J(q)^T F
```

The attractive term pulls the end effector toward the target. The repulsive term pushes it away from the obstacle. A small tangential term helps reduce local-minimum behavior.

Current benchmark:

| Method | Success | Final error | Collision count | Minimum clearance | Path length | Control effort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| task_space_pd_direct | `1` | `0.000023` | `0` | `0.019138` | `1.692152` | `1773.004414` |
| potential_field_jt | `1` | `0.000010` | `0` | `0.026155` | `2.000163` | `13735.629422` |

Interpretation: in this first scenario both methods reach the target, while the potential-field controller keeps slightly more clearance from the obstacle. It also uses a longer path and substantially higher control effort. This is a useful early result because it shows the tradeoff rather than only reporting success.

Main outputs:

* `results/plots/obstacle_avoidance_trajectory.png`
* `results/tables/obstacle_reaching_benchmark.csv`
* `results/videos/obstacle_avoidance_potential_field.mp4`

### Residual PPO

The residual learning stage compares two action parameterizations.

Vanilla PPO:

```text
observation -> PPO -> joint torque
```

Residual PPO:

```text
observation -> PPO -> residual torque
controller_prior + residual torque -> joint torque
```

The controller prior is the task-space PD plus Jacobian transpose controller. The residual policy learns a correction on top of this prior.

Current residual action interface:

```text
final_action = controller_prior + residual_scale * residual_action
```

## PPO Smoke Results

A short single-run PPO smoke test used `4096` timesteps.

| Method | Success rate | Final error | Steps | Return | Control effort |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla PPO | `0.0` | `2.540648` | `500.0` | `-777.734674` | `4846.776400` |
| residual PPO | `1.0` | `0.047742` | `246.0` | `-119.548819` | `1372.930290` |

In this smoke run, vanilla PPO did not solve the task, while residual PPO reached the target threshold.

Main outputs:

* `results/plots/ppo_training_comparison.png`
* `results/tables/ppo_training_comparison.csv`
* `results/videos/residual_ppo_reaching.mp4`
* `results/policies/vanilla_ppo_reaching.zip`
* `results/policies/residual_ppo_reaching.zip`

## Sample-Efficiency Study

A small multi-seed study compared `vanilla_ppo` and `residual_ppo` over three training budgets and three seeds.

Training budgets:

* `1024` timesteps
* `2048` timesteps
* `4096` timesteps

Seeds:

* `3`
* `7`
* `11`

Summary:

| Method | Timesteps | Mean success rate | Mean final error | Mean steps |
| --- | ---: | ---: | ---: | ---: |
| vanilla PPO | `1024` | `0.000000` | `1.633425` | `500.000000` |
| vanilla PPO | `2048` | `0.000000` | `1.336820` | `500.000000` |
| vanilla PPO | `4096` | `0.333333` | `1.680695` | `430.333333` |
| residual PPO | `1024` | `1.000000` | `0.047033` | `206.333333` |
| residual PPO | `2048` | `0.666667` | `0.050971` | `318.333333` |
| residual PPO | `4096` | `0.666667` | `0.059561` | `340.666667` |

Interpretation: in this small study, residual PPO solves the reaching task more reliably at low sample counts than vanilla PPO. However, the residual PPO results are not perfectly monotonic across training budgets, which suggests noticeable variance from short runs and a small seed count. This should be treated as early evidence, not a final sample-efficiency conclusion.

Main outputs:

* `results/plots/ppo_sample_efficiency.png`
* `results/tables/ppo_sample_efficiency_summary.csv`
* `results/tables/ppo_sample_efficiency_detail.csv`
* `results/policies/sample_efficiency/`

## Videos

The project currently includes three demonstration videos:

| Video | Description |
| --- | --- |
| `results/videos/mujoco_reaching_pd.mp4` | MuJoCo fixed-target reaching with joint-space PD |
| `results/videos/obstacle_avoidance_potential_field.mp4` | Potential-field obstacle avoidance rollout |
| `results/videos/residual_ppo_reaching.mp4` | Trained residual PPO reaching rollout |

The videos use headless matplotlib rendering where needed. This avoids relying on a GUI/OpenGL context while still visualizing the simulated arm motion.

## Test Coverage

The project includes tests for:

* kinematics
* trajectory generation
* classical controllers
* MuJoCo XML loading
* potential-field helpers
* lightweight reaching environment
* Gymnasium adapter

Current verification:

```text
31 tests passed
syntax ok: 36 Python files
```

## Limitations

This is an early-stage mini-lab. The current results are useful for development and portfolio evidence, but they are not yet a polished research study.

Main limitations:

* The robot is a simple planar 2-link arm, not a full 3D manipulator.
* PPO experiments use short training budgets.
* The sample-efficiency study uses only three seeds.
* Target distributions are narrow.
* Obstacle avoidance is currently demonstrated in a simple circular-obstacle setting.
* Residual PPO has not yet been tested under disturbances, model mismatch, contact, or more complex tasks.
* The current benchmark should not be interpreted as proof that residual PPO is always better than vanilla PPO.

## Future Work

Recommended next steps:

1. Extend kinematics to a 3-link planar arm.
2. Add broader target sampling for reaching.
3. Run longer PPO training with more seeds.
4. Add learning curves instead of only final evaluation metrics.
5. Compare residual PPO against the controller prior alone across varied targets.
6. Add MuJoCo obstacle geometry and evaluate obstacle avoidance in simulation.
7. Test residual policies under controller gain mismatch or link-length perturbations.
8. Add a final README figure layout for GitHub presentation.

## Portfolio Summary

This project demonstrates a full robotics mini-lab pipeline:

```text
kinematics -> trajectory generation -> control -> MuJoCo simulation -> obstacle avoidance -> residual RL
```

The main technical idea is that classical robot control can provide useful structure for learning-based manipulation. Instead of making PPO learn low-level reaching from scratch, the residual formulation gives PPO a stable controller prior and asks it to learn corrective actions. Early experiments on a planar reaching task suggest that this residual setup is easier to train than vanilla PPO at low sample counts.

The strongest current portfolio claim is:

> I built a robot arm manipulation mini-lab that connects analytical kinematics, Jacobian-based control, MuJoCo simulation, potential-field obstacle avoidance, and residual PPO. The project investigates how classical controller priors can make learning-based reaching easier than direct torque learning from scratch.
