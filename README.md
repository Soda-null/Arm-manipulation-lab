# Arm Manipulation Lab: 3D Spatial Manipulator Edition

This repository is a staged robotics mini-lab for learning 3D manipulator
kinematics, Jacobian-based inverse kinematics, trajectory generation, control,
and MuJoCo simulation.

## Motivation

The project is designed to move from basic manipulator math to
simulation-based robot control. The code favors readable NumPy
implementations, small examples, and validation scripts over a polished
framework. It is meant to grow step by step into portfolio evidence for
robotics and mechanical engineering study.

## Current Scope

This pass establishes the project architecture and implements the first 3D
kinematics utilities:

- Coordinate-frame transforms and homogeneous transforms.
- A simple serial-arm model with revolute joints.
- Forward kinematics for a staged 3D arm.
- Geometric Jacobian computation.
- Damped least-squares inverse kinematics for position targets.
- Matplotlib examples that save early result figures.

MuJoCo, obstacle avoidance, visual servoing, and RL are intentionally left as
future stages.

## Roadmap

- [x] Stage 0: Project skeleton
- [x] Stage 1: 3D transformations and coordinate frames
- [x] Stage 2: Forward kinematics
- [x] Stage 3: Jacobian and numerical validation
- [x] Stage 4: Jacobian-based inverse kinematics
- [x] Stage 5: Trajectory generation
- [ ] Stage 6: Control
- [x] Stage 7A: Minimal MuJoCo reaching integration
- [x] Stage 7B: MuJoCo GIF export and FK model validation
- [ ] Stage 8: Obstacle avoidance
- [ ] Stage 9: Visual servoing
- [ ] Stage 10: Residual RL extension

## Expected Results

Figures:

- `results/figures/transform_demo.png`
- `results/figures/fk_demo.png`
- `results/figures/ik_convergence.png`
- `results/figures/ik_final_pose.png`
- `results/figures/trajectory_tracking_demo.png`
- `results/figures/interactive_ik_viewer.png`
- `results/figures/mujoco_reaching_demo.png`
- `results/figures/mujoco_fk_validation.png`
- `results/figures/mujoco_trajectory_tracking_demo.png`
- `results/figures/mujoco_control_sweep.png`
- `results/figures/mujoco_pd_control_comparison.png`
- `results/figures/robot_shell_demo.png`
- `results/gifs/trajectory_tracking_demo.gif`
- `results/gifs/mujoco_reaching_demo.gif`
- `results/gifs/mujoco_trajectory_tracking_demo.gif`
- `results/gifs/robot_shell_tracking_demo.gif`
- `results/tables/mujoco_fk_validation.csv`
- `results/tables/mujoco_trajectory_tracking.csv`
- `results/tables/mujoco_control_sweep.csv`
- `results/tables/mujoco_pd_control_comparison.csv`

Future placeholders:

- GIFs: `results/gifs/`
- Videos: `results/videos/`
- Tables: `results/tables/`

## How To Run

From this repository root:

```bash
pip install -r requirements.txt
python examples/01_transform_demo.py
python examples/02_forward_kinematics_demo.py
python examples/03_jacobian_validation_demo.py
python examples/04_inverse_kinematics_reaching_demo.py
python examples/05_trajectory_tracking_demo.py
python examples/06_mujoco_reaching_demo.py
python examples/07_interactive_ik_viewer.py
python examples/08_trajectory_animation_demo.py
python examples/09_mujoco_model_validation_demo.py
python examples/10_mujoco_trajectory_tracking_demo.py
python examples/11_mujoco_control_sweep.py
python examples/12_mujoco_pd_control_demo.py
python examples/13_visual_shell_demo.py
pytest
```

For a non-interactive smoke test:

```bash
python examples/07_interactive_ik_viewer.py --save-only
```

The MuJoCo example is optional because it requires the `mujoco` Python package:

```bash
python examples/06_mujoco_reaching_demo.py
```

If MuJoCo is not installed, the script prints a friendly future-stage message.

## Repository Philosophy

- Write readable robotics code before optimizing.
- Keep each stage small enough to validate.
- Prefer NumPy and Matplotlib for the early math.
- Keep failure analysis visible instead of hiding rough edges.
- Avoid adding heavy dependencies before the project needs them.
