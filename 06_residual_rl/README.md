# Stage 06: Residual RL

Goal:
Compare vanilla PPO against controller-prior residual PPO for reaching and obstacle avoidance.

Scripts:

* `train_vanilla_ppo.py`
* `train_residual_ppo.py`
* `evaluate_policies.py`

Expected outputs:

* `results/plots/ppo_training_curves.png`
* `results/tables/residual_rl_comparison.csv`
* `results/videos/residual_policy_demo.mp4`

Status:
Initial residual-control interface benchmark, small PPO smoke training, and a small multi-seed sample-efficiency study complete.

Next:
Run longer multi-seed PPO training with broader targets and report variance more rigorously.
