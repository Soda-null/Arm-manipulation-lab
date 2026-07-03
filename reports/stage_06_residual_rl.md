# Stage 06: Residual RL

## Goal

Compare vanilla PPO against controller-prior residual PPO for sample efficiency, final error, smoothness, and control effort.

## Planned Scripts

* `06_residual_rl/train_vanilla_ppo.py`
* `06_residual_rl/train_residual_ppo.py`
* `06_residual_rl/evaluate_policies.py`

## Expected Outputs

* training curves
* benchmark table
* demo video

## Status

Initial residual-control interface benchmark, small PPO smoke training, and small multi-seed sample-efficiency study complete.

## Notes

Generated first-pass Stage 06 outputs:

* `results/plots/residual_rl_interface_benchmark.png`
* `results/plots/ppo_training_comparison.png`
* `results/plots/ppo_sample_efficiency.png`
* `results/tables/residual_rl_interface_benchmark.csv`
* `results/tables/vanilla_ppo_metrics.csv`
* `results/tables/residual_ppo_metrics.csv`
* `results/tables/ppo_training_comparison.csv`
* `results/tables/ppo_sample_efficiency_detail.csv`
* `results/tables/ppo_sample_efficiency_summary.csv`
* `results/policies/vanilla_ppo_reaching.zip`
* `results/policies/residual_ppo_reaching.zip`

Current PPO training remains short and should not be treated as final. In a 3-seed grid over 1024, 2048, and 4096 timesteps, residual PPO solved the task more reliably at low sample counts than vanilla PPO, but results are not perfectly monotonic due to short runs and small seed count. Longer multi-seed training and broader target distributions remain future work before making strong claims.
