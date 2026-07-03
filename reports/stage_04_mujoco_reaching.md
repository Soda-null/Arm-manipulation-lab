# Stage 04: MuJoCo Reaching

## Goal

Create/load a simple planar arm XML, run joint-space PD reaching, and record a reaching video.

## Planned Scripts

* `04_mujoco_reaching/inspect_model.py`
* `04_mujoco_reaching/reaching_pd_demo.py`
* `04_mujoco_reaching/record_video.py`

## Expected Outputs

* MuJoCo reaching video
* reaching metrics table

## Status

Initial implementation complete for model inspection, fixed-target joint-PD reaching, and video recording.

## Notes

Generated first-pass Stage 04 outputs:

* `results/plots/mujoco_reaching_pd.png`
* `results/tables/mujoco_reaching_metrics.csv`
* `results/videos/mujoco_reaching_pd.mp4`

Added MuJoCo XML loading tests covering model dimensions and expected object names.
