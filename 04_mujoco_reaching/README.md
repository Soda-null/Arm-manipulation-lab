# Stage 04: MuJoCo Reaching

Goal:
Create a simple planar arm MuJoCo model, inspect the model, run reaching demos, and record videos.

Scripts:

* `inspect_model.py`
* `reaching_pd_demo.py`
* `record_video.py`

Expected outputs:

* `results/videos/mujoco_reaching_pd.mp4`
* `results/tables/mujoco_reaching_metrics.csv`
* `results/plots/mujoco_reaching_pd.png`

Status:
Initial implementation complete for XML loading, fixed-target joint-PD reaching, and MP4 video recording.

Next:
Add task-space control variants inside MuJoCo and compare reaching metrics.
