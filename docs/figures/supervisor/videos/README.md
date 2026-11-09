# Policy videos (illustrative retrained runs)

**Every clip is an illustrative retrained run (same config and seed as the claim cell), not one of the analysed runs.** The claim runs never saved their policies (`upload_ckpt` was false; d151), so one centered/uncentered pair per robot was retrained with `jobs_r4.py video-runs` (campaign `r7videoruns-90413106`, block `r7vid`, checkpoints on R2) and rendered here. A retrained run does not reproduce its logged tracking exactly; both numbers are given.

## Seed choice (no cherry-picking)

Per robot: the centered claim seed at the **lower median** (rank 10 of 20) of the logged final tracking (mean of the last 5 evals, claim seeds 3000-3019, the values behind the supervisor figures). 20 is even and the two middle seeds are equidistant from the median, so the lower one is taken by rule. `selection.json` has the numbers.

- Go1: seed **3002**, logged centered tracking 0.8829 (uncentered partner, same seed, logged 0.301).
- G1: seed **3012**, logged centered tracking 0.8174 (uncentered partner, same seed, logged 0.298).

## What the clips show (read from the clips and the per-segment velocities below)

- Go1 centered walks every segment (realised vx 0.74-0.75 against 0.8 commanded, wz 0.51 against 0.5). Go1 uncentered stays on the floor (root z 0.07-0.08 m, velocities about 0): the standing-floor failure of the claim cell.
- G1 centered walks upright throughout (root z 0.74-0.75 m); its second forward segment realises vx 0.49 against 0.8, so the tracking is good, not perfect (final tracking 0.81).
- G1 uncentered stands upright and shuffles for about the first 10 s without following the commands, then collapses and stays down (root z 0.12 m in the last two segments; torso tilt stays under the 70 deg reset limit, so no reset fires). This is one clip of one seed; its tracking value (0.30) is the floor shared by all 20 uncentered claim seeds.

## Clips

### go1_centered_seed3002.mp4 / .gif

- arm: centered, offset +1; robot Go1; seed 3002; job `r7vid_go1_g0.999_o1_value_s3002`
- checkpoint: R2 mjlab_ssp_centering/r7videoruns-90413106/runs/r7vid_go1_g0.999_o1_value_s3002/model_final.pt (local copy mjlab_centering/results/video_runs/r7vid_go1_g0.999_o1_value_s3002/model_final.pt)
- final tracking (last-5 evals): **0.8804** this retrained run; 0.8829 logged in the claim cell (same seed)
- trained on GPU `NVIDIA L40`, torch 2.15.0+cu132; rendered with torch 2.15.0+cu132, MUJOCO_GL=egl
- sizes: mp4 2.75 MB, gif 8.0 MB
- realised velocities per command segment (body frame, mean over the segment):

  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.75, vy -0.04, wz -0.02, min root z 0.30 m
  - turn left: command (+0.4, +0.0, +0.5) -> vx +0.43, vy +0.00, wz +0.51, min root z 0.30 m
  - sideways (left): command (+0.0, +0.4, +0.0) -> vx +0.07, vy +0.29, wz +0.02, min root z 0.30 m
  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.74, vy -0.02, wz -0.02, min root z 0.29 m
  - stop: command (+0.0, +0.0, +0.0) -> vx +0.04, vy -0.01, wz +0.02, min root z 0.28 m

### go1_uncentered_seed3002.mp4 / .gif

- arm: uncentered, offset +1; robot Go1; seed 3002; job `r7vid_go1_g0.999_o1_none_s3002`
- checkpoint: R2 mjlab_ssp_centering/r7videoruns-90413106/runs/r7vid_go1_g0.999_o1_none_s3002/model_final.pt (local copy mjlab_centering/results/video_runs/r7vid_go1_g0.999_o1_none_s3002/model_final.pt)
- final tracking (last-5 evals): **0.3182** this retrained run; 0.3011 logged in the claim cell (same seed)
- trained on GPU `NVIDIA L40`, torch 2.15.0+cu132; rendered with torch 2.15.0+cu132, MUJOCO_GL=egl
- sizes: mp4 0.94 MB, gif 5.91 MB
- realised velocities per command segment (body frame, mean over the segment):

  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.01, vy +0.01, wz -0.00, min root z 0.07 m
  - turn left: command (+0.4, +0.0, +0.5) -> vx +0.01, vy +0.01, wz +0.01, min root z 0.08 m
  - sideways (left): command (+0.0, +0.4, +0.0) -> vx +0.00, vy +0.00, wz -0.00, min root z 0.08 m
  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx -0.00, vy +0.01, wz -0.01, min root z 0.08 m
  - stop: command (+0.0, +0.0, +0.0) -> vx +0.00, vy -0.01, wz +0.01, min root z 0.08 m

### g1_centered_seed3012.mp4 / .gif

- arm: centered, offset +1; robot G1; seed 3012; job `r7vid_g1_g0.99_o1_value_s3012`
- checkpoint: R2 mjlab_ssp_centering/r7videoruns-90413106/runs/r7vid_g1_g0.99_o1_value_s3012/model_final.pt (local copy mjlab_centering/results/video_runs/r7vid_g1_g0.99_o1_value_s3012/model_final.pt)
- final tracking (last-5 evals): **0.8088** this retrained run; 0.8174 logged in the claim cell (same seed)
- trained on GPU `NVIDIA L40`, torch 2.15.0+cu132; rendered with torch 2.15.0+cu132, MUJOCO_GL=egl
- sizes: mp4 2.49 MB, gif 7.57 MB
- realised velocities per command segment (body frame, mean over the segment):

  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.70, vy -0.01, wz -0.03, min root z 0.74 m
  - turn left: command (+0.4, +0.0, +0.5) -> vx +0.38, vy +0.02, wz +0.42, min root z 0.75 m
  - sideways (left): command (+0.0, +0.4, +0.0) -> vx +0.04, vy +0.26, wz +0.01, min root z 0.75 m
  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.49, vy +0.02, wz +0.03, min root z 0.74 m
  - stop: command (+0.0, +0.0, +0.0) -> vx +0.17, vy -0.04, wz -0.02, min root z 0.75 m

### g1_uncentered_seed3012.mp4 / .gif

- arm: uncentered, offset +1; robot G1; seed 3012; job `r7vid_g1_g0.99_o1_none_s3012`
- checkpoint: R2 mjlab_ssp_centering/r7videoruns-90413106/runs/r7vid_g1_g0.99_o1_none_s3012/model_final.pt (local copy mjlab_centering/results/video_runs/r7vid_g1_g0.99_o1_none_s3012/model_final.pt)
- final tracking (last-5 evals): **0.3013** this retrained run; 0.2976 logged in the claim cell (same seed)
- trained on GPU `NVIDIA L40`, torch 2.15.0+cu132; rendered with torch 2.15.0+cu132, MUJOCO_GL=egl
- sizes: mp4 1.81 MB, gif 8.14 MB
- realised velocities per command segment (body frame, mean over the segment):

  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx +0.13, vy +0.03, wz -0.63, min root z 0.74 m
  - turn left: command (+0.4, +0.0, +0.5) -> vx +0.03, vy -0.00, wz -0.41, min root z 0.74 m
  - sideways (left): command (+0.0, +0.4, +0.0) -> vx +0.04, vy +0.02, wz -0.22, min root z 0.74 m
  - forward 0.8 m/s: command (+0.8, +0.0, +0.0) -> vx -0.25, vy -0.15, wz +0.21, min root z 0.12 m
  - stop: command (+0.0, +0.0, +0.0) -> vx -0.01, vy -0.01, wz +0.03, min root z 0.13 m

## Command sequence (fixed, body frame; never resampled)

forward 0.8 m/s (4 s, cmd vx +0.8 vy +0.0 wz +0.0) -> turn left (3 s, cmd vx +0.4 vy +0.0 wz +0.5) -> sideways (left) (3 s, cmd vx +0.0 vy +0.4 wz +0.0) -> forward 0.8 m/s (3 s, cmd vx +0.8 vy +0.0 wz +0.0) -> stop (3 s, cmd vx +0.0 vy +0.0 wz +0.0)

Commands are (vx m/s, vy m/s, wz rad/s). 'turn left' is a walking turn (vx 0.4, wz +0.5, the top of the trained yaw range). The heading/world-frame/standing command modes are switched off so the script is exact.

## Render settings

- 1280x720, 30 fps (control step 0.02 s; frames are taken whenever simulated time passes k/30 s), H.264 yuv420p crf 22; GIF 640 px wide, 12 fps, 96-colour palette (shrinks automatically if over 10 MB).
- mjlab commit 8bb2bc9e, `ManagerBasedRlEnv(render_mode='rgb_array')` with mjlab's `OffscreenRenderer`; tracking camera on the robot root (`ViewerConfig.OriginType.ASSET_ROOT`; Go1 distance 2.6 m / G1 3.6 m, elevation -18/-14 deg, azimuth 150 deg). One env, deterministic (actor mean action, `stochastic_output=False`), observation normaliser and encoder from the checkpoint and `config.json` (raw encoder, as in the claim cells). mjlab's `fell_over` termination (torso tilt > 70 deg) stays active as in training, so a fall past that angle would reset the robot in place; a robot that collapses without crossing it (the uncentered G1) simply stays down.
- Overlay (top-left): robot, arm, seed, final tracking, retrained-run label; (bottom-left): time and current command.

## Regenerate

```bash
cd mjlab_centering
.venv/bin/python analysis/render_policy_videos.py all
```

This downloads the four checkpoints (read-only) into `results/video_runs/`, re-derives the seeds from the local mirrors (`selection.json` is reused unless `--reselect`), renders the mp4 and gif files and rewrites this README. To retrain the checkpoints: `python jobs_r4.py video-runs` then launch the campaign with `orchestrator.py` (see decision 151).

## Cost of the retrains

One community L40 pod ($0.69/h), 2026-11-06 03:37-04:56 UTC (four runs in parallel, 9,000 iterations in total): **$0.92** from the RunPod billing API (`analysis/spend.py --since 2026-11-06T03:00:00Z`), against a $2 cap. The orchestrator's own ledger stopped at $0.46 because its process was interrupted before the runs finished; the billing figure is the one to use. No pod was left running.
