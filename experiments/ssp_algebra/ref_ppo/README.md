# ref_ppo: POPGym's own PPO reference (handoff 08 step 1)

Throwaway setup that was built in `/tmp/popgym_ref` (lost on reboot). Saved here so it can be rebuilt.
See `runs/RUN_007_ssp_algebra_controller.md` section 3 for the interpretation.

## Rebuild

```bash
uv venv --python 3.14 /tmp/popgym_ref
# CPU torch first (the frozen file pins torch==2.14.1+cpu)
uv pip install --python /tmp/popgym_ref/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python /tmp/popgym_ref/bin/python -r requirements-frozen.txt --no-deps
OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_ref.py --env RepeatPreviousEasy --model GRU --steps 400000 \
    --workers 4 --envs-per-worker 8
```

`requirements-frozen.txt` is the full `uv pip freeze` (63 packages: ray 2.59.0, torch 2.14.1+cpu, popgym 1.0.7,
gymnasium 1.2.2). popgym's own baseline extra declares NO dependencies; dnc, einops, opt_einsum, scipy, pillow,
matplotlib, imageio, scikit-learn, pandas, tqdm were all installed by hand, and are in the freeze.
Do NOT install this into the repo `.venv` (ray downgrades gymnasium 1.3.0 -> 1.2.2).

## What `run_ref.py` is and why it exists

POPGym's `popgym/baselines/ppo.py` does not run on ray 2.59. `run_ref.py` bypasses its `ray.tune.run` driver and
builds `PPOConfig` directly from POPGym's own config dict and its own `ray_models` classes, with the OLD API stack
requested explicitly (`enable_rl_module_and_learner=False`, `enable_env_runner_and_connector_v2=False`) because the
shipped models are ModelV2. API drift handled in `run_ref.py`:

- `num_workers` / `num_envs_per_worker` -> `num_env_runners` / `num_envs_per_env_runner`
- `horizon` removed
- `local_dir=` is a hard error in `ray.tune.run` (renamed `storage_path`) -- avoided by not using Tune

Edits that were ALSO made to the installed `popgym/baselines/ppo.py` while trying to run it unmodified (NOT used by
`run_ref.py`, recorded only so they are not rediscovered):

1. `local_dir=storage_path,` -> `storage_path=storage_path,`
2. `trial_name_creator` crashed because a single-element `grid_search` resolves to a bare tuple/str; patched by
   unwrapping `trial.config["env"]` and `trial.config["model"]["custom_model"]`. After that it still failed
   (`'str' object has no attribute '__name__'`), so the Tune driver was abandoned.

## Results

- `smoke_40k_RepeatPreviousEasy_GRU.json`: the only JSON `run_ref.py` actually wrote (2 workers x 4 envs, 41k steps,
  `-0.513 -> -0.257`).
- The two real runs were killed by wall-clock caps (`timeout`) before the script's final write, so their curves were
  only printed. Transcribed from the console output, NOT from a result file: see `results_transcribed.md`.
