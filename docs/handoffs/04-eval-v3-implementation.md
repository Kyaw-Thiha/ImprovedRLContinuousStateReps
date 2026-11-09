# Handoff 04: eval_v3 implementation and run procedure

Written 2026-10-06. The eval_v3 code is complete locally. No pod has been launched for this phase and no eval_v3 artifact has been written to R2. Local gates: `pytest` 108 passed, 2 skipped; `scripts/smoke_test_eval_v3.py` passed with `--no-r2` and tiny budgets.

## Purpose and code

RUN_003 produced the comparable DQN headline results but did not save policies. eval_v3 saves the final policy, tests robustness without retraining, measures deployment properties, and tests whether DQN replay/target settings explain SSP collapse.

- `scripts/eval_v3_layout.py`: the only source of eval_v3 R2 keys and seed conventions.
- `rl/policy_io.py`: safe tensor/plain-data policy save/load, representation state restoration, exact greedy replay verification.
- `experiments/trial_cartpole_dqn.py`: optional environment seeding, final policy save, probe Q/action-gap diagnostics, final returns, verification replay.
- `scripts/make_probe_states.py`: deterministic probe states and pooled observation-noise scale.
- `scripts/run_policy_pass.py`, `scripts/policy_pass_worker.py`: retraining, policy verification/upload, summaries, and DQN-settings variants.
- `rl/perturbations.py`: CartPole perturbation conditions and seeded rollout semantics.
- `scripts/run_robustness.py`, `scripts/analyze_robustness.py`: resumable robustness jobs and bootstrap/Holm analysis.
- `scripts/run_deployment.py`: model size, simulated fp16/int8 retention, and quiet CPU latency.
- `scripts/launch_cell_pod.py`: existing search launch plus eval modes (`--job policy_pass`, `robustness`, `deployment`).

## R2 layout

Keys are built through `scripts/eval_v3_layout.py`; do not hand-write them.

```text
stage0_cartpole/eval_v3/
  _code/<content-hashed tarball>
  _shared/probe_states.npz
  <study>/
    policies/seed_<SEED>.pt
    train_runs.jsonl
    train_summary.json
    manifest.json
    robustness_runs.jsonl
    deployment.json
  dqn_settings/<study>/<variant>/
    train_runs.jsonl
    train_summary.json
    manifest.json
```

`<study>` is `dqn_<rep>_<centering>[_deep]_v4_hparam_search`. The local mirror is `outputs/eval_v3/`. eval_v3 never writes `refilter_v2/`, `best_params/`, `optuna_dbs/` or their backups. Normal policy seeds are 3000-3019; DQN-settings variants use 3000-3009 and save no policies.

A normal policy record contains the seed, last-5 score, last checkpoint, histories, final-policy returns, probe history, timing, policy key/hash/bytes, and `policy_verified`. A policy is complete only after exact replay from the saved file (`verify_max_abs_diff == 0.0`) and successful upload (or local existence with `--no-r2`). Robustness records contain study, seed, condition/family, episode returns, mean/std/min/p10. Deployment output has `size`, `quantization`, and `latency`. Manifests record params, versions, seeds and policy checksums.

## Offline commands

Generate the shared probe file without R2:

```bash
.venv/bin/python scripts/make_probe_states.py --local-root /tmp/eval_v3 --no-push
```

Run a local policy pass with a local winner YAML:

```bash
.venv/bin/python scripts/run_policy_pass.py --cell ssp/none/deep \
  --params-yaml /path/dqn_ssp_none_deep_v4_hparam_search.yaml \
  --local-root /tmp/eval_v3 --no-r2 --workers 4
```

Run robustness and analysis:

```bash
.venv/bin/python scripts/run_robustness.py --cell ssp/none/deep \
  --local-root /tmp/eval_v3 --no-r2 --workers 4
.venv/bin/python scripts/analyze_robustness.py --local-root /tmp/eval_v3 \
  --cells ssp/none/deep,tile_coding/none/deep --out-dir /tmp/eval-analysis
```

Run deployment (latency should use a quiet pod):

```bash
.venv/bin/python scripts/run_deployment.py --cell ssp/none/deep \
  --local-root /tmp/eval_v3 --no-r2 --sections size,quant --quant-episodes 20
.venv/bin/python scripts/run_deployment.py --cell ssp/none/deep \
  --local-root /tmp/eval_v3 --no-r2 --sections latency --latency-calls 2000
```

The offline end-to-end smoke test is:

```bash
.venv/bin/python scripts/smoke_test_eval_v3.py
```

## Pod launch and run order

Dry-run one job before creating anything:

```bash
.venv/bin/python scripts/launch_cell_pod.py --job policy_pass \
  --cells ssp/none/deep --dry-run
```

Actual eval modes accept `--job-args`, for example:

```bash
.venv/bin/python scripts/launch_cell_pod.py --job policy_pass \
  --cells ssp/none/deep --job-args "--workers 4 --n-seeds 20"
.venv/bin/python scripts/launch_cell_pod.py --job robustness \
  --cells ssp/none/deep --job-args "--workers 4 --n-seeds 20"
.venv/bin/python scripts/launch_cell_pod.py --job deployment \
  --cells ssp/none/deep --job-args "--sections size,quant --quant-episodes 20"
```

Run one cell first and time it. Then:

1. Run policy pass for the eight primary robustness cells: SSP/tile × none/value × linear/deep.
2. Check every seed is verified and `train_summary.json` plus `manifest.json` exist.
3. Run robustness and deployment for those cells; run latency separately on a quiet pod; analyze robustness.
4. Run the other cells, including `simple` and discrete as reference cells. Add `tile_coding/value/deep` once its RUN_003 winner lands on R2.
5. Run DQN-settings variants: buffer 5000/50000 and target update 10/100/500 for SSP/tile × none × linear/deep, ten paired seeds. Do this after the baseline policy pass and before any size sweep.

`parallel_jobs.run_jobs` restores JSONL and skips completed keys, so a restart resumes. Failed records are retried; persistent failure is fatal rather than silently partial. The launcher self-deletes a pod only after a successful runner exit; failure leaves it for inspection.

## Protocol limitations and decisions

- Only final policies are saved; no peak/intermediate checkpoint is saved.
- The probe `noise_sigma` is policy-independent, pooled from 50 seeded scripted-controller episodes with 10% random actions: approximately `[0.4341, 0.3061, 0.0272, 0.2106]`. The 0.01/0.05/0.1 observation-noise levels are mild for pole angle; inspect the pilot before adding 0.25/0.5.
- `random_action_0.05` replaces the chosen action with a uniform action, so with two actions the wrong-action rate is approximately 2.5%.
- Delay 1/2 fills initial applied actions with the reset-observation decision.
- The trial verification replay uses the trial's own `steps` cap; production is 500, while tiny smoke runs use 80.
- Winner trial resolution is: explicit `--winner-trial`, YAML winner field, R2 refilter summary, then warning/default 0. Check the selected trial before a real pass.
- `watch_cells.py` only understands search/refilter pods and is not an eval-v3 cost guard. Until an eval-specific watcher exists, monitor eval pods manually and enforce a time cap. Never touch `cerb-*` pods; create pods one at a time.
- Cost estimates from earlier timings are UNVERIFIED: policy pass about 8 pod-hours (~$1), robustness about 3–5 pod-hours, DQN-settings about 6 pod-hours, and policy storage about 2 GB.

## Pre-launch checklist

- Confirm `tile_coding/value/deep` has a winner and final-eval summary on R2.
- Run the offline smoke test and `pytest` from the exact code snapshot.
- Dry-run the launcher and inspect the redacted body.
- Confirm the code tarball contains `rl/policy_io.py`, `rl/perturbations.py`, `scripts/eval_v3_layout.py`, `make_probe_states.py`, and all three runners, and contains no credentials.
- Generate/check `_shared/probe_states.npz` once.
- Confirm `.env` credentials and RunPod capacity; never touch `cerb-*` pods.
- Time one primary cell before launching all cells.
- After each policy pass, check verified count, policy checksums, summary and manifest before scheduling robustness/deployment.
