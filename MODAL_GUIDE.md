# Running Experiments on Modal

All commands assume the `rl` conda environment and are run from the **project root**.

```bash
conda activate rl
```

---

## 1. One-time setup

```bash
modal token new     # opens browser — only needed once per machine
```

Upload existing A2C best params so DQN warm-start can read them:

```bash
modal volume put rl-outputs \
  outputs/hparam_search/best/ \
  outputs/hparam_search/best/
```

For running on **RunPod CPU** instead of Modal (cheaper for Stage-0, see [RQG_COSTS]):

```bash
# On a fresh RunPod CPU pod (Ubuntu 22.04 base, any size):
# 1. Copy the repo + your local .env to the pod
# 2. Run the reproducible setup:
bash scripts/setup_runpod.sh

# The setup script is idempotent — safe to re-run on every pod start.
# It installs uv, creates .venv, installs requirements-stage0.txt + boto3,
# and verifies R2 connectivity.
```

---

## 2. Hparam search

**Preferred: all 9 conditions in parallel (9 containers, wall time = slowest single run)**

```bash
# --detach keeps jobs running even if your terminal closes or machine shuts down
modal run --detach modal_runner.py::dqn_all
modal run --detach modal_runner.py::a2c_all
```

**Single condition (for debugging or re-running one):**

```bash
modal run --detach modal_runner.py::dqn --representation discrete --centering none
modal run --detach modal_runner.py::a2c --representation tile_coding --centering value
```

Defaults: `--n-trials 500 --top-k 10 --reeval-seeds 10` (single seed per trial during
the search itself; the multi-seed step happens post-hoc on the top-k trials only —
see docs/02-experimental-design.md). Override with flags if needed.

Interrupted runs resume automatically — Optuna study persists in the Volume.

---

## 3. Download results

```bash
# Trial CSVs + YAML best params (into local outputs/)
modal volume get rl-outputs outputs/ ./outputs/

# Optuna DB (for analysis / optuna-dashboard)
modal volume get rl-outputs optuna_studies.db ./optuna_studies_modal.db

# Browse what's on the volume
modal volume ls rl-outputs
modal volume ls rl-outputs outputs/hparam_search/best/
```

---

## 4. Persistence to Cloudflare R2

Artifacts (Optuna DB, best_params YAMLs, tuned CSVs, code snapshots) are
periodically pushed to `rl-ssp-centering` R2 bucket during the search. See
`scripts/sync_to_r2_README.md` for bucket layout and subcommand reference.

| What | When |
|---|---|
| `optuna_studies_dqn.db` + `best_params/*.yaml` | Every 10 completed trials (configurable via `--sync-every-n-trials N`) |
| Full phase sync + code_snapshot.tar.gz | Run `scripts/sync_to_r2.py phase <stage>` manually after each Phase completes |
| Resume from R2 | `rclone copyto :r2remote:...optuna_studies_dqn.db ./optuna_studies_dqn.db` (or use awscli/boto3 directly — rclone has signing issues with R2+SigV4) |

**Why not rclone?** rclone 1.74+ uses AWS SDK Go v2 which adds `Amz-Sdk-*` headers that Cloudflare R2's SigV4 validator rejects with 403 Forbidden. boto3 (used internally by `sync_to_r2.py`) and awscli both work fine.

---

## 5. Final ablation runs (local, after downloading best params)

Best params are read from `outputs/hparam_search/best/`.

```bash
# All 9 conditions × both models, 20 seeds each (script default — see
# Colas, Sigaud & Oudeyer arXiv:1806.08295 in docs/06-related-work.md for why)
python scripts/run_best_ablation.py --n-seeds 20

# Single condition
python scripts/run_best_ablation.py --representation discrete --centering none --n-seeds 20
```

After completion, push to R2:
```bash
.venv/bin/python scripts/sync_to_r2.py phase stage0_cartpole
```

---

## Reference

| Representation | `--representation` |
|---|---|
| One-hot / Discrete | `discrete` |
| Spatial Semantic Pointer | `ssp` |
| Tile Coding | `tile_coding` |

| Centering | `--centering` |
|---|---|
| None | `none` |
| Simple (running avg) | `simple` |
| Value-based | `value` |

**CPU per container:** `cpu=4`, `n_jobs=2` (2 Optuna workers per container). Same total cost as cpu=2/n_jobs=1 — halves wall time. Change both together if adjusting.  
**Timeout per container:** 24 h (`timeout=86400`). No documented Modal hard cap.  
**Volume name:** `rl-outputs`  
**Artifacts on volume:** `/vol/outputs/hparam_search/{dqn,a2c}/{rep}/{centering}/`  
**Best params on volume:** `/vol/outputs/hparam_search/best/`  
**Optuna DB on volume:** `/vol/optuna_studies.db`

[RQG_COSTS]: RunPod CPU pricing (Sep 2026): cpu3c 16-vCPU @ $0.48/h, cpu3c 32-vCPU @ $0.96/h. See https://runpod.io/pricing for current rates.
