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

Defaults: `--n-trials 100 --n-seeds 10`. Override with flags if needed.

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

## 4. Final ablation runs (local, after downloading best params)

Best params are read from `outputs/hparam_search/best/`.

```bash
# All 9 conditions × both models, 10 seeds each
python scripts/run_best_ablation.py --n-seeds 10

# Single condition
python scripts/run_best_ablation.py --representation discrete --centering none --n-seeds 10
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
