# Refactoring Plan: ImprovedRLContinuousStateReps

## Context

The codebase has grown organically: one `trial_cartpole.py` core, several copy-paste runner scripts per experiment configuration, NNI-based hyperparameter search, and a two-tier data processing pipeline (legacy hardcoded + newer argparse-based). The goal is to scale to 3×3 AC + 3×3 DQN experiments with hyperparameter tuning, without changing any internal RL/Nengo/SSP implementations. We are also restructuring the project layout so it's easier to navigate and extend.

---

## What I Found

### Current Pain Points

1. **Runner script explosion**: 6+ `run_*.py` scripts that are 95% identical, differing only in hardcoded hyperparams, rep type, and centering mode. Adding DQN = another wave of scripts.

2. **Config is scattered**: Hyperparameters live in:
   - NNI YAML files (`config_cartpole_discrete.yml`) for search spaces
   - Hardcoded defaults in `trial_cartpole.py` `params()`
   - Hardcoded best-found values inside each `run_*.py`
   - Command-line args for `eta` and `condition` in centering scripts

3. **Already using NNI** (not Optuna yet): `exp_cartpole_discrete.py`, `exp_cartpole_placessp.py` use NNI with search spaces in YAML. This is the existing hyperparameter optimization infrastructure.

4. **Data processing is two-tier**:
   - Legacy: `parse_trial_metadata.py`, `merge_episodic_rwds.py` — hardcoded paths, hardcoded column lists
   - Modern: `parse_reward_centering_metadata.py`, `merge_reward_centering_rwds.py` — argparse, flexible directory traversal
   - The modern pattern is the right one; legacy just needs to be unified under it

5. **Plotting is reasonably well organized** — `ablation_plotting.py` is a clean library with CLI in `plot_cartpole_ablation.py`. Minor gaps:
   - Hardcoded color schemes assume exactly 3 centering modes
   - `solve_threshold=495.0` is CartPole-specific (fine for now, but needs parameterization for new tasks)

6. **Unnecessary package nesting**: `network/rlnet/` — `network/` has no `__init__.py` and no purpose beyond wrapping `rlnet/`. All imports spell out `network.rlnet` for no benefit.

7. **`sspspace.py` misplaced**: 34KB SSP math file lives at the `rlnet/` root but conceptually belongs in `representations/` alongside `ssp.py` which wraps it.

8. **`utils.py` mixed concerns**: Math helpers (`softmax`, `sparsity_to_x_intercept`, `next_power_of_2`), visualization (`rend`, `save_gifs`), and analysis (`get_ac_output`) all in one file.

---

## Target Project Layout

```
ImprovedRLContinuousStateReps/
├── rl/                              # was network/rlnet/ — renamed, one less nesting level
│   ├── __init__.py
│   ├── utils.py                     # math only: softmax, sparsity_to_x_intercept, next_power_of_2
│   ├── viz.py                       # rend, save_gifs, get_ac_output (split from utils.py)
│   ├── networks/                    # ActorCritic, ActorCriticLDN, LDN — unchanged
│   ├── representations/
│   │   ├── sspspace.py              # moved from rl/ root — code untouched
│   │   ├── ssp.py
│   │   ├── tilecoding.py
│   │   ├── normal.py
│   │   ├── onehot.py
│   │   ├── onehottransform.py       # kept as-is (verify if used before promoting)
│   │   ├── vsa.py
│   │   └── __init__.py
│   └── rules/                       # TD0, TD0Center, TDn, TDL, TDt, TD0iG — unchanged
│
├── conf/                            # Hydra configs (new)
│   ├── base.yaml
│   ├── experiment/
│   │   ├── actor_critic.yaml
│   │   └── dqn.yaml                 # future
│   ├── representation/
│   │   ├── discrete.yaml
│   │   ├── tile_coding.yaml
│   │   └── ssp.yaml
│   ├── centering/
│   │   ├── none.yaml
│   │   ├── simple.yaml
│   │   └── value.yaml
│   └── hparam_search/
│       ├── optuna_discrete.yaml
│       ├── optuna_tile_coding.yaml
│       └── optuna_ssp.yaml
│
├── scripts/                         # unified experiment runners (new)
│   ├── run_experiment.py            # Hydra-driven, N seeds for a given config composition
│   ├── run_hparam_search.py         # Optuna search, writes best params back to conf/
│   └── run_ablation.py             # sweeps all rep × centering combos
│
├── experiments/                     # was cartpoleExperiments/
│   ├── trial_cartpole.py            # ACTrial — light refactor only (see below)
│   ├── a2c_baseline_cartpole.py
│   └── plotting/
│       ├── ablation_plotting.py     # minor update: generalize color schemes
│       ├── plot_cartpole_ablation.py
│       └── README.md
│
├── data/                            # was cartpoleData/
│   ├── process.py                   # unified entry point: parse metadata + merge rewards
│   ├── merge_reward_centering_rwds.py   # kept, becomes canonical merge script
│   ├── parse_reward_centering_metadata.py  # kept, generalized
│   ├── merge_episodic_rwds.py       # retire (keep for historical data only)
│   ├── parse_trial_metadata.py      # retire (keep for historical data only)
│   └── raw/, processed/, ...        # existing output data — untouched
│
└── REFACTORING_PLAN.md
```

---

## What Changes and How

### 1. `network/rlnet/` → `rl/` (file moves, no code changes)

| Action | Detail |
|--------|--------|
| Rename `network/rlnet/` → `rl/` | Drop the `network/` wrapper entirely |
| Move `rl/sspspace.py` → `rl/representations/sspspace.py` | Code untouched; import in `ssp.py` simplifies from `from ..sspspace` to `from .sspspace` |
| Split `rl/utils.py` → `rl/utils.py` + `rl/viz.py` | Move `rend`, `save_gifs`, `get_ac_output` into `viz.py` |
| Update `rl/__init__.py` | Re-export SSPSpace classes from new location |
| Update 9 experiment files | `import network.rlnet as net` → `import rl as net` |
| Update 1 import in `trial_cartpole.py` | `from network.rlnet.utils import ...` → `from rl.utils import ...` (and add `from rl.viz import ...` if needed) |

**Internal implementations in `networks/`, `representations/`, `rules/` are not touched.**

### 2. Unified Config with Hydra

Each leaf config holds best-known hyperparameters for that condition (post-tuning). Hydra composes them at runtime:

```bash
python scripts/run_experiment.py +experiment=actor_critic +representation=ssp +centering=value
```

### 3. Single Unified Experiment Runner (replaces 6+ `run_*.py`)

`scripts/run_experiment.py` — reads Hydra config, instantiates trial class, runs N seeds, saves to auto-derived output path.

`scripts/run_ablation.py` — sweeps all rep × centering combos; one command to run the full 3×3 grid.

`scripts/run_hparam_search.py` — Optuna search for a given representation, writes best params into `conf/`.

### 4. Optuna + Hydra (replaces NNI)

NNI scripts (`exp_cartpole_*.py`, `config_cartpole_*.yml`) deleted. Search spaces move into `conf/hparam_search/`. Optuna study artifacts saved in SQLite for resumability.

### 5. Data Processing Unification

`data/process.py` becomes the single entry point (wraps the modern `merge_reward_centering_rwds.py` pattern). Legacy scripts kept in place but not extended.

### 6. W&B Integration

Added as live training tracker inside `BaseTrial`. Additive — existing CSV pipeline unchanged. Logs per-episode reward and metadata during training.

### 7. Trial Class Hierarchy

```
BaseTrial (abstract)           — env setup, seed loop, data saving, W&B logging
  ├── ACTrial(BaseTrial)       — existing evaluate() logic moved here, AC-specific params
  └── DQNTrial(BaseTrial)      — future
```

`trial_cartpole.py` refactor is **light**: extract shared scaffolding into `BaseTrial`, leave `evaluate()` logic untouched.

### 8. Output Directory Convention

Auto-derived from Hydra config composition:
```
outputs/{experiment}/{representation}/{centering}/seed_{n}/
```
No manual path management.

---

## Files Changing

| File | Change |
|------|--------|
| `network/rlnet/` (whole dir) | Moved/renamed → `rl/` |
| `network/rlnet/sspspace.py` | Moved → `rl/representations/sspspace.py` (code untouched) |
| `network/rlnet/utils.py` | Split → `rl/utils.py` (math) + `rl/viz.py` (visualization/analysis) |
| `network/rlnet/__init__.py` | Update re-exports for new sspspace location |
| `cartpoleExperiments/run_*.py` (6 files) | Deleted — replaced by `scripts/run_experiment.py` + Hydra configs |
| `cartpoleExperiments/exp_cartpole_*.py` | Deleted — replaced by `scripts/run_hparam_search.py` + Optuna |
| `cartpoleExperiments/config_cartpole_*.yml` | Deleted — replaced by `conf/hparam_search/*.yaml` |
| `cartpoleExperiments/trial_cartpole.py` | Light refactor: extract `BaseTrial`, update import paths |
| `cartpoleData/parse_reward_centering_metadata.py` | Generalize — remove experiment-specific defaults |
| `cartpoleData/merge_reward_centering_rwds.py` | Generalize — becomes canonical merge script |
| `cartpoleExperiments/plotting/ablation_plotting.py` | Minor: generalize color schemes for >3 conditions |

## Files NOT Changing (content)

| File | Reason |
|------|--------|
| `rl/networks/*.py` | Research implementations — untouched |
| `rl/representations/*.py` | Research implementations — untouched |
| `rl/rules/*.py` | Research implementations — untouched |
| `experiments/plotting/plot_cartpole_ablation.py` | CLI already good |
| `data/merge_episodic_rwds.py` | Retired, not extended |
| `data/parse_trial_metadata.py` | Retired, not extended |

---

## Decisions Made

1. **Project layout**: `network/rlnet/` → `rl/`; `cartpoleExperiments/` → `experiments/`; `cartpoleData/` → `data/`
2. **`sspspace.py`**: Moved into `rl/representations/` — conceptually belongs with SSP rep, code untouched
3. **`utils.py`**: Split into `utils.py` (math) and `viz.py` (visualization/analysis)
4. **HPO**: Migrate NNI → Optuna + Hydra (`hydra-optuna-sweeper`). NNI scripts deleted.
5. **W&B**: Add as live training tracker. CSV pipeline stays as-is — W&B is additive.
6. **Output dirs**: Auto-derived from Hydra config composition.
7. **DQN**: `BaseTrial` abstract class → `ACTrial` and `DQNTrial` both inherit it.

---

## Verification

After refactoring:
1. `python scripts/run_experiment.py +experiment=actor_critic +representation=discrete +centering=none seed=0` — same results as current `run_trial_cartpole_repeats.py`
2. `python scripts/run_hparam_search.py +representation=discrete n_trials=5` — Optuna search completes, outputs best params
3. `python data/process.py --data-dir outputs/ac/discrete/none/` — produces metadata CSV and rewards CSV
4. Existing `experiments/plotting/plot_cartpole_ablation.py` on processed data — plots render correctly

---

## Open Items / Future Considerations

- **`onehottransform.py`**: Absent from `representations/__init__.py` `__all__` — verify if used anywhere before deciding to promote or remove
- **Dynamic eta / gamma curriculum**: Future feature — will need a `schedule/` config group in Hydra
- **Additional tasks beyond CartPole**: `conf/task/cartpole.yaml` slot implied by structure; `experiments/` naming is already task-agnostic
- **Parallel seed execution**: Optuna SQLite backend enables multi-process trials; decide concurrency strategy when implementing
