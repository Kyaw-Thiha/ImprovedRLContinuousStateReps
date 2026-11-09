"""
DQN hyperparameter search using Optuna for one (representation, centering,
architecture) cell.

Every cell is tuned fully independently — this project no longer locks a
representation's hyperparameters from a centering=none search and only
sweeps the centering step size for the other two centering modes (see
docs/02-experimental-design.md's "Hyperparameter tuning methodology" and
docs/07-decision-log.md entry 11 for why that staged design was retired).

Methodology follows Antonin Raffin's two-part Optuna guide almost verbatim:
  - Part 1 (concepts): https://araffin.github.io/post/hyperparam-tuning/
  - Part 2 (practice):  https://araffin.github.io/post/optuna/
Concretely:
  1. Single seed per trial during the search (seed=trial.number) — not
     averaged over multiple seeds. Part 2 calls multi-seed-per-trial
     "expensive... and makes it difficult to prune out bad trials early";
     Eimer, Lindauer & Raffin (arXiv:2306.01324) found more than ~3 tuning
     seeds can make the search harder without reliably better results.
  2. A separate deterministic evaluation env (see
     experiments/trial_cartpole_dqn.py's _run_eval_episodes), checked in
     periodically — not the noisy training-reward trace — matching Part 2's
     TrialEvalCallback pattern.
  3. TPESampler + MedianPruner (not HyperbandPruner), with an explicit
     "never prune before 1/3 of budget" warmup, per Part 2's recipe.
  4. 500 trials per cell (rl-baselines3-zoo's own default budget, per Part 2).
  5. Post-hoc: the top --top-k trials are re-evaluated with --reeval-seeds
     fresh seeds each, and the most robust one (by mean, tie-broken toward
     lower std) becomes the cell's winner — the concrete implementation of
     Part 2's "re-evaluate the top trials to find lucky seeds" tip, which
     isn't itself automated in rl-baselines3-zoo's own code.

Results are persisted in optuna_studies.db so searches can be interrupted and
resumed. Best params are written to outputs/hparam_search/best/dqn_{rep}_{centering}.yaml.

Usage:
    python scripts/run_hparam_search_dqn.py --representation discrete --centering none
    python scripts/run_hparam_search_dqn.py --representation ssp --centering value --n-trials 500 --n-jobs 4
    python scripts/run_hparam_search_dqn.py --representation tile_coding --centering simple --resume

Search spaces are documented in conf/hparam_search/optuna_dqn_{rep}.yaml.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from optuna.trial import TrialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole_dqn import DQNTrial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel_jobs as pj  # noqa: E402  (parallel, resumable, R2-backed job runner)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "optuna_studies.db")
DATA_DIR = os.path.join(REPO_ROOT, "outputs", "hparam_search")
BEST_DIR = os.path.join(DATA_DIR, "best")

# NOTE (2026-10-04): the old reeval picked its candidates from trials tied at the
# max search score (CartPole's 500-step cap saturates the single-checkpoint metric)
# and scored them with the online training return. Both were replaced — see the
# "Post-hoc re-evaluation" section below (LAST_K, POOL_*, STAGE1_*) and
# docs/handoffs/02-dqn-v4-greedy-refilter-plan.md.
A2C_BEST_DIR = BEST_DIR  # A2C best params live in the same directory

BASE_PARAMS = dict(
    trials=1000,
    steps=500,
    env="CartPole-v1",
    learnTrials=1000,
    dynamic_epsilon=True,
    state_dis=0.99,
    reward_center_beta=0.001,
    reward_center_eta=1.0,
    reward_center_init=0.0,
    verbose=False,
    gifs=False,
)

# DQN-specific defaults (used in warm-start trial when A2C best params exist)
DQN_DEFAULTS = dict(
    buffer_size=5000,
    batch_size=64,
    target_update_freq=10,
    learning_starts=1000,
    train_freq=4,
)


# H2 arms (d131): name -> normalize_state. "tile_h2" is the tile-coding reference with the same trial budget; it is
# scale-invariant per dimension, so it needs no normalize flag. No H2 arm is warm-started (equal treatment).
H2_SSP = {"ssp_h2": True, "ssp_h2raw": False}
H2_ALL = ("ssp_h2", "ssp_h2raw", "tile_h2")


def _load_a2c_best(representation: str, centering: str) -> dict:
    path = os.path.join(A2C_BEST_DIR, f"{representation}_{centering}.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = yaml.safe_load(f)
    return payload.get("params", {})


def _enqueue_warmstart(study: optuna.Study, representation: str, centering: str) -> bool:
    if representation in H2_ALL:
        return False  # d131: H2 arms start cold and equal; the old A2C-derived start would favour one arm
    a2c = _load_a2c_best(representation, centering)
    if not a2c:
        return False

    trial_params = {}

    # Shared params
    # Cap lr to DQN's max for this representation — A2C often finds lr > 0.1
    # (A2C's own lr ceiling is 1.0, vs DQN's Adam-capped range) which is
    # outside DQN's range. discrete's own DQN search ceiling was raised to 0.2
    # (see sample_representation_params) after both DQN (lr=0.094, close to
    # the old 0.1 ceiling) and A2C's independent search (lr=0.939, 94% to ITS
    # 1.0 ceiling) showed the same boundary-pinning signal for discrete
    # specifically — other reps keep the original 0.1 cap since they didn't
    # show that signal.
    lr_cap = 0.2 if representation == "discrete" else 1e-1
    if "eps" in a2c:
        trial_params["eps"] = float(a2c["eps"])
    if "lr" in a2c:
        trial_params["lr"] = min(float(a2c["lr"]), lr_cap)

    # Representation-specific shared params.
    # For discrete and tile_coding, A2C may choose large representation sizes that are
    # very expensive for DQN (huge replay buffer). Cap to DQN-feasible sizes so the
    # warm-start trial doesn't run for hours before Hyperband can prune it.
    if representation == "discrete" and "n_bins" in a2c:
        trial_params["n_bins"] = min(int(a2c["n_bins"]), 15)  # 15^4=50k vs 23^4=280k obs_dim
    elif representation == "ssp":
        if "length_scale" in a2c:
            trial_params["length_scale"] = float(a2c["length_scale"])
        if "n_rotates" in a2c:
            trial_params["n_rotates"] = int(a2c["n_rotates"])
    elif representation == "tile_coding":
        if "num_tilings" in a2c:
            trial_params["num_tilings"] = min(int(a2c["num_tilings"]), 16)  # cap: 16 vs 32
        if "tiles_per_dim" in a2c:
            tpd = a2c["tiles_per_dim"]
            trial_params["tiles_per_dim"] = int(tpd[0]) if isinstance(tpd, (list, tuple)) else int(tpd)
        if "iht_size" in a2c:
            trial_params["iht_size"] = min(int(a2c["iht_size"]), 32768)  # cap: 32768 vs 65536+

    # DQN-specific defaults (RL Zoo CartPole starting point)
    trial_params.update(DQN_DEFAULTS)

    # Centering params
    if centering == "simple" and "reward_center_beta" in a2c:
        trial_params["reward_center_beta"] = float(a2c["reward_center_beta"])
    elif centering == "value" and "reward_center_eta" in a2c:
        trial_params["reward_center_eta"] = float(a2c["reward_center_eta"])

    study.enqueue_trial(trial_params)
    print(f"Warm-start trial enqueued from A2C best params ({representation}/{centering})")
    return True


def sample_representation_params(trial: optuna.Trial, representation: str) -> dict:
    # Adam-compatible lr range: 1e-4 to 1e-1 (vs 1e-4 to 1.0 for NLMS)
    if representation == "discrete":
        # lr ceiling raised 0.1->0.2 after the first DQN search found the
        # winning lr=0.094 sitting close to the old ceiling, AND A2C's
        # independent search on the same representation found lr=0.939 (94%
        # of the way to ITS much higher 1.0 ceiling) — two separate searches
        # both pointing at the same "wants to go higher" signal for discrete
        # specifically. n_bins is left alone: DQN's own search found an
        # interior optimum (19, not the max 23), unlike A2C which pinned at
        # 23 — DQN's linear/combinatorial-sparsity tradeoff seems to bite
        # sooner, so widening n_bins further is more likely to hurt than help.
        return dict(
            rep_="Discrete",
            n_bins=trial.suggest_categorical("n_bins", [7, 11, 15, 19, 23]),
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 0.2, log=True),
        )
    elif representation == "ssp":
        return dict(
            rep_="PlaceSSP",
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 0.2, log=True),
            length_scale=trial.suggest_float("length_scale", 0.1, 1.0, log=True),
            # n_scales fixed back to 5 (was briefly a free [3,8] dimension) —
            # ssp_dim = 2*(n_rotates*n_scales)*(domain_dim+1) + 1 (verified in
            # sspspace's HexagonalSSPSpace source), i.e. n_rotates and n_scales
            # are perfectly multiplicative/interchangeable for dimensionality.
            # Searching both let TPE draw e.g. n_rotates=20,n_scales=7 (dim
            # 1401) in the SAME 80-trial budget as n_rotates=4,n_scales=3 (dim
            # 121) — an 11x spread in representation size. Large-dim draws need
            # more gradient steps to warm up, but HyperbandPruner's first rung
            # (150 episodes) judges everything on the same yardstick, so those
            # draws got pruned immediately regardless of true potential
            # (confirmed directly: a trial with n_rotates=20 was killed at
            # episode 450 with value 9.4). Fixing n_scales removes the
            # redundant search dimension (no representational-capacity loss —
            # n_rotates alone already spans the same dimensionality range) and
            # collapses the size-variance spread back to what n_rotates alone
            # produces, which is what let this same search find 486.69 with
            # n_rotates=19 before. lr ceiling also raised 0.1->0.2 (matching
            # discrete's earlier fix): this run's own winner independently hit
            # lr=0.09, 90% of the old 0.1 ceiling — same boundary-pinning
            # signal, never actioned for ssp specifically until now.
            n_rotates=trial.suggest_int("n_rotates", 4, 20),
            n_scales=5,
        )
    elif representation in H2_SSP:
        # H2 (d131, REPORT_006): the CartPole length-scale test. Identical to the "ssp" space except (a) the length
        # scale is searched over [0.01, 1] (RUN_005: the cliff needs ~0.042 in raw units, 2.4x below the old floor of
        # 0.1) and (b) `normalize_state` is fixed per arm: True divides each dimension by DQNTrial.state_scale =
        # [4.8, 10, 0.418, 10] before encoding; False is the unnormalized wide-range control.
        return dict(
            rep_="PlaceSSP",
            normalize_state=H2_SSP[representation],
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 0.2, log=True),
            length_scale=trial.suggest_float("length_scale", 0.01, 1.0, log=True),
            n_rotates=trial.suggest_int("n_rotates", 4, 20),
            n_scales=5,
        )
    elif representation in ("tile_coding", "tile_h2"):
        tiles_per_dim = trial.suggest_categorical("tiles_per_dim", [10, 12, 16])
        return dict(
            rep_="TileCoding",
            num_tilings=trial.suggest_categorical("num_tilings", [16, 32]),
            tiles_per_dim=(tiles_per_dim,) * 4,
            iht_size=trial.suggest_categorical("iht_size", [32768, 65536, 131072]),
            tile_state_indices=(0, 1, 2, 3),
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")


def sample_dqn_params(trial: optuna.Trial) -> dict:
    # Per Sep 22 discussion: batch_size is hardcoded to 64 (DQN_DEFAULTS below).
    # Marginal expected gain from tuning batch_size — standard CartPole
    # practice and 64 sits in the sweet spot for our linear DQN. Removes one
    # search dimension per trial.
    return dict(
        buffer_size=trial.suggest_categorical("buffer_size", [1000, 5000, 10000]),
        # Per smoke-test finding (2026-09-22): target_update_freq >= 500 caused
        # Q to systematically underestimate during the bootstrap window (target
        # network stayed at random-init for ~1500 env steps), trapping Normal /
        # PlaceSSP / Discrete reps in a degenerate policy. Cap at 100.
        target_update_freq=trial.suggest_categorical("target_update_freq", [10, 50, 100]),
        # Per smoke-test finding: learning_starts >= 500 wasted the first 1000
        # random steps; learning_starts=200 worked. Lower the floor to 100.
        learning_starts=trial.suggest_categorical("learning_starts", [100, 500, 1000]),
    )


def sample_centering_params(trial: optuna.Trial, centering: str) -> dict:
    params = dict(reward_center_mode=centering)
    if centering == "simple":
        params["reward_center_beta"] = trial.suggest_float("reward_center_beta", 1e-5, 1e-1, log=True)
    elif centering == "value":
        params["reward_center_eta"] = trial.suggest_float("reward_center_eta", 1e-4, 1.0, log=True)
    return params


def _best_filename(representation: str, centering: str, architecture: str, variant: str = "") -> str:
    # "linear" keeps the original filename (backward-compatible with results
    # already on disk/R2 and with run_best_ablation.py); "deep" gets its own
    # suffix so it never collides with the linear results. `variant` (e.g. "_v2")
    # is appended last so a re-filtered result never overwrites the original file.
    suffix = "" if architecture == "linear" else f"_{architecture}"
    return f"dqn_{representation}_{centering}{suffix}{variant}.yaml"


def _cell_data_dir(representation: str, centering: str, architecture: str) -> str:
    """Relative-to-DATA_DIR path for one cell's per-trial CSV artifacts
    (rewards.csv/states.csv/values.csv, written per trial by
    trial_cartpole_dqn.py's evaluate()). Architecture-suffixed like
    _best_filename so linear/deep runs for the same (representation,
    centering) never share a directory — otherwise mirroring this path to an
    R2 prefix for CSV backup (see sync_to_r2.py's cmd_trial --csv-dir) would
    let a linear and a deep pod's trial folders collide there, even though
    they never collide locally (each pod only ever runs one architecture).
    """
    suffix = "" if architecture == "linear" else f"_{architecture}"
    return os.path.join("dqn", representation, f"{centering}{suffix}")


_TRIAL_USER_ATTRS = (
    "eval_history", "terminal_reward", "total_time", "build_time", "train_time", "eval_time",
    "env_steps", "n_updates", "n_params", "n_active", "dimensionality",
)


def _jsonable(v):
    """numpy scalars / NaN -> plain JSON-safe values for Optuna user_attrs."""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def make_objective(representation: str, centering: str, architecture: str,
                    n_evaluations: int, n_eval_episodes: int):
    """Every (representation, centering) cell samples its OWN full set of
    representation + DQN + centering hyperparameters — no locking from a
    separate centering=none search. See module docstring.
    """
    data_dir = os.path.join(DATA_DIR, _cell_data_dir(representation, centering, architecture))
    os.makedirs(data_dir, exist_ok=True)

    eval_freq = max(1, BASE_PARAMS["trials"] // n_evaluations)

    def objective(trial: optuna.Trial) -> float:
        dqn = DQNTrial()
        params = dict(BASE_PARAMS)
        params.update(sample_representation_params(trial, representation))
        params.update(DQN_DEFAULTS)
        params["architecture"] = architecture
        params.update(sample_centering_params(trial, centering))
        params["eval_freq"] = eval_freq
        params["n_eval_episodes"] = n_eval_episodes

        eval_idx = 0

        def _pruning_callback(mean_eval_return):
            nonlocal eval_idx
            eval_idx += 1
            trial.report(mean_eval_return, eval_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        dqn._pruning_callback = _pruning_callback
        try:
            # seed=trial.number (not averaged over multiple seeds): matches
            # Part 2's own recipe ("we do not seed the trial" — one training
            # run per trial). trial.number still gives reproducibility
            # without the search-time seed-averaging Part 2 warns against.
            metadata = dqn.run(
                seed=trial.number,
                data_dir=data_dir,
                pre_comment=f"optuna trial={trial.number} rep={representation} "
                            f"centering={centering} arch={architecture}",
                **params,
            )
        finally:
            dqn._pruning_callback = None

        # Extra per-trial metadata for greedy-checkpoint re-ranking and cost comparison
        # (added 2026-10-04). Purely additive: the returned objective value is unchanged,
        # so trials resumed into an existing v4 study stay comparable with the old ones.
        for _k in _TRIAL_USER_ATTRS:
            if _k in metadata:
                trial.set_user_attr(_k, _jsonable(metadata[_k]))

        return float(metadata["terminal_eval_reward"])

    return objective


# ---------------------------------------------------------------------------
# Post-hoc re-evaluation — redesigned 2026-10-04.
# See docs/handoffs/02-dqn-v4-greedy-refilter-plan.md for the full rationale.
#
# What changed vs. the original reeval:
#   * It now scores each reeval run with the SAME greedy-eval metric the
#     search used (eval_freq/n_eval_episodes are set), instead of the online
#     epsilon-greedy training return it used to read.
#   * Candidates are ranked by the mean of the last LAST_K greedy checkpoints
#     (read from the Optuna study's intermediate values: free, no R2 scraping)
#     over ALL completed trials, not only those tied at the max score.
#   * Staged filter: STAGE1_SEEDS seeds for every pool candidate, keep the
#     better STAGE1_KEEP_FRAC, then top up survivors to the full seed count.
# The search objective itself (last greedy checkpoint) is unchanged.
# ---------------------------------------------------------------------------
LAST_K = 5                  # checkpoints averaged for ranking AND for each reeval seed's score
STABLE_THRESHOLD = 475.0    # "stable-from" tie-break: earliest checkpoint after which all >= this
POOL_QUANTILE = 0.10        # candidate pool = top 10% of ranked completed trials...
POOL_MIN = 20               # ...but never fewer than this...
POOL_MAX = 20               # ...or more than this. 40 -> 20 on 2026-10-05: the pilot found config sd ~30
                            # vs seed sd ~90 among top candidates, so a bigger pool buys little.
STAGE1_SEEDS = 3
STAGE1_KEEP_FRAC = 1.0      # 0.5 -> 1.0 on 2026-10-05: a 3-seed cut (SE ~52) is a coin flip here, so the
                            # default is now flat pool x reeval-seeds (stage 1 just runs the first 3 seeds).
REEVAL_SEED_OFFSET = 1000   # offset away from search-phase seeds (0..n_trials-1)
REEVAL_ATTR_PREFIX = "reeval_v2"   # study user_attr namespace; old "reeval_*" attrs are left untouched
REEVAL_RESULTS_FILE = "reeval_v2_results.json"
REEVAL_RUNS_FILE = "reeval_v2_runs.jsonl"        # one line per finished (candidate, seed) run; resumable
REEVAL_VARIANT = "_v2"      # suffix on the best_params yaml so the original files are never overwritten
# Final eval of each cell's filter winner on FRESH seeds (never used in search 0..499 or reeval 1000+).
FINAL_SEED_OFFSET = 2000
FINAL_SEEDS = 20            # Colas et al.: >= 20 seeds for a reliable std estimate
FINAL_EVAL_EPISODES = 100   # greedy episodes of the FINAL policy after training (secondary metric)
FINAL_RUNS_FILE = "final_eval_runs.jsonl"
FINAL_SUMMARY_FILE = "final_eval_summary.json"
SOLVED_THRESHOLD = 475.0    # CartPole-v1 "solved" mean return
R2_REFILTER_PREFIX = "stage0_cartpole/refilter_v2"   # NEW prefix; nothing existing is overwritten


def last_k_mean(history, k: int = LAST_K) -> float:
    h = [float(x) for x in history]
    return float(np.mean(h[-k:]))


def stable_from_index(history, threshold: float = STABLE_THRESHOLD) -> int:
    """Earliest checkpoint index i such that every checkpoint j >= i is >= threshold.
    Returns len(history) when even the last checkpoint is below threshold. Smaller is better."""
    idx = len(history)
    for i in range(len(history) - 1, -1, -1):
        if history[i] >= threshold:
            idx = i
        else:
            break
    return idx


def rank_trials_by_checkpoints(study: optuna.Study, n_evaluations: int, last_k: int = LAST_K,
                               seed: int = 0) -> list:
    """All COMPLETE trials that have exactly n_evaluations greedy checkpoints, best first.

    Sort keys: last-K mean (desc), last-8 mean (desc), stable-from index (asc), then a
    seeded random number, so the order is fully deterministic for a given set of trials.
    Pruned trials (fewer checkpoints) are excluded.
    """
    rng = random.Random(seed)
    rows = []
    for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        hist = [t.intermediate_values[s] for s in sorted(t.intermediate_values)]
        if len(hist) != n_evaluations:
            continue
        rows.append({
            "trial": t,
            "number": t.number,
            "history": hist,
            "last1": float(hist[-1]),
            "last_k": last_k_mean(hist, last_k),
            "last8": last_k_mean(hist, 8),
            "stable_from": stable_from_index(hist),
            "tiebreak": rng.random(),
        })
    rows.sort(key=lambda r: (-r["last_k"], -r["last8"], r["stable_from"], r["tiebreak"]))
    return rows


def select_reeval_pool(study: optuna.Study, n_evaluations: int, last_k: int = LAST_K,
                       quantile: float = POOL_QUANTILE, pool_min: int = POOL_MIN,
                       pool_max: int = POOL_MAX) -> list:
    """Top `quantile` of the ranked completed trials, clamped to [pool_min, pool_max]."""
    ranked = rank_trials_by_checkpoints(study, n_evaluations, last_k)
    if not ranked:
        raise RuntimeError(
            f"No COMPLETE trials with {n_evaluations} intermediate values in study "
            f"{study.study_name!r} — cannot build a reeval pool.")
    n = len(ranked)
    pool_size = min(n, pool_max, max(pool_min, math.ceil(quantile * n)))
    pool = ranked[:pool_size]
    cut = pool[-1]
    key = lambda r: (r["last_k"], r["last8"], r["stable_from"])
    n_tied = sum(1 for r in ranked if key(r) == key(cut))
    n_tied_taken = sum(1 for r in pool if key(r) == key(cut))
    print(f"  reeval pool: {pool_size} of {n} ranked completed trials "
          f"(quantile={quantile}, min={pool_min}, max={pool_max}); cutoff last-{last_k} mean="
          f"{cut['last_k']:.1f}; {n_tied_taken} of {n_tied} trials tied with the cutoff on "
          f"every sort key were taken (seeded random among those).")
    return pool


def _run_reeval_seed(params: dict, data_dir: str, seed_idx: int, tag: str,
                     eval_freq: int, n_eval_episodes: int, last_k: int,
                     seed_offset: int = REEVAL_SEED_OFFSET, final_eval_episodes: int = 0) -> dict:
    dqn = DQNTrial()
    extra = {"final_eval_episodes": int(final_eval_episodes)} if final_eval_episodes else {}
    metadata = dqn.run(
        seed=seed_offset + seed_idx,
        data_dir=data_dir,
        pre_comment=f"{tag} seed={seed_idx}",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        **{"init_seed": -1, **params, **extra},   # default: initial weights follow the run seed
    )
    hist = [float(x) for x in metadata["eval_history"]]
    rec = {
        "seed": seed_offset + seed_idx,
        "score": last_k_mean(hist, last_k),
        "last1": hist[-1],
        "eval_history": hist,
        "terminal_reward": float(metadata["terminal_reward"]),
        "total_time": float(metadata["total_time"]),
        "train_time": float(metadata["train_time"]),
        "eval_time": float(metadata["eval_time"]),
    }
    for k in ("build_time", "env_steps", "n_updates", "n_params", "n_active", "dimensionality",
              "final_eval_reward"):
        if k in metadata:
            rec[k] = _jsonable(metadata[k])
    return rec


def reevaluate_top_trials(study: optuna.Study, representation: str, centering: str,
                           architecture: str, n_reeval_seeds: int, n_evaluations: int,
                           n_eval_episodes: int, last_k: int = LAST_K,
                           quantile: float = POOL_QUANTILE, pool_min: int = POOL_MIN,
                           pool_max: int = POOL_MAX, stage1_seeds: int = STAGE1_SEEDS,
                           stage1_keep: float = STAGE1_KEEP_FRAC, workers: int = 1,
                           r2_prefix: str = None) -> tuple[dict, float, float, int]:
    """Post-hoc validation (Raffin Part 2: "re-evaluate the top trials multiple times
    to find out which ones were lucky seeds"), staged, parallel and resumable.

    Each seed's score is the mean of its last `last_k` GREEDY eval checkpoints (same
    evaluation settings as the search). Stage 1 gives every pool candidate
    `stage1_seeds` seeds; the better `stage1_keep` fraction survive and are topped up
    to `n_reeval_seeds` (with the default keep=1.0 this is simply pool x n_reeval_seeds).
    Winner = best mean (rounded to 1 point), ties broken toward lower std (np.std,
    unchanged from the original reeval).

    Every (candidate, seed) run is one job for `workers` processes; each finished run is
    appended to reeval_v2_runs.jsonl (and pushed to R2 under `r2_prefix` when given), and
    finished runs are skipped on restart.

    Returns (winning_params, mean_score, std_score, winning_search_trial_number).
    """
    t_start = pj.time.time()
    pool = select_reeval_pool(study, n_evaluations, last_k, quantile, pool_min, pool_max)
    data_dir = os.path.join(DATA_DIR, _cell_data_dir(representation, centering, architecture), "reeval_v2")
    os.makedirs(data_dir, exist_ok=True)
    runs_path = os.path.join(data_dir, REEVAL_RUNS_FILE)
    results_path = os.path.join(data_dir, REEVAL_RESULTS_FILE)
    r2_runs = f"{r2_prefix}/{REEVAL_RUNS_FILE}" if r2_prefix else None
    r2_results = f"{r2_prefix}/{REEVAL_RESULTS_FILE}" if r2_prefix else None
    eval_freq = max(1, BASE_PARAMS["trials"] // n_evaluations)

    cands = []
    for rank, row in enumerate(pool):
        t = row["trial"]
        params = dict(BASE_PARAMS)
        params.update(DQN_DEFAULTS)
        params.update(trial_params_from_search_params(representation, centering, t.params, architecture))
        cands.append({"rank": rank, "row": row, "trial": t, "params": params, "by_seed": {}})
    by_trial = {c["trial"].number: c for c in cands}

    def _runs(c):
        return [c["by_seed"][s] for s in sorted(c["by_seed"])]

    def _stats(c, first=None):
        scores = [r["score"] for r in _runs(c)[:first]]
        return float(np.mean(scores)), float(np.std(scores))

    def _jobs(c, seed_indices):
        t = c["trial"]
        tag = (f"reeval_v2 rank={c['rank']} search_trial={t.number} rep={representation} "
               f"centering={centering} arch={architecture}")
        return [dict(kind="reeval", trial=t.number, rank=c["rank"], seed_idx=s, params=c["params"],
                     eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, last_k=last_k,
                     seed_offset=REEVAL_SEED_OFFSET, final_eval_episodes=0, tag=tag, data_dir=data_dir)
                for s in seed_indices]

    def _run_stage(cs, seed_indices_fn):
        job_list = [j for c in cs for j in _jobs(c, seed_indices_fn(c))]
        if not job_list:
            return
        recs = pj.run_jobs(job_list, pj.eval_worker, pj.job_key, runs_path, workers,
                           r2_key=r2_runs, label="reeval")
        for (_kind, trial, s), rec in recs.items():
            by_trial[trial]["by_seed"][s] = rec

    def _persist(stage: str, survivors):
        surv_ids = {c["rank"] for c in survivors}
        out = {
            "stage": stage, "last_k": last_k, "n_reeval_seeds": n_reeval_seeds,
            "stage1_seeds": stage1_seeds, "stage1_keep": stage1_keep,
            "n_eval_episodes": n_eval_episodes, "eval_freq": eval_freq,
            "seed_offset": REEVAL_SEED_OFFSET, "wall_min": (pj.time.time() - t_start) / 60,
            "pod": pj.pod_info(workers), "candidates": [],
        }
        for c in cands:
            if not c["by_seed"]:
                continue
            m, s = _stats(c)
            out["candidates"].append({
                "rank": c["rank"], "search_trial": c["trial"].number,
                "search_score": float(c["trial"].value),
                "search_last_k": c["row"]["last_k"], "search_last1": c["row"]["last1"],
                "search_stable_from": c["row"]["stable_from"],
                "n_seeds_run": len(c["by_seed"]), "mean": m, "std": s,
                "survived_stage1": c["rank"] in surv_ids,
                "runs": _runs(c), "params": c["trial"].params,
            })
        with open(results_path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=False)
        if r2_results:
            pj.push_file(r2_results, results_path)

    stage1_n = max(1, min(stage1_seeds, n_reeval_seeds))
    print(f"  stage 1: {stage1_n} seeds x {len(cands)} candidates ({workers} workers)", flush=True)
    _run_stage(cands, lambda c: range(stage1_n))
    for c in cands:
        m, s = _stats(c, stage1_n)
        print(f"  stage1 rank={c['rank']} (search trial #{c['trial'].number}, search last-{last_k} "
              f"{c['row']['last_k']:.1f}): mean={m:.2f} std={s:.2f} over {stage1_n} seeds")
    _persist("stage1", cands)

    keep_n = max(1, math.ceil(len(cands) * stage1_keep))
    survivors = sorted(cands, key=lambda c: (-_stats(c, stage1_n)[0], _stats(c, stage1_n)[1]))[:keep_n]
    print(f"  stage 2: keeping {len(survivors)} of {len(cands)}; topping up to {n_reeval_seeds} seeds",
          flush=True)
    _run_stage(survivors, lambda c: range(stage1_n, n_reeval_seeds))
    for c in survivors:
        m, s = _stats(c)
        # Same line format the original reeval printed (RUN_002's log-recovery procedure greps it).
        print(f"  reeval rank={c['rank']} (search trial #{c['trial'].number}, search score "
              f"{c['trial'].value:.2f}): mean={m:.2f} std={s:.2f} over {len(c['by_seed'])} seeds")
    _persist("final", survivors)

    # Best by mean; ties within 1 point broken toward lower std (unchanged rule).
    best = max(survivors, key=lambda c: (round(_stats(c)[0], 0), -_stats(c)[1]))
    m, s = _stats(best)
    return best["trial"].params, m, s, best["trial"].number


def final_evaluate(study_name: str, representation: str, centering: str, architecture: str,
                   winner_trial: int, winner_params: dict, reeval_mean: float, reeval_std: float,
                   n_seeds: int, n_evaluations: int, n_eval_episodes: int, last_k: int,
                   final_eval_episodes: int, workers: int, seed_offset: int = FINAL_SEED_OFFSET,
                   r2_prefix: str = None) -> dict:
    """Final evaluation of the filter winner on `n_seeds` FRESH seeds (greedy, same eval settings
    as the search). Primary per-seed score = mean of the last `last_k` greedy checkpoints; also
    records last-1, a longer greedy eval of the final policy, the online return, timings and size.
    Reported as mean +/- np.std across seeds (unchanged std convention), with per-seed values and the
    share of seeds that reached SOLVED_THRESHOLD alongside. Resumable like the filter.
    """
    t_start = pj.time.time()
    data_dir = os.path.join(DATA_DIR, _cell_data_dir(representation, centering, architecture), "final_eval_v2")
    os.makedirs(data_dir, exist_ok=True)
    runs_path = os.path.join(data_dir, FINAL_RUNS_FILE)
    summary_path = os.path.join(data_dir, FINAL_SUMMARY_FILE)
    eval_freq = max(1, BASE_PARAMS["trials"] // n_evaluations)

    params = dict(BASE_PARAMS)
    params.update(DQN_DEFAULTS)
    params.update(trial_params_from_search_params(representation, centering, winner_params, architecture))
    tag = (f"final_eval_v2 search_trial={winner_trial} rep={representation} "
           f"centering={centering} arch={architecture}")
    jobs = [dict(kind="final", trial=int(winner_trial), rank=0, seed_idx=s, params=params,
                 eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, last_k=last_k,
                 seed_offset=seed_offset, final_eval_episodes=final_eval_episodes, tag=tag,
                 data_dir=data_dir) for s in range(n_seeds)]
    r2_runs = f"{r2_prefix}/{FINAL_RUNS_FILE}" if r2_prefix else None
    recs = pj.run_jobs(jobs, pj.eval_worker, pj.job_key, runs_path, workers, r2_key=r2_runs,
                       label="final-eval")
    runs = [recs[("final", int(winner_trial), s)] for s in range(n_seeds)]

    def _col(k):
        return [float(r[k]) for r in runs if r.get(k) is not None]

    scores = _col("score")
    fin = _col("final_eval_reward")
    summary = {
        "study": study_name, "representation": representation, "centering": centering,
        "architecture": architecture, "winner_search_trial": int(winner_trial),
        "winner_search_params": winner_params,
        "filter_winner": {"mean": reeval_mean, "std": reeval_std},
        "final_eval": {
            "n_seeds": n_seeds, "seed_offset": seed_offset, "last_k": last_k,
            "n_eval_episodes": n_eval_episodes, "final_eval_episodes": final_eval_episodes,
            "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)), "score_max": float(np.max(scores)),
            "per_seed_scores": scores,
            "share_solved_lastk": float(np.mean([s >= SOLVED_THRESHOLD for s in scores])),
            "last1_mean": float(np.mean(_col("last1"))), "last1_std": float(np.std(_col("last1"))),
            "per_seed_last1": _col("last1"),
            "final_policy_eval_mean": float(np.mean(fin)) if fin else None,
            "final_policy_eval_std": float(np.std(fin)) if fin else None,
            "per_seed_final_policy_eval": fin,
            "share_solved_final_policy": float(np.mean([x >= SOLVED_THRESHOLD for x in fin])) if fin else None,
            "online_return_mean": float(np.mean(_col("terminal_reward"))),
        },
        "cost": {
            "train_time_mean_s": float(np.mean(_col("train_time"))),
            "eval_time_mean_s": float(np.mean(_col("eval_time"))),
            "total_time_mean_s": float(np.mean(_col("total_time"))),
            "note": "run times are measured with `workers` runs sharing one pod, i.e. contended",
            "dimensionality": runs[0].get("dimensionality"), "n_params": runs[0].get("n_params"),
            "n_active": runs[0].get("n_active"), "env_steps_mean": float(np.mean(_col("env_steps"))) if _col("env_steps") else None,
            "n_updates_mean": float(np.mean(_col("n_updates"))) if _col("n_updates") else None,
        },
        "pod": pj.pod_info(workers), "wall_min": (pj.time.time() - t_start) / 60,
        "runs": runs,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if r2_prefix:
        pj.push_file(f"{r2_prefix}/{FINAL_SUMMARY_FILE}", summary_path)
    return summary


def trial_params_from_search_params(representation: str, centering: str, search_params: dict,
                                     architecture: str = "linear") -> dict:
    params = dict(search_params)
    params["architecture"] = architecture

    if representation == "discrete":
        params["rep_"] = "Discrete"
    elif representation == "ssp":
        params["rep_"] = "PlaceSSP"
    elif representation in H2_SSP:
        params["rep_"] = "PlaceSSP"
        params["normalize_state"] = H2_SSP[representation]
        params["n_scales"] = 5
    elif representation in ("tile_coding", "tile_h2"):
        params["rep_"] = "TileCoding"
        if isinstance(params.get("tiles_per_dim"), int):
            params["tiles_per_dim"] = [params["tiles_per_dim"]] * 4
        params["tile_state_indices"] = [0, 1, 2, 3]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    params["reward_center_mode"] = centering
    return params


def save_best_params(representation: str, centering: str, best_value: float, best_params: dict,
                      architecture: str = "linear", best_std: float = None,
                      variant: str = "") -> None:
    os.makedirs(BEST_DIR, exist_ok=True)
    best_path = os.path.join(BEST_DIR, _best_filename(representation, centering, architecture, variant))

    payload = {
        "model_type": "dqn",
        "representation": representation,
        "centering": centering,
        "architecture": architecture,
        "best_value": float(best_value),
        "selection": "greedy_lastk_staged_reeval_v2" if variant else "legacy_online_reeval",
        # Std across the winning candidate's n_reeval_seeds seeds — without
        # this, a report of best_value alone can't distinguish a robust
        # config from a lucky-but-noisy one, which is exactly what the whole
        # reeval step exists to catch. None for the rare case this is called
        # outside reevaluate_top_trials (e.g. a hypothetical future caller
        # that only has a single-seed value).
        "best_std": float(best_std) if best_std is not None else None,
        "params": trial_params_from_search_params(representation, centering, best_params, architecture),
    }

    with open(best_path, "w") as f:
        f.write("# Best DQN params from Optuna search. Intended for scripts/run_best_ablation.py\n")
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    print(f"Best params written to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna DQN hyperparameter search")
    parser.add_argument("--representation", choices=["discrete", "ssp", "tile_coding", *H2_ALL], required=True)
    parser.add_argument("--centering", choices=["none", "simple", "value"], required=True)
    parser.add_argument("--architecture", choices=["linear", "deep"], default="linear",
                        help="'linear' (Q=w.phi(s)) or 'deep' (2 hidden layers of 64 units, "
                             "ReLU) — its own independent search, not a sampled dimension.")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Optuna trials for this cell (default: 500, rl-baselines3-zoo's own default budget)")
    parser.add_argument("--n-evaluations", type=int, default=15,
                        help="Deterministic-eval checkpoints spread across each trial's episode budget")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Episodes per eval checkpoint (mean return is what's reported to Optuna)")
    parser.add_argument("--n-startup-trials", type=int, default=10,
                        help="Random trials before TPE starts modeling correlations / before pruning is allowed")
    parser.add_argument("--top-k", type=int, default=10,
                        help="DEPRECATED, ignored (kept so existing launch commands don't break). "
                             "The reeval pool is now set by --pool-quantile/--pool-min/--pool-max.")
    parser.add_argument("--reeval-seeds", type=int, default=10,
                        help="Total seeds per surviving candidate in the post-hoc re-evaluation step")
    parser.add_argument("--last-k", type=int, default=LAST_K,
                        help="Greedy checkpoints averaged for ranking and for each reeval seed's score")
    parser.add_argument("--pool-quantile", type=float, default=POOL_QUANTILE,
                        help="Reeval pool = top this fraction of ranked completed trials")
    parser.add_argument("--pool-min", type=int, default=POOL_MIN, help="Minimum reeval pool size")
    parser.add_argument("--pool-max", type=int, default=POOL_MAX, help="Maximum reeval pool size")
    parser.add_argument("--stage1-seeds", type=int, default=STAGE1_SEEDS,
                        help="Seeds every pool candidate gets before the first cut")
    parser.add_argument("--stage1-keep", type=float, default=STAGE1_KEEP_FRAC,
                        help="Fraction of the pool that survives the first cut")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trial workers")
    parser.add_argument("--resume", action="store_true", help="Print existing progress and resume")
    parser.add_argument("--db-path", default=None,
                        help="Path to Optuna SQLite DB (default: optuna_studies.db). "
                             "Use a separate DB for DQN to avoid schema conflicts with A2C's older DB.")
    parser.add_argument("--sync-every-n-trials", type=int, default=10,
                        help="Push Optuna DB + best_params to R2 every N completed trials "
                             "(0 disables). See scripts/sync_to_r2_README.md.")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable R2 syncing entirely (overrides --sync-every-n-trials).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker PROCESSES for the re-filter and final-eval runs (use the pod's "
                             "vCPU count). The search itself is controlled by --n-jobs and is unchanged.")
    parser.add_argument("--final-eval", action="store_true",
                        help="After the re-filter, evaluate the winner on --final-seeds fresh seeds (greedy).")
    parser.add_argument("--final-seeds", type=int, default=FINAL_SEEDS)
    parser.add_argument("--final-seed-offset", type=int, default=FINAL_SEED_OFFSET)
    parser.add_argument("--final-eval-episodes", type=int, default=FINAL_EVAL_EPISODES,
                        help="Greedy episodes of the final policy per final-eval seed (0 disables)")
    parser.add_argument("--idle-after", action="store_true",
                        help="Pod mode: never exit (RunPod restarts the entrypoint on any exit); idle when "
                             "finished or on error instead.")
    parser.add_argument("--self-delete", action="store_true",
                        help="With --idle-after: when everything is safely on R2, delete this pod via the "
                             "RunPod API (needs RUNPOD_POD_ID and RUNPOD_API_KEY in the pod env).")
    args = parser.parse_args()

    if args.db_path is not None:
        global DB_PATH
        DB_PATH = args.db_path

    os.makedirs(DATA_DIR, exist_ok=True)

    arch_suffix = "" if args.architecture == "linear" else f"_{args.architecture}"
    # STUDY_GEN isolates this Optuna study (and its R2 per-study key) from any
    # pre-existing history under an older generation's name. Bump whenever a
    # change makes old trial values incomparable to new ones, or makes an old
    # study's sampled params inconsistent with the current search space.
    # v3->v4 (this bump): full methodology replacement, not a search-space
    # tweak — old v3 trial values are the mean of a multi-seed rolling
    # TRAINING-reward trace under HyperbandPruner; v4 trial values are a
    # single-seed deterministic EVAL-checkpoint return under MedianPruner, and
    # centering != "none" cells now sample their own representation/DQN
    # hyperparameters instead of locking them from a centering=none search
    # (see docs/02-experimental-design.md). None of this is comparable to v3.
    STUDY_GEN = "v4"
    study_name = f"dqn_{args.representation}_{args.centering}{arch_suffix}_{STUDY_GEN}_hparam_search"
    storage = f"sqlite:///{DB_PATH}"

    # TPESampler + MedianPruner, following araffin.github.io/post/optuna/'s
    # recipe: no trial can be pruned before 1/3 of its evaluation checkpoints
    # (n_warmup_steps), and both sampler/pruner share n_startup_trials so
    # early trials are pure random search before TPE/pruning kick in.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=args.n_startup_trials,
                                            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials,
                                            n_warmup_steps=args.n_evaluations // 3),
        load_if_exists=True,
    )

    n_existing = len(study.trials)

    # Record the ABSOLUTE trial-count target once, persisted on the study
    # itself (survives restarts since it's stored in the RDB backend). Without
    # this, a pod/container restart after a successful, completed run (Runpod
    # pods restart their entrypoint on any process exit, success or failure —
    # there's no "run once and stop" primitive) would see n_existing trials
    # already recorded and, since main() always requested n_trials MORE with
    # no memory of prior invocations, add another full n_trials on top —
    # indefinitely, every restart. Observed in production: tile_coding and ssp
    # both overshot their 80-trial target by ~3x this way before we noticed
    # and manually terminated the pods. Setting the target once and computing
    # `remaining` against it makes every invocation past the first a no-op
    # once the target is reached.
    TARGET_ATTR = "target_n_trials"
    if TARGET_ATTR not in study.user_attrs:
        study.set_user_attr(TARGET_ATTR, n_existing + args.n_trials)
    target_n_trials = study.user_attrs[TARGET_ATTR]

    # Stale RUNNING trials are dead: this pod owns the study (one pod per cell, n_jobs=1) and nothing of
    # ours is running yet, so a RUNNING row can only be left over from a pod that was terminated mid-trial.
    # Mark them FAIL and raise the target by the same number so the study still ends with `target`
    # FINISHED trials (remaining = target - len(study.trials), and len() counts FAIL/RUNNING rows).
    stale = [t.number for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))]
    if stale:
        for number in stale:
            study.tell(number, state=TrialState.FAIL)
        target_n_trials += len(stale)
        study.set_user_attr(TARGET_ATTR, target_n_trials)
        study.set_user_attr("stale_running_failed",
                            int(study.user_attrs.get("stale_running_failed", 0)) + len(stale))
        print(f"Marked {len(stale)} stale RUNNING trial(s) {stale} as FAIL; target raised to "
              f"{target_n_trials} so {target_n_trials - len(stale)} finished trials are still reached.")

    warm_started = False
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already recorded")
        if args.resume:
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if completed:
                print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")
            else:
                print("No completed trials yet.")
    else:
        # Fresh study — enqueue a warm-start trial from A2C's best params for
        # this exact (representation, centering) cell. Every centering mode
        # now samples its own full representation+DQN+centering params, so
        # this applies uniformly (previously skipped for centering != "none",
        # back when only the centering step size was sampled there).
        warm_started = _enqueue_warmstart(study, args.representation, args.centering)

    objective = make_objective(args.representation, args.centering, args.architecture,
                                args.n_evaluations, args.n_eval_episodes)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # R2 sync callback — fires after every N completed trials. Uses fire-and-forget
    # subprocess so the search isn't blocked on upload latency. Deliberately
    # omits --best-params-file: this cell's yaml is only ever written once, at
    # completion (save_best_params(), below), so there's nothing new to push
    # for it mid-search. DOES include --csv-dir on every sync — per-trial
    # rewards/states/values CSVs are written incrementally throughout the
    # search (one trial at a time), unlike best_params, so periodic syncing
    # of them is meaningful (protects against losing a pod's trial history if
    # it's lost mid-search, not just its final summary).
    SYNC_SCRIPT = str(Path(__file__).parent / "sync_to_r2.py")
    CSV_DIR = _cell_data_dir(args.representation, args.centering, args.architecture)
    sync_state = {"last": 0, "every": args.sync_every_n_trials, "enabled": not args.no_sync}

    def maybe_sync(study, _trial):
        if not sync_state["enabled"] or sync_state["every"] <= 0:
            return
        n_done = sum(1 for t in study.trials if t.state.is_finished())
        if n_done - sync_state["last"] >= sync_state["every"]:
            sync_state["last"] = n_done
            try:
                subprocess.Popen(
                    [sys.executable, SYNC_SCRIPT, "trial", "--db-path", DB_PATH,
                     "--csv-dir", CSV_DIR],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,  # survive parent process death
                )
            except FileNotFoundError:
                # sync_to_r2.py missing or .env missing — skip silently
                pass

    # Use n_existing (captured before _enqueue_warmstart) rather than a fresh
    # len(study.trials) here: study.enqueue_trial() immediately creates a
    # WAITING-state trial row, so re-reading the count after enqueueing but
    # before any optimize() call double-counts it and undershoots the target
    # by one (verified: --n-trials 5 on a fresh study only ran 4).
    remaining = max(0, target_n_trials - n_existing)
    if remaining == 0:
        print(f"Target of {target_n_trials} trials already reached "
              f"({len(study.trials)} in study) — nothing to do this invocation "
              f"(likely a pod/container restart after a previously-completed run).")
    else:
        if warm_started and args.n_jobs > 1:
            # Run the enqueued warm-start trial single-threaded to avoid a race
            # condition where multiple workers simultaneously claim Trial#0.
            study.optimize(objective, n_trials=1, n_jobs=1, show_progress_bar=True,
                           callbacks=[maybe_sync])
            remaining = max(0, target_n_trials - len(study.trials))

        if remaining > 0:
            study.optimize(objective, n_trials=remaining, n_jobs=args.n_jobs,
                           show_progress_bar=True, callbacks=[maybe_sync])

    print(f"\nSearch complete. Total trials in study: {len(study.trials)} (target {target_n_trials})")
    print(f"Raw search winner (single seed, may be a lucky seed): "
          f"value={study.best_value:.2f}  params={study.best_params}")

    # Post-hoc re-evaluation (araffin.github.io/post/optuna/'s "find lucky
    # seeds" tip). Cached on the study itself so a pod restart after this
    # already ran doesn't re-spend the full reeval. Finer-grained resume (per run) comes from the
    # JSONL that is restored from R2.
    # New namespace (reeval_v2_*): the original "reeval_*" attrs from the old online-metric
    # reeval are left untouched on the study, and the new result is written to a *_v2.yaml.
    REEVAL_DONE_ATTR = f"{REEVAL_ATTR_PREFIX}_done"
    BEST_TRIAL_ATTR = f"{REEVAL_ATTR_PREFIX}_best_trial"
    r2_prefix = None if args.no_sync else f"{R2_REFILTER_PREFIX}/{study_name}"
    SYNC_OK = {"ok": True}

    def _final_sync() -> bool:
        """Push DB + this cell's v2 best-params yaml + CSV tree; True only if every upload succeeded."""
        if args.no_sync:
            return True
        best_params_filename = _best_filename(args.representation, args.centering, args.architecture,
                                              variant=REEVAL_VARIANT)
        try:
            r = subprocess.run(
                [sys.executable, SYNC_SCRIPT, "trial", "--db-path", DB_PATH,
                 "--best-params-file", best_params_filename, "--csv-dir", CSV_DIR],
                check=False,
            )
            return r.returncode == 0
        except FileNotFoundError:
            return False

    if study.user_attrs.get(REEVAL_DONE_ATTR) and BEST_TRIAL_ATTR in study.user_attrs:
        best_params = study.user_attrs[f"{REEVAL_ATTR_PREFIX}_best_params"]
        best_value = study.user_attrs[f"{REEVAL_ATTR_PREFIX}_best_value"]
        best_std = study.user_attrs[f"{REEVAL_ATTR_PREFIX}_best_std"]
        best_trial = int(study.user_attrs[BEST_TRIAL_ATTR])
        print("Greedy (v2) post-hoc re-evaluation already recorded for this study — reusing it.")
    else:
        print(f"\nRe-evaluating (greedy eval, last-{args.last_k} checkpoints, staged "
              f"{args.stage1_seeds}->{args.reeval_seeds} seeds, keep {args.stage1_keep:.0%}, pool top "
              f"{args.pool_quantile:.0%} clamped to [{args.pool_min}, {args.pool_max}], "
              f"{args.workers} workers)...", flush=True)
        best_params, best_value, best_std, best_trial = reevaluate_top_trials(
            study, args.representation, args.centering, args.architecture,
            n_reeval_seeds=args.reeval_seeds, n_evaluations=args.n_evaluations,
            n_eval_episodes=args.n_eval_episodes, last_k=args.last_k,
            quantile=args.pool_quantile, pool_min=args.pool_min, pool_max=args.pool_max,
            stage1_seeds=args.stage1_seeds, stage1_keep=args.stage1_keep,
            workers=args.workers, r2_prefix=r2_prefix,
        )
        study.set_user_attr(f"{REEVAL_ATTR_PREFIX}_best_params", best_params)
        study.set_user_attr(f"{REEVAL_ATTR_PREFIX}_best_value", best_value)
        study.set_user_attr(f"{REEVAL_ATTR_PREFIX}_best_std", best_std)
        study.set_user_attr(BEST_TRIAL_ATTR, int(best_trial))
        study.set_user_attr(REEVAL_DONE_ATTR, True)

    print(f"Post-hoc winner: mean={best_value:.2f} std={best_std:.2f} params={best_params}")

    save_best_params(args.representation, args.centering, best_value, best_params,
                      architecture=args.architecture, best_std=best_std, variant=REEVAL_VARIANT)

    # Sync AFTER save_best_params so R2 gets the just-written best-params YAML. --best-params-file
    # scopes this to ONLY this cell's own file — see sync_to_r2.py's cmd_trial docstring for the
    # multi-pod clobbering bug this fixes. --csv-dir also picks up the reeval/ subfolder's CSVs.
    SYNC_OK["ok"] &= _final_sync()

    if args.final_eval:
        print(f"\nFinal eval: winner (search trial #{best_trial}) on {args.final_seeds} fresh seeds "
              f"(offset {args.final_seed_offset}), greedy, {args.workers} workers...", flush=True)
        summary = final_evaluate(
            study_name, args.representation, args.centering, args.architecture,
            winner_trial=best_trial, winner_params=best_params,
            reeval_mean=best_value, reeval_std=best_std, n_seeds=args.final_seeds,
            n_evaluations=args.n_evaluations, n_eval_episodes=args.n_eval_episodes,
            last_k=args.last_k, final_eval_episodes=args.final_eval_episodes,
            workers=args.workers, seed_offset=args.final_seed_offset, r2_prefix=r2_prefix,
        )
        fe = summary["final_eval"]
        print(f"FINAL EVAL (greedy last-{args.last_k} mean over {fe['n_seeds']} seeds): "
              f"{fe['score_mean']:.1f} +/- {fe['score_std']:.1f}  "
              f"(min {fe['score_min']:.0f}, max {fe['score_max']:.0f}, "
              f"solved {fe['share_solved_lastk']:.0%}); filter winner had {best_value:.1f} +/- {best_std:.1f}")
        SYNC_OK["ok"] &= _final_sync()   # also uploads the final-eval CSVs

    print("CELL COMPLETE" + ("" if SYNC_OK["ok"] else " (WARNING: at least one R2 upload failed)"), flush=True)
    if args.idle_after:
        _idle_forever(delete_ok=args.self_delete and SYNC_OK["ok"] and not args.no_sync)


def _idle_forever(delete_ok: bool = False):
    """Pod mode: RunPod restarts the entrypoint whenever the process exits, so never exit. If asked and
    everything is safely on R2, delete this pod through the RunPod API first (cost guard)."""
    import time
    import urllib.error
    import urllib.request
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if delete_ok and pod and key:
        req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}", method="DELETE",
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": "curl/8.5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                print(f"[self-delete] DELETE pod {pod} -> HTTP {r.status}", flush=True)
        except urllib.error.HTTPError as e:
            print(f"[self-delete] DELETE pod {pod} -> HTTP {e.code} {e.read().decode()[:200]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[self-delete] failed: {e!r}", flush=True)
    elif delete_ok:
        print("[self-delete] RUNPOD_POD_ID/RUNPOD_API_KEY not in env — idling instead.", flush=True)
    print("idling (pod mode) — all results are on R2; delete this pod.", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if "--idle-after" in sys.argv:
            # Do NOT exit: RunPod would restart the entrypoint in a loop. Idle so the failure is inspectable.
            print("ERROR (see traceback above) — idling instead of exiting (--idle-after).", flush=True)
            import time
            while True:
                time.sleep(3600)
        raise
