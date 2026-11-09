"""
A2C hyperparameter search using Optuna for a representation/centering condition.

Follows the same methodology as scripts/run_hparam_search_dqn.py — see that
module's docstring and docs/02-experimental-design.md for the full rationale —
based on Antonin Raffin's two-part Optuna guide:
  - Part 1 (concepts): https://araffin.github.io/post/hyperparam-tuning/
  - Part 2 (practice):  https://araffin.github.io/post/optuna/
Concretely: single seed per trial during search (seed=trial.number, not
averaged), TPESampler+MedianPruner with an explicit "never prune before 1/3
of budget" warmup, 500 trials per cell, and a post-hoc top-k/reeval-seeds
re-evaluation step to filter lucky-seed winners.

One methodology point from the DQN version is NOT implemented here: a
separate deterministic evaluation environment. ACTrial's network
(rl/networks/acBasic_nengo.py's ActorCritic) is a Nengo simulation whose `step()`
always runs the learning rule as part of advancing the simulator — there is
no inference-only/no-learning call to evaluate a frozen policy on a held-out
env without also training on it. Absent that, this script still scores/prunes
on the rolling mean of the last 100 *training* episodes (noisier than a true
held-out eval, same as this project's pre-revamp DQN search did). Revisit if
ACTrial ever gets a genuine eval-only path.

Results are persisted in optuna_studies.db (SQLite) so searches can be
interrupted and resumed without losing progress. Raw trial artifacts are written
to outputs/hparam_search/a2c/{representation}/{centering}/. Best params are written
to outputs/hparam_search/best/{representation}_{centering}.yaml when the search
completes.

Usage:
    python scripts/run_hparam_search_a2c_nengo.py --representation discrete --centering none
    python scripts/run_hparam_search_a2c_nengo.py --representation ssp --centering value --n-trials 500
    python scripts/run_hparam_search_a2c_nengo.py --representation discrete --centering none --resume

Search spaces are documented in conf/hparam_search/optuna_{rep}.yaml.
"""

import sys, os, argparse
import numpy as np
import optuna
import yaml
from optuna.trial import TrialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole_nengo import ACTrial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "optuna_studies.db")
DATA_DIR = os.path.join(REPO_ROOT, "outputs", "hparam_search")
BEST_DIR = os.path.join(DATA_DIR, "best")

# Base params held fixed during search.
BASE_PARAMS = dict(
    trials=1000,
    steps=500,
    env="CartPole-v1",
    n_done=1,
    learnTrials=1000,
    rule="TD0",
    dynamic_epsilon=True,
    act_dis=0.9,
    state_dis=0.99,
    on_policy_override=False,
    force_rho_one=True,
    reward_center_beta=0.001,
    reward_center_eta=1.0,
    reward_center_init=0.0,
    verbose=False,
    gifs=False,
)


def sample_representation_params(trial: optuna.Trial, representation: str) -> dict:
    """Sample hyperparameters for the given representation."""
    if representation == "discrete":
        return dict(
            rep_="Discrete",
            n_bins=trial.suggest_categorical("n_bins", [7, 11, 15, 19, 23]),
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 1.0, log=True),
        )
    elif representation == "ssp":
        return dict(
            rep_="PlaceSSP",
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 1.0, log=True),
            length_scale=trial.suggest_float("length_scale", 0.05, 5.0, log=True),
            n_rotates=trial.suggest_int("n_rotates", 4, 16),
        )
    elif representation == "tile_coding":
        tiles_per_dim = trial.suggest_categorical("tiles_per_dim", [6, 8, 10, 12, 16])
        return dict(
            rep_="TileCoding",
            num_tilings=trial.suggest_categorical("num_tilings", [8, 16, 32]),
            tiles_per_dim=(tiles_per_dim,) * 4,
            iht_size=trial.suggest_categorical("iht_size", [16384, 32768, 65536, 131072]),
            tile_state_indices=(0, 1, 2, 3),
            eps=trial.suggest_float("eps", 0.01, 0.5, log=True),
            lr=trial.suggest_float("lr", 1e-4, 1.0, log=True),
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")


def sample_centering_params(trial: optuna.Trial, centering: str) -> dict:
    """Sample reward-centering hyperparameters for the given centering mode."""
    params = dict(reward_center_mode=centering)

    if centering == "none":
        return params
    if centering == "simple":
        params["reward_center_beta"] = trial.suggest_float("reward_center_beta", 1e-5, 1e-1, log=True)
        return params
    if centering == "value":
        params["reward_center_eta"] = trial.suggest_float("reward_center_eta", 1e-4, 1.0, log=True)
        return params

    raise ValueError(f"Unknown centering mode: {centering}")


def make_objective(representation: str, centering: str):
    """Single seed per trial (seed=trial.number, not averaged) — see module
    docstring. No separate eval env (ACTrial/Nengo limitation, also noted in
    the module docstring); prunes/scores on the rolling 100-episode TRAINING
    reward, reported every episode, same signal as before.
    """
    # Keep pytry's per-run ACTrial artifacts grouped by the 3x3 HPO condition.
    data_dir = os.path.join(DATA_DIR, "a2c", representation, centering)
    os.makedirs(data_dir, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        # Each trial gets its own ACTrial instance — required for thread safety
        # when study.optimize is called with n_jobs > 1.
        ac = ACTrial()
        params = dict(BASE_PARAMS)
        params.update(sample_representation_params(trial, representation))
        params.update(sample_centering_params(trial, centering))

        def _pruning_callback(ep_idx, rolling_reward):
            trial.report(rolling_reward, ep_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        ac._pruning_callback = _pruning_callback
        try:
            metadata = ac.run(
                seed=trial.number,
                data_dir=data_dir,
                pre_comment=f"optuna trial={trial.number} rep={representation} centering={centering}",
                **params,
            )
        finally:
            ac._pruning_callback = None

        return float(metadata["terminal_reward"])

    return objective


def reevaluate_top_trials(study: optuna.Study, representation: str, centering: str,
                           top_k: int, n_reeval_seeds: int) -> tuple[dict, float, float]:
    """Post-hoc validation (Part 2: "re-evaluate the top trials multiple times
    to find out which ones were lucky seeds"). Re-runs each of the search's
    top-k trials with n_reeval_seeds fresh seeds and picks the winner by mean
    (ties broken toward lower std). See run_hparam_search_dqn.py's twin
    function for the DQN version of this same step.
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No COMPLETE trials in study '{study.study_name}' to re-evaluate.")

    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]
    data_dir = os.path.join(DATA_DIR, "a2c", representation, centering, "reeval")
    os.makedirs(data_dir, exist_ok=True)

    results = []
    for rank, t in enumerate(top_trials):
        params = dict(BASE_PARAMS)
        params.update(trial_params_from_search_params(representation, centering, t.params))

        rewards = []
        for seed in range(n_reeval_seeds):
            ac = ACTrial()
            metadata = ac.run(
                seed=1000 + seed,  # offset away from search-phase seeds (0..n_trials-1)
                data_dir=data_dir,
                pre_comment=f"reeval rank={rank} search_trial={t.number} "
                            f"rep={representation} centering={centering} seed={seed}",
                **params,
            )
            rewards.append(metadata["terminal_reward"])

        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        print(f"  reeval rank={rank} (search trial #{t.number}, search score {t.value:.2f}): "
              f"mean={mean_r:.2f} std={std_r:.2f} over {n_reeval_seeds} seeds")
        results.append((t.params, mean_r, std_r))

    best_params, best_mean, best_std = max(results, key=lambda r: (round(r[1], 0), -r[2]))
    return best_params, best_mean, best_std


def trial_params_from_search_params(representation: str, centering: str, search_params: dict) -> dict:
    """Convert Optuna's searched params into ACTrial-ready params."""
    params = dict(search_params)

    if representation == "discrete":
        params["rep_"] = "Discrete"
    elif representation == "ssp":
        params["rep_"] = "PlaceSSP"
    elif representation == "tile_coding":
        params["rep_"] = "TileCoding"
        if isinstance(params.get("tiles_per_dim"), int):
            params["tiles_per_dim"] = [params["tiles_per_dim"]] * 4
        params["tile_state_indices"] = [0, 1, 2, 3]
    else:
        raise ValueError(f"Unknown representation: {representation}")

    params["reward_center_mode"] = centering
    return params


def save_best_params(representation: str, centering: str, best_value: float, best_params: dict) -> None:
    """Write condition-specific best params without mutating conf/ defaults."""
    os.makedirs(BEST_DIR, exist_ok=True)
    best_path = os.path.join(BEST_DIR, f"{representation}_{centering}.yaml")

    payload = {
        "representation": representation,
        "centering": centering,
        "best_value": float(best_value),
        "params": trial_params_from_search_params(representation, centering, best_params),
    }

    with open(best_path, "w") as f:
        f.write("# Best params from Optuna search. Intended for scripts/run_best_ablation.py\n")
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    print(f"Best params written to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna A2C hyperparameter search")
    parser.add_argument(
        "--representation", choices=["discrete", "ssp", "tile_coding"], required=True, help="Which representation to tune"
    )
    parser.add_argument(
        "--centering", choices=["none", "simple", "value"], required=True, help="Which reward-centering mode to tune"
    )
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Optuna trials for this cell (default: 500, rl-baselines3-zoo's own default budget)")
    parser.add_argument("--n-startup-trials", type=int, default=10,
                        help="Random trials before TPE starts modeling correlations / before pruning is allowed")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Post-hoc: re-evaluate this many top search trials with --reeval-seeds fresh seeds each")
    parser.add_argument("--reeval-seeds", type=int, default=10,
                        help="Seeds per top-k trial in the post-hoc re-evaluation step")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trial workers (4-6 recommended on 8-core machines)")
    parser.add_argument(
        "--resume", action="store_true", help="Print existing study progress and resume (study always persists to DB)"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # _v2: study renamed/versioned when this switched from multi-seed-averaged
    # trials to single-seed-per-trial + post-hoc re-evaluation — old trial
    # values (means over 5 seeds) aren't comparable to new ones (single seed).
    study_name = f"{args.representation}_{args.centering}_v2_hparam_search"
    storage = f"sqlite:///{DB_PATH}"

    # TPESampler + MedianPruner, following araffin.github.io/post/optuna/'s
    # recipe: no trial can be pruned before 1/3 of its episode budget
    # (n_warmup_steps, in episode units here since every episode is reported —
    # see module docstring on why there's no separate eval-checkpoint count).
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=args.n_startup_trials,
                                            multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials,
                                            n_warmup_steps=BASE_PARAMS["trials"] // 3),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already completed")
        if args.resume:
            completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
            if completed_trials:
                print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")
            else:
                print("Current best: none yet; existing trials have not completed successfully.")

    objective = make_objective(args.representation, args.centering)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    remaining = max(0, args.n_trials - n_existing)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=args.n_jobs, show_progress_bar=True)

    print(f"\nSearch complete. Total trials in study: {len(study.trials)}")
    print(f"Raw search winner (single seed, may be a lucky seed): "
          f"value={study.best_value:.2f}  params={study.best_params}")

    # Post-hoc re-evaluation (araffin.github.io/post/optuna/'s "find lucky
    # seeds" tip). Cached on the study itself so a re-run after this already
    # happened doesn't re-spend top_k*reeval_seeds full training runs.
    REEVAL_DONE_ATTR = "reeval_done"
    if study.user_attrs.get(REEVAL_DONE_ATTR):
        best_params = study.user_attrs["reeval_best_params"]
        best_value = study.user_attrs["reeval_best_value"]
        best_std = study.user_attrs["reeval_best_std"]
        print("Post-hoc re-evaluation already recorded for this study — reusing it.")
    else:
        print(f"\nRe-evaluating top {args.top_k} trials with {args.reeval_seeds} seeds each...")
        best_params, best_value, best_std = reevaluate_top_trials(
            study, args.representation, args.centering, args.top_k, args.reeval_seeds,
        )
        study.set_user_attr("reeval_best_params", best_params)
        study.set_user_attr("reeval_best_value", best_value)
        study.set_user_attr("reeval_best_std", best_std)
        study.set_user_attr(REEVAL_DONE_ATTR, True)

    print(f"Post-hoc winner: mean={best_value:.2f} std={best_std:.2f} params={best_params}")

    save_best_params(args.representation, args.centering, best_value, best_params)


if __name__ == "__main__":
    main()
