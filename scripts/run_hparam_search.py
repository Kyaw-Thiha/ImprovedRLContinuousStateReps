"""
Hyperparameter search using Optuna for a representation/centering condition.

Results are persisted in optuna_studies.db (SQLite) so searches can be
interrupted and resumed without losing progress. Raw trial artifacts are written
to outputs/hparam_search/{representation}/{centering}/. Best params are written
to outputs/hparam_search/best/{representation}_{centering}.yaml when the search
completes.

Usage:
    python scripts/run_hparam_search.py --representation discrete --centering none
    python scripts/run_hparam_search.py --representation ssp --centering value --n-trials 50
    python scripts/run_hparam_search.py --representation tile_coding --centering simple --n-seeds 5
    python scripts/run_hparam_search.py --representation discrete --centering none --resume

Search spaces are documented in conf/hparam_search/optuna_{rep}.yaml.
"""

import sys, os, argparse
import numpy as np
import optuna
import yaml
from optuna.trial import TrialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole import ACTrial

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


def sample_params(trial: optuna.Trial, representation: str, centering: str) -> dict:
    params = sample_representation_params(trial, representation)
    params.update(sample_centering_params(trial, centering))
    return params


def make_objective(representation: str, centering: str, n_seeds: int):
    # Keep pytry's per-run ACTrial artifacts grouped by the 3x3 HPO condition.
    data_dir = os.path.join(DATA_DIR, representation, centering)
    os.makedirs(data_dir, exist_ok=True)

    episodes_per_seed = BASE_PARAMS["trials"]

    def objective(trial: optuna.Trial) -> float:
        # Each trial gets its own ACTrial instance — required for thread safety
        # when study.optimize is called with n_jobs > 1.
        ac = ACTrial()
        params = dict(BASE_PARAMS)
        params.update(sample_params(trial, representation, centering))

        rewards = []
        for seed in range(n_seeds):
            ep_offset = seed * episodes_per_seed

            def _pruning_callback(ep_idx, rolling_reward, _offset=ep_offset):
                trial.report(rolling_reward, _offset + ep_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            ac._pruning_callback = _pruning_callback
            try:
                metadata = ac.run(
                    seed=seed,
                    data_dir=data_dir,
                    pre_comment=f"optuna trial={trial.number} rep={representation} centering={centering} seed={seed}",
                    **params,
                )
                rewards.append(metadata["terminal_reward"])
            except optuna.exceptions.TrialPruned:
                raise
            finally:
                ac._pruning_callback = None

        return float(np.mean(rewards))

    return objective


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
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument(
        "--representation", choices=["discrete", "ssp", "tile_coding"], required=True, help="Which representation to tune"
    )
    parser.add_argument(
        "--centering", choices=["none", "simple", "value"], required=True, help="Which reward-centering mode to tune"
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds to average per Optuna trial (fewer = faster search)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trial workers (4-6 recommended on 8-core machines)")
    parser.add_argument(
        "--resume", action="store_true", help="Print existing study progress and resume (study always persists to DB)"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    study_name = f"{args.representation}_{args.centering}_hparam_search"
    storage = f"sqlite:///{DB_PATH}"

    # load_if_exists=True means re-running without --resume safely continues the study.
    # MedianPruner: don't prune until 5 trials complete and episode 200 has passed,
    # so agents have enough time to warm up before being compared.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=200),
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

    objective = make_objective(args.representation, args.centering, args.n_seeds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    print(f"\nSearch complete.")
    print(f"Best value : {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    save_best_params(args.representation, args.centering, study.best_value, study.best_params)


if __name__ == "__main__":
    main()
