"""
Hyperparameter search using Optuna for a given representation type.

Results are persisted in optuna_studies.db (SQLite) so searches can be
interrupted and resumed without losing progress. Best params are written
back to conf/representation/{rep}.yaml when the search completes.

Usage:
    python scripts/run_hparam_search.py --representation discrete
    python scripts/run_hparam_search.py --representation ssp --n-trials 50
    python scripts/run_hparam_search.py --representation tile_coding --n-seeds 5
    python scripts/run_hparam_search.py --representation discrete --resume

Search spaces are documented in conf/hparam_search/optuna_{rep}.yaml.
"""

import sys, os, argparse
import numpy as np
import optuna
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from trial_cartpole import ACTrial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_REP_DIR = os.path.join(REPO_ROOT, "conf", "representation")
DB_PATH = os.path.join(REPO_ROOT, "optuna_studies.db")
DATA_DIR = os.path.join(REPO_ROOT, "outputs", "hparam_search")

# Base params held fixed during search — centering off, standard AC settings
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
    reward_center_mode="none",
    reward_center_beta=0.001,
    reward_center_eta=1.0,
    reward_center_init=0.0,
    verbose=False,
    gifs=False,
)


def sample_params(trial: optuna.Trial, representation: str) -> dict:
    """Sample hyperparameters for the given representation."""
    if representation == "discrete":
        return dict(
            rep_="Discrete",
            n_bins=19,
            eps=trial.suggest_float("eps", 0.1, 0.8),
            lr=trial.suggest_float("lr", 0.001, 0.5, log=True),
        )
    elif representation == "ssp":
        return dict(
            rep_="PlaceSSP",
            eps=trial.suggest_float("eps", 0.1, 0.8),
            lr=trial.suggest_float("lr", 0.001, 0.5, log=True),
            length_scale=trial.suggest_float("length_scale", 0.1, 2.0),
            n_rotates=trial.suggest_int("n_rotates", 4, 12),
        )
    elif representation == "tile_coding":
        return dict(
            rep_="TileCoding",
            num_tilings=trial.suggest_int("num_tilings", 4, 32),
            tiles_per_dim=(8, 8, 8, 8),
            iht_size=65536,
            tile_state_indices=(0, 1, 2, 3),
            eps=trial.suggest_float("eps", 0.1, 0.8),
            lr=trial.suggest_float("lr", 0.001, 0.5, log=True),
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")


def make_objective(representation: str, n_seeds: int):
    ac = ACTrial()

    def objective(trial: optuna.Trial) -> float:
        rep_params = sample_params(trial, representation)

        rewards = []
        for seed in range(n_seeds):
            metadata = ac.run(
                seed=seed,
                data_dir=DATA_DIR,
                pre_comment=f"optuna trial={trial.number} rep={representation} seed={seed}",
                **BASE_PARAMS,
                **rep_params,
            )
            rewards.append(metadata["terminal_reward"])

        return float(np.mean(rewards))

    return objective


def save_best_params(representation: str, best_params: dict) -> None:
    """Merge best searched params into the representation's conf YAML."""
    conf_path = os.path.join(CONF_REP_DIR, f"{representation}.yaml")

    with open(conf_path) as f:
        current = yaml.safe_load(f)

    current.update(best_params)

    with open(conf_path, "w") as f:
        f.write(f"# Best params from Optuna search\n")
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)

    print(f"Best params written to {conf_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument(
        "--representation", choices=["discrete", "ssp", "tile_coding"], required=True,
        help="Which representation to tune"
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Seeds to average per Optuna trial (fewer = faster search)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Print existing study progress and resume (study always persists to DB)"
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    study_name = f"{args.representation}_hparam_search"
    storage = f"sqlite:///{DB_PATH}"

    # load_if_exists=True means re-running without --resume safely continues the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already completed")
        if args.resume:
            print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")

    objective = make_objective(args.representation, args.n_seeds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nSearch complete.")
    print(f"Best value : {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    save_best_params(args.representation, study.best_params)


if __name__ == "__main__":
    main()
