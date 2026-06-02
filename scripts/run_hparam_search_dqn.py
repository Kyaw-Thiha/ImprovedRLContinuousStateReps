"""
DQN hyperparameter search using Optuna for a representation/centering condition.

Speed-ups over the A2C search:
  1. Warm-start: the first trial is seeded from A2C's optimal lr/eps (and
     representation-specific params) so the shared search space converges faster.
  2. HyperbandPruner: eliminates the bottom 2/3 of trials at each bracket,
     cutting ~60% more wasted steps than the MedianPruner used for A2C.
  3. Recommend n_seeds=3 during search (vs 5 for A2C); use 5 only for final ablation.

Results are persisted in optuna_studies.db so searches can be interrupted and
resumed. Best params are written to outputs/hparam_search/best/dqn_{rep}_{centering}.yaml.

Usage:
    python scripts/run_hparam_search_dqn.py --representation discrete --centering none
    python scripts/run_hparam_search_dqn.py --representation ssp --centering value --n-trials 60 --n-seeds 3 --n-jobs 4
    python scripts/run_hparam_search_dqn.py --representation tile_coding --centering simple --resume

Search spaces are documented in conf/hparam_search/optuna_dqn_{rep}.yaml.
"""

import argparse
import os
import sys

import numpy as np
import optuna
import yaml
from optuna.trial import TrialState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole_dqn import DQNTrial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "optuna_studies.db")
DATA_DIR = os.path.join(REPO_ROOT, "outputs", "hparam_search")
BEST_DIR = os.path.join(DATA_DIR, "best")
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
    buffer_size=10000,
    batch_size=64,
    target_update_freq=100,
    learning_starts=500,
    train_freq=1,
)


def _load_a2c_best(representation: str, centering: str) -> dict:
    path = os.path.join(A2C_BEST_DIR, f"{representation}_{centering}.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = yaml.safe_load(f)
    return payload.get("params", {})


def _enqueue_warmstart(study: optuna.Study, representation: str, centering: str):
    a2c = _load_a2c_best(representation, centering)
    if not a2c:
        return

    trial_params = {}

    # Shared params
    if "eps" in a2c:
        trial_params["eps"] = float(a2c["eps"])
    if "lr" in a2c:
        trial_params["lr"] = float(a2c["lr"])

    # Representation-specific shared params
    if representation == "discrete" and "n_bins" in a2c:
        trial_params["n_bins"] = int(a2c["n_bins"])
    elif representation == "ssp":
        if "length_scale" in a2c:
            trial_params["length_scale"] = float(a2c["length_scale"])
        if "n_rotates" in a2c:
            trial_params["n_rotates"] = int(a2c["n_rotates"])
    elif representation == "tile_coding":
        if "num_tilings" in a2c:
            trial_params["num_tilings"] = int(a2c["num_tilings"])
        if "tiles_per_dim" in a2c:
            tpd = a2c["tiles_per_dim"]
            trial_params["tiles_per_dim"] = int(tpd[0]) if isinstance(tpd, (list, tuple)) else int(tpd)
        if "iht_size" in a2c:
            trial_params["iht_size"] = int(a2c["iht_size"])

    # DQN-specific defaults (RL Zoo CartPole starting point)
    trial_params.update(DQN_DEFAULTS)

    # Centering params
    if centering == "simple" and "reward_center_beta" in a2c:
        trial_params["reward_center_beta"] = float(a2c["reward_center_beta"])
    elif centering == "value" and "reward_center_eta" in a2c:
        trial_params["reward_center_eta"] = float(a2c["reward_center_eta"])

    study.enqueue_trial(trial_params, user_attrs={"source": f"a2c_warmstart_{representation}_{centering}"})
    print(f"Warm-start trial enqueued from A2C best params ({representation}/{centering})")


def sample_representation_params(trial: optuna.Trial, representation: str) -> dict:
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


def sample_dqn_params(trial: optuna.Trial) -> dict:
    return dict(
        buffer_size=trial.suggest_categorical("buffer_size", [1000, 5000, 10000, 50000]),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        target_update_freq=trial.suggest_categorical("target_update_freq", [10, 50, 100, 500]),
        learning_starts=trial.suggest_categorical("learning_starts", [100, 500, 1000]),
    )


def sample_centering_params(trial: optuna.Trial, centering: str) -> dict:
    params = dict(reward_center_mode=centering)
    if centering == "simple":
        params["reward_center_beta"] = trial.suggest_float("reward_center_beta", 1e-5, 1e-1, log=True)
    elif centering == "value":
        params["reward_center_eta"] = trial.suggest_float("reward_center_eta", 1e-4, 1.0, log=True)
    return params


def make_objective(representation: str, centering: str, n_seeds: int):
    data_dir = os.path.join(DATA_DIR, "dqn", representation, centering)
    os.makedirs(data_dir, exist_ok=True)

    episodes_per_seed = BASE_PARAMS["trials"]

    def objective(trial: optuna.Trial) -> float:
        dqn = DQNTrial()
        params = dict(BASE_PARAMS)
        params.update(sample_representation_params(trial, representation))
        params.update(sample_dqn_params(trial))
        params.update(sample_centering_params(trial, centering))

        rewards = []
        for seed in range(n_seeds):
            ep_offset = seed * episodes_per_seed

            def _pruning_callback(ep_idx, rolling_reward, _offset=ep_offset):
                trial.report(rolling_reward, _offset + ep_idx)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            dqn._pruning_callback = _pruning_callback
            try:
                metadata = dqn.run(
                    seed=seed,
                    data_dir=data_dir,
                    pre_comment=f"optuna trial={trial.number} rep={representation} centering={centering} seed={seed}",
                    **params,
                )
                rewards.append(metadata["terminal_reward"])
            except optuna.exceptions.TrialPruned:
                raise
            finally:
                dqn._pruning_callback = None

        return float(np.mean(rewards))

    return objective


def trial_params_from_search_params(representation: str, centering: str, search_params: dict) -> dict:
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
    os.makedirs(BEST_DIR, exist_ok=True)
    best_path = os.path.join(BEST_DIR, f"dqn_{representation}_{centering}.yaml")

    payload = {
        "model_type": "dqn",
        "representation": representation,
        "centering": centering,
        "best_value": float(best_value),
        "params": trial_params_from_search_params(representation, centering, best_params),
    }

    with open(best_path, "w") as f:
        f.write("# Best DQN params from Optuna search. Intended for scripts/run_best_ablation.py\n")
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    print(f"Best params written to {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Optuna DQN hyperparameter search")
    parser.add_argument("--representation", choices=["discrete", "ssp", "tile_coding"], required=True)
    parser.add_argument("--centering", choices=["none", "simple", "value"], required=True)
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per trial (3 recommended for search)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel trial workers")
    parser.add_argument("--resume", action="store_true", help="Print existing progress and resume")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    study_name = f"dqn_{args.representation}_{args.centering}_hparam_search"
    storage = f"sqlite:///{DB_PATH}"

    # HyperbandPruner: aggressively eliminates bottom 2/3 of trials at each bracket.
    # min_resource=50 gives agents time to warm up before being pruned.
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=50,
            max_resource=BASE_PARAMS["trials"],
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"Study '{study_name}': {n_existing} trials already recorded")
        if args.resume:
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if completed:
                print(f"Current best: {study.best_value:.2f}  params: {study.best_params}")
            else:
                print("No completed trials yet.")
    else:
        # Fresh study — enqueue warm-start trial from A2C best params
        _enqueue_warmstart(study, args.representation, args.centering)

    objective = make_objective(args.representation, args.centering, args.n_seeds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    print(f"\nSearch complete.")
    print(f"Best value : {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")

    save_best_params(args.representation, args.centering, study.best_value, study.best_params)


if __name__ == "__main__":
    main()
