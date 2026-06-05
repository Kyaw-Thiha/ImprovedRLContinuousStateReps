"""
Run the full 3x3 ablation grid using Optuna-selected params per condition.

Expected input files are produced by scripts/run_hparam_search.py:
    outputs/hparam_search/best/{representation}_{centering}.yaml

Usage:
    python scripts/run_best_ablation.py
    python scripts/run_best_ablation.py --n-seeds 20
    python scripts/run_best_ablation.py --representation discrete --centering value
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole import ACTrial
from experiments.trial_cartpole_dqn import DQNTrial
from run_hparam_search import BASE_PARAMS, BEST_DIR
from run_hparam_search_dqn import BASE_PARAMS as DQN_BASE_PARAMS


REPRESENTATIONS = ["discrete", "ssp", "tile_coding"]
CENTERINGS = ["none", "simple", "value"]


def load_best_params(best_dir: str, representation: str, centering: str, model_type: str = "actor_critic") -> dict:
    prefix = "dqn_" if model_type == "dqn" else ""
    best_path = os.path.join(best_dir, f"{prefix}{representation}_{centering}.yaml")
    if not os.path.exists(best_path):
        search_script = "run_hparam_search_dqn.py" if model_type == "dqn" else "run_hparam_search.py"
        raise FileNotFoundError(
            f"Missing tuned params: {best_path}. Run scripts/{search_script} "
            f"--representation {representation} --centering {centering} first."
        )

    with open(best_path) as f:
        payload = yaml.safe_load(f)

    return payload["params"]


def output_dir(output_root: str, model_type: str, params: dict) -> str:
    return os.path.abspath(os.path.join(
        output_root,
        model_type,
        params["rep_"],
        params["reward_center_mode"],
    ))


def main():
    parser = argparse.ArgumentParser(description="Run tuned 3x3 CartPole actor-critic ablation")
    parser.add_argument("--representation", choices=REPRESENTATIONS, help="Run one representation only")
    parser.add_argument("--centering", choices=CENTERINGS, help="Run one centering mode only")
    parser.add_argument("--best-dir", default=BEST_DIR, help="Directory containing best-param YAML files")
    parser.add_argument("--output-root", default=os.path.join("outputs", "tuned"), help="Where final runs are saved")
    parser.add_argument("--model-type", choices=["actor_critic", "dqn"], default="actor_critic", help="Which model to run")
    parser.add_argument("--n-seeds", type=int, default=20, help="Seeds per condition")
    parser.add_argument("--start-seed", type=int, default=0, help="Resume from this seed index (skip already-completed seeds)")
    parser.add_argument("--trials", type=int, default=1000, help="Episodes per seed")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    args = parser.parse_args()

    representations = [args.representation] if args.representation else REPRESENTATIONS
    centerings = [args.centering] if args.centering else CENTERINGS

    if args.model_type == "dqn":
        trial_runner = DQNTrial()
        base_params = dict(DQN_BASE_PARAMS)
    else:
        trial_runner = ACTrial()
        base_params = dict(BASE_PARAMS)

    total = len(representations) * len(centerings)
    done = 0

    for representation in representations:
        for centering in centerings:
            done += 1
            params = dict(base_params)
            params.update(load_best_params(args.best_dir, representation, centering, args.model_type))
            params["trials"] = args.trials
            params["learnTrials"] = args.trials
            params["steps"] = args.steps

            data_dir = output_dir(args.output_root, args.model_type, params)
            os.makedirs(data_dir, exist_ok=True)

            print(f"\n[{done}/{total}] representation={representation} centering={centering}")
            print(f"  data_dir={data_dir}")
            print(f"  params={params}")

            terminal_rewards = []
            for seed in range(args.start_seed, args.n_seeds):
                metadata = trial_runner.run(
                    seed=seed,
                    data_dir=data_dir,
                    pre_comment=f"tuned rep={representation}, centering={centering}, seed={seed}",
                    **params,
                )
                terminal_rewards.append(metadata["terminal_reward"])
                print(
                    f"  seed={seed} terminal_reward={metadata['terminal_reward']:.1f} "
                    f"episodes_to_learn={metadata['episodes_to_learn']}"
                )

            print(f"  mean terminal reward over {args.n_seeds} seeds: {float(np.mean(terminal_rewards)):.1f}")

    print(f"\nTuned ablation complete: {total} configurations run.")


if __name__ == "__main__":
    main()
