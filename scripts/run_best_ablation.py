"""
Run the full 3x3 ablation grid using Optuna-selected params per condition.

Expected input files are produced by scripts/run_hparam_search_a2c_nengo.py (A2C) or
scripts/run_hparam_search_dqn.py (DQN):
    outputs/hparam_search/best/{representation}_{centering}.yaml

Usage:
    python scripts/run_best_ablation.py
    python scripts/run_best_ablation.py --n-seeds 20
    python scripts/run_best_ablation.py --representation discrete --centering value
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

from experiments.trial_cartpole_dqn import DQNTrial
# BEST_DIR is the same directory the A2C search uses; importing it from the DQN script avoids
# pulling in the Nengo-dependent A2C modules (they're imported lazily in main() when needed).
from run_hparam_search_dqn import BASE_PARAMS as DQN_BASE_PARAMS, BEST_DIR


REPRESENTATIONS = ["discrete", "ssp", "tile_coding"]
CENTERINGS = ["none", "simple", "value"]


def load_best_params(best_dir: str, representation: str, centering: str, model_type: str = "actor_critic",
                     architecture: str = "linear", variant: str = "") -> dict:
    prefix = "dqn_" if model_type == "dqn" else ""
    # Same naming rule as run_hparam_search_dqn._best_filename: "linear" has no suffix,
    # "deep" is suffixed, `variant` (e.g. "_v2") goes last.
    arch_suffix = "_deep" if (model_type == "dqn" and architecture == "deep") else ""
    var_suffix = variant if model_type == "dqn" else ""
    best_path = os.path.join(best_dir, f"{prefix}{representation}_{centering}{arch_suffix}{var_suffix}.yaml")
    if not os.path.exists(best_path):
        search_script = "run_hparam_search_dqn.py" if model_type == "dqn" else "run_hparam_search_a2c_nengo.py"
        raise FileNotFoundError(
            f"Missing tuned params: {best_path}. Run scripts/{search_script} "
            f"--representation {representation} --centering {centering} first."
        )

    with open(best_path) as f:
        payload = yaml.safe_load(f)

    return payload["params"]


def output_dir(output_root: str, model_type: str, params: dict) -> str:
    # Deep DQN gets its own directory so it never collides with linear DQN results.
    if model_type == "dqn" and params.get("architecture") == "deep":
        model_type = "dqn_deep"
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
    parser.add_argument("--architecture", choices=["linear", "deep"], default="linear",
                        help="DQN only: which architecture's tuned params to load and run")
    parser.add_argument("--variant", default="",
                        help="DQN only: best-params filename suffix, e.g. _v2 for the greedy re-filtered winners")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Added to every seed. For the DQN final eval use >= 2000 "
                             "(search used 0..499, reeval 1000+) so seeds are fresh")
    parser.add_argument("--n-evaluations", type=int, default=15, help="DQN only: greedy-eval checkpoints per run")
    parser.add_argument("--n-eval-episodes", type=int, default=10, help="DQN only: episodes per checkpoint")
    parser.add_argument("--final-eval-episodes", type=int, default=100,
                        help="DQN only: greedy episodes of the final policy after training")
    parser.add_argument("--last-k", type=int, default=5,
                        help="DQN only: checkpoints averaged for the primary greedy score")
    parser.add_argument("--fixed-init", action="store_true",
                        help="DQN only: use the legacy fixed init (seed 0) for every run, as the search did")
    args = parser.parse_args()

    representations = [args.representation] if args.representation else REPRESENTATIONS
    centerings = [args.centering] if args.centering else CENTERINGS

    if args.model_type == "dqn":
        trial_runner = DQNTrial()
        base_params = dict(DQN_BASE_PARAMS)
    else:
        # Legacy A2C path only: these import Nengo, which the Stage-0 env does not install.
        from experiments.trial_cartpole_nengo import ACTrial
        from run_hparam_search_a2c_nengo import BASE_PARAMS
        trial_runner = ACTrial()
        base_params = dict(BASE_PARAMS)

    total = len(representations) * len(centerings)
    done = 0

    for representation in representations:
        for centering in centerings:
            done += 1
            params = dict(base_params)
            params.update(load_best_params(args.best_dir, representation, centering, args.model_type,
                                           args.architecture, args.variant))
            params["trials"] = args.trials
            params["learnTrials"] = args.trials
            params["steps"] = args.steps
            if args.model_type == "dqn":
                # Greedy eval ON (the plain DQN path never set eval_freq, so it only produced the
                # online return). Same eval_freq rule as the search.
                params["eval_freq"] = max(1, args.trials // args.n_evaluations)
                params["n_eval_episodes"] = args.n_eval_episodes
                params["final_eval_episodes"] = args.final_eval_episodes
                # Initial weights follow the run seed unless --fixed-init (the search used fixed init=0).
                params["init_seed"] = 0 if args.fixed_init else -1

            data_dir = output_dir(args.output_root, args.model_type, params)
            os.makedirs(data_dir, exist_ok=True)

            print(f"\n[{done}/{total}] representation={representation} centering={centering}")
            print(f"  data_dir={data_dir}")
            print(f"  params={params}")

            terminal_rewards = []
            seed_log = os.path.join(data_dir, "final_eval_seeds.jsonl")
            for seed in range(args.start_seed, args.n_seeds):
                metadata = trial_runner.run(
                    seed=args.seed_offset + seed,
                    data_dir=data_dir,
                    pre_comment=f"tuned rep={representation}, centering={centering}, seed={seed}",
                    **params,
                )
                terminal_rewards.append(metadata["terminal_reward"])
                if args.model_type == "dqn":
                    # Primary metric = mean of the last K greedy checkpoints (same definition as the
                    # search ranking and the reeval); also keep last-1, the long final-policy eval,
                    # the online return and cost metadata. See docs/handoffs/02-*.md.
                    hist = [float(x) for x in metadata["eval_history"]]
                    rec = {
                        "seed": args.seed_offset + seed,
                        "score_lastk": float(np.mean(hist[-args.last_k:])),
                        "last1": hist[-1],
                        "final_eval_reward": float(metadata["final_eval_reward"]),
                        "terminal_reward_online": float(metadata["terminal_reward"]),
                        "eval_history": hist,
                        "total_time": float(metadata["total_time"]),
                        "train_time": float(metadata["train_time"]),
                        "eval_time": float(metadata["eval_time"]),
                        "env_steps": int(metadata["env_steps"]),
                        "n_updates": int(metadata["n_updates"]),
                        "n_params": int(metadata["n_params"]),
                        "dimensionality": int(metadata["dimensionality"]),
                    }
                    with open(seed_log, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"  seed={rec['seed']} greedy last-{args.last_k}={rec['score_lastk']:.1f} "
                          f"last1={rec['last1']:.1f} final_eval={rec['final_eval_reward']:.1f} "
                          f"online={rec['terminal_reward_online']:.1f}")
                else:
                    print(
                        f"  seed={seed} terminal_reward={metadata['terminal_reward']:.1f} "
                        f"episodes_to_learn={metadata['episodes_to_learn']}"
                    )

            if args.model_type == "dqn" and os.path.exists(seed_log):
                with open(seed_log) as f:
                    recs = [json.loads(line) for line in f if line.strip()]
                sc = [r["score_lastk"] for r in recs]
                print(f"  greedy last-{args.last_k} over {len(recs)} seeds: "
                      f"mean={float(np.mean(sc)):.1f} std={float(np.std(sc)):.1f}")
            else:
                print(f"  mean terminal reward over {args.n_seeds} seeds: {float(np.mean(terminal_rewards)):.1f}")

    print(f"\nTuned ablation complete: {total} configurations run.")


if __name__ == "__main__":
    main()
