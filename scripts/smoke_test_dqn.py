"""
DQN smoke test — runs a hand-picked hyperparameter configuration across
all four representations × N seeds, on classic CartPole-v1.

Goal: prove DQN learns at all reps *before* committing compute to a full
Optuna sweep. Used as a gating check; supports retry with different
hyperparameters to find a working config.

Pass criterion (per attempt): at least 2 of 4 reps hit terminal_reward > 200.

Post-fix defaults (2026-09-22): target_update_freq=25, learning_starts=200,
eps=0.05 — these are the values identified by the smoke-test investigation
as resolving the prior 8-10 reward collapse. Use --attempt to vary on retries.

train_freq=4 with target_update_freq=25 keeps target sync cadence constant in
env-time (25 grad-steps * 4 env-steps/grad = 100 env-steps between target syncs,
matching the pre-change baseline of 1 grad-step * 100 env-steps = 100 env-steps).

Usage:
  .venv/bin/python scripts/smoke_test_dqn.py --attempt 1
  .venv/bin/python scripts/smoke_test_dqn.py --attempt 2 --lr 1e-3
  .venv/bin/python scripts/smoke_test_dqn.py --attempt 3 --lr 2.5e-4 \\
      --target-update-freq 50 --learning-starts 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments"))

from experiments.trial_cartpole_dqn import DQNTrial


BASE = dict(
    trials=1000,
    steps=500,
    env="CartPole-v1",
    learnTrials=1000,
    dynamic_epsilon=False,  # disable the buggy decay schedule; use fixed eps
    state_dis=0.99,
    reward_center_beta=0.001,
    reward_center_eta=1.0,
    reward_center_init=0.0,
    verbose=False,
    gifs=False,
)

# Post-fix DQN defaults (2026-09-22).
COMMON_DQN_DEFAULTS = dict(
    lr=5e-4,
    eps=0.05,
    buffer_size=50000,
    batch_size=64,
    target_update_freq=25,
    learning_starts=200,
    train_freq=4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    max_grad_norm=10.0,
)

PER_REP = {
    "normal": dict(rep_="Normal"),
    "discrete": dict(rep_="Discrete", n_bins=7),
    "ssp": dict(rep_="PlaceSSP", length_scale=0.4, n_rotates=8, n_scales=5),
    "tile_coding": dict(
        rep_="TileCoding",
        num_tilings=8,
        tiles_per_dim=(12, 12, 12, 12),
        iht_size=4096,
        tile_state_indices=(0, 1, 2, 3),
    ),
}

PASS_THRESHOLD = 200.0   # terminal rolling-100 reward
PASS_MIN_REPS = 2        # require at least N of 4 reps to pass


def run_one(rep: str, seed: int, data_dir: str, overrides: dict, rep_overrides: dict) -> dict:
    params = dict(BASE)
    common = dict(COMMON_DQN_DEFAULTS)
    common.update(overrides)
    params.update(common)
    params.update(rep_overrides.get(rep, PER_REP[rep]))
    params["reward_center_mode"] = "none"

    trial = DQNTrial()
    start = time.time()
    metadata = trial.run(
        seed=seed,
        data_dir=data_dir,
        pre_comment=f"smoke-test rep={rep} seed={seed}",
        **params,
    )
    elapsed = time.time() - start
    return {
        "rep": rep,
        "seed": seed,
        "terminal_reward": metadata["terminal_reward"],
        "episodes_to_learn": metadata.get("episodes_to_learn"),
        "build_time": metadata["build_time"],
        "total_time": metadata["total_time"],
        "avg_trial_time": metadata["avg_trial_time"],
        "elapsed": elapsed,
        "dimensionality": metadata["dimensionality"],
    }


def main():
    parser = argparse.ArgumentParser(description="DQN smoke test with retry")
    parser.add_argument("--attempt", type=int, default=1,
                        help="Attempt number (used to namespace outputs)")
    parser.add_argument("--reps", nargs="+",
                        default=["normal", "discrete", "ssp", "tile_coding"],
                        help="Representations to test")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Seeds per rep (default 2 — smoke only, fast)")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Episodes per run (default 1000)")
    parser.add_argument("--data-root", default=os.path.join("outputs", "smoke_test_dqn"))
    parser.add_argument("--out", default=None, help="JSON summary path")
    # Hyperparam overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-update-freq", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD)
    parser.add_argument("--pass-min-reps", type=int, default=PASS_MIN_REPS)
    # Per-rep param overrides (apply to whichever rep matches the key)
    parser.add_argument("--rep-config", action="append", default=[],
                        help="Per-rep JSON override, e.g. --rep-config '{\"discrete\":{\"n_bins\":5}}'. Repeatable.")
    args = parser.parse_args()

    BASE["trials"] = args.trials

    # Apply overrides
    overrides = {}
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.eps is not None:
        overrides["eps"] = args.eps
    if args.buffer_size is not None:
        overrides["buffer_size"] = args.buffer_size
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.target_update_freq is not None:
        overrides["target_update_freq"] = args.target_update_freq
    if args.learning_starts is not None:
        overrides["learning_starts"] = args.learning_starts

    # Per-rep config overrides (e.g. --rep-config '{"discrete":{"n_bins":5}}')
    import json as _json
    rep_overrides = {}
    for spec in args.rep_config:
        try:
            data = _json.loads(spec)
        except Exception as e:
            print(f"WARNING: could not parse --rep-config '{spec}': {e}")
            continue
        for k, v in data.items():
            rep_overrides[k] = dict(PER_REP.get(k, {}))
            rep_overrides[k].update(v)

    data_dir = os.path.join(args.data_root, f"attempt_{args.attempt}")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print(f"DQN smoke test — attempt {args.attempt}")
    print("=" * 70)
    print(f"reps           : {args.reps}")
    print(f"seeds/rep      : {args.seeds}")
    print(f"episodes       : {BASE['trials']}  (pass: rolling-100 > {args.pass_threshold} on ≥{args.pass_min_reps} reps)")
    print(f"data dir       : {data_dir}")
    print(f"overrides      : {overrides if overrides else 'none (using post-fix defaults)'}")
    print()

    results = []
    overall_start = time.time()
    for rep in args.reps:
        for seed in range(args.seeds):
            print(f"[{rep}/seed{seed}] running...", flush=True)
            try:
                r = run_one(rep, seed, data_dir, overrides, rep_overrides)
            except Exception as e:
                print(f"[{rep}/seed{seed}] FAILED: {type(e).__name__}: {e}")
                r = {
                    "rep": rep, "seed": seed,
                    "terminal_reward": float("nan"),
                    "episodes_to_learn": None,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed": 0.0,
                    "dimensionality": -1,
                }
            print(
                f"[{rep}/seed{seed}] done in {r['elapsed']:.1f}s | "
                f"terminal_reward={r['terminal_reward']:.2f}  "
                f"episodes_to_learn={r['episodes_to_learn']}  "
                f"phi_dim={r['dimensionality']}"
            )
            results.append(r)

    overall_elapsed = time.time() - overall_start

    print()
    print("=" * 70)
    print(f"Summary — attempt {args.attempt}")
    print("=" * 70)
    by_rep = {}
    for r in results:
        by_rep.setdefault(r["rep"], []).append(r["terminal_reward"])

    passed_reps = []
    for rep, rewards in by_rep.items():
        arr = np.array(rewards, dtype=float)
        # Treat NaN as failed
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            print(f"  [FAIL] {rep:12s}  ALL RUNS FAILED")
            continue
        mean, std = float(valid.mean()), float(valid.std())
        per_rep_pass = bool((valid > args.pass_threshold).all())
        marker = "PASS" if per_rep_pass else "FAIL"
        if per_rep_pass:
            passed_reps.append(rep)
        print(f"  [{marker}] {rep:12s}  mean={mean:7.2f}  std={std:6.2f}  rewards={[f'{x:.1f}' for x in valid]}")

    print()
    print(f"Total elapsed: {overall_elapsed:.1f}s")
    print(f"Passed reps: {len(passed_reps)}/{len(by_rep)} (need ≥{args.pass_min_reps})")

    overall_pass = len(passed_reps) >= args.pass_min_reps
    if overall_pass:
        print(f"OVERALL: PASS — {len(passed_reps)} reps cleared threshold. Proceed to Optuna sweep.")
    else:
        print(f"OVERALL: FAIL — only {len(passed_reps)} rep(s) cleared threshold.")
        print(f"         Consider adjusting --lr, --target-update-freq, or --learning-starts on next attempt.")

    summary = {
        "attempt": args.attempt,
        "config": {
            "reps": args.reps, "seeds": args.seeds, "episodes": args.trials,
            "pass_threshold": args.pass_threshold,
            "pass_min_reps": args.pass_min_reps,
            "overrides": overrides,
        },
        "results": results,
        "passed_reps": passed_reps,
        "passed": overall_pass,
        "total_elapsed_s": overall_elapsed,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to {args.out}")
    else:
        default_out = os.path.join(args.data_root, f"attempt_{args.attempt}.json")
        with open(default_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to {default_out}")

    sys.exit(0 if overall_pass else 2)


if __name__ == "__main__":
    main()
