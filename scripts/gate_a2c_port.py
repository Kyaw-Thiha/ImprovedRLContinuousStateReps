"""Validation gate for the NumPy A2C port: old winner params, fresh seeds, OLD online metric.

For each cell, runs the legacy winner config (outputs/hparam_search/best/<rep>_<centering>.yaml, read only)
on N fresh seeds with the port and scores each run with the old metric (mean return over the last 100 training
episodes, epsilon exploration included). The old `best_value` in the yaml is the mean of 10 re-evaluation
seeds of the best-of-top-10 trial, so it is slightly optimistic.

Pass rule (fixed in advance): the old value lies inside the bootstrap 95% CI of the port's mean over seeds,
OR |port mean - old value| <= 2 standard errors.

Local only, writes nothing except stdout:
    OMP_NUM_THREADS=1 .venv/bin/python scripts/gate_a2c_port.py --cells ssp/none,tile_coding/none
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CELLS = "ssp/none,tile_coding/none,discrete/none,tile_coding/value"


def _run(job):
    from experiments.trial_cartpole_a2c import load_legacy_winner, train_a2c

    rep, cen, seed, trials = job
    params, old = load_legacy_winner(rep, cen)
    params["trials"] = trials
    params["learnTrials"] = trials
    out = train_a2c(params, seed=seed)
    return rep, cen, seed, out["terminal_reward"], out["train_time"], out["env_steps"], out["final_eps"]


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=1000, help="old re-evaluation used seeds 1000+")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    from experiments.trial_cartpole_a2c import load_legacy_winner

    cells = [tuple(c.split("/")) for c in args.cells.split(",")]
    jobs = [(r, c, args.seed_offset + i, args.trials) for r, c in cells for i in range(args.seeds)]
    results = {cell: [] for cell in cells}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for f in as_completed(futs):
            rep, cen, seed, score, secs, steps, eps = f.result()
            results[(rep, cen)].append((seed, score, secs, steps, eps))
            print(f"  {rep}/{cen} seed {seed}: online={score:6.1f}  {secs:5.1f}s  {steps} env steps  final eps={eps:.3f}", flush=True)
    print(f"\nwall {time.time() - t0:.0f}s with {args.workers} workers\n")

    all_pass = True
    for (rep, cen), rows in results.items():
        scores = np.array([r[1] for r in sorted(rows)])
        _, old = load_legacy_winner(rep, cen)
        mean, sd = scores.mean(), scores.std()
        se = sd / np.sqrt(len(scores))
        lo, hi = bootstrap_ci(scores)
        ok = (lo <= old <= hi) or abs(mean - old) <= 2 * se
        all_pass &= ok
        print(
            f"{rep}/{cen}: port {mean:6.1f} +/- {sd:5.1f} (SE {se:4.1f}, 95% CI [{lo:.1f}, {hi:.1f}])  "
            f"old {old:6.1f}  diff {mean - old:+6.1f}  -> {'PASS' if ok else 'FAIL'}"
        )
        print(f"    per-seed: {np.round(scores, 0).astype(int).tolist()}")
    print("\nGATE:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
