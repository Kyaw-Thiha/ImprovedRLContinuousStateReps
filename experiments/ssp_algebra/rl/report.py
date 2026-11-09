"""Aggregate results/rl_pilot_<env>_<arm>_s<seed>.jsonl into a table (markdown + JSON).

    python report.py [--horizon 1000000] [--horizon 3000000] [--env RepeatPreviousEasy]

Definitions (all from the curve rows: one row per PPO iteration, ``episode_return_mean`` = RLlib's mean over the last 100
episodes, i.e. POPGym's own reported quantity):
  final      mean over the rows with 0.9*H < timesteps <= H  (the last 10% of the first H steps), then mean +- sample std over seeds
  steps_to_T first row whose return >= T = random + frac*(optimal - random), per seed; "k/n" seeds reached; median over reaching seeds
  aulc       trapezoid area of return vs steps over [0, H] divided by H (mean return along the curve; the t=0 point is the
             first row's value), mean +- std over seeds
  wall       cumulative wall seconds at the last row with timesteps <= H (6 runs shared the machine: contended wall time)
"""

import argparse
import glob
import json
import os
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ARM_ORDER = ["gru", "fwp", "ssp", "ssp_aux", "ssp_oracle4"]


def load(env):
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, f"rl_pilot_{env}_*_s*.jsonl"))):
        base = os.path.basename(f)[len(f"rl_pilot_{env}_"):-len(".jsonl")]
        arm, seed = base.rsplit("_s", 1)
        rows = [json.loads(l) for l in open(f) if l.strip()]
        it = [r for r in rows if r["row"] == "iter" and r["episode_return_mean"] is not None]
        if it:
            runs.setdefault(arm, {})[int(seed)] = it
    return runs


def metrics(rows, H, thr):
    rows = [r for r in rows if r["timesteps"] <= H]
    if not rows:
        return None
    ts = np.array([0.0] + [r["timesteps"] for r in rows], float)
    y = np.array([rows[0]["episode_return_mean"]] + [r["episode_return_mean"] for r in rows], float)
    reached = ts[-1]
    last = [r["episode_return_mean"] for r in rows if r["timesteps"] > 0.9 * H]
    out = {"reached_steps": float(reached), "complete": bool(reached >= 0.95 * H), "final": float(np.mean(last)) if last else None,
           "aulc": float(np.trapezoid(y, ts) / reached), "wall_s": float(rows[-1]["wall_s"]), "mmer": float(y.max())}
    for name, t in thr.items():
        hit = [r["timesteps"] for r in rows if r["episode_return_mean"] >= t]
        out["steps_to_" + name] = float(hit[0]) if hit else None
    return out


def table(runs, H, thr, arms=None):
    lines, summ = [], {}
    lines.append(f"| arm | seeds done | final return (last 10% of {H/1e6:g}M) mean ± std | steps to {thr['T50']:.3f} (50%) | steps to {thr['T90']:.3f} (90%) | AULC | wall to H (h, contended) | steps/s | MMER |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arm in (arms or [a for a in ARM_ORDER if a in runs]):
        ms = {s: metrics(r, H, thr) for s, r in sorted(runs[arm].items())}
        ms = {s: m for s, m in ms.items() if m and m["complete"]}
        if not ms:
            continue
        fin = [m["final"] for m in ms.values()]
        au = [m["aulc"] for m in ms.values()]
        wall = [m["wall_s"] for m in ms.values()]
        mm = [m["mmer"] for m in ms.values()]
        sd = lambda v: statistics_std(v)

        def reach(k):
            v = [m[k] for m in ms.values()]
            ok = [x for x in v if x is not None]
            return f"{len(ok)}/{len(v)} (median {st.median(ok)/1e3:.0f}k)" if ok else f"0/{len(v)} (never)"
        lines.append(f"| {arm} | {len(ms)} | {np.mean(fin):+.3f} ± {sd(fin):.3f} | {reach('steps_to_T50')} | {reach('steps_to_T90')} | "
                     f"{np.mean(au):+.3f} ± {sd(au):.3f} | {np.mean(wall)/3600:.2f} | {np.mean([m['reached_steps']/m['wall_s'] for m in ms.values()]):.0f} | {np.mean(mm):+.3f} |")
        summ[arm] = {"seeds": sorted(ms), "per_seed": {str(s): m for s, m in ms.items()},
                     "final_mean": float(np.mean(fin)), "final_std": sd(fin), "aulc_mean": float(np.mean(au)), "aulc_std": sd(au)}
    return "\n".join(lines), summ


def statistics_std(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="RepeatPreviousEasy")
    p.add_argument("--horizon", type=float, action="append")
    args = p.parse_args()
    ev = json.load(open(os.path.join(RES, "rl_env_returns.json")))[args.env]
    rnd, opt = ev["random_mean"], ev["optimal_mean"]
    thr = {"T50": rnd + 0.5 * (opt - rnd), "T90": rnd + 0.9 * (opt - rnd)}
    runs = load(args.env)
    out = {"env": args.env, "random": rnd, "optimal": opt, "thresholds": thr, "horizons": {}}
    for H in args.horizon or [1e6]:
        tab, summ = table(runs, H, thr)
        print(f"\n## horizon {H:g} steps (random {rnd:.3f}, optimal {opt:.3f})\n\n{tab}\n")
        out["horizons"][str(int(H))] = summ
    json.dump(out, open(os.path.join(RES, f"rl_pilot_{args.env}_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
