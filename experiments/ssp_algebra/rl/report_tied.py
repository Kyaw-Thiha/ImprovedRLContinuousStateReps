"""Tables for the tied-addressing round: rl_tied_* runs next to the first pilot's gru / ssp / ssp_aux / fwp (rl_pilot_*).

    /tmp/popgym_ref/bin/python report_tied.py --horizon 1000000 --horizon 2000000 --horizon 3000000
"""
import argparse
import glob
import json
import os

import numpy as np

import report as R

ENV = "RepeatPreviousEasy"


def load_prefix(prefix):
    runs = {}
    for f in sorted(glob.glob(os.path.join(R.RES, f"{prefix}_{ENV}_*_s*.jsonl"))):
        base = os.path.basename(f)[len(f"{prefix}_{ENV}_"):-len(".jsonl")]
        arm, seed = base.rsplit("_s", 1)
        rows = [json.loads(l) for l in open(f) if l.strip()]
        it = [r for r in rows if r["row"] == "iter" and r["episode_return_mean"] is not None]
        if it:
            runs.setdefault(arm, {})[int(seed)] = it
    return runs


def trajectory(runs, key="lag_bias"):
    out = {}
    for arm, seeds in runs.items():
        for s, rows in sorted(seeds.items()):
            pts = [(r["timesteps"], r["addr"][key]) for r in rows if r.get("addr") and key in r["addr"]]
            if pts:
                out[(arm, s)] = pts
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=float, action="append")
    p.add_argument("--prefix", default="rl_tied", help="result-file prefix to load (e.g. rl_bank for the lag-bank round)")
    args = p.parse_args()
    ev = json.load(open(os.path.join(R.RES, "rl_env_returns.json")))[ENV]
    rnd, opt = ev["random_mean"], ev["optimal_mean"]
    thr = {"T50": rnd + 0.5 * (opt - rnd), "T90": rnd + 0.9 * (opt - rnd)}
    base = load_prefix("rl_pilot")
    tied = load_prefix(args.prefix)
    runs = {"gru": base["gru"], "ssp (untied, pilot)": base["ssp"], "ssp_aux (untied, pilot)": base["ssp_aux"], **tied}
    order = ["gru", "ssp (untied, pilot)", "ssp_aux (untied, pilot)"]
    order += [k for k in ("untied_scaled", "tied", "tied_aux", "tied_lag4", "tied_bank", "tied_bank_aux") if k in tied]
    order += [k for k in tied if k not in order]
    out = {"thresholds": thr, "random": rnd, "optimal": opt, "horizons": {}, "trajectory": {}}
    for H in args.horizon or [1e6]:
        tab, summ = R.table(runs, H, thr, arms=[a for a in order if a in runs])
        print(f"\n## horizon {H:g}\n\n{tab}\n")
        for arm, v in summ.items():
            print(f"{arm}: final {[round(m['final'],3) for m in v['per_seed'].values()]}  "
                  f"T50 {[None if m['steps_to_T50'] is None else int(m['steps_to_T50']/1e3) for m in v['per_seed'].values()]}k  "
                  f"T90 {[None if m['steps_to_T90'] is None else int(m['steps_to_T90']/1e3) for m in v['per_seed'].values()]}k  "
                  f"aulc {[round(m['aulc'],3) for m in v['per_seed'].values()]}")
        out["horizons"][str(int(H))] = summ
    print("\n## address-bias (lag) trajectory, index units; ramp_w for the untied pilot arms is the ramp weight")
    for (arm, s), pts in sorted(trajectory(tied).items()):
        sel = [pts[i] for i in sorted({0, len(pts) // 4, len(pts) // 2, 3 * len(pts) // 4, len(pts) - 1})]
        allv = [v for _, v in pts]
        print(f"{arm} s{s}: " + "  ".join(f"{t//1000}k:{v:+.3f}" for t, v in sel) +
              f"   range [{min(allv):+.3f},{max(allv):+.3f}]")
        out["trajectory"][f"{arm}_s{s}"] = pts
    print("\n## lag probe (h=0, t=25) and |lag head w| at the last row")
    for (arm, s), rows in sorted({(a, s): r for a, d in tied.items() for s, r in d.items()}.items()):
        ad = rows[-1].get("addr") or {}
        print(arm, s, {k: round(v, 3) for k, v in ad.items()})
    json.dump(out, open(os.path.join(R.RES, f"{args.prefix}_RepeatPreviousEasy_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
