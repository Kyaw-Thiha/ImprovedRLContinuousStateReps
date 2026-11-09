"""Per-condition summary of hold-out rows from one or more sweep JSONL files (what RUN_008/009 used ad hoc).

    ../../.venv/bin/python summarize_runs.py results/sweep_dg_ws8_s375.jsonl results/sweep_learn3_ws8.jsonl
    ../../.venv/bin/python summarize_runs.py results/sweep_r0fix.jsonl --variant index --splits id extrap_T

One line per (variant, arm label): converged score (``conv_*``, mean of the last 3 checkpoints) +- standard error over hold-out
seeds for each requested split, the per-seed ``id`` scores, the *updates* at which the mean ``id`` curve first reaches 0.90 / 0.99
(coarse: it is the evaluation checkpoint), tokens to 50% (``t50``, only if every seed reaches it), the winning config's lr /
lam_init / temp, and any pretraining tokens.  Only ``phase == "final"`` rows are used (search rows are config draws, not results).
Do NOT pass files that contain the same arm label trained under different protocols (e.g. the 6000-update baselines next to the
1500-update ones): the rows would be merged.
"""

import argparse
import json

import numpy as np


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def ms(v, p=3):
    v = np.array(v, float)
    return f"{v.mean():.{p}f}±{v.std() / np.sqrt(len(v)):.{p}f}"


def updates_to(rr, thr, split="id"):
    steps = [c["step"] for c in rr[0]["curve"]]
    mean = np.array([[c[split] for c in r["curve"]] for r in rr]).mean(0)
    i = next((i for i, v in enumerate(mean) if v >= thr), None)
    return "never" if i is None else str(steps[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--splits", nargs="*", default=["id", "extrap_T", "extrap_lag"])
    ap.add_argument("--phase", default="final")
    a = ap.parse_args()
    groups = {}
    for f in a.files:
        for r in rows(f):
            if r.get("phase") != a.phase or (a.variant and r["variant"] != a.variant):
                continue
            groups.setdefault((r["variant"], r["arm"], f), []).append(r)
    for (variant, arm, f), rr in sorted(groups.items()):
        s = f"{variant}/{arm:34s} n={len(rr)} "
        for sp in a.splits:
            key = "conv_" + sp
            if key in rr[0]["summary"]:
                s += f" {sp}={ms([r['summary'][key] for r in rr])}"
        s += f"  upd->0.9={updates_to(rr, 0.9):>5s} ->0.99={updates_to(rr, 0.99):>5s}"
        t50 = [r["summary"].get("t50_id") for r in rr]
        s += f"  t50={'%.0fk' % (np.mean(t50) / 1e3) if all(t50) else 'not all seeds'}"
        s += f"  seeds_id={[round(r['summary']['conv_id'], 2) for r in rr]}"
        c = rr[0]["cfg"]
        s += "  cfg{" + ",".join(f"{k}={c[k]:.4g}" if isinstance(c[k], float) else f"{k}={c[k]}" for k in ("lr", "lam_init", "temp") if k in c) + "}"
        if rr[0].get("pretrain_tokens"):
            s += f" pretrain_tok={rr[0]['pretrain_tokens']}"
        print(s + f"   <{f.split('/')[-1]}>")


if __name__ == "__main__":
    main()
