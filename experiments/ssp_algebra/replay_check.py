"""Regression replay: does the CURRENT code reproduce rows stored by an earlier run, bit for bit?

    OMP_NUM_THREADS=1 ../../.venv/bin/python replay_check.py [--tol 1e-6]

Why this exists (RUN_011 follow-up): between RUN_010 and RUN_011 I edited ``arms_learned.py``, ``ssp_ops.py``,
``harness.py`` and ``tasks_*.py``.  Nothing in the folder is committed, so there is no earlier version to diff against;
the only evidence that those edits left old behaviour unchanged is that stored result rows can be *regenerated*.  Every
stored row carries its own ``cfg``, ``seed``, ``budget``, ``steps`` and ``eval_steps``, and a CPU run at one thread is
deterministic, so a replay that matches proves the default code paths are unchanged, and a mismatch is a regression
that must be bisected (in a scratch copy, never in place) before any new number is trusted.

What is replayed (each row is a different code path):
  * ``s2_t16_a05_b024``       learned index arm with the width schedule (``SSPIndexLearned``, ``set_width``)
  * ``s1_idx_floor`` / ssp    rung-0 index arm (``arms.SSPArm``) -- untouched code, a control for the harness itself
  * ``s4_r3_aux``             rung-3 pointer arm, default ``halting="learned"`` -- the arm I added ``counter``/``cue_hop`` to
  * ``s1_idx_floor`` / gru    a baseline, same reason as rung 0

The task shape is recovered from the row: index rows from RUN_010 ran at ``ep_len=16`` (the default), which is also the
finding that RUN_010's "T=24/32/48/64" rows were really T=16 (tokens per update ~ 512).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

import harness as H
import run_sweep as RS

torch.set_num_threads(1)          # `run_sweep.run_job` does this in its worker; without it the replay is not like-for-like
try:
    torch.set_num_interop_threads(1)
except RuntimeError:              # already initialised: harmless, the setting is process-wide
    pass

RES = pathlib.Path(__file__).with_name("results")

# (file, arm label filter or None, variant, task_kw, robust)
CASES = [
    ("sweep_s2_t16_a05_b024.jsonl", None, "index", {"ep_len": 16}, False),
    ("sweep_s1_idx_floor.jsonl", "ssp", "index", {"ep_len": 16}, False),
    ("sweep_s1_idx_floor.jsonl", "gru", "index", {"ep_len": 16}, False),
    ("sweep_s4_r3_aux.jsonl", None, "pointer", {}, True),
]


def pick_row(path, arm_filter):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("phase") == "final" and (arm_filter is None or r["arm"] == arm_filter)]
    rows.sort(key=lambda r: r["seed"])
    return rows[0]


def replay(row, variant, task_kw, robust):
    task = RS.make_task(variant, **task_kw)
    sp = H.splits_for(variant, task, robust=robust)
    new = H.run_one(row.get("arm_real", row["arm"]), task, sp, row["budget"], row["cfg"], row["seed"], row["steps"],
                    64, 300, eval_steps=row.get("eval_steps"), pool=row.get("pool"))
    return new


def compare(old, new, tol):
    worst, where = 0.0, None
    if len(old["curve"]) != len(new["curve"]):
        return float("inf"), "curve length %d vs %d" % (len(old["curve"]), len(new["curve"]))
    for a, b in zip(old["curve"], new["curve"]):
        for k in a:
            if k in ("step", "tokens"):
                if a[k] != b.get(k):
                    return float("inf"), f"{k} {a[k]} vs {b.get(k)} at step {a['step']}"
                continue
            d = abs(float(a[k]) - float(b[k]))
            if d > worst:
                worst, where = d, f"{k} at step {a['step']}"
    return worst, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--only", default=None, help="substring of the file name to restrict the replay")
    ap.add_argument("--no-control", action="store_true", help="skip the same-code replay-vs-replay control")
    args = ap.parse_args()
    bad = 0
    for fname, arm_filter, variant, task_kw, robust in CASES:
        if args.only and args.only not in fname:
            continue
        old = pick_row(RES / fname, arm_filter)
        new = replay(old, variant, task_kw, robust)
        again = None if args.no_control else replay(old, variant, task_kw, robust)
        worst, where = compare(old, new, args.tol)
        ok = worst <= args.tol
        bad += (not ok)
        ctrl = float("nan")
        if again is not None:
            ctrl, _ = compare(new, again, args.tol)
        print(f"{'OK  ' if ok else 'DIFF'} {fname:32s} arm={old['arm']:34s} seed={old['seed']} "
              f"stored-vs-replay={worst:.3e} ({where})  replay-vs-replay={ctrl:.3e}  "
              f"conv_id old {old['summary']['conv_id']:.6f} new {new['summary']['conv_id']:.6f}",
              flush=True)
    print("replay:", "ALL ROWS REPRODUCED" if not bad else
          f"{bad} row(s) differ -- compare against the replay-vs-replay column before calling it a regression")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
