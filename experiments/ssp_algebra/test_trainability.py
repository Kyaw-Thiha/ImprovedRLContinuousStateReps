"""Baseline trainability gate (handoff 09 section 5.4, step C).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_trainability.py

RUN_007's `tf_full` reached only 0.37 on index recall and 0.54 on delay -- tasks it can represent trivially -- so
there was no valid upper-bound baseline (RUN_007 section 4.5, bug 7.14/limitation 2).  A number from an arm that
cannot pass its own trainability gate must not be used in any claim.

The gate is deliberately generous: an *easy* version of the task (ep_len 8), 800 updates, and the arm gets several
learning rates, because the question is "can this architecture learn this at all", not "with the one config that
happened to be drawn".  The threshold (0.95 in-distribution) is never lowered; if an arm fails, its optimisation is
fixed instead.
"""

import json
import pathlib

import numpy as np
import torch

import arms as A
import harness as H
import tasks_c

THRESHOLD = 0.95
LRS = (1e-2, 3e-3, 1e-3)


def _try(arm_name, steps=800, ep_len=8, batch=64):
    task = tasks_c.IndexTask(n_items=8, ep_len=ep_len)
    sp = H.splits_for("index", task)
    best = (None, -1.0, None)
    for lr in LRS:
        arm, info = H.build_any(arm_name, task, 256, 0, {"hidden": 64})
        curve, _ = H.train_arm(arm, task, sp, seed=0, steps=steps, batch=batch, lr=lr, eval_every=steps)
        got = curve[-1]["id"]
        if got > best[1]:
            best = (lr, got, info["knob"])
        print(f"    {arm_name} lr={lr:g} -> id={got:.3f}", flush=True)
    return best


def test_baselines_are_trainable(arms=("gru", "tf_full", "fwp")):
    rows = []
    for arm_name in arms:
        lr, got, knob = _try(arm_name)
        rows.append((arm_name, lr, got, knob))
        print(f"  {arm_name}: best id={got:.3f} at lr={lr:g} ({knob})")
    bad = [r for r in rows if r[2] < THRESHOLD]
    # record the measurement so `analysis.py` can exclude a baseline that failed instead of silently using it
    out = pathlib.Path(__file__).with_name("results") / "trainability.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"threshold": THRESHOLD, "steps": 800, "ep_len": 8,
                               "arms": {n: {"best_id": float(g), "lr": lr, "knob": knob, "passed": bool(g >= THRESHOLD)}
                                        for n, lr, g, knob in rows}}, indent=2))
    print(f"  -> {out}")
    assert not bad, ("baseline(s) failed the trainability gate on easy index: "
                     + ", ".join(f"{n} id={g:.3f}" for n, _, g, _ in bad))
    return rows


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    print(f"  trainability gate: ep_len=8, 800 updates, threshold id >= {THRESHOLD}")
    try:
        test_baselines_are_trainable()
        print("PASS baselines_are_trainable")
    except AssertionError as e:
        print(f"FAIL baselines_are_trainable: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
