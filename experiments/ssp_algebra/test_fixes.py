"""Checks added after the RUN_007 review (docs/handoffs/09).  ``test_harness.py`` is intentionally left untouched.

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_fixes.py

Each check was run on the broken code first (RED) and must pass on the fixed code.  They never loosen a threshold:
if a check fails, the code is wrong.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import torch
import torch.nn as nn

import arms as A
import harness as H
import tasks_b
import tasks_c

HERE = os.path.dirname(os.path.abspath(__file__))


def test_seed_stability():
    """The config draw for a given (variant, arm, cfg_idx) must not depend on the process's hash salt."""
    snippet = textwrap.dedent("""
        import json, sys, types
        sys.path.insert(0, %r)
        import run_sweep as R
        args = types.SimpleNamespace(variants=["index", "pointer"], arms=["gru", "ssp"], budget=256, tag="t",
                                     steps=10, batch=8, eval_every=5, task_kw={})
        jobs = R.build_search_jobs(args, 3)
        print(json.dumps([j["cfg"] for j in jobs], sort_keys=True))
    """ % HERE)
    outs = []
    for salt in ("1", "2", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt, OMP_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-400:]
        outs.append(r.stdout.strip().splitlines()[-1])
    same = len(set(outs)) == 1
    print(f"  config draws identical across hash salts = {same}")
    assert same, "config draw depends on PYTHONHASHSEED: resuming a partial search in a new process would mix configs"


class _Pass(nn.Module):
    """Replaces the readout so the logits ARE the cleanup scores (the first ``n`` features of the readout input)."""

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[..., : self.n]


def test_pointer_oracle_readout(min_acc=0.99):
    """With exact codes, no forgetting and a pass-through readout, the arm must answer f^h(s) for every h in 1..8."""
    task = tasks_b.PointerTask(n_nodes=8, n_query=8, train_hops=(1, 2), max_hops=8)
    arm, _ = A.build_arm("ssp", task, budget=512, seed=0, decay=False, hidden=8)
    arm.mix = _Pass(task.n_nodes)
    rng = np.random.default_rng(0)
    worst, rows = 1.0, []
    for h in range(1, 9):
        b = task.batch(64, task.ep_len, rng=rng, hops=(h,))
        with torch.no_grad():
            logits = arm(b, task.features(b))
        pred = logits.argmax(-1)
        m = b["mask"]
        acc = ((pred == b["target"]) & m).sum().item() / m.sum().item()
        rows.append(f"h{h}={acc:.2f}")
        worst = min(worst, acc)
    print("  oracle readout by hop: " + " ".join(rows))
    assert worst >= min_acc, f"the hand-wired pointer arm does not compute f^h(s): worst hop accuracy {worst:.3f}"


def main():
    torch.set_num_threads(1)
    checks = [("seed_stability", test_seed_stability),
              ("pointer_oracle_readout", test_pointer_oracle_readout)]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
