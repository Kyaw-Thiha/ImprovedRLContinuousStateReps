"""The default-path guard: every arm configuration that existed before RUN_012 must still produce bit-identical output.

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_default_path_hash.py

Why this file exists.  During RUN_012 every edit to ``arms_learned.py`` / ``ssp_ops.py`` / ``harness.py`` was followed by
a forward-pass hash of the plain ``fwp`` and the rung-1 SSP arm (identity, recipe-prior and random read heads) on index
and delay.  That check caught three silently dropped lines.  The script lived in ``/tmp`` and was lost in a wipe; the
*expected hashes below were recorded before any edit of this round* (printed in the session), so the guard can be
rebuilt without the old code.  It is a verification asset: never change the expected values to make it pass -- a
mismatch means a default path changed, and the right response is to fix the code.

Scope: the hash covers the logits of ``fwp`` (``hidden=16``), ``ssp_index_learned`` with ``recipes.slope_only_cfg()``
(minus ``lr``) and with ``rung=1`` alone, on ``IndexTask(8, 16)`` and ``DelayTask(8, 16, lags 1-4)``, batch 8, rng 0,
seed 0, budget 256.  It does not cover the pointer arm, the new arms (``fwp_delta``, ``assoc_*``) or any non-default
knob; those have their own tests.
"""

import hashlib

import numpy as np
import torch

import arms_learned as L
import recipes
import tasks_c

EXPECTED = {
    "index/fwp": "0b3dcb6ddd496dd9",
    "index/ssp_slope": "df62f23242feec61",
    "index/ssp_rand": "9cb68520043b71cf",
    "delay/fwp": "24d70cbbe744ce4c",
    "delay/ssp_slope": "dce35df374905ed6",
    "delay/ssp_rand": "f9fae7875a2efbcc",
}


def _h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def measure():
    out = {}
    for tname, task in (("index", tasks_c.IndexTask(n_items=8, ep_len=16)),
                        ("delay", tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2, 3, 4)))):
        b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
        cfgs = {"fwp": dict(hidden=16), "ssp_slope": dict(**recipes.slope_only_cfg()), "ssp_rand": dict(rung=1)}
        for k, c in cfgs.items():
            name = "fwp" if k == "fwp" else "ssp_index_learned"
            c = {kk: v for kk, v in c.items() if kk not in ("lr",)}
            arm, _ = L.build_learned(name, task, budget=256, seed=0, **c)
            with torch.no_grad():
                out[f"{tname}/{k}"] = _h(arm(b, task.features(b)))
    return out


def test_default_paths_are_bit_identical():
    got = measure()
    bad = {k: (EXPECTED[k], got[k]) for k in EXPECTED if got.get(k) != EXPECTED[k]}
    for k in EXPECTED:
        print(f"  {k:18s} expected {EXPECTED[k]}  got {got[k]}  {'ok' if k not in bad else 'MISMATCH'}")
    assert not bad, f"a default path changed: {bad}"


if __name__ == "__main__":
    test_default_paths_are_bit_identical()
    print("test_default_path_hash: all 6 default paths bit-identical")
