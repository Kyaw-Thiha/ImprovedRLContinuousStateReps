#!/usr/bin/env python3
"""DQN-settings sweep for one cell: run all five variants back-to-back on ONE pod.

RUN_004's DQN-settings test is buffer/target-update variants on {tile_coding, ssp} x none x
{linear, deep}, 10 seeds each, paired with the baseline pass on the same seeds. Each variant is a
separate `run_policy_pass.py` invocation, so the obvious approach costs 5 pods per cell (20 pods, 20
image pulls, each at risk from the Docker Hub anonymous rate limit). One pod per cell runs all five
sequentially instead, which also keeps the comparison on one machine.

Resumable: `run_policy_pass.py` skips (seed, variant) records already on R2, so a re-run only fills
gaps.

    python scripts/run_dqn_settings_sweep.py --cell tile_coding/none/linear --n-seeds 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

# (buffer_size, target_update_freq); baseline is (5000, 10) and is not re-run here.
VARIANTS = [
    (50000, None),
    (None, 100),
    (None, 500),
    (50000, 100),
    (50000, 500),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", required=True, help="e.g. tile_coding/none/linear")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=0, help="0 = match the pod's vCPU count")
    args = ap.parse_args(argv)

    workers = args.workers or (os.cpu_count() or 4)
    t0 = time.time()
    failed = []
    for i, (buf, tu) in enumerate(VARIANTS, 1):
        cmd = [sys.executable, os.path.join(HERE, "run_policy_pass.py"),
               "--cell", args.cell, "--workers", str(workers),
               "--n-seeds", str(args.n_seeds), "--seed-offset", str(args.seed_offset)]
        if buf:
            cmd += ["--variant-buffer-size", str(buf)]
        if tu:
            cmd += ["--variant-target-update-freq", str(tu)]
        label = f"buf{buf or '-'}_tu{tu or '-'}"
        print(f"[dqn-settings {i}/{len(VARIANTS)}] {label}: {' '.join(cmd[1:])}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        for ln in [x for x in (p.stdout or "").splitlines() if "DONE" in x or "variant=" in x][-2:]:
            print(f"    {ln}", flush=True)
        if p.returncode != 0:
            failed.append(label)
            print(f"    FAILED rc={p.returncode}: {(p.stderr or '')[-200:]}", flush=True)
    print(f"[dqn-settings] {args.cell} done in {time.time() - t0:.0f}s", flush=True)
    if failed:
        print("[dqn-settings] failed variants:", ",".join(failed), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
