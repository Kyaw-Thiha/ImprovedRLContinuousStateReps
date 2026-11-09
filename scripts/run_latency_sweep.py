#!/usr/bin/env python3
"""Latency sweep: run the deployment `latency` section for many cells on ONE pod.

Why: `run_deployment.py --cell X --sections latency` handles a single cell, so doing all 27 cells the
obvious way costs 27 pods -- 27 image pulls through Docker Hub's anonymous rate limit, which already
killed pods twice during RUN_004. Latency also has to be measured on a machine that is otherwise idle,
and one pod doing every cell back-to-back is both cheaper and *more* comparable than 27 different hosts.

Writes each cell's latency block into its own `deployment.json` (the runner merges it), exactly as the
per-cell command would.

    python scripts/run_latency_sweep.py --latency-calls 2000
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

import eval_v3_layout as layout  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", default="all", help="'all' or a comma list")
    ap.add_argument("--latency-calls", type=int, default=2000)
    ap.add_argument("--latency-seed-idx", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="accepted and ignored: the launcher always passes --workers to eval jobs")
    args = ap.parse_args(argv)

    if args.cells == "all":
        cells = list(layout.ALL_CELLS) + [c for c in layout.A2C_CELLS if c not in layout.ALL_CELLS]
    else:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    t0 = time.time()
    failed = []
    for i, cell in enumerate(cells, 1):
        cmd = [sys.executable, os.path.join(HERE, "run_deployment.py"), "--cell", cell,
               "--sections", "latency", "--latency-calls", str(args.latency_calls),
               "--latency-seed-idx", str(args.latency_seed_idx)]
        print(f"[latency {i}/{len(cells)}] {' '.join(cmd[1:])}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        tail = [ln for ln in (p.stdout or "").splitlines() if "latency" in ln.lower()][-2:]
        for ln in tail:
            print(f"    {ln}", flush=True)
        if p.returncode != 0:
            failed.append(cell)
            print(f"    FAILED rc={p.returncode}: {(p.stderr or '')[-200:]}", flush=True)
    print(f"[latency] done in {time.time() - t0:.0f}s; {len(cells) - len(failed)}/{len(cells)} ok", flush=True)
    if failed:
        print("[latency] failed:", ",".join(failed), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
