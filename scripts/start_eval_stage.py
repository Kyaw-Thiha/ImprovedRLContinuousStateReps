#!/usr/bin/env python3
"""Start the next eval_v3 stage for cells that are ready for it.

`robustness` and `deployment` both need a cell's saved policies, i.e. a finished policy pass
(`train_summary.json` + `manifest.json` on R2). Rather than waiting for all 27 cells and then
launching 54 pods at once -- which is the worst possible pattern for the Docker Hub anonymous pull
limit that already killed six pods -- this starts a stage only for cells that are ready, capped at
`--max-pods` concurrent new pods per invocation. Because each runner is resumable (JSONL keyed by
(policy seed, condition)), re-running is safe; use `watch_eval_v3.py --relaunch-missing` to replace
pods that die on an image-pull failure.

Dry-run by default; pass --go to actually create pods.

    python scripts/start_eval_stage.py --stage robustness                 # report only
    python scripts/start_eval_stage.py --stage robustness --go --max-pods 8
    python scripts/start_eval_stage.py --stage deployment --go --job-args "--sections size,quant"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import eval_v3_layout as layout  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


watch = _load("watch_eval_v3", "scripts/watch_eval_v3.py")


def expected_robustness_records(n_seeds: int = 20) -> int:
    try:
        import rl.perturbations as pert
        n_cond = len(pert.ALL_CONDITIONS)
    except Exception:  # noqa: BLE001
        n_cond = 21
    return n_seeds * n_cond


def stage_done(env, s3, stage: str, cell: str, n_seeds: int = 20) -> tuple:
    """(done, detail) for a stage. Robustness has no summary file, so it is counted."""
    study = layout.study_name(cell)
    if stage == "robustness":
        blob, _ = watch.get_obj(s3, env["R2_BUCKET"], layout.robustness_key(study))
        n = len([ln for ln in (blob or b"").decode().splitlines() if ln.strip()])
        want = expected_robustness_records(n_seeds)
        return n >= want, f"{n}/{want} records"
    blob, _ = watch.get_obj(s3, env["R2_BUCKET"], layout.deployment_key(study))
    if not blob:
        return False, "no deployment.json"
    import json
    try:
        d = json.loads(blob.decode())
    except json.JSONDecodeError:
        return False, "unreadable deployment.json"
    have = [s for s in ("size", "quantization", "latency") if s in d]
    return ("size" in have and "quantization" in have), f"sections={','.join(have) or 'none'}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=["robustness", "deployment"])
    ap.add_argument("--include-a2c", action="store_true", default=True)
    ap.add_argument("--no-a2c", dest="include_a2c", action="store_false")
    ap.add_argument("--cells", default="", help="explicit comma list, else all 27")
    ap.add_argument("--max-pods", type=int, default=8, help="max pods to create in this invocation")
    ap.add_argument("--sizes", default="4,2")
    ap.add_argument("--job-args", default="")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args(argv)

    env = watch.load_env()
    s3 = watch.r2_client(env)
    cells = [c for c in args.cells.split(",") if c] or watch.cells_for_job(args.include_a2c)

    ready, blocked_policy, already, done = [], [], [], []
    for cell in cells:
        pol_done = watch.progress(env, s3, "policy_pass", cell)["done"]
        s_done, detail = stage_done(env, s3, args.stage, cell, args.n_seeds)
        if s_done:
            done.append(cell)
        elif pol_done:
            ready.append((cell, detail))
        else:
            blocked_policy.append(cell)

    print(f"stage={args.stage}  cells={len(cells)}")
    print(f"  policies ready, stage to run : {len(ready)}")
    print(f"  policies not finished yet    : {len(blocked_policy)}")
    print(f"  stage already complete       : {len(done)}")
    for cell, detail in ready:
        print(f"    READY {cell} ({detail})")
    if not ready:
        print("nothing to start.")
        return 0

    launch = ready[: args.max_pods]
    if len(ready) > len(launch):
        print(f"  (capping at --max-pods {args.max_pods}; {len(ready) - len(launch)} left for the next call)")
    if not args.go:
        print(f"\ndry-run: would launch {len(launch)} pod(s) for stage {args.stage}:")
        print("  " + ",".join(c for c, _ in launch))
        return 0

    for cell, _ in launch:
        cmd = [sys.executable, os.path.join(HERE, "watch_eval_v3.py")]  # placeholder to keep imports honest
        cmd = [sys.executable, os.path.join(HERE, "launch_cell_pod.py"), "--job", args.stage,
               "--cells", cell, "--sizes", args.sizes]
        if args.job_args:
            cmd += ["--job-args", args.job_args]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        for line in (p.stdout or "").splitlines():
            if "CREATED" in line or "NOT CREATED" in line:
                print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
