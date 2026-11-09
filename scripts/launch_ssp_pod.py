"""Launch ONE RunPod CPU pod that runs an ssp_algebra job (experiments/ssp_algebra/pod_job.py).

Reuses the existing machinery unchanged (image, tarball builder, R2 code upload, job bootstrap with
self-delete-on-success, registry auth) from launch_cell_pod / launch_pilot_pod; it only avoids
``parse_cell`` because these jobs are not DQN cells.  ``launch_cell_pod.py`` is deliberately not edited.

    python scripts/launch_ssp_pod.py --name probe --prefix ssp_algebra/probe/ --probe --dry-run
    python scripts/launch_ssp_pod.py --name probe --prefix ssp_algebra/probe/ --probe
    python scripts/launch_ssp_pod.py --name b128 --prefix ssp_algebra/r0fix_b128/ --sizes 16,8,4 \\
        --sweep "--mode sweep --variants index --tag r0fix_b128 --budget 128 --dense-eval --configs 4 --seeds 3 \\
                 --steps 1500 --workers 4"

One pod per invocation, never any existing pod touched.  A pod that exits 0 deletes itself; a failed one idles so the
log can be read (and its partial results are already on R2).  Check ``runpodctl pod list`` afterwards.
"""

import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_cell_pod as lc  # noqa: E402
import launch_pilot_pod as lp  # noqa: E402

CODE_PREFIX = "ssp_algebra/_code/"
SCRIPT = "experiments/ssp_algebra/pod_job.py"


def build_body(env, name, vcpus, code_key, prefix, probe, sweep):
    args = f"--prefix {prefix}" + (" --probe" if probe else f" -- {sweep}")
    return {
        "name": f"ssp-{name}".replace("_", "-"),
        "image": lp.IMAGE,
        "cloud": "SECURE",
        "disk": 20,
        "cpu": {"id": lp.CPU_FLAVOR, "vcpuCount": vcpus},
        "entrypoint": ["python", "-c", lc.JOB_BOOTSTRAP],
        "env": {**{k: env[k] for k in lp.R2_KEYS}, "EVAL_CODE_KEY": code_key, "JOB_SCRIPT": SCRIPT,
                "JOB_ARGS": args},
        **lc.registry_auth(env),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prefix", required=True, help="R2 prefix for this pod's results, e.g. ssp_algebra/r0fix_b128/")
    ap.add_argument("--sizes", default="4", help="vCPU counts to try in order (4 is the size existing pods use)")
    ap.add_argument("--probe", action="store_true", help="print versions + run the fast gates, then exit")
    ap.add_argument("--sweep", default="", help="run_sweep.py arguments (quoted)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.probe and not a.sweep:
        sys.exit("give --probe or --sweep '<run_sweep args>'")
    env = lp.load_env()
    missing = [k for k in (*lp.R2_KEYS, "RUNPOD_API_KEY") if not env.get(k)]
    if missing:
        sys.exit(f"missing in .env: {missing}")
    sizes = [int(s) for s in a.sizes.split(",")]
    if a.dry_run:
        body = build_body(env, a.name, sizes[0], "<code key>", a.prefix, a.probe, a.sweep)
        print(f"POST {lp.API}")
        print(json.dumps(lc.redact(body), indent=2))
        print("code tarball would be", f"{len(lp.make_code_tarball()) / 1024:.0f} KB", "->", CODE_PREFIX)
        return
    code_key = lc.upload_code(env, CODE_PREFIX)
    for n in sizes:
        try:
            out = lc.create_pod(env, build_body(env, a.name, n, code_key, a.prefix, a.probe, a.sweep))
            print(f"CREATED pod id={out.get('id')} name={out.get('name')} vcpus={n}", flush=True)
            return
        except urllib.error.HTTPError as e:
            print(f"{n} vCPUs: HTTP {e.code}: {e.read().decode()[:300]}", flush=True)
            if e.code in (401, 403):
                sys.exit("authentication failed - check RUNPOD_API_KEY")
        except Exception as e:  # noqa: BLE001
            print(f"{n} vCPUs: {e!r}", flush=True)
        time.sleep(2)
    sys.exit("NOT CREATED (capacity?) - retry later")


if __name__ == "__main__":
    main()
