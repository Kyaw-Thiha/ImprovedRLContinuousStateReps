"""Pod-side entrypoint for the ssp_algebra sweeps (run by ``scripts/launch_ssp_pod.py`` through the existing
eval_v3 job bootstrap, which extracts the code tarball to /project and self-deletes the pod only on exit 0).

    python experiments/ssp_algebra/pod_job.py --prefix ssp_algebra/r0fix_b128/ --probe
    python experiments/ssp_algebra/pod_job.py --prefix ssp_algebra/r0fix_b128/ -- --mode sweep --variants index ...

What it adds over calling ``run_sweep.py`` directly:
  * **resume**: pulls any existing ``*.jsonl`` under ``--prefix`` from R2 first, so a re-launched pod continues
    instead of redoing (the sweep's own resume key is variant/arm/cfg_idx/seed/phase, so a distinct ``--tag`` per
    invocation is still required: budget and pool are NOT in that key);
  * **durability**: uploads results every ``--sync-every`` seconds and once more at exit, *including when the sweep
    fails*, so an idle-for-inspection pod never holds the only copy;
  * ``--probe``: prints interpreter / torch / numpy versions and runs the fast correctness gates, so a pod image that
    cannot run this code is found in a minute, not after a sweep.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def s3():
    import boto3

    e = os.environ
    return boto3.client("s3", endpoint_url=f"https://{e['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                        aws_access_key_id=e["R2_ACCESS_KEY_ID"], aws_secret_access_key=e["R2_SECRET_ACCESS_KEY"])


def pull(prefix):
    client, bucket = s3(), os.environ["R2_BUCKET"]
    RESULTS.mkdir(exist_ok=True)
    n = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            if name.endswith((".jsonl", ".json")) and "/" not in name:
                client.download_file(bucket, obj["Key"], str(RESULTS / name))
                n += 1
    print(f"[pod_job] resumed {n} file(s) from {prefix}", flush=True)


def push(prefix, tag):
    client, bucket = s3(), os.environ["R2_BUCKET"]
    n = 0
    for p in sorted(RESULTS.glob(f"*{tag}*")):
        if p.suffix in (".jsonl", ".json"):
            client.upload_file(str(p), bucket, prefix + p.name)
            n += 1
    print(f"[pod_job] pushed {n} file(s) to {prefix}", flush=True)
    return n


def probe():
    import numpy
    import torch

    print(f"[probe] python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}  "
          f"cpus {os.cpu_count()}", flush=True)
    rc = 0
    for script in ("test_fixes.py", "test_measurement.py", "test_analysis.py"):
        r = subprocess.run([sys.executable, script], cwd=HERE, env={**os.environ, "OMP_NUM_THREADS": "1"})
        print(f"[probe] {script} -> exit {r.returncode}", flush=True)
        rc |= r.returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="R2 key prefix for this invocation's results (ends with /)")
    ap.add_argument("--tag", default=None, help="sweep tag, used to select which result files to upload")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sync-every", type=int, default=300)
    ap.add_argument("sweep_args", nargs=argparse.REMAINDER, help="after '--': arguments for run_sweep.py")
    a = ap.parse_args()
    prefix = a.prefix if a.prefix.endswith("/") else a.prefix + "/"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if a.probe:
        rc = probe()
        if rc == 0:
            RESULTS.mkdir(exist_ok=True)
            (RESULTS / "probe_ok.json").write_text('{"ok": true}')
            push(prefix, "probe")
        sys.exit(rc)

    args = [x for x in a.sweep_args if x != "--"]
    tag = a.tag or (args[args.index("--tag") + 1] if "--tag" in args else "main")
    pull(prefix)
    stop = threading.Event()

    def loop():
        while not stop.wait(a.sync_every):
            try:
                push(prefix, tag)
            except Exception as ex:  # noqa: BLE001  (a sync failure must not kill the sweep)
                print("[pod_job] sync failed:", repr(ex), flush=True)

    threading.Thread(target=loop, daemon=True).start()
    t0 = time.time()
    rc = subprocess.call([sys.executable, "run_sweep.py", *args], cwd=HERE)
    stop.set()
    push(prefix, tag)                                   # always, including after a failure
    print(f"[pod_job] sweep exit {rc} after {time.time() - t0:.0f}s", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
