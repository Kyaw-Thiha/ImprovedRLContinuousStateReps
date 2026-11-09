"""Launch, monitor and collect batches of ssp_algebra RunPod jobs from one JSON job list.

Replaces the throw-away /tmp scripts used during RUN_008/009.  One job = one 4-vCPU CPU pod running
``experiments/ssp_algebra/pod_job.py`` -> ``run_sweep.py``; results are uploaded to R2 under ``ssp_algebra/<name>/`` and the pod
self-deletes on success (see ``launch_ssp_pod.py``).  Nothing here touches pods that do not start with ``ssp-``.

    python scripts/ssp_pod_batch.py launch  jobs.json [--dry-run]    # creates the pods, one at a time
    python scripts/ssp_pod_batch.py status  jobs.json                # rows on R2 per job + my live pods
    python scripts/ssp_pod_batch.py collect jobs.json [--timeout 3500]   # waits, then downloads sweep_<tag>.jsonl (+ winners)

Job file (a JSON list); every field but ``name``, ``variant`` and ``arms`` is optional:

    {"name": "dg2-ws8-s375", "variant": "index", "arms": ["ssp_index_learned"], "label": "ssp_index_learned_r1_ws8_s375",
     "arm_cfg": {"rung":1,"init":"random","length_scale":1.0,"ls_start":8.0,"ls_mode":"sched","width_steps":375},
     "extra": "--ep-len 32 --pool 64 --robust", "configs": 6, "seeds": 5, "steps": 1500, "budget": 256, "tag": "dg2_ws8_s375"}

Rules that bit us before: ``name`` uses hyphens (it becomes the pod and R2 prefix); ``tag`` must be unique per job and per protocol
(the sweep's resume key omits budget, pool, steps, robust, ep-len); ``arm_cfg`` is passed as compact JSON (no spaces); ``--label``
needs exactly one arm; expected rows = len(arms) * (configs + seeds).  Launching is slow when ``experiments/ssp_algebra/results`` is
large because every launch rebuilds and uploads the code tarball; move old results out of the way first if it gets over ~50 MB.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import launch_cell_pod as lc  # noqa: E402
import launch_pilot_pod as lp  # noqa: E402

RESULTS = ROOT / "experiments" / "ssp_algebra" / "results"


def load_jobs(path):
    jobs = json.load(open(path))
    names = set()
    for j in jobs:
        for k in ("name", "variant", "arms"):
            if k not in j:
                sys.exit(f"job missing '{k}': {j}")
        if j["name"] in names:
            sys.exit(f"duplicate job name {j['name']}")
        names.add(j["name"])
        j.setdefault("tag", j["name"].replace("-", "_"))
        j.setdefault("configs", 6)
        j.setdefault("seeds", 5)
        j.setdefault("steps", 1500)
        j.setdefault("budget", 256)
        j.setdefault("extra", "")
        j.setdefault("mode", "sweep")
        j.setdefault("rows", len(j["arms"]) * (j["configs"] + j["seeds"]))
        if j.get("label") and len(j["arms"]) != 1:
            sys.exit(f"{j['name']}: --label needs exactly one arm")
    return jobs


def sweep_args(j):
    s = (f"--mode {j['mode']} --variants {j['variant']} --arms {' '.join(j['arms'])} --tag {j['tag']} --budget {j['budget']} "
         f"--dense-eval --configs {j['configs']} --seeds {j['seeds']} --steps {j['steps']} --workers 4")
    if j.get("label"):
        s += f" --label {j['label']}"
    if j.get("arm_cfg"):
        s += " --arm-cfg " + json.dumps(j["arm_cfg"], separators=(",", ":"))
    if j["extra"]:
        s += " " + j["extra"]                     # later flags win in argparse, so extra may override --steps etc.
    return s


def r2_rows(s3, bucket, j):
    fn = f"{'sweep' if j['mode'] == 'sweep' else 'ablation'}_{j['tag']}.jsonl"
    try:
        body = s3.get_object(Bucket=bucket, Key=f"ssp_algebra/{j['name']}/{fn}")["Body"].read().decode()
        return fn, body, sum(1 for l in body.splitlines() if l.strip())
    except Exception:  # noqa: BLE001
        return fn, None, 0


def my_pods(env):
    req = urllib.request.Request(lp.API, headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}", "User-Agent": lp.UA,
                                                  "Accept": "application/json"})
    pods = json.loads(urllib.request.urlopen(req, timeout=30).read())
    pods = pods if isinstance(pods, list) else pods.get("pods", [])
    return sorted(p["name"] for p in pods if str(p.get("name", "")).startswith("ssp-"))


def cmd_launch(jobs, a):
    failed = []
    for j in jobs:
        cmd = [sys.executable, str(HERE / "launch_ssp_pod.py"), "--name", j["name"], "--prefix", f"ssp_algebra/{j['name']}/",
               "--sizes", "4", "--sweep", sweep_args(j)]
        if a.dry_run:
            print(j["name"], "->", sweep_args(j))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        out = [l for l in (r.stdout + r.stderr).splitlines() if "CREATED" in l or "HTTP" in l or "rror" in l]
        print(j["name"], "->", out[-1][:120] if out else "??", flush=True)
        if not any("CREATED pod" in l for l in out):
            failed.append(j["name"])
        time.sleep(3)                             # one pod creation at a time
    print("FAILED:", failed or "none")


def cmd_status(jobs, env, s3):
    done = 0
    for j in jobs:
        _, _, n = r2_rows(s3, env["R2_BUCKET"], j)
        done += n >= j["rows"]
        print(f"  {j['name']:28s} {n:3d}/{j['rows']}")
    print(f"finished {done}/{len(jobs)}; my pods alive: {my_pods(env)}")


def cmd_collect(jobs, env, s3, timeout):
    RESULTS.mkdir(parents=True, exist_ok=True)
    got, t0 = set(), time.time()
    while time.time() - t0 < timeout and len(got) < len(jobs):
        for j in jobs:
            if j["name"] in got:
                continue
            fn, body, n = r2_rows(s3, env["R2_BUCKET"], j)
            if n >= j["rows"]:
                (RESULTS / fn).write_text(body)
                got.add(j["name"])
                try:
                    s3.download_file(env["R2_BUCKET"], f"ssp_algebra/{j['name']}/winners_{j['tag']}.json",
                                     str(RESULTS / f"winners_{j['tag']}.json"))
                except Exception:  # noqa: BLE001
                    pass
                print(f"[{time.time() - t0:5.0f}s] done {j['name']} ({n} rows)", flush=True)
        if len(got) < len(jobs):
            time.sleep(20)
    missing = [j["name"] for j in jobs if j["name"] not in got]
    print("ALL DONE" if not missing else f"TIMED OUT; incomplete: {missing}")
    print("my pods alive:", my_pods(env))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["launch", "status", "collect"])
    ap.add_argument("jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=int, default=3500)
    a = ap.parse_args()
    jobs = load_jobs(a.jobs)
    if a.cmd == "launch" and a.dry_run:
        cmd_launch(jobs, a)
        return
    env = lp.load_env()
    s3 = lc.s3_client(env)
    if a.cmd == "launch":
        cmd_launch(jobs, a)
    elif a.cmd == "status":
        cmd_status(jobs, env, s3)
    else:
        cmd_collect(jobs, env, s3, a.timeout)


if __name__ == "__main__":
    main()
