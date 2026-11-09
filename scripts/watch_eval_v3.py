#!/usr/bin/env python3
"""Cost/safety watcher for eval_v3 pods.

`scripts/watch_cells.py` only understands the RUN_002 search pods (it polls
`refilter_v2/<study>/final_eval_summary.json` and derives `dqn-v2-*` names), so eval_v3 pods had to be
watched by hand. This is that watcher: it lists every pod named `eval-v3-*`, maps it back to its
(job, cell) by regenerating the expected names, and reports the job's progress from its R2 marker.

Completion markers (what each runner writes LAST, per its code):
  policy_pass  train_summary.json + manifest.json present; progress = seeds whose record is verified
  robustness   robustness_runs.jsonl with n_seeds * 21 records (21 conditions, see rl.perturbations)
  deployment   deployment.json with the requested sections present

Safety rules:
  * only pods whose name starts with `eval-v3-` are ever reported or deleted;
  * `cerb-*` pods (and anything else) are listed as "not ours" and never touched;
  * deletion only happens with `--delete-finished`, only for completed jobs, and only after
    `--grace-min` minutes since the marker appeared on R2.

Usage
-----
    .venv/bin/python scripts/watch_eval_v3.py                     # report only
    .venv/bin/python scripts/watch_eval_v3.py --delete-finished   # reap completed jobs
    .venv/bin/python scripts/watch_eval_v3.py --logs              # also tail the last log line
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

JOBS = ("policy_pass", "robustness", "deployment")
PREFIX = "eval-v3-"


def load_env() -> dict:
    env = {}
    with open(os.path.join(REPO, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    return env


def runpodctl() -> str:
    exe = shutil.which("runpodctl") or os.path.expanduser("~/.local/bin/runpodctl")
    if not os.path.exists(exe):
        sys.exit("runpodctl not found")
    return exe


def _ctl(key: str, *args: str, timeout: int = 60):
    p = subprocess.run([runpodctl(), *args], capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "RUNPOD_API_KEY": key})
    return p.stdout.strip(), p.stderr.strip(), p.returncode


def list_pods(key: str, attempts: int = 3) -> list:
    """All pods, unioned over a few calls.

    A single `pod list` intermittently returns a truncated/incomplete set (it once showed 13 of 19
    live pods). Trusting one call made the supervisor believe live cells had died and create
    duplicate pods, so retry and union by id.
    """
    seen = {}
    last_err = ""
    for i in range(attempts):
        out, err, rc = _ctl(key, "pod", "list")
        if rc != 0:
            last_err = err
            time.sleep(2)
            continue
        try:
            for p in json.loads(out or "[]"):
                seen[p.get("id")] = p
        except json.JSONDecodeError:
            last_err = f"unexpected output: {out[:200]}"
        if len(seen) > 0 and i == 0:
            time.sleep(3)   # one confirmation pass; the fleet is small, so this is cheap
    if not seen:
        sys.exit(f"runpodctl pod list failed: {last_err[:300]}")
    return list(seen.values())


def expected_names() -> dict:
    """{(job, cell): pod name} for every eval_v3 job and cell."""
    import eval_v3_layout as layout
    cells = list(layout.ALL_CELLS) + [c for c in layout.A2C_CELLS if c not in layout.ALL_CELLS]
    out = {}
    for job in JOBS:
        for cell in cells:
            rep, cen, arch = layout.parse_cell(cell)
            out[(job, cell)] = f"{PREFIX}{job}-{rep}-{cen}-{arch}".replace("_", "-")
    return out


def r2_client(env):
    import boto3
    return boto3.client("s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])


def get_obj(s3, bucket, key):
    try:
        r = s3.get_object(Bucket=bucket, Key=key)
        return r["Body"].read(), r["LastModified"]
    except Exception:  # noqa: BLE001
        return None, None


def progress(env, s3, job: str, cell: str) -> dict:
    """Marker state + progress counters for one (job, cell)."""
    import eval_v3_layout as layout
    study = layout.study_name(cell)
    out = {"done": False, "detail": "", "modified": None}
    if job == "policy_pass":
        runs, _runs_m = get_obj(s3, env["R2_BUCKET"], layout.train_runs_key(study))
        summary, m1 = get_obj(s3, env["R2_BUCKET"], layout.train_summary_key(study))
        manifest, m2 = get_obj(s3, env["R2_BUCKET"], layout.manifest_key(study))
        n = verified = 0
        if runs:
            for line in runs.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    verified += 1 if json.loads(line).get("policy_verified") else 0
                except json.JSONDecodeError:
                    pass
        mods = [m for m in (_runs_m, m1, m2) if m is not None]
        out.update(done=bool(summary and manifest), modified=max(mods) if mods else None,
                   detail=f"{verified}/{n} seeds verified")
    elif job == "robustness":
        # Robustness has no summary file, so completeness is counted. This must agree with
        # start_eval_stage.stage_done(), otherwise the supervisor believes a finished cell still needs
        # work and relaunches it forever (it did exactly that before this was fixed).
        blob, m = get_obj(s3, env["R2_BUCKET"], layout.robustness_key(study))
        n = len([l for l in (blob or b"").decode().splitlines() if l.strip()])
        try:
            import rl.perturbations as pert
            n_cond = len(pert.ALL_CONDITIONS)
        except Exception:  # noqa: BLE001
            n_cond = 21
        want = layout.POLICY_N_SEEDS * n_cond
        out.update(done=n >= want, modified=m, detail=f"{n}/{want} records")
    else:
        blob, m = get_obj(s3, env["R2_BUCKET"], layout.deployment_key(study))
        sections = []
        if blob:
            try:
                d = json.loads(blob.decode())
                sections = [s for s in ("size", "quantization", "latency") if s in d]
            except json.JSONDecodeError:
                pass
        out.update(done=("size" in sections and "quantization" in sections), modified=m,
                   detail=f"sections={','.join(sections) or 'none'}")
    return out


def age_min(runtime: str) -> float:
    """Pod runtime string -> minutes (best effort).

    RunPod returns e.g. 'Rented by User: Tue Oct 06 2026 18:36:31 GMT+0000 (Coordinated Universal Time)'.
    """
    if not runtime:
        return float("nan")
    s = runtime.split(": ", 1)[1] if runtime.startswith("Rented by User: ") else runtime
    for fmt in ("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)",
                "%a %b %d %Y %H:%M:%S GMT%z (Coordinated Universal Time)"):
        try:
            t = datetime.strptime(s, fmt)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - t).total_seconds() / 60.0
        except (ValueError, TypeError):
            continue
    return float("nan")


def cells_for_job(include_a2c: bool = False):
    import eval_v3_layout as layout
    cells = list(layout.ALL_CELLS)
    if include_a2c:
        cells += [c for c in layout.A2C_CELLS if c not in layout.ALL_CELLS]
    return cells


def relaunch_state_path(path=None):
    return path or "/tmp/eval_v3_supervisor.json"


def load_relaunch_state(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def save_relaunch_state(path, state):
    try:
        with open(path, "w") as fh:
            json.dump(state, fh, indent=2)
    except OSError:
        pass


def relaunch(cells, job, sizes, dry_run=False):
    """Shell out to the launcher for a cell list (one pod per cell, sequential)."""
    cmd = [sys.executable, os.path.join(HERE, "launch_cell_pod.py"), "--job", job,
           "--cells", ",".join(cells), "--sizes", sizes]
    if dry_run:
        print(f"  would run: {' '.join(cmd)}")
        return 0
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in (p.stdout or "").splitlines():
        if "CREATED" in line or "NOT CREATED" in line:
            print(f"  {line}")
    return p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delete-finished", action="store_true")
    ap.add_argument("--dedupe", action="store_true",
                    help="for each (job, cell) with several live pods, keep the oldest and delete the rest")
    ap.add_argument("--grace-min", type=float, default=12.0)
    ap.add_argument("--max-minutes", type=float, default=180.0, help="flag pods older than this")
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--job", default="policy_pass", choices=list(JOBS),
                    help="which job the supervisor covers (with --relaunch-missing)")
    ap.add_argument("--include-a2c", action="store_true",
                    help="also cover the 9 A2C cells (default: the 18 DQN cells only)")
    ap.add_argument("--relaunch-missing", action="store_true",
                    help="relaunch cells that have no live pod and no completion marker")
    ap.add_argument("--sizes", default="4,2", help="vCPU sizes to try, for relaunches")
    ap.add_argument("--max-relaunch", type=int, default=3, help="relaunch attempts per cell")
    ap.add_argument("--min-idle-min", type=float, default=30.0,
                    help="only relaunch a cell with no live pod if its R2 marker is this stale")
    ap.add_argument("--state", default=None, help="relaunch counter file (default /tmp/eval_v3_supervisor.json)")
    args = ap.parse_args(argv)

    env = load_env()
    key = env.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing from .env")
    s3 = r2_client(env)
    names = expected_names()
    reverse = {v: k for k, v in names.items()}

    pods = list_pods(key)
    rows = []
    for p in pods:
        name = p.get("name", "")
        owner = reverse.get(name)
        row = {"id": p.get("id"), "name": name, "status": p.get("runtimeStatus"),
               "cost_per_hr": p.get("costPerHr"), "minutes": age_min(p.get("lastStatusChange")),
               "ours": bool(owner)}
        if owner:
            job, cell = owner
            row.update(job=job, cell=cell)
            row.update(progress(env, s3, job, cell))
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        ours = [r for r in rows if r["ours"]]
        others = [r for r in rows if not r["ours"]]
        print(f"{len(ours)} eval-v3 pod(s), {len(others)} other pod(s) (never touched)\n")
        for r in ours:
            flag = ""
            if r.get("done"):
                flag = "DONE"
            elif r["minutes"] == r["minutes"] and r["minutes"] > args.max_minutes:
                flag = "OVER-TIME-CAP"
            print(f"{r['id']:>16}  {r['cell']:<26} {r['job']:<12} {str(r['status']):<12} "
                  f"{r['minutes']:6.1f} min  {'DONE' if r.get('done') else 'running':<8} {r.get('detail','')} {flag}")
            if args.logs and (flag or not r.get("done")):
                out, err, rc = _ctl(key, "pod", "logs", str(r["id"]))
                tail = (out or err).splitlines()[-2:]
                for line in tail:
                    print(f"      | {line[:200]}")
        for r in others:
            print(f"{r['id']:>16}  {r['name']:<40} NOT OURS (skipped)")

    if args.relaunch_missing:
        # "done" MUST come from R2, not from the pod list: a finished pod self-deletes, so a completed
        # cell has no pod and would otherwise look missing and be relaunched on every cycle. That bug
        # (plus a name-convention mismatch) caused dozens of duplicate pods during this run.
        state_path = relaunch_state_path(args.state)
        state = load_relaunch_state(state_path)
        live = {(r["job"], r["cell"]) for r in rows if r["ours"]}
        now = datetime.now(timezone.utc)
        missing = []
        for c in cells_for_job(args.include_a2c):
            if (args.job, c) in live:
                continue
            pr = progress(env, s3, args.job, c)
            if pr["done"]:
                continue
            act = pr.get("modified")
            if act is not None and (now - act).total_seconds() / 60.0 < args.min_idle_min:
                print(f"  skip {c}: R2 active {(now - act).total_seconds()/60:.1f} min ago")
                continue
            missing.append(c)
        todo = []
        for c in missing:
            k = f"{args.job}:{c}"
            if state.get(k, 0) >= args.max_relaunch:
                print(f"  give up on {args.job} {c}: {state.get(k)} relaunches already")
                continue
            todo.append(c)
        if todo:
            print(f"\nrelaunching {len(todo)} cell(s) with no live pod and no completion marker:")
            print("  " + ", ".join(todo))
            relaunch(todo, args.job, args.sizes)
            for c in todo:
                k = f"{args.job}:{c}"
                state[k] = state.get(k, 0) + 1
            save_relaunch_state(state_path, state)
        else:
            print(f"\nno missing pods for job {args.job}")

    if args.dedupe:
        # Two pods for one (job, cell) are NOT necessarily duplicates: a cell's DQN-settings variants
        # share the same pod name because the launcher's name only encodes (job, cell). So group by
        # JOB_ARGS and only delete pods that duplicate another pod's exact arguments -- an ambiguous
        # group is left alone.
        by_key = {}
        for r in rows:
            if r["ours"]:
                by_key.setdefault((r["job"], r["cell"]), []).append(r)

        def _job_args(pod_id):
            out, err, rc = _ctl(key, "pod", "get", str(pod_id))
            try:
                return json.loads(out).get("env", {}).get("JOB_ARGS")
            except Exception:  # noqa: BLE001
                return None

        for (job, cell), rs in sorted(by_key.items()):
            if len(rs) < 2:
                continue
            sigs = [(_job_args(r["id"]), r) for r in rs]
            groups = {}
            for sig, r in sigs:
                groups.setdefault(sig, []).append(r)
            paired = {sig: g for sig, g in groups.items() if len(g) > 1}
            if not paired:
                print(f"{cell} {job}: {len(rs)} pods with distinct JOB_ARGS -> not duplicates, skipping")
                continue
            for sig, g in paired.items():
                g.sort(key=lambda r: (1, 0.0) if r["minutes"] != r["minutes"] else (0, -r["minutes"]))
                keep, dupes = g[0], g[1:]
                print(f"{cell} {job}: {len(g)} pods with identical args -> keeping {keep['id']} "
                      f"({keep['minutes']:.1f} min), deleting {len(dupes)}")
                for d in dupes:
                    out, err, rc = _ctl(key, "pod", "delete", str(d["id"]))
                    print(f"  delete {d['id']} ({d['minutes']:.1f} min): rc={rc} {out or err}"[:160])

    if args.delete_finished:
        import eval_v3_layout as layout
        now = datetime.now(timezone.utc)
        for r in [x for x in rows if x["ours"] and x.get("done")]:
            m = r.get("modified")
            if m is not None:
                mins = (now - m).total_seconds() / 60.0
                if mins < args.grace_min:
                    print(f"skip {r['id']} ({r['cell']} {r['job']}): marker only {mins:.1f} min old "
                          f"(< grace {args.grace_min})")
                    continue
            out, err, rc = _ctl(key, "pod", "delete", str(r["id"]))
            print(f"delete {r['id']} ({r['cell']} {r['job']}): rc={rc} {out or err}"[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
