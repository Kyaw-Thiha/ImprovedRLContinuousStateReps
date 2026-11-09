"""
Watcher / cost guard for the chained cell pods (scripts/launch_cell_pod.py).

For each cell it polls R2 (stage0_cartpole/refilter_v2/<study>/...) and the RunPod pod list, prints one
line whenever something changes, prints EVERY error (an earlier watcher swallowed R2 read errors and
reported a false stall), and deletes a cell's pod
  * when its final_eval_summary.json and v2 best-params yaml are on R2 (the pod normally deletes itself
    first; a 404 here just means it already did), or
  * when the per-cell time cap is hit (the pod is deleted even if R2 reads keep failing).

    python scripts/watch_cells.py --cells discrete/none/linear,ssp/none/deep --max-hours 6
"""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_cell_pod as lc  # noqa: E402
import launch_pilot_pod as lp  # noqa: E402

PREFIX = "stage0_cartpole/refilter_v2"


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")


def api(env, method, path):
    req = urllib.request.Request(f"https://api.runpod.io/v2/{path}", method=method,
                                 headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}",
                                          "User-Agent": lp.UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body.strip() else None)


def list_pods(env):
    _, data = api(env, "GET", "pods")
    if isinstance(data, dict):
        data = data.get("pods") or data.get("data") or []
    return {p.get("name"): p for p in data}


def n_lines(s3, bucket, key):
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    except Exception as e:  # noqa: BLE001
        if "NoSuchKey" in repr(e) or "404" in repr(e):
            return 0
        raise
    return sum(1 for ln in body.splitlines() if ln.strip() and '"error"' not in ln)


def exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:  # noqa: BLE001
        if "404" in repr(e) or "NoSuchKey" in repr(e) or "Not Found" in repr(e):
            return False
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--max-hours", type=float, default=8.0, help="per-cell time cap; the pod is deleted at the cap")
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--grace-min", type=float, default=12.0,
                    help="after a cell's results are on R2, wait this long for the pod to finish its last "
                         "sync and delete itself before the watcher deletes it")
    a = ap.parse_args()
    env = lp.load_env()
    s3, bucket = lc.s3_client(env), env["R2_BUCKET"]
    cells = [c for c in a.cells.split(",") if c]
    pod_name = {c: lc.build_body(env, c, 4, "x", 500)["name"] for c in cells}
    t0 = time.time()
    state = {c: None for c in cells}
    finished = set()
    done_since = {}
    while len(finished) < len(cells):
        try:
            pods = list_pods(env)
        except Exception as e:  # noqa: BLE001
            pods = None
            print(f"{now()} UTC RunPod list problem: {e!r}"[:200], flush=True)
        for c in cells:
            if c in finished:
                continue
            study = lc.study_name(c)
            rep, cen, arch = lc.parse_cell(c)
            yaml_key = f"stage0_cartpole/best_params/dqn_{rep}_{cen}{'' if arch == 'linear' else '_deep'}_v2.yaml"
            try:
                n_reeval = n_lines(s3, bucket, f"{PREFIX}/{study}/reeval_v2_runs.jsonl")
                n_final = n_lines(s3, bucket, f"{PREFIX}/{study}/final_eval_runs.jsonl")
                done = exists(s3, bucket, f"{PREFIX}/{study}/final_eval_summary.json") and exists(s3, bucket, yaml_key)
            except Exception as e:  # noqa: BLE001
                print(f"{now()} UTC {c}: R2 read problem: {e!r}"[:200], flush=True)
                n_reeval = n_final = done = None
            pod = None if pods is None else pods.get(pod_name[c])
            pod_s = "unknown" if pods is None else (f"{pod.get('runtimeStatus', pod.get('desiredStatus', '?'))}({pod.get('id')})" if pod else "gone")
            cur = (n_reeval, n_final, done, pod_s)
            if cur != state[c]:
                print(f"{now()} UTC {c}: reeval runs={n_reeval} final runs={n_final} done={done} pod={pod_s}", flush=True)
                state[c] = cur
            capped = (time.time() - t0) > a.max_hours * 3600
            if done and c not in done_since:
                done_since[c] = time.time()
                print(f"{now()} UTC {c}: results on R2; giving the pod {a.grace_min:.0f} min to finish "
                      f"syncing and self-delete", flush=True)
            gone = pods is not None and pod is None
            settled = bool(done) and (gone or time.time() - done_since[c] >= a.grace_min * 60)
            if settled or capped:
                if capped and not done:
                    print(f"{now()} UTC {c}: TIME CAP {a.max_hours} h reached before completion", flush=True)
                if settled and gone:
                    print(f"{now()} UTC {c}: pod already gone (self-delete worked)", flush=True)
                if pod:
                    try:
                        st, _ = api(env, "DELETE", f"pods/{pod['id']}")
                        print(f"{now()} UTC {c}: deleted pod {pod['id']} (HTTP {st})", flush=True)
                    except urllib.error.HTTPError as e:
                        print(f"{now()} UTC {c}: DELETE -> HTTP {e.code} {e.read().decode()[:150]}", flush=True)
                finished.add(c)
        time.sleep(a.poll)
    print(f"{now()} UTC all {len(cells)} cells finished or capped.", flush=True)


if __name__ == "__main__":
    main()
