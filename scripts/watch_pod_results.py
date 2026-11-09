"""
Cost-guard watcher for pilot pods: waits until an R2 results JSONL has N finished runs, then DELETES
the RunPod pod. Always deletes the pod at the time cap, even if R2 reads keep failing, and prints every
error instead of swallowing it (an earlier watcher hid R2 read errors and reported a false stall while
the pod sat idle, billing).

    python scripts/watch_pod_results.py --pod <id> --key stage0_cartpole/pilot/init_centering_runs.jsonl \
        --expected 40 --max-min 60
"""
import argparse, json, os, sys, time, urllib.request, urllib.error, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = {}
for l in open(os.path.join(REPO, ".env")):
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); env[k] = v.strip().strip('"').strip("'")


def delete_pod(pod):
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}", method="DELETE",
        headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}", "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return f"DELETE -> HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return f"DELETE -> HTTP {e.code} {e.read().decode()[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--expected", type=int, required=True); ap.add_argument("--max-min", type=float, default=60)
    a = ap.parse_args()
    import boto3
    s3 = boto3.client("s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])
    t0, last_n, errs = time.time(), -1, 0
    reason = None
    while True:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
        try:
            body = s3.get_object(Bucket=env["R2_BUCKET"], Key=a.key)["Body"].read().decode()
            rs = [json.loads(x) for x in body.splitlines() if x.strip()]
            n = sum(1 for r in rs if "error" not in r); e = len(rs) - n; errs = 0
            if n != last_n:
                print(f"{now} UTC finished {n}/{a.expected}, run errors {e}", flush=True); last_n = n
            if n >= a.expected:
                reason = "complete"; break
        except Exception as ex:
            errs += 1
            msg = repr(ex)[:160]
            if "NoSuchKey" in msg and errs < 3:
                pass  # results file simply not written yet
            else:
                print(f"{now} UTC R2 read problem #{errs}: {msg}", flush=True)
        if time.time() - t0 > a.max_min * 60:
            reason = f"time cap {a.max_min} min reached at {last_n}/{a.expected}"; break
        time.sleep(45)
    print("stopping:", reason, flush=True)
    print(delete_pod(a.pod), flush=True)


if __name__ == "__main__":
    main()
