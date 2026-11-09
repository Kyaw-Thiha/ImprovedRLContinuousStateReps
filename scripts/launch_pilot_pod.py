"""
Launch the greedy-rank pilot (scripts/pilot_greedy_rank_check.py) on a RunPod CPU pod.

Why no new Docker image: pushing the 6 GB dependency layer over a home connection takes about an
hour, to ship ~300 KB of changed code. Instead this script uploads a small code tarball
(experiments/ rl/ scripts/ conf/) to R2 under a NEW key and starts the existing, already-pulled
`kevinkyawthiha/dqn-ssp-centering:stage0` image with an ENTRYPOINT override that downloads the
tarball, extracts it over /project, and runs the pilot.

Reads RUNPOD_API_KEY and the R2_* credentials from the repo .env (never printed). Creates ONE pod
via RunPod REST v2 (POST https://api.runpod.io/v2/pods — `runpodctl pod create` cannot set the vCPU
count). Tries 16, then 8, then 4 vCPUs (CPU capacity is intermittent); `--sizes` overrides.

The pod:
  - reads the two Optuna study DBs from R2 (read-only),
  - writes results ONLY under stage0_cartpole/pilot/ on R2 (new prefix; nothing existing is touched),
  - resumes from stage0_cartpole/pilot/pilot_runs.jsonl after a restart,
  - idles when finished (RunPod restarts the entrypoint on exit otherwise) — fetch results with
        python scripts/pilot_greedy_rank_check.py --analyze --pull-results
    then DELETE the pod.

Usage:
    python scripts/launch_pilot_pod.py --dry-run        # redacted request, no upload, no pod
    python scripts/launch_pilot_pod.py --upload-only    # just upload the code tarball
    python scripts/launch_pilot_pod.py
"""

import argparse
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.runpod.io/v2/pods"
IMAGE = "kevinkyawthiha/dqn-ssp-centering:stage0"
CPU_FLAVOR = "cpu3c"   # same flavor the search pods used (4 vCPU = $0.12/hr there)
R2_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
CODE_KEY = "stage0_cartpole/pilot/pilot_code.tar.gz"
CODE_DIRS = ("experiments", "rl", "scripts", "conf")
# Cloudflare (in front of api.runpod.io) rejects Python's default urllib User-Agent with HTTP 403 / error 1010.
UA = "curl/8.5.0"

# Runs inside the pod as the ENTRYPOINT: fetch code from R2, extract over /project, exec the pilot.
BOOTSTRAP = r"""
import os, sys, tarfile, boto3
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'])
s3.download_file(e['R2_BUCKET'], '__CODE_KEY__', '/tmp/pilot_code.tgz')
with tarfile.open('/tmp/pilot_code.tgz') as t:
    t.extractall('/project')
os.chdir('/project')
args = e.get('PILOT_ARGS', '--r2 --idle-after --workers 4').split()
print('[bootstrap] code extracted; running pilot with:', args, flush=True)
os.execvp('python', ['python', e.get('PILOT_SCRIPT', 'scripts/pilot_greedy_rank_check.py')] + args)
""".replace("__CODE_KEY__", CODE_KEY)


def load_env():
    env = {}
    with open(os.path.join(REPO_ROOT, ".env")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    return env


def make_code_tarball():
    buf = io.BytesIO()

    def keep(ti):
        n = ti.name
        if ("__pycache__" in n or n.endswith((".pyc", ".db", ".jsonl", ".png", ".pdf", ".env"))
                or "/outputs/" in n or "/output/" in n or ".s3cfg" in n or "/.env" in n):
            return None
        return ti

    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d in CODE_DIRS:
            t.add(os.path.join(REPO_ROOT, d), arcname=d, filter=keep)
    return buf.getvalue()


def upload_code(env):
    import boto3
    data = make_code_tarball()
    s3 = boto3.client(
        "s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])
    s3.put_object(Bucket=env["R2_BUCKET"], Key=CODE_KEY, Body=data)
    print(f"uploaded code tarball ({len(data)/1024:.0f} KB) to R2 key {CODE_KEY}")


def build_body(env, vcpus, script="scripts/pilot_greedy_rank_check.py", name="dqn-pilot-greedy", extra_args=""):
    return {
        "name": f"{name}-{vcpus}vcpu",
        "image": IMAGE,
        "cloud": "SECURE",
        "disk": 20,
        "cpu": {"id": CPU_FLAVOR, "vcpuCount": vcpus},
        "entrypoint": ["python", "-c", BOOTSTRAP],
        "env": {**{k: env[k] for k in R2_KEYS}, "PILOT_ARGS": f"--r2 --idle-after --workers {vcpus}" + (f" {extra_args}" if extra_args else ""),
                "PILOT_SCRIPT": script},
    }


def redact(body):
    b = json.loads(json.dumps(body))
    b["env"] = {k: ("<redacted>" if k in R2_KEYS else v) for k, v in b["env"].items()}
    b["entrypoint"] = [b["entrypoint"][0], b["entrypoint"][1], "<bootstrap: download code tarball from R2, extract to /project, exec pilot>"]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,8,4", help="vCPU counts to try in order (powers of two)")
    ap.add_argument("--script", default="scripts/pilot_greedy_rank_check.py", help="script (inside the code tarball) the pod runs")
    ap.add_argument("--name", default="dqn-pilot-greedy", help="pod name prefix")
    ap.add_argument("--extra-args", default="", help="extra args appended to the pod script's arguments")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--upload-only", action="store_true")
    a = ap.parse_args()
    env = load_env()
    missing = [k for k in R2_KEYS if not env.get(k)]
    if missing:
        sys.exit(f"missing in .env: {missing}")
    sizes = [int(s) for s in a.sizes.split(",")]

    if a.dry_run:
        print(f"POST {API}")
        print(json.dumps(redact(build_body(env, sizes[0], a.script, a.name, a.extra_args)), indent=2))
        print("code tarball would be", f"{len(make_code_tarball())/1024:.0f} KB", "->", CODE_KEY)
        print("RUNPOD_API_KEY in .env:", "yes" if env.get("RUNPOD_API_KEY") else "NO (launch will fail)")
        return
    upload_code(env)
    if a.upload_only:
        return
    if not env.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY not found in .env")

    for n in sizes:
        body = build_body(env, n, a.script, a.name, a.extra_args)
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}", "Content-Type": "application/json",
                     "User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            print(f"created pod with {n} vCPUs: id={out.get('id')} name={out.get('name')}")
            return
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:400]
            print(f"{n} vCPUs: HTTP {e.code}: {msg}")
            if e.code in (401, 403):
                sys.exit("authentication failed — check RUNPOD_API_KEY")
        except Exception as e:
            print(f"{n} vCPUs: {e!r}")
    sys.exit("no size could be created; retry later (CPU capacity is intermittent)")


if __name__ == "__main__":
    main()
