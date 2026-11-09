"""
Launch one RunPod CPU pod per DQN cell that runs the WHOLE chained pipeline:

    top-up search (to the cell's target, stale RUNNING trials failed + compensated)
      -> greedy re-filter (pool 20 x 10 seeds, N parallel workers, resumable JSONL on R2)
      -> final eval of the winner (20 fresh seeds, greedy)
      -> everything pushed to R2, then the pod deletes itself (or idles if that is not possible).

Same delivery trick as launch_pilot_pod.py (no Docker image push: the 6 GB dependency layer is on
the existing `:stage0` image; a small code tarball is fetched from R2 at pod start), except that
the tarball key is content-hashed so already-running pods never see a different code version.

R2 locations written (all NEW; nothing existing is overwritten except each cell's own Optuna DB key,
which only gains rows/attributes and which --backup-dbs copies first):
    stage0_cartpole/refilter_v2/<study>/{reeval_v2_runs.jsonl,reeval_v2_results.json,
                                         final_eval_runs.jsonl,final_eval_summary.json}
    stage0_cartpole/best_params/dqn_<rep>_<centering>[_deep]_v2.yaml

Usage:
    python scripts/launch_cell_pod.py --cells discrete/none/linear --dry-run
    python scripts/launch_cell_pod.py --backup-dbs                 # copy all 18 v4 DBs to a backup prefix
    python scripts/launch_cell_pod.py --cells discrete/none/linear,ssp/none/deep
    python scripts/launch_cell_pod.py --cells all
Cell syntax: <representation>/<centering>/<linear|deep>.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_pilot_pod as lp  # noqa: E402  (reuses env loading, tarball builder, API constants)

REPS = ("discrete", "ssp", "tile_coding")
H2_REPS = ("ssp_h2", "ssp_h2raw", "tile_h2")  # d131: CartPole length-scale test arms (not part of ALL_CELLS)
CENTERINGS = ("none", "simple", "value")
ARCHS = ("linear", "deep")
ALL_CELLS = [f"{r}/{c}/{a}" for r in REPS for c in CENTERINGS for a in ARCHS]
CODE_PREFIX = "stage0_cartpole/refilter_v2/_code/"
BACKUP_PREFIX = "stage0_cartpole/backup_2026-10-05/optuna_dbs/"

# Runs inside the pod as the ENTRYPOINT: fetch code (R2), extract over /project, hand over to the normal
# cell entrypoint (restores the study DB from R2, then execs run_hparam_search_dqn.py with CELL_ARGS).
BOOTSTRAP = r"""
import os, sys, tarfile, boto3
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'])
s3.download_file(e['R2_BUCKET'], e['CODE_KEY'], '/tmp/cell_code.tgz')
with tarfile.open('/tmp/cell_code.tgz') as t:
    t.extractall('/project')
os.chdir('/project')
args = e['CELL_ARGS'].split()
print('[bootstrap] code', e['CODE_KEY'], 'extracted; cell args:', args, flush=True)
os.execvp('bash', ['bash', 'scripts/entrypoint_dqn_search.sh'] + args)
"""

# eval_v3 jobs (policy pass / robustness / deployment): run ONE script, then delete the pod only if it
# exited 0 (the script itself has already pushed everything to R2); otherwise idle so the failure can be
# inspected (RunPod restarts an exited entrypoint in a loop, so never just exit).
JOB_SCRIPTS = {"policy_pass": "scripts/run_policy_pass.py", "robustness": "scripts/run_robustness.py",
               "deployment": "scripts/run_deployment.py",
               # sweeps: one pod for many cells / many variants, to limit image pulls
               "latency": "scripts/run_latency_sweep.py",
               "dqn_settings": "scripts/run_dqn_settings_sweep.py"}
JOB_BOOTSTRAP = r"""
import os, sys, tarfile, subprocess, time, boto3
import urllib.request, urllib.error
e = os.environ
s3 = boto3.client('s3', endpoint_url='https://%s.r2.cloudflarestorage.com' % e['R2_ACCOUNT_ID'],
                  aws_access_key_id=e['R2_ACCESS_KEY_ID'], aws_secret_access_key=e['R2_SECRET_ACCESS_KEY'])
s3.download_file(e['R2_BUCKET'], e['EVAL_CODE_KEY'], '/tmp/cell_code.tgz')
with tarfile.open('/tmp/cell_code.tgz') as t:
    t.extractall('/project')
os.chdir('/project')
cmd = [sys.executable, e['JOB_SCRIPT']] + e['JOB_ARGS'].split()
print('[bootstrap] code', e['EVAL_CODE_KEY'], 'extracted; running', cmd, flush=True)
rc = subprocess.call(cmd)
print('[bootstrap] job exited with status', rc, flush=True)
pod, key = e.get('RUNPOD_POD_ID'), e.get('RUNPOD_API_KEY')
if rc == 0 and pod and key:
    req = urllib.request.Request('https://api.runpod.io/v2/pods/%s' % pod, method='DELETE',
                                 headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'curl/8.5.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print('[self-delete] DELETE pod', pod, '-> HTTP', r.status, flush=True)
    except urllib.error.HTTPError as ex:
        print('[self-delete] DELETE pod', pod, '-> HTTP', ex.code, ex.read().decode()[:200], flush=True)
    except Exception as ex:
        print('[self-delete] failed:', repr(ex), flush=True)
elif rc == 0:
    print('[self-delete] RUNPOD_POD_ID/RUNPOD_API_KEY not in env - idling instead.', flush=True)
else:
    print('[bootstrap] FAILED - leaving the pod up for inspection.', flush=True)
while True:
    time.sleep(3600)
"""


def parse_cell(cell: str):
    rep, cen, arch = cell.split("/")
    assert rep in REPS + H2_REPS and cen in CENTERINGS and arch in ARCHS + ("a2c",), cell  # "a2c": eval_v3 A2C-port cells
    return rep, cen, arch


def study_name(cell: str) -> str:
    rep, cen, arch = parse_cell(cell)
    if arch == "a2c":
        return f"a2c_{rep}_{cen}_port"
    return f"dqn_{rep}_{cen}{'' if arch == 'linear' else '_deep'}_v4_hparam_search"


def s3_client(env):
    import boto3
    return boto3.client(
        "s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])


def upload_code(env, prefix: str = CODE_PREFIX) -> str:
    """Upload the code tarball under a content-hashed key; returns the key."""
    data = lp.make_code_tarball()
    key = f"{prefix}cell_code_{hashlib.sha256(data).hexdigest()[:12]}.tar.gz"
    s3 = s3_client(env)
    try:
        s3.head_object(Bucket=env["R2_BUCKET"], Key=key)
        print(f"code tarball already on R2: {key} ({len(data)/1024:.0f} KB)")
    except Exception:  # noqa: BLE001
        s3.put_object(Bucket=env["R2_BUCKET"], Key=key, Body=data)
        print(f"uploaded code tarball ({len(data)/1024:.0f} KB): {key}")
    return key


def backup_dbs(env):
    """Server-side copy of every cell's v4 Optuna DB to a backup prefix (never overwrites)."""
    s3, bucket = s3_client(env), env["R2_BUCKET"]
    for cell in ALL_CELLS:
        name = study_name(cell)
        src = f"stage0_cartpole/optuna_dbs/optuna_studies_{name}.db"
        dst = f"{BACKUP_PREFIX}optuna_studies_{name}.db"
        try:
            s3.head_object(Bucket=bucket, Key=dst)
            print(f"exists, skipping: {dst}")
            continue
        except Exception:  # noqa: BLE001
            pass
        s3.copy_object(Bucket=bucket, Key=dst, CopySource={"Bucket": bucket, "Key": src})
        print(f"backed up {name}")


def cell_args(cell: str, vcpus: int, n_trials: int, extra: str = "") -> str:
    rep, cen, arch = parse_cell(cell)
    a = f"--representation {rep} --centering {cen} --architecture {arch} --n-trials {n_trials} " \
        f"--workers {vcpus} --final-eval --idle-after --self-delete"
    return a + (f" {extra}" if extra else "")


def registry_auth(env) -> dict:
    """Optional RunPod container-registry credential, referenced by id.

    Anonymous Docker Hub pulls share a per-IP rate limit, and hitting it ("toomanyrequests") killed six
    pods during the first eval_v3 launch wave. Create the credential once with

        runpodctl registry create --name dockerhub --username <dockerhub-user>   # password from prompt/stdin

    so the secret never enters this repo, then put its id in .env as
    RUNPOD_REGISTRY_AUTH_ID=<id from `runpodctl registry list`>. Absent, pods pull anonymously
    (previous behaviour).
    """
    rid = env.get("RUNPOD_REGISTRY_AUTH_ID")
    return {"containerRegistryAuthId": rid} if rid else {}


def build_body(env, cell: str, vcpus: int, code_key: str, n_trials: int, extra: str = ""):
    rep, cen, arch = parse_cell(cell)
    return {
        "name": f"dqn-v2-{rep}-{cen}-{arch}".replace("_", "-"),
        "image": lp.IMAGE,
        "cloud": "SECURE",
        "disk": 20,
        "cpu": {"id": lp.CPU_FLAVOR, "vcpuCount": vcpus},
        "entrypoint": ["python", "-c", BOOTSTRAP],
        "env": {**{k: env[k] for k in lp.R2_KEYS}, "CODE_KEY": code_key,
                "CELL_ARGS": cell_args(cell, vcpus, n_trials, extra)},
        **registry_auth(env),
    }


# Runners that take a single --cell. The sweeps take --cells / their own selection instead.
JOB_TAKES_CELL = {"policy_pass": True, "robustness": True, "deployment": True,
                  "dqn_settings": True, "latency": False}


def job_args(job: str, cell: str, vcpus: int, extra: str = "") -> str:
    """`--cell` and `--workers`, then any extra runner args.

    The launcher's own `--workers` is omitted when the caller already passes one, so the runner never
    receives a duplicate flag (argparse would silently keep the last one and the pod would be sized
    differently from its vCPU count).
    """
    bits = [f"--cell {cell}"] if JOB_TAKES_CELL.get(job, True) else []
    if "--workers" not in extra:
        bits.append(f"--workers {vcpus}")
    if extra:
        bits.append(extra)
    return " ".join(bits)


def variant_slug(extra: str) -> str:
    """`--variant-buffer-size 50000 --variant-target-update-freq 100` -> `buf50000-tu100`.

    Without this every DQN-settings variant pod shares the baseline pod's name, which made the
    variants indistinguishable from duplicates (and the duplicate reaper had to be taught to tell
    them apart by comparing JOB_ARGS).
    """
    buf = tu = None
    toks = extra.split()
    for i, t in enumerate(toks):
        if t in ("--variant-buffer-size", "--variant-target-update-freq") and i + 1 < len(toks):
            (buf := toks[i + 1]) if t.endswith("buffer-size") else (tu := toks[i + 1])
    bits = [f"buf{buf}"] if buf else []
    if tu:
        bits.append(f"tu{tu}")
    return "-".join(bits)


def build_job_body(env, job: str, cell: str, vcpus: int, code_key: str, extra: str = ""):
    rep, cen, arch = parse_cell(cell)
    slug = variant_slug(extra)
    # `.replace` applies to the whole name: the job name ("policy_pass") and rep ("tile_coding") both
    # contain underscores, and dropping the replace made pods unrecognisable to the watcher.
    name = (f"eval-v3-{job}-{rep}-{cen}-{arch}" + (f"-{slug}" if slug else "")).replace("_", "-")
    return {
        "name": name,
        "image": lp.IMAGE,
        "cloud": "SECURE",
        "disk": 20,
        "cpu": {"id": lp.CPU_FLAVOR, "vcpuCount": vcpus},
        "entrypoint": ["python", "-c", JOB_BOOTSTRAP],
        "env": {**{k: env[k] for k in lp.R2_KEYS}, "EVAL_CODE_KEY": code_key,
                "JOB_SCRIPT": JOB_SCRIPTS[job], "JOB_ARGS": job_args(job, cell, vcpus, extra)},
        **registry_auth(env),
    }


def redact(body):
    b = json.loads(json.dumps(body))
    b["env"] = {k: ("<redacted>" if k in lp.R2_KEYS else v) for k, v in b["env"].items()}
    if "JOB_SCRIPT" in b["env"]:
        b["entrypoint"] = ["python", "-c", "<bootstrap: fetch code tarball from R2, extract to /project, run "
                                           "JOB_SCRIPT, self-delete the pod on exit status 0, else idle>"]
    else:
        b["entrypoint"] = ["python", "-c", "<bootstrap: fetch code tarball from R2, extract to /project, run entrypoint>"]
    return b


def create_pod(env, body):
    req = urllib.request.Request(
        lp.API, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {env['RUNPOD_API_KEY']}", "Content-Type": "application/json",
                 "User-Agent": lp.UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def run_job_mode(env, a, cells, sizes):
    """eval_v3 pods: same CPU pod pattern, code under layout.code_prefix(), one runner script per pod."""
    import eval_v3_layout as layout
    if not cells:
        sys.exit("give --cells")
    for c in cells:
        parse_cell(c)
    if a.dry_run:
        print(f"POST {lp.API}")
        for c in cells:
            print(json.dumps(redact(build_job_body(env, a.job, c, sizes[0], "<code key>", a.job_args)), indent=2))
        print("code tarball would be", f"{len(lp.make_code_tarball())/1024:.0f} KB", "->", layout.code_prefix())
        print("RUNPOD_API_KEY in .env:", "yes" if env.get("RUNPOD_API_KEY") else "NO")
        return
    if not env.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY not found in .env")
    code_key = upload_code(env, layout.code_prefix())
    for c in cells:
        done = False
        for n in sizes:
            try:
                out = create_pod(env, build_job_body(env, a.job, c, n, code_key, a.job_args))
                print(f"CREATED {a.job} {c}: pod id={out.get('id')} name={out.get('name')} vcpus={n}", flush=True)
                done = True
                break
            except urllib.error.HTTPError as e:
                print(f"{c} {n} vCPUs: HTTP {e.code}: {e.read().decode()[:300]}", flush=True)
                if e.code in (401, 403):
                    sys.exit("authentication failed — check RUNPOD_API_KEY")
            except Exception as e:  # noqa: BLE001
                print(f"{c} {n} vCPUs: {e!r}", flush=True)
        if not done:
            print(f"NOT CREATED {c} (capacity?) — retry later", flush=True)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma list of rep/centering/arch, or 'all'")
    ap.add_argument("--sizes", default="4", help="vCPU counts to try in order (default 4: search pods' size)")
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--cell-args", default="", help="extra CLI args appended to the search script (search jobs only)")
    ap.add_argument("--backup-dbs", action="store_true", help="copy all 18 v4 DBs to the backup prefix and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--job", choices=["search", "policy_pass", "robustness", "deployment",
                                      "latency", "dqn_settings"], default="search",
                    help="search (default): the original chained search/filter/final-eval pod. The others run "
                         "the eval_v3 runner of that name (code under the eval_v3 _code prefix).")
    ap.add_argument("--job-args", default="", help="extra CLI args appended to the eval_v3 runner")
    a = ap.parse_args()
    env = lp.load_env()
    missing = [k for k in lp.R2_KEYS if not env.get(k)]
    if missing:
        sys.exit(f"missing in .env: {missing}")
    if a.backup_dbs:
        backup_dbs(env)
        return
    cells = ALL_CELLS if a.cells == "all" else [c for c in a.cells.split(",") if c]
    if a.job != "search":
        run_job_mode(env, a, cells, [int(s) for s in a.sizes.split(",")])
        return
    if not cells:
        sys.exit("give --cells (or --backup-dbs)")
    for c in cells:
        parse_cell(c)
    sizes = [int(s) for s in a.sizes.split(",")]

    if a.dry_run:
        print(f"POST {lp.API}")
        for c in cells:
            print(json.dumps(redact(build_body(env, c, sizes[0], "<code key>", a.n_trials, a.cell_args)), indent=2))
        print("code tarball would be", f"{len(lp.make_code_tarball())/1024:.0f} KB")
        print("RUNPOD_API_KEY in .env:", "yes" if env.get("RUNPOD_API_KEY") else "NO")
        return
    if not env.get("RUNPOD_API_KEY"):
        sys.exit("RUNPOD_API_KEY not found in .env")
    code_key = upload_code(env)
    for c in cells:
        done = False
        for n in sizes:
            body = build_body(env, c, n, code_key, a.n_trials, a.cell_args)
            try:
                out = create_pod(env, body)
                print(f"CREATED {c}: pod id={out.get('id')} name={out.get('name')} vcpus={n}", flush=True)
                done = True
                break
            except urllib.error.HTTPError as e:
                print(f"{c} {n} vCPUs: HTTP {e.code}: {e.read().decode()[:300]}", flush=True)
                if e.code in (401, 403):
                    sys.exit("authentication failed — check RUNPOD_API_KEY")
            except Exception as e:  # noqa: BLE001
                print(f"{c} {n} vCPUs: {e!r}", flush=True)
        if not done:
            print(f"NOT CREATED {c} (capacity?) — retry later", flush=True)
        time.sleep(2)   # one creation at a time (parallel creates were intermittently blocked before)


if __name__ == "__main__":
    main()
