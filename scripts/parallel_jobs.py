"""
Parallel, resumable job runner for the DQN re-filter and final-eval steps.

Why this exists (see docs/handoffs/02-dqn-v4-greedy-refilter-plan.md and the 2026-10-05 build notes):
the original reeval ran every (candidate, seed) training run one at a time in a single process
(3 of a 4-vCPU pod's cores idle) and only persisted results at the end of a stage, so a hung or
restarted pod redid everything. Here:

  * every job is one (candidate, seed) run; N worker processes (spawn) execute them in parallel;
  * every finished job is appended to a JSONL file immediately and, when R2 is available, the file
    is pushed to R2 right away;
  * on start the JSONL is restored from R2 (a restarted pod has a wiped disk) and jobs whose key is
    already recorded are skipped, so a hang or restart loses at most `workers` runs;
  * a failed job (exception) is recorded with an "error" field, is NOT counted as done, and is retried
    in a later round; a dead worker process (e.g. OOM) breaks the pool, which is rebuilt.

Only runs within this module's rounds are retried; a job that keeps failing after `max_rounds`
raises RuntimeError so the caller does not silently score a candidate on partial data.
"""

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
from concurrent.futures.process import BrokenProcessPool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------------
# R2 helpers (never raise: a failed push must not kill a multi-hour run; the caller sees the bool)
# ---------------------------------------------------------------------------------------------
def _env():
    env = dict(os.environ)
    if "R2_ACCOUNT_ID" not in env:
        try:
            with open(os.path.join(REPO_ROOT, ".env")) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env.setdefault(k, v.strip().strip('"').strip("'"))
        except FileNotFoundError:
            pass
    return env


def r2_client():
    """(s3, bucket) or (None, None) when no R2 credentials are available."""
    env = _env()
    if not all(env.get(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")):
        return None, None
    import boto3
    s3 = boto3.client(
        "s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])
    return s3, env["R2_BUCKET"]


def pull_file(key: str, local: str) -> str:
    """Download `key` to `local`. Returns 'ok', 'missing' (key does not exist) or 'error'."""
    s3, bucket = r2_client()
    if s3 is None:
        return "error"
    try:
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        s3.download_file(bucket, key, local)
        return "ok"
    except Exception as e:  # noqa: BLE001
        msg = repr(e)
        if "404" in msg or "NoSuchKey" in msg or "Not Found" in msg:
            return "missing"
        print(f"[r2] pull {key} FAILED: {msg[:200]}", flush=True)
        return "error"


def repair_jsonl(path: str) -> int:
    """Rewrite `path` keeping only parseable lines (each newline-terminated). A pod killed mid-write can
    leave a truncated last line WITHOUT a newline; appending to it would glue the next record onto the
    garbage and lose that record too. Returns the number of dropped lines."""
    if not os.path.exists(path):
        return 0
    good, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                good.append(line.rstrip("\n"))
            except json.JSONDecodeError:
                bad += 1
    if bad or (good and not open(path).read().endswith("\n")):
        with open(path, "w") as f:
            f.writelines(g + "\n" for g in good)
    return bad


def push_file(key: str, local: str) -> bool:
    s3, bucket = r2_client()
    if s3 is None:
        return False
    try:
        s3.upload_file(local, bucket, key)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[r2] push {key} FAILED: {repr(e)[:200]}", flush=True)
        return False


# ---------------------------------------------------------------------------------------------
# JSONL bookkeeping
# ---------------------------------------------------------------------------------------------
def read_jsonl(path: str) -> list:
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A pod killed mid-write can leave a truncated last line; that job just reruns.
                    print(f"[jobs] ignoring unparseable line in {path}", flush=True)
    return out


def completed_records(path: str, key_fn) -> dict:
    """key -> record for every non-error line (the last one wins if duplicated)."""
    return {key_fn(r): r for r in read_jsonl(path) if "error" not in r}


def init_worker():
    """Run in each worker process: single-threaded torch, no tqdm spam."""
    os.environ["TQDM_DISABLE"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def run_jobs(jobs: list, worker_fn, key_fn, jsonl_path: str, workers: int,
             r2_key: str = None, label: str = "jobs", max_rounds: int = 3) -> dict:
    """Run every job whose key is not yet recorded; return {key: record} for ALL jobs.

    jobs      : list of picklable dicts (consumed by worker_fn in a spawned process)
    worker_fn : top-level function job -> record dict (must include whatever key_fn needs); it
                should catch its own exceptions and return {..., "error": repr(e)} instead of raising
    key_fn    : record/job -> hashable key (must give the same key for a job and its record)
    r2_key    : if set, the JSONL is restored from R2 first (when absent locally) and pushed after
                every finished job.
    """
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    if r2_key and not os.path.exists(jsonl_path):
        st = pull_file(r2_key, jsonl_path)
        print(f"[{label}] restore {r2_key}: {st}", flush=True)
    dropped = repair_jsonl(jsonl_path)
    if dropped:
        print(f"[{label}] dropped {dropped} truncated line(s) from {jsonl_path}", flush=True)

    ctx = mp.get_context("spawn")
    for rnd in range(1, max_rounds + 1):
        done = completed_records(jsonl_path, key_fn)
        todo = [j for j in jobs if key_fn(j) not in done]
        if rnd == 1:
            print(f"[{label}] {len(jobs)} jobs: {len(jobs) - len(todo)} already recorded, "
                  f"{len(todo)} to run with {workers} workers", flush=True)
        if not todo:
            break
        if rnd > 1:
            print(f"[{label}] round {rnd}: retrying {len(todo)} jobs that did not complete", flush=True)
        n_done, t0 = 0, time.time()
        try:
            with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                        initializer=init_worker) as ex, open(jsonl_path, "a") as out:
                futs = [ex.submit(worker_fn, j) for j in todo]
                for fut in cf.as_completed(futs):
                    try:
                        rec = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as e:  # noqa: BLE001
                        print(f"[{label}] worker raised {e!r}", flush=True)
                        continue
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    n_done += 1
                    if "error" in rec:
                        print(f"[{label}] [{n_done}/{len(todo)}] ERROR {rec['error'][:120]}", flush=True)
                    else:
                        el = (time.time() - t0) / 60
                        print(f"[{label}] [{n_done}/{len(todo)}] {_brief(rec)}  ({el:.1f} min elapsed)",
                              flush=True)
                    if r2_key:
                        push_file(r2_key, jsonl_path)
        except BrokenProcessPool:
            print(f"[{label}] a worker process died (OOM?); rebuilding the pool", flush=True)
    if r2_key and os.path.exists(jsonl_path):
        push_file(r2_key, jsonl_path)

    done = completed_records(jsonl_path, key_fn)
    missing = [j for j in jobs if key_fn(j) not in done]
    if missing:
        raise RuntimeError(f"[{label}] {len(missing)} of {len(jobs)} jobs still failing after "
                           f"{max_rounds} rounds, e.g. key={key_fn(missing[0])!r}")
    return {key_fn(j): done[key_fn(j)] for j in jobs}


def _brief(rec: dict) -> str:
    bits = []
    for k in ("trial", "rank", "seed_idx", "seed"):
        if k in rec:
            bits.append(f"{k}={rec[k]}")
    if "score" in rec:
        bits.append(f"score={rec['score']:.1f}")
    if "last1" in rec:
        bits.append(f"last1={rec['last1']:.0f}")
    return " ".join(bits)


# ---------------------------------------------------------------------------------------------
# The concrete job used by the DQN re-filter (kind="reeval") and final eval (kind="final")
# ---------------------------------------------------------------------------------------------
def job_key(x: dict) -> tuple:
    """Same key for a job dict and the record its worker returns."""
    return (x["kind"], int(x["trial"]), int(x["seed_idx"]))


def eval_worker(job: dict) -> dict:
    """One full DQN training run with greedy eval checkpoints. Runs in a spawned worker process.
    Never raises: failures come back as {..., "error": ...} so the parent can record and retry them."""
    meta = {k: job[k] for k in ("kind", "trial", "rank", "seed_idx")}
    data_dir = os.path.join(job["data_dir"], f"{job['kind']}_t{job['trial']}_s{job['seed_idx']}")
    try:
        os.makedirs(data_dir, exist_ok=True)
        import run_hparam_search_dqn as m
        _silence_tqdm()
        res = m._run_reeval_seed(
            job["params"], data_dir, job["seed_idx"], job["tag"], job["eval_freq"],
            job["n_eval_episodes"], job["last_k"], seed_offset=job["seed_offset"],
            final_eval_episodes=job["final_eval_episodes"])
        return {**meta, **res}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {**meta, "error": repr(e) + " | " + traceback.format_exc()[-400:]}


def pod_info(workers: int) -> dict:
    """One-line description of where/how a result was produced (agreed metadata)."""
    import platform
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch
        tv = torch.__version__
    except Exception:  # noqa: BLE001
        tv = None
    return {"pod_id": os.environ.get("RUNPOD_POD_ID"), "cpu_model": cpu, "n_cpu": os.cpu_count(),
            "workers": workers, "torch": tv, "python": platform.python_version()}


def _silence_tqdm():
    """The trial loop does `from tqdm import tqdm; for ... in tqdm(range(n))` (no disable flag) and the
    TQDM_DISABLE env var is not honored by every tqdm version, so 4 workers spray progress bars into the
    pod's log pipe (suspected contributor to the old 'Bug #2' hangs). Wrap it with disable=True."""
    import functools
    import sys
    mod = sys.modules.get("experiments.trial_cartpole_dqn")
    if mod is not None and not getattr(mod.tqdm, "_silenced", False):
        quiet = functools.partial(mod.tqdm, disable=True)
        quiet._silenced = True
        mod.tqdm = quiet
