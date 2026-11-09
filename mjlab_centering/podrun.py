"""Run a job manifest on one machine with N parallel training processes (the pod-side queue, no R2 control plane).

Measured on an A40: 5 concurrent 2048-env processes give ~3.5x the throughput of one (2.3-2.6 s/iteration each, GPU
at 100%, 5 GB), so a pod runs several jobs at once.

    python podrun.py --manifest manifests/w1.json --out /project/results/w1 --procs 6 [--blocks eta,sslen]

A job may carry `extra: {"after": "<mother_job_id>"}`: it starts once the mother's `ckpt_it<save_at>.pt` exists and
is resumed from it (shock branching). Mothers must precede their branches in the manifest. Finished jobs (final.json)
are skipped, so the queue can be restarted safely.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _r2_exists(rel: str) -> bool:
  try:
    from mjlab_centering import r2 as _r2

    return _r2.exists(rel)
  except Exception:
    return False


def _r2_fetch(rel: str, dest: Path) -> bool:
  """Download a mother checkpoint from R2 (boto3 writes to a temp file and renames, so a partial file never appears)."""
  try:
    from mjlab_centering import r2 as _r2

    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(_r2.download_file(rel, str(dest)))
  except Exception:
    return False


CLAIM_TTL_S = 4 * 3600.0  # longer than the longest job (2500 iterations, about 2.5 h at a bad 0.2 it/s)


def claim_is_free(existing_ts: float | None, now: float, ttl: float = CLAIM_TTL_S) -> bool:
  """A job is free unless another pod claimed it recently. Shared across pods through R2 (d140)."""
  return existing_ts is None or (now - existing_ts) > ttl


def _r2_claim(rel: str, jid: str) -> bool:
  """Take (or find taken) the R2 claim for a job. True = we own it, False = another live pod owns it."""
  try:
    from mjlab_centering import r2 as _r2

    key = f"{rel}/claims/{jid}.json"
    now = time.time()
    try:
      body = _r2.client().get_object(Bucket=_r2.bucket(), Key=_r2.key(key))["Body"].read()
      existing = float(json.loads(body).get("ts", 0.0))
    except Exception:  # noqa: BLE001 - no claim yet
      existing = None
    if not claim_is_free(existing, now):
      # A claim from a pod that no longer exists is stale no matter how recent: the lease only protects against a LIVE
      # second runner, and a preempted pod never runs its release path (seen 2026-11-08, six jobs frozen at ~850).
      owner = json.loads(body).get("owner") if existing is not None else None
      # Fail CLOSED: an unreachable pod list means "I cannot tell", not "everyone is gone". Inside a pod this call has
      # repeatedly failed (runpodctl is not on the bootstrap's PATH), which made every pod reclaim every claim and run
      # the same job two or three times at once (measured 2026-11-09: three pods on one seed, all billed).
      alive: set[str] | None
      try:
        from mjlab_centering import pods as _pods
        alive = {str(p.get("id")) for p in _pods.list_pods()}
        if not alive:
          alive = None
      except Exception:  # noqa: BLE001
        alive = None
      if owner and alive is not None and owner not in alive:
        print(f"[podrun] reclaiming {jid}: its claim belongs to pod {owner}, which is gone", flush=True)
      else:
        return False
    owner = os.environ.get("RUNPOD_POD_ID") or os.uname().nodename
    _r2.put_json(key, {"ts": now, "owner": owner})
    return True
  except Exception:  # noqa: BLE001 - R2 trouble must never stop a run
    return True


def mother_prefix(extra: dict, r2: str | None) -> str | None:
  """R2 campaign that holds a branch's mother: `extra.after_campaign` if given (branch-only manifests), else this campaign."""
  return (extra or {}).get("after_campaign") or r2


def run_job(job: dict, out_root: Path, py: str, env: dict, log_lock: threading.Lock, r2: str | None = None) -> dict:
  jid = job["job_id"]
  out = out_root / jid
  if (out / "final.json").exists():
    return {"job_id": jid, "status": "skipped"}
  # d1019 (was d168): resume from the rolling checkpoint this job mirrored, so a preempted pod continues instead of restarting.
  resumed = False
  if r2 and not job.get("resume"):
    ck_dest = out / "ckpt_latest.pt"
    if _r2_fetch(f"{r2}/runs/{jid}/ckpt_latest.pt", ck_dest):
      job = {**job, "resume": str(ck_dest)}
      resumed = True
      # keep the existing curve: iter.jsonl is appended to, so the pre-resume rows must be present locally or the next
      # upload would replace the mirrored file with only the post-resume part
      _r2_fetch(f"{r2}/runs/{jid}/iter.jsonl", out / "iter.jsonl")
      _r2_fetch(f"{r2}/runs/{jid}/eval.jsonl", out / "eval.jsonl")
      with log_lock:
        print(f"[podrun] {jid} resuming from the mirrored checkpoint", flush=True)
  if not resumed and ((out / "iter.jsonl").exists() or (out / "eval.jsonl").exists()):
    # A previous attempt died mid-run with no checkpoint: stale rows would be appended to by the new attempt (iter
    # restarts at 0, the eval curve mixes two policies). Start clean; only finished runs (final.json) are ever reused.
    for f in ("iter.jsonl", "eval.jsonl", "ckpt_it*.pt"):
      for path in out.glob(f):
        path.unlink()
  if r2 and _r2_exists(f"{r2}/runs/{jid}/DONE"):  # a replacement pod must not redo what R2 already certifies
    return {"job_id": jid, "status": "skipped_r2_done"}
  if r2 and not _r2_claim(r2, jid):  # d140: another pod is running it right now
    return {"job_id": jid, "status": "skipped_inflight"}
  ex = job.get("extra") or {}
  after = ex.get("after")
  if after:
    ck = out_root / after / f"ckpt_it{job['shock_at']}.pt"
    # Branch-only manifests (d116) name the finished campaign that holds the mother: its checkpoint is already on R2, so a
    # failed fetch is final after a few tries instead of waiting hours for a mother that is not coming.
    src = mother_prefix(ex, r2)
    fixed = bool(ex.get("after_campaign"))
    t0, tries = time.time(), 0
    while not ck.exists():
      if src and _r2_fetch(f"{src}/runs/{after}/ckpt_it{job['shock_at']}.pt", ck):  # mother finished on another pod
        break
      tries += 1
      if fixed and tries >= 5:
        return {"job_id": jid, "status": "mother_missing"}
      if time.time() - t0 > 6 * 3600:
        return {"job_id": jid, "status": "mother_timeout"}
      time.sleep(20)
    time.sleep(5)  # let the checkpoint finish flushing
    job = {**job, "resume": str(ck)}
  out.mkdir(parents=True, exist_ok=True)
  for attempt in (1, 2):
    t0 = time.time()
    with open(out_root / f"{jid}.log", "a") as lf:
      cmd = [py, "-m", "mjlab_centering.train", "--job", json.dumps(job), "--out", str(out), "--quiet"]
      if r2:  # mirror every artefact to R2 as the run produces it, so a dead pod loses nothing
        cmd += ["--r2", r2]  # r2 = the campaign id: artefacts land in <campaign>/runs/<jid>/ like the worker path
      rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    if rc == 0 and (out / "final.json").exists():
      if r2:  # DONE marker is the durable record; the claim is only a lease while running
        try:
          from mjlab_centering import r2 as _r2
          _r2.client().delete_object(Bucket=_r2.bucket(), Key=_r2.key(f"{r2}/claims/{jid}.json"))
        except Exception:  # noqa: BLE001
          pass
      return {"job_id": jid, "status": "ok", "wall_s": time.time() - t0, "attempt": attempt}
    with log_lock:
      print(f"[podrun] {jid} failed rc={rc} attempt {attempt}", flush=True)
  # Release the claim: a job that failed did not run, and leaving the lease behind made every later pod skip all 30
  # jobs as "in_flight" for four hours (d1013). Only a *successful* job should keep its claim, and even then DONE covers it.
  if r2:
    try:
      from mjlab_centering import r2 as _r2
      _r2.client().delete_object(Bucket=_r2.bucket(), Key=_r2.key(f"{r2}/claims/{jid}.json"))
    except Exception:  # noqa: BLE001
      pass
  # A failed run leaves nothing in R2, so the only way to diagnose it is the per-job log. Upload it (d1012).
  if r2:
    try:
      from mjlab_centering import r2 as _r2
      _r2.upload_file(str(out_root / f"{jid}.log"), f"{r2}/logs/fail_{jid}.log")
    except Exception as e:  # noqa: BLE001
      print(f"[podrun] could not upload the failure log for {jid}: {e!r}", flush=True)
  return {"job_id": jid, "status": "failed"}


def main() -> None:
  # One queue runner per machine: two runners (e.g. a keeper restart racing an orphaned one) would run the same job
  # twice, and the two attempts append to the same JSONL, interleaving iteration counters (seen on pod 1, 2026-11-01).
  import fcntl
  lock = open("/tmp/podrun.lock", "w")
  try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    print("[podrun] another queue runner holds /tmp/podrun.lock; exiting", flush=True)
    return
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--procs", type=int, default=6)
  ap.add_argument("--blocks", default=None)
  ap.add_argument("--python", default=sys.executable)
  ap.add_argument("--r2", default=None, help="R2 campaign id (e.g. r4sweep-ab12cd34); each job mirrors to <campaign>/runs/<job_id>/ under mjlab_ssp_centering/. `auto` uses the manifest's own campaign field, which is what a warm kept pod needs (d165): the queue it receives is a *different* campaign from the one it was launched with, and its runs must land under that campaign for the readouts to find them.")
  a = ap.parse_args()
  man = json.loads(Path(a.manifest).read_text())
  jobs = man["jobs"]
  if a.r2 == "auto":
    a.r2 = man.get("campaign")
    if not a.r2:
      raise SystemExit("--r2 auto needs a `campaign` field in the queue manifest")
    print(f"[podrun] r2 campaign from the manifest: {a.r2}", flush=True)
  if a.blocks:
    keep = set(a.blocks.split(","))
    jobs = [j for j in jobs if j["block"] in keep or (j.get("extra") or {}).get("after") in {k["job_id"] for k in jobs if k["block"] in keep}]
  out_root = Path(a.out)
  out_root.mkdir(parents=True, exist_ok=True)
  env = {**os.environ, "MUJOCO_GL": "disable"}
  lock = threading.Lock()
  results = []
  with ThreadPoolExecutor(max_workers=a.procs) as ex:
    futs = [ex.submit(run_job, j, out_root, a.python, env, lock, a.r2) for j in jobs]
    for f in futs:
      r = f.result()
      results.append(r)
      with lock:
        print(f"[podrun] {r['job_id']} -> {r['status']}", flush=True)
      (out_root / "queue_status.json").write_text(json.dumps(results, indent=1))
  print(f"[podrun] finished: {sum(r['status'] in ('ok','skipped') for r in results)}/{len(results)} ok", flush=True)


if __name__ == "__main__":
  main()
