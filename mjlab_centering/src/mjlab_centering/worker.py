"""Pod worker: claim jobs from an R2 campaign, run them, mark them done, self-delete when the queue drains.

The worker never trusts local state: the campaign manifest and the `DONE` markers on R2 are authoritative, and
every claim is an atomic conditional put (see r2.claim). A restarted pod resumes at the next unclaimed job; a
run that dies mid-way leaves a stale claim that the orchestrator (or another pod) can take over after the TTL.

    python -m mjlab_centering.worker --campaign c1-abc123          # run until the queue drains
    python -m mjlab_centering.worker --campaign c1-abc123 --max-jobs 1
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

from mjlab_centering import r2
from mjlab_centering.jobs import build_jobs
from mjlab_centering.train import build_spec, run


def _heartbeat_loop(campaign: str, job_id: str, pod: str, stop: threading.Event) -> None:
  while not stop.wait(60):
    r2.heartbeat(campaign, job_id, pod)


def _safe(fn, *a, **kw) -> None:
  try:
    fn(*a, **kw)
  except Exception as e:  # R2 hiccups must not kill the pod
    print(f"[worker] non-fatal: {fn.__name__}: {e!r}", flush=True)


def clear_stale(out_dir: str) -> list[str]:
  """Drop artefacts of an earlier attempt at the same job before starting a new one.

  A re-claimed job (pod restart, claim expiry) starts again at iteration 0. Without this, the new attempt appends to
  the old `iter.jsonl`/`eval.jsonl` and mirrors the interleaved file to the same R2 key, which mixes two policies on
  one iteration axis (seen on `pilot_g1_raw_none_o1_s1000`: 500 duplicated iterations). `podrun` already did this;
  the worker path did not.
  """
  from pathlib import Path

  d = Path(out_dir)
  if not d.exists():
    return []
  removed = []
  for name in ("iter.jsonl", "eval.jsonl", "final.json", "DONE"):
    f = d / name
    if f.exists():
      f.unlink()
      removed.append(name)
  for f in d.glob("ckpt_it*.pt"):
    f.unlink()
    removed.append(f.name)
  return removed


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True, help="campaign id, e.g. c1-abc123")
  ap.add_argument("--max-jobs", type=int, default=0, help="stop after N jobs (0 = until drained)")
  ap.add_argument("--self-delete", action="store_true")
  ap.add_argument("--idle-after", action="store_true", help="sleep instead of exiting when drained")
  ap.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "/project/results"))
  a = ap.parse_args()

  pod = os.environ.get("RUNPOD_POD_ID") or socket.gethostname()
  n_done = 0
  n_failed_streak = 0
  print(f"[worker] pod={pod} campaign={a.campaign}", flush=True)

  while True:
    try:
      manifest = r2.get_json(f"{a.campaign}/manifest.json")
    except Exception as e:
      print(f"[worker] manifest read failed, retrying: {e!r}", flush=True)
      time.sleep(30)
      continue
    if manifest is None:
      print(f"[worker] no manifest at {a.campaign}/manifest.json", flush=True)
      return
    jobs = [build_spec(j) for j in manifest["jobs"]]
    jobs.sort(key=lambda j: (j.priority, j.job_id))
    done = r2.done_set(a.campaign)
    pending = [j for j in jobs if j.job_id not in done]
    if not pending:
      print(f"[worker] queue drained ({len(done)} done)", flush=True)
      break

    ran_one = False
    for spec in pending:
      if r2.exists(f"{a.campaign}/runs/{spec.job_id}/DONE"):
        continue
      if not r2.claim(a.campaign, spec.job_id, pod):
        continue
      print(f"[worker] claimed {spec.job_id}", flush=True)
      stop = threading.Event()
      th = threading.Thread(target=_heartbeat_loop, args=(a.campaign, spec.job_id, pod, stop), daemon=True)
      th.start()
      try:
        out_dir = os.path.join(a.results_dir, spec.job_id)
        dropped = clear_stale(out_dir)
        if dropped:
          print(f"[worker] {spec.job_id}: cleared stale attempt ({', '.join(dropped)})", flush=True)
        run(spec, out_dir, r2_campaign=a.campaign)
        r2.release(a.campaign, spec.job_id)  # DONE is authoritative; the claim is no longer needed
        n_done += 1
        n_failed_streak = 0
        ran_one = True
      except Exception as e:
        n_failed_streak += 1
        print(f"[worker] job {spec.job_id} FAILED ({n_failed_streak}/3): {e!r}", flush=True)
        import traceback

        traceback.print_exc()
        _safe(r2.put_json, f"{a.campaign}/runs/{spec.job_id}/FAILED", {"error": repr(e), "pod": pod})
        _safe(r2.release, a.campaign, spec.job_id)
      finally:
        stop.set()
        th.join(timeout=2)
      if n_failed_streak >= 3:
        # Do NOT exit: an exiting entrypoint makes RunPod restart the container, which re-pays the whole bootstrap
        # (~95 min) and loses the warm mjlab/kernel cache. Park and retry instead.
        print("[worker] 3 consecutive job failures: parking this pod for 10 min before retrying", flush=True)
        _safe(r2.put_json, f"{a.campaign}/runs/_pod_failures/{pod}.json", {"streak": n_failed_streak})
        time.sleep(600)
        n_failed_streak = 0
      if a.max_jobs and n_done >= a.max_jobs:
        break
    if a.max_jobs and n_done >= a.max_jobs:
      break
    if not ran_one:
      # everything is claimed by live pods right now; wait for one to finish
      time.sleep(60)

  if a.self_delete and os.environ.get("RUNPOD_POD_ID") and os.environ.get("RUNPOD_API_KEY"):
    try:
      from mjlab_centering.orchestrator import delete_pod

      delete_pod(os.environ["RUNPOD_POD_ID"], os.environ["RUNPOD_API_KEY"])
    except Exception as e:
      print(f"[worker] self-delete failed: {e!r}", flush=True)
  if a.idle_after:
    while True:
      time.sleep(3600)


if __name__ == "__main__":
  sys.exit(main())
