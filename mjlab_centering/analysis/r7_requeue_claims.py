"""Requeue a campaign's jobs whose pods died before finishing.

A pod claims a job by writing ``<campaign>/claims/<job>.json``; the claim is the only thing that stops a fresh fleet from
picking the job up again. When pods are lost (preemption, an emptied account, a reboot) the claims outlive them, so the jobs
must be handed back explicitly. This deletes the claim of every job that has no ``DONE`` marker, which is exactly the set
that needs re-running: finished jobs keep their claims and their ``DONE``, and are never re-run.

    python analysis/r7_requeue_claims.py --campaign r7rs-419904b0 --dry-run
    python analysis/r7_requeue_claims.py --campaign r7rs-419904b0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from mjlab_centering import r2  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--dry-run", action="store_true", help="list what would be requeued, touch nothing")
  a = ap.parse_args()

  base = r2.key("")  # list_keys returns full keys; delete_key takes a path relative to the bucket root
  claims = [k[len(base):] for k in r2.list_keys(f"{a.campaign}/claims") if k.endswith(".json")]
  done = {k.split("/")[-2] for k in r2.list_keys(f"{a.campaign}/runs") if k.endswith("/DONE")}
  stranded = []
  for key in claims:
    job = key.rsplit("/", 1)[-1][:-5]
    if job not in done:
      stranded.append((job, key))
  print(f"{len(claims)} claims, {len(done)} done, {len(stranded)} stranded")
  for job, key in sorted(stranded):
    if a.dry_run:
      print("  would requeue", job)
    else:
      r2.delete_key(key)
      print("  requeued", job)
  if not a.dry_run and stranded:
    print("done: a fresh fleet will re-claim these jobs from the start")


if __name__ == "__main__":
  main()
