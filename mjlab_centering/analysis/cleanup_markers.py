"""Delete DONE markers that belong to jobs no longer in the campaign manifest.

    python analysis/cleanup_markers.py --campaign night1-3e71b4 [--apply]

Decision 92 retired twelve penalty-10 G1 cells by writing administrative `DONE` markers so that pods holding a stale
pending list would skip them. Those markers are not completions: they inflate `done_set` and every count derived from it.
Once the campaign has drained (no pod can still be holding a stale list), they must be deleted before any final count.
This lists them by default and deletes them only with `--apply`.
"""

from __future__ import annotations

import argparse

from mjlab_centering import r2


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--apply", action="store_true")
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  live = {j["job_id"] for j in man["jobs"]}
  done = r2.done_set(a.campaign)
  orphans = sorted(done - live)
  print(f"{len(done)} DONE markers, {len(live)} jobs in the manifest, {len(orphans)} orphan marker(s)")
  for j in orphans:
    print("  ", j)
  if orphans and a.apply:
    c = r2.client()
    for j in orphans:
      c.delete_object(Bucket=r2.bucket(), Key=r2.key(f"{a.campaign}/runs/{j}/DONE"))
    print(f"deleted {len(orphans)} marker(s); the campaign now reports {len(done) - len(orphans)} completions")
  elif orphans:
    print("dry run; pass --apply to delete")


if __name__ == "__main__":
  main()
