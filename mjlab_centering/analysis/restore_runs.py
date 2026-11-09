"""Restore every run that mirrored itself to R2 into a local results tree (authoritative over pod snapshots).

`podrun --r2` makes each job upload iter.jsonl / eval.jsonl / final.json / DONE under
`<prefix>/runs/<job_id>/...`, so a finished run is recoverable even if the pod is deleted before its tarball lands.
This script finds those keys, groups by job id, and prefers a run with `DONE` (then the newest final.json, then the
newest iter.jsonl). It also tolerates the doubled project prefix produced when a caller passes a prefix that already
contains `mjlab_ssp_centering/`.

    python analysis/restore_runs.py --out results/merged_r2 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mjlab_centering import r2

FILES = ("iter.jsonl", "eval.jsonl", "final.json")


def scan(client, bucket: str, prefixes: tuple[str, ...] = ()) -> dict[str, dict[str, tuple[str, object]]]:
  """{job_id: {filename: (key, obj)}} over every `.../runs/<job_id>/<file>` key in the bucket."""
  found: dict[str, dict[str, tuple[str, object]]] = defaultdict(dict)
  roots = [r2.key(p) for p in prefixes] if prefixes else [r2.key("")]
  for root in roots:
    _scan_root(client, bucket, root, found)
  return found


def _scan_root(client, bucket: str, root: str, found: dict) -> None:
  token = None
  while True:
    kw = {"Bucket": bucket, "Prefix": root}
    if token:
      kw["ContinuationToken"] = token
    resp = client.list_objects_v2(**kw)
    for o in resp.get("Contents", []):
      parts = o["Key"].split("/")
      if "runs" not in parts:
        continue
      i = len(parts) - 1 - parts[::-1].index("runs")  # last "runs"
      if i + 2 >= len(parts) + 1 or i + 1 >= len(parts):
        continue
      job = parts[i + 1]
      fname = parts[-1]
      if fname not in FILES + ("DONE",) or len(parts) != i + 3:
        continue
      prev = found[job].get(fname)
      if prev is None or o["LastModified"] > prev[1]["LastModified"]:
        found[job][fname] = (o["Key"], o)
    if not resp.get("IsTruncated"):
      break
    token = resp.get("NextContinuationToken")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/merged_r2")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--prefix", default="", help="comma list of campaign prefixes to restore (default: everything)")
  a = ap.parse_args()
  client, bucket = r2.client(), r2.bucket()
  prefixes = tuple(p.strip() for p in a.prefix.split(",") if p.strip())
  found = scan(client, bucket, prefixes)
  out = Path(a.out)
  summary = {}
  for job, files in sorted(found.items()):
    done = "DONE" in files
    have = [f for f in FILES if f in files]
    summary[job] = {"done": done, "files": have}
    if a.dry_run:
      continue
    d = out / job
    d.mkdir(parents=True, exist_ok=True)
    for f in have:
      client.download_file(bucket, files[f][0], str(d / f))
    if done:
      (d / "DONE").write_text("{}")
  if not a.dry_run:
    (out / "restore_report.json").write_text(json.dumps(summary, indent=1))
  print(f"{len(found)} job ids in R2; {sum(1 for v in summary.values() if v['done'])} finished")
  partial = [j for j, v in summary.items() if not v["done"] and "iter.jsonl" in v["files"]]
  if partial:
    print(f"{len(partial)} unfinished runs also mirrored (kept for partial-curve analysis)")


if __name__ == "__main__":
  main()
