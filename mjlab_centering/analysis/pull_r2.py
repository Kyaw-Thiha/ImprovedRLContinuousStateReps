"""Pull every pod snapshot from R2 and unpack it locally, so results survive pod deletion.

    python analysis/pull_r2.py --out results/from_r2 [--prefix pods/]

Each pod pushes `mjlab_ssp_centering/pods/<tag>/_project_results.tgz` every 5 minutes and `_FINAL_results.tgz` just
before it deletes itself; this script takes the newest of the two per tag and extracts it into <out>/<tag>/.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

from mjlab_centering import r2


def latest_per_tag(client, bucket: str, prefix: str) -> dict[tuple[str, str], tuple[str, object]]:
  """Newest object per (pod tag, file name). Keyed by name too, because one pod can push several different tarballs
  (results, linear-slice outputs, the final snapshot) and they must not shadow each other."""
  best: dict[tuple[str, str], tuple[str, object]] = {}
  token = None
  while True:
    kw = {"Bucket": bucket, "Prefix": r2.key(prefix)}
    if token:
      kw["ContinuationToken"] = token
    resp = client.list_objects_v2(**kw)
    for o in resp.get("Contents", []):
      parts = o["Key"].split("/")
      if len(parts) < 3 or not parts[-1].endswith(".tgz"):
        continue
      k = (parts[-2], parts[-1])
      if k not in best or o["LastModified"] > best[k][1]["LastModified"]:
        best[k] = (o["Key"], o)
    if not resp.get("IsTruncated"):
      break
    token = resp.get("NextContinuationToken")
  return best


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/from_r2")
  ap.add_argument("--prefix", default="pods/")
  a = ap.parse_args()
  client, bucket = r2.client(), r2.bucket()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  best = latest_per_tag(client, bucket, a.prefix)
  if not best:
    print("nothing in R2 yet")
    return
  for (tag, name), (key, o) in sorted(best.items()):
    dest = out / f"{tag}__{name}"
    client.download_file(bucket, key, str(dest))
    target = out / tag
    target.mkdir(exist_ok=True)
    with tarfile.open(dest) as t:
      t.extractall(target)
    print(f"{tag}/{name}: {o['Size']} bytes {o['LastModified']:%H:%M} -> {target}")


if __name__ == "__main__":
  main()
