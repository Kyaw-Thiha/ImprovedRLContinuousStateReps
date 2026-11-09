"""Pick the centering step size from the eta sweep (preregistration: chosen on the sweep cell, then frozen).

    python analysis/pick_eta.py --campaign e1-xxxxxxxx
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

from mjlab_centering import r2


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--write", default=None, help="write the chosen eta to this path")
  a = ap.parse_args()
  scores: dict[float, list[float]] = defaultdict(list)
  for k in r2.list_keys(f"{a.campaign}/runs/"):
    if not k.endswith("eval.jsonl"):
      continue
    job = k.split("/runs/")[1].rsplit("/", 1)[0]
    m = re.search(r"eta([0-9.]+)$", job)
    if not m:
      continue
    eta = float(m.group(1))
    body = r2.client().get_object(Bucket=r2.bucket(), Key=k)["Body"].read().decode()
    rows = [json.loads(ln) for ln in body.splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["iter"])
    if len(rows) < 5:
      print(f"  {job}: only {len(rows)} evals, skipped")
      continue
    j = [r["J_task"] for r in rows]
    scores[eta].append(float(np.mean(j[-5:])))
    print(f"  {job}: evals={len(rows)} last5={np.mean(j[-5:]):.4f}")
  if not scores:
    raise SystemExit("no complete eta cells yet")
  means = {e: float(np.mean(v)) for e, v in scores.items()}
  best = max(means, key=lambda e: (means[e], len(scores[e])))
  print("eta means:", {k: round(v, 4) for k, v in sorted(means.items())})
  print("chosen eta:", best)
  if a.write:
    with open(a.write, "w") as f:
      json.dump({"eta": best, "means": means, "n": {k: len(v) for k, v in scores.items()}}, f, indent=1)


if __name__ == "__main__":
  main()
