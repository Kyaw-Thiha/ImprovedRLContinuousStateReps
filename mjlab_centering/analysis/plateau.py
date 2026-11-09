"""Plateau gate: choose the common run length for every arm (pre-registered rule).

Rule (docs/mjlab_centering/01-preregistration.md): the plateau point is the first iteration at which the 100-iteration
running mean of the reference arm's eval J_task changes by less than 3% over the following 100 iterations, averaged
over the two reference seeds. If it is later than 400, every run's length becomes that point.

    python analysis/plateau.py --campaign b1-4d73b6d1
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mjlab_centering import r2


def plateau_point(it: np.ndarray, j: np.ndarray, window: int = 100, tol: float = 0.03) -> int:
  """First iteration where mean(J[next window]) is within tol of mean(J[prev window]), and stays there."""
  if len(it) < 4:
    return -1
  for i in range(len(it)):
    t = it[i]
    prev = j[(it >= t - window) & (it <= t)]
    nxt = j[(it > t) & (it <= t + window)]
    if len(prev) < 2 or len(nxt) < 2:
      continue
    base = abs(float(np.mean(prev))) + 1e-9
    if abs(float(np.mean(nxt)) - float(np.mean(prev))) / base < tol:
      # must hold for the rest of the run
      rest = j[(it > t) & (it <= t + 2 * window)]
      if len(rest) < 2 or abs(float(np.mean(rest)) - float(np.mean(prev))) / base < tol:
        return int(t)
  return int(it[-1])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--jobs", default="plateau_ref_s2000,plateau_ref_s2001")
  a = ap.parse_args()
  points = []
  for job in a.jobs.split(","):
    try:
      body = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{a.campaign}/runs/{job}/eval.jsonl"))
    except Exception:
      print(f"{job}: no eval.jsonl yet")
      continue
    rows = [json.loads(ln) for ln in body["Body"].read().decode().splitlines() if ln.strip()]
    if not rows:
      continue
    rows.sort(key=lambda r: r["iter"])
    it = np.array([r["iter"] for r in rows], float)
    j = np.array([r["J_task"] for r in rows], float)
    p = plateau_point(it, j)
    points.append(p)
    print(f"{job}: n_evals={len(it)} final={j[-5:].mean():.4f} plateau_it={p}")
  if points:
    rec = int(np.ceil(np.median(points) / 25.0) * 25)
    print(json.dumps({"recommended_iters": max(rec, 400), "seed_points": points}))


if __name__ == "__main__":
  main()
