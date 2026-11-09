"""Round-1 pilot readout (decision 78), from a campaign's per-run R2 mirrors.

    python analysis/pilot_readout.py --campaign pilot1-9f3a1c [--partial]

Groups the pilot cells by (robot, center, offset) and prints the last-5-eval means, the count that walks
(`track_lin_frac` >= 0.5), and the paired centering gap per robot and offset. Also prints the G1 wrapper replication
(no-time-outs vs time-outs-kept) as a separate block.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from mjlab_centering import r2

WALK = 0.5


def _rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def latest_attempt(rows: list[dict]) -> tuple[list[dict], int]:
  """Keep only the rows of the newest attempt.

  A job that is re-claimed (pod restart, claim expiry) restarts at iteration 0 and appends to the same mirrored JSONL,
  so the file can hold several attempts (seen: 500 duplicated iterations on `pilot_g1_raw_none_o1_s1000`). Mixing them
  averages two policies and two iteration axes. The newest attempt starts at the last row whose `iter` is 0.
  """
  starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]
  if len(starts) <= 1:
    return rows, max(1, len(starts))
  return rows[starts[-1]:], len(starts)


def _last5(ev: list[dict], key: str) -> float:
  v = [r[key] for r in ev[-5:]]
  return float(np.mean(v)) if v else float("nan")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  groups: dict[tuple, list[dict]] = defaultdict(list)
  reps: dict[str, list[dict]] = defaultdict(list)
  for job in man["jobs"]:
    jid = job["job_id"]
    ev_raw = _rows(f"{a.campaign}/runs/{jid}/eval.jsonl")
    it_raw = _rows(f"{a.campaign}/runs/{jid}/iter.jsonl")
    it, n_it = latest_attempt(it_raw)
    ev, n_ev = latest_attempt(ev_raw)
    ev = sorted(ev, key=lambda r: r["iter"])
    done = r2.exists(f"{a.campaign}/runs/{jid}/DONE")
    if max(n_it, n_ev) > 1:
      print(f"  [integrity] {jid}: {max(n_it, n_ev)} attempts in the mirror; using the newest only")
    rec = {"jid": jid, "seed": job["seed"], "done": done, "iters": len(it),
           "track": _last5(ev, "track_lin_frac"), "J": _last5(ev, "J_task"),
           "std": float(it[-1].get("action_std", np.nan)) if it else float("nan"),
           "falls": float(it[-1].get("fall_rate_per_1k", np.nan)) if it else float("nan")}
    if job["block"] == "g1rep":
      reps["legacy" if "legacy" in jid else "timeouts"].append(rec)
    else:  # the gamma is part of the cell: pilot (gamma .999) and pilot99 (gamma .99) must never be pooled
      groups[(job["robot"], job["center"], job["offset"], job["gamma"], job["penalty"])].append(rec)

  print(f"campaign {a.campaign}: {len(man['jobs'])} jobs\n")
  print(f"{'cell':40s} {'n':>2s} {'done':>4s} {'track':>6s} {'J_task':>7s} {'std':>5s} {'walk':>5s}")
  for (robot, center, off, gam, pen), recs in sorted(groups.items()):
    n = len(recs)
    nd = sum(r["done"] for r in recs)
    tr = np.nanmean([r["track"] for r in recs])
    jt = np.nanmean([r["J"] for r in recs])
    sd = np.nanmean([r["std"] for r in recs])
    wk = sum(r["track"] >= WALK for r in recs)
    print(f"{robot}/{center}/o{off:g}/g{gam:g}/p{pen:g}".ljust(46) + f" {n:2d} {nd:4d} {tr:6.3f} {jt:7.4f} {sd:5.2f} {wk:2d}/{n}")

  print("\npaired centering gap (value - none), same seeds, J_task:")
  for robot, gam, pen in sorted({(k[0], k[3], k[4]) for k in groups}):
    for off in sorted({k[2] for k in groups if (k[0], k[3], k[4]) == (robot, gam, pen)}):
      v = {r["seed"]: r for r in groups.get((robot, "value", off, gam, pen), [])}
      n0 = {r["seed"]: r for r in groups.get((robot, "none", off, gam, pen), [])}
      shared = sorted(set(v) & set(n0))
      d = [v[s]["J"] - n0[s]["J"] for s in shared]
      dt = [v[s]["track"] - n0[s]["track"] for s in shared]
      if d:
        print(f"  {robot} g{gam:g} offset {off:g} p{pen:g}: n={len(d)} dJ={np.mean(d):+.4f} {[round(x, 4) for x in d]}  "
              f"dtrack={np.mean(dt):+.3f}")

  if reps:
    print("\nG1 wrapper replication (offset 0, no centering):")
    for k, recs in sorted(reps.items()):
      print(f"  {k:9s} n={len(recs)} " + ", ".join(
        f"s{r['seed']}: iters {r['iters']:4d} track {r['track']:.3f} J {r['J']:.4f} std {r['std']:.2f}"
        for r in sorted(recs, key=lambda r: r["seed"])))


if __name__ == "__main__":
  main()
