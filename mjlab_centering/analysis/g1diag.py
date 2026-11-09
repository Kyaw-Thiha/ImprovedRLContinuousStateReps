"""G1 diagnosis readout (decision 73), straight from the campaign's R2 mirrors.

    python analysis/g1diag.py --campaign g1diag-7c41e2

For each job: how far it got, the last training row (offset_now, r_bar, action std, falls), the eval curve, and the
pre-registered verdict - "walks" iff the mean `track_lin_frac` of the last 5 evals is >= 0.5 (standing floor 0.267,
the failed uncentered-at-offset-c reference 0.29-0.35).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from mjlab_centering import r2

WALK = 0.5
STANDING_TRACK = 0.267


def _jsonl(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--walk", type=float, default=WALK)
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  for job in man["jobs"]:
    jid = job["job_id"]
    ev = sorted(_jsonl(f"{a.campaign}/runs/{jid}/eval.jsonl"), key=lambda r: r["iter"])
    it = _jsonl(f"{a.campaign}/runs/{jid}/iter.jsonl")
    done = r2.exists(f"{a.campaign}/runs/{jid}/DONE")
    print(f"\n== {jid}  (offset {job['offset']}, center {job['center']}, eta {job['eta']}, "
          f"r_bar_init {job['r_bar_init']}, {job['iters']} iters)  DONE={done}")
    if it:
      last = it[-1]
      print(f"   iter rows {len(it):5d}  latest iter {last['iter']:4d}  offset_now {last.get('offset_now')}  "
            f"r_bar {last.get('r_bar'):.4f}  action_std {last.get('action_std'):.3f}  "
            f"falls/1k {last.get('fall_rate_per_1k', float('nan')):.2f}  resets {last.get('resets')}")
      fps = [r.get("fps") for r in it[-50:] if r.get("fps")]
      if fps:
        print(f"   fps {np.mean(fps):.0f}")
    if not ev:
      print("   no evals yet")
      continue
    tr = [r["track_lin_frac"] for r in ev]
    jt = [r["J_task"] for r in ev]
    for i in range(0, len(ev), max(1, len(ev) // 8)):
      print(f"   iter {ev[i]['iter']:4d}  track_lin {tr[i]:.3f}  J_task {jt[i]:.4f}  "
            f"falls/1k {ev[i].get('fall_rate_per_1k', float('nan')):.1f}")
    tail_tr = float(np.mean(tr[-5:]))
    tail_jt = float(np.mean(jt[-5:]))
    walk = tail_tr >= a.walk
    print(f"   last-5 mean: track_lin {tail_tr:.3f} (standing {STANDING_TRACK:.3f}), J_task {tail_jt:.4f}  "
          f"-> {'WALKS' if walk else 'does NOT walk'}")


if __name__ == "__main__":
  main()
