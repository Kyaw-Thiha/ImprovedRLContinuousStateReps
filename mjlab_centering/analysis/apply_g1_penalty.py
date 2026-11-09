"""Apply the decision-82/84 rule to the overnight campaign's G1 cells, once the penalty-10 arm has finished.

    python analysis/apply_g1_penalty.py --from fix1-b7c40a --to night1-3e71b4 [--wait-min 120]

If `fix_g1_pen10_none_o0_s1000` walks (last-5 track_lin >= 0.5), the G1 cells keep penalty 10. If it does not, the
pre-registered rule applies - penalty = 130 x the standing per-step reward, i.e. 5.7 for G1 - and the G1 jobs in the
target campaign that have not started yet are rewritten. Rewriting is safe: workers re-read the manifest every loop.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from mjlab_centering import r2

ROOT = Path(__file__).resolve().parents[1]
PEN10 = "fix_g1_pen10_none_o0_s1000"
WALK = 0.5
RULE_PENALTY = 5.7


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--from", dest="src", required=True)
  ap.add_argument("--to", dest="dst", required=True)
  ap.add_argument("--wait-min", type=float, default=120.0)
  a = ap.parse_args()

  t0 = time.time()
  while time.time() - t0 < a.wait_min * 60:
    if r2.exists(f"{a.src}/runs/{PEN10}/DONE"):
      break
    print("[penalty] waiting for the penalty-10 arm", flush=True)
    time.sleep(120)
  ev = rows(f"{a.src}/runs/{PEN10}/eval.jsonl")
  if not ev:
    print("[penalty] no penalty-10 result; leaving the manifest untouched")
    return
  track = float(np.mean([e["track_lin_frac"] for e in ev[-5:]]))
  walks = track >= WALK
  print(f"[penalty] penalty-10 arm: last-5 track {track:.3f} -> {'walks' if walks else 'does NOT walk'}")

  man = r2.get_json(f"{a.dst}/manifest.json")
  done = r2.done_set(a.dst)
  changed = []
  if not walks:
    # Before spending 12 jobs on G1 centering, prove the rule's own penalty walks: one G1 seed at 5.7, offset 0,
    # priority 1 so it runs before the priority-2 centering cells. (Decision 90.)
    chk = "w2g1chk_pen57_none_o0_s1000"
    if not any(j["job_id"] == chk for j in man["jobs"]) and chk not in done:
      base = next(j for j in man["jobs"] if j["job_id"].startswith("w2g1c_"))
      new = dict(base)
      new.update({"job_id": chk, "center": "none", "penalty": RULE_PENALTY, "eta": 0.0,
                  "r_bar_init": "zero", "offset": 0.0, "gamma": 0.99, "priority": 1})
      man["jobs"].append(new)
      changed.append(chk)
    for j in man["jobs"]:
      if j["robot"] == "g1" and j["job_id"] not in done and j["penalty"] != RULE_PENALTY:
        j["penalty"] = RULE_PENALTY
        changed.append(j["job_id"])
    r2.put_json(f"{a.dst}/manifest.json", man)
    (ROOT / "manifests" / "night1.json").write_text(json.dumps(man, indent=1))
  verdict = {"penalty10_track": track, "penalty10_walks": walks,
             "rewritten": changed, "g1_penalty": 10.0 if walks else RULE_PENALTY}
  (ROOT / "results" / "g1_penalty_verdict.json").write_text(json.dumps(verdict, indent=1))
  print(json.dumps(verdict, indent=1)[:800])


if __name__ == "__main__":
  main()
