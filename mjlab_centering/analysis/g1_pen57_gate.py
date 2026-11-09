"""Cancel the G1 centering cells if the pre-registered 5.7 penalty does not walk either.

    python analysis/g1_pen57_gate.py --campaign night1-3e71b4 --wait-min 300

Waits for `w2g1chk_pen57_none_o0_s1000`. If its last-5 `track_lin` is >= 0.5, the 12 G1 centering cells stand. If not,
they are moved to priority 99 (they keep their place in the manifest but never get claimed) and the verdict is written
to `results/g1_pen57_verdict.json`, so the campaign does not spend 12 runs on a cell that cannot walk.
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
CHK = "w2g1chk_pen57_none_o0_s1000"
WALK = 0.5


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--wait-min", type=float, default=300.0)
  a = ap.parse_args()
  t0 = time.time()
  while time.time() - t0 < a.wait_min * 60:
    if r2.exists(f"{a.campaign}/runs/{CHK}/DONE"):
      break
    print("[pen57] waiting for the 5.7 verification seed", flush=True)
    time.sleep(120)
  ev = rows(f"{a.campaign}/runs/{CHK}/eval.jsonl")
  if not ev:
    print("[pen57] no result; leaving the G1 cells alone")
    return
  track = float(np.mean([e["track_lin_frac"] for e in ev[-5:]]))
  walks = track >= WALK
  print(f"[pen57] last-5 track {track:.3f} -> {'walks' if walks else 'does NOT walk'}")
  verdict = {"track": track, "walks": walks, "cancelled": []}
  if not walks:
    man = r2.get_json(f"{a.campaign}/manifest.json")
    done = r2.done_set(a.campaign)
    for j in man["jobs"]:
      if j["robot"] == "g1" and "g1c" in j["block"] and j["job_id"] not in done:
        j["priority"] = 99
        verdict["cancelled"].append(j["job_id"])
    r2.put_json(f"{a.campaign}/manifest.json", man)
    (ROOT / "manifests" / "night1.json").write_text(json.dumps(man, indent=1))
  (ROOT / "results" / "g1_pen57_verdict.json").write_text(json.dumps(verdict, indent=1))
  print(f"[pen57] cancelled {len(verdict['cancelled'])} G1 centering cells" if verdict["cancelled"] else "[pen57] G1 cells stand")


if __name__ == "__main__":
  main()
