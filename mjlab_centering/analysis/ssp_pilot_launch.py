"""Build (and print) the Stage-B pilot manifest from d130's gate, once the length-scale sweep is complete.

    python analysis/ssp_pilot_launch.py

Prints CAMPAIGN=/MANIFEST=/JOBS=/ITERS= for the shell keeper, or REFUSE=<reason> with exit 3. The gate is d106 as
amended by d138: the best (width, length scale) must walk on at least 2 of the 3 search seeds; otherwise Stage B is
skipped and the result is "SSP does not walk at any width or length scale tried".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
import jobs_r4  # noqa: E402
import ssp_readout  # noqa: E402


def parse_best(best: str) -> tuple[int, float]:
  """"ssp4x4@0.1" -> (4, 0.1)."""
  n = int(best[3:best.index("x")])
  return n, float(best.split("@")[1])


def build(root: Path, out_dir: Path) -> dict:
  """Apply the gate and, if it passes, write the pilot manifest. Raises SystemExit(3) with a REFUSE message otherwise."""
  r = ssp_readout.width_report(root)
  if not r.get("ls_sweep", {}).get("done"):
    raise SystemExit(f"REFUSE=length-scale sweep incomplete: {r.get('verdict')}")
  if not r.get("stage_b_gate_passes"):
    raise SystemExit(f"REFUSE=gate fails ({r.get('best_walks')}): Stage B skipped")
  n, ls = parse_best(r["best_overall"])
  jobs = jobs_r4.ssp_pilot(n, ls)
  jobs_r4.check(jobs)
  camp = jobs_r4.campaign_id("r6sspp", jobs)
  path = Path(out_dir) / f"{camp}.json"
  path.write_text(json.dumps({"campaign": camp, "jobs": jobs}, indent=1))
  return {"best": r["best_overall"], "campaign": camp, "manifest": str(path), "jobs": len(jobs),
          "iters": sum(j["iters"] for j in jobs)}


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=str(ROOT / "results" / "merged_r2"))
  ap.add_argument("--out-dir", default=str(ROOT / "manifests"))
  a = ap.parse_args()
  b = build(Path(a.root), Path(a.out_dir))
  print(f"BEST={b['best']}")
  print(f"CAMPAIGN={b['campaign']}")
  print(f"MANIFEST={b['manifest']}")
  print(f"JOBS={b['jobs']}")
  print(f"ITERS={b['iters']}")

if __name__ == "__main__":
  main(sys.argv[1:])
