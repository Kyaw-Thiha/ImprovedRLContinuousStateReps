"""Build (and print) the manifest for d138's unconditional length-scale sweep at the best SSP width.

    python analysis/ssp_ls_launch.py [--root results/merged_r2]

Prints two KEY=VALUE lines for the shell keeper:
    CAMPAIGN=<campaign id>
    MANIFEST=<path to the manifest json>
It refuses (non-zero exit, a REFUSE line) when the width stage is not complete, when the sweep is already done, or when
the best width cannot be determined — the keeper then waits another round.
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


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=str(ROOT / "results" / "merged_r2"))
  a = ap.parse_args()
  r = ssp_readout.width_report(Path(a.root))
  if "best_at_ls0.3" not in r:
    print(f"REFUSE=width stage incomplete: {r.get('verdict')}")
    raise SystemExit(3)
  if r.get("ls_sweep", {}).get("done"):
    print("REFUSE=sweep already complete")
    raise SystemExit(3)
  n = int(r["best_at_ls0.3"][3:r["best_at_ls0.3"].index("x")])
  jobs = jobs_r4.ssp_ls_sweep(n)
  jobs_r4.check(jobs)
  camp = jobs_r4.campaign_id("r6lss", jobs)
  path = ROOT / "manifests" / f"{camp}.json"
  path.write_text(json.dumps({"campaign": camp, "jobs": jobs}, indent=1))
  print(f"BEST_WIDTH={n}")
  print(f"CAMPAIGN={camp}")
  print(f"MANIFEST={path}")
  print(f"JOBS={len(jobs)}")
  print(f"ITERS={sum(j['iters'] for j in jobs)}")


if __name__ == "__main__":
  main(sys.argv[1:])
