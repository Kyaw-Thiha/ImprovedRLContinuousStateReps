"""Render the Stage-B pilot readout as the markdown table REPORT_006 needs (d130 rules).

    python analysis/ssp_pilot_table.py --n 2 --ls 1.0 [--root results/merged_r2]

Pairs every SSP pilot cell with the raw cell on the same seed and offset, prints per-cell walks/tracking, the
difference-in-differences interaction per offset, and the escalation decision exactly as d130 fixed it (escalate a cell
when |delta walks| >= 2 of 5, or both arms at an offset when |interaction| > 0.10 tracking).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import ssp_readout  # noqa: E402


def table(rep: dict) -> str:
  lines = ["| offset | arm | n pairs | SSP tracking | raw tracking | dtrack | walks SSP | walks raw |",
           "|---|---|---|---|---|---|---|---|"]
  for off, rec in rep["offsets"].items():
    for c in ("none", "value"):
      r = rec[c]
      lines.append(f"| {off} | {c} | {r['n_pairs']} | {r['mean_ssp']:.3f} | {r['mean_raw']:.3f} | "
                   f"{r['mean_delta']:+.3f} | {r['walks_ssp']} | {r['walks_raw']} |")
  return "\n".join(lines)


def summary(rep: dict) -> str:
  out = []
  for off, rec in rep["offsets"].items():
    it = rec["interaction"]
    out.append(f"- offset {off}: interaction (DiD) {it['mean']:+.3f} tracking, se {it['se']:.3f}, on {it['n_seeds']} seeds")
  out.append(f"- escalation: {rep['verdict']}")
  return "\n".join(out)


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--n", type=int, required=True)
  ap.add_argument("--ls", type=float, required=True)
  ap.add_argument("--root", default=str(ROOT / "results" / "merged_r2"))
  a = ap.parse_args()
  rep = ssp_readout.pilot_report(a.n, a.ls, Path(a.root))
  print(table(rep))
  print()
  print(summary(rep))
  print()
  print("```json")
  print(json.dumps({k: v for k, v in rep.items() if k != "offsets"}, indent=1))
  print("```")


if __name__ == "__main__":
  main(sys.argv[1:])
