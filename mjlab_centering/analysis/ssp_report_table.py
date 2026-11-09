"""Render d129/d138's width + sweep readout as the markdown table REPORT_006 needs, so the numbers are never retyped.

    python analysis/ssp_report_table.py [--root results/merged_r2] [--pilot]

Always run the restore first when the runs are only on R2:
    python analysis/restore_runs.py --out results/merged_r2 --prefix r6sspwidthqueue-
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import ssp_readout  # noqa: E402


def width_table(rep: dict) -> str:
  rows = []
  for k in sorted(rep["cells"], key=lambda k: (k != "raw", k)):
    c = rep["cells"][k]
    rows.append(f"| {k} | {c['n']}/{c['of']} | {c['mean']:.3f} | {c['walks']}/{c['of']} |")
  head = "| cell | seeds | mean tracking | walks |\n|---|---|---|---|"
  return head + "\n" + "\n".join(rows)


def summary_lines(rep: dict) -> list[str]:
  out = []
  if "best_at_ls0.3" in rep:
    out.append(f"- best width at ls 0.3: **{rep['best_at_ls0.3']}** "
               f"(gap to raw {rep.get('gap_to_raw', float('nan')):+.3f}; closes gap within 0.03: {rep.get('closes_gap_d105i')})")
  out.append(f"- 8x8 pilot triggered (4x4 beats 2x2): {rep.get('run_8x8_pilot_d105iv')}")
  for k in sorted(k for k in rep if k.startswith("ssp_beats_rff_")):
    out.append(f"- {k.replace('ssp_beats_rff_', 'SSP beats matched RFF at ')}: {rep[k]}")
  if rep.get("ls_sweep"):
    sw = rep["ls_sweep"]
    vals = ", ".join(f"{ls}: {v['mean']:.3f} ({v['walks']}/{v['n']})" if v else f"{ls}: missing"
                     for ls, v in sw["values"].items())
    out.append(f"- length-scale sweep at {sw['width']}x{sw['width']}: {vals}")
  if rep.get("best_overall"):
    out.append(f"- best overall: **{rep['best_overall']}**, walks {rep['best_walks']}")
  if "stage_b_gate_passes" in rep:
    out.append(f"- Stage-B gate: {'PASSES' if rep['stage_b_gate_passes'] else 'FAILS'}")
  out.append(f"- verdict: {rep['verdict']}")
  return out


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=str(ROOT / "results" / "merged_r2"))
  a = ap.parse_args()
  rep = ssp_readout.width_report(Path(a.root))
  print(width_table(rep))
  print()
  print("\n".join(summary_lines(rep)))
  print()
  print("```json")
  print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1))
  print("```")


if __name__ == "__main__":
  main(sys.argv[1:])
