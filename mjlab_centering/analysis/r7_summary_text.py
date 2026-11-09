"""Print the prose for REPORT_007's abstract and conclusions from the readout JSON, so the words match the numbers.

    python analysis/r7_summary_text.py [--json results/r7_claim.json]

Reads the verdicts and cell summaries produced by `r7_report.py claim --json` and emits the sentences the report needs,
with the n, the walking counts, the means and the paired intervals taken from the data. Nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"vn": "VN", "sc": "SC", "rs": "RS"}
MEAN = {"vn": "value normalization", "sc": "simple centering", "rs": "return-std scaling"}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default=str(ROOT / "results" / "r7_claim.json"))
  a = ap.parse_args()
  d = json.loads(Path(a.json).read_text())
  s = d["summary"]
  v = {(x["arm"], x["offset"]): x for x in d["verdicts"]}
  print("### abstract / conclusions sentences\n")
  for arm in ("vn", "sc"):
    for off in (0.1, 1.0):
      k = f"{off:g}_{arm}"
      if k not in s or (arm, off) not in v:
        continue
      c, w = s[k], v[(arm, off)]
      pv = w["vs_value"]
      print(f"{LABEL[arm]} at c = {off:g}: n={c['n']}, {c['walks']} walk, tracking {c['track']:.3f} (J_task {c['J']:.4f}); "
            f"paired tracking against value centering {pv['mean']:+.3f} [{pv['ci'][0]:+.3f}, {pv['ci'][1]:+.3f}] "
            f"-> **{w['reading'].split(' (')[0]}**.")
  print("\n### grade sentence")
  ns = sorted({row["n"] for row in s.values()})
  print(f"n per cell: {', '.join(str(x) for x in ns)}; " +
        ("claim grade (20 pairs)" if all(x >= 20 for x in ns) else "pilot grade (< 20 pairs)"))


if __name__ == "__main__":
  main()
