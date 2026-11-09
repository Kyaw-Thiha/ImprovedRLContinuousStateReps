"""Rewrite the pilot tables in docs/REPORT_009.md from a fresh readout.

    python analysis/r9_inject.py [--campaign r9pilot-3ae3cc84]

Only the region between the report's PILOT_TABLES markers is replaced; every number comes from r9_report.py's own output
so the report cannot drift from the run files.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parent / "docs" / "REPORT_009.md"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  a = ap.parse_args()
  out = subprocess.run([sys.executable, str(ROOT / "analysis" / "r9_report.py"), "--campaign", a.campaign,
                        "--pull", "--json", str(ROOT / "results" / "r9_pilot.json")],
                       capture_output=True, text=True, check=True).stdout
  body = out.split("\n", 1)[1].strip()  # drop the campaign header line
  s = REPORT.read_text()
  i, j = s.index("<!-- PILOT_TABLES_START -->"), s.index("<!-- PILOT_TABLES_END -->")
  s = s[:i] + "<!-- PILOT_TABLES_START -->\n" + body + "\n<!-- PILOT_TABLES_END -->" + s[j + len("<!-- PILOT_TABLES_END -->"):]
  REPORT.write_text(s)
  print(f"wrote {REPORT}")


if __name__ == "__main__":
  main()
