"""Rewrite the round-9b tables in docs/REPORT_009.md from a fresh readout (the analogue of r9_inject.py).

    python analysis/r9_leak_inject.py [--campaign r9aleak-faa5c3d5]

Only the region between the report's LEAK_TABLES markers is replaced; every number comes from
analysis/r9_leak_report.py's own output.
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
  ap.add_argument("--campaign", default="r9aleak-faa5c3d5")
  a = ap.parse_args()
  out = subprocess.run([sys.executable, str(ROOT / "analysis" / "r9_leak_report.py"), "--campaign", a.campaign,
                        "--pull", "--json", str(ROOT / "results" / "r9_leak.json")],
                       capture_output=True, text=True, check=True).stdout
  body = out.split("\n", 1)[1].strip()  # drop the campaign header
  s = REPORT.read_text()
  i, j = s.index("<!-- LEAK_TABLES_START -->"), s.index("<!-- LEAK_TABLES_END -->")
  s = (s[:i] + "<!-- LEAK_TABLES_START -->\n" + body + "\n<!-- LEAK_TABLES_END -->"
       + s[j + len("<!-- LEAK_TABLES_END -->"):])
  REPORT.write_text(s)
  print(f"wrote {REPORT}")


if __name__ == "__main__":
  main()
