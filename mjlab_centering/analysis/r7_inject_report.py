"""Inject the generated claim readout into REPORT_007 (between the markers), so the report cannot drift from the data.

    python analysis/r7_inject_report.py [--report ../docs/REPORT_007.md]

It replaces the content between `<!-- CLAIM_TABLES_START/END -->` with the current `r7_report.py claim --timing` output and
the block between `<!-- VERDICTS_START/END -->` with the mechanical verdicts from `results/r7_claim.json`. Run by the
finisher; safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = ROOT.parent / "docs" / "REPORT_007.md"


def replace_between(text: str, start: str, end: str, body: str) -> str:
  i, j = text.index(start) + len(start), text.index(end)
  return text[:i] + "\n" + body.strip() + "\n" + text[j:]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--report", default=str(REPORT))
  ap.add_argument("--campaign", default="r7vnsc-a565805a")
  ap.add_argument("--json", default="results/r7_claim.json")
  ap.add_argument("--marker", default="CLAIM_TABLES", help="marker family to replace: CLAIM_TABLES or RS_TABLES")
  a = ap.parse_args()
  rep = subprocess.run([sys.executable, str(HERE / "r7_report.py"), "claim", "--campaigns", a.campaign, "--timing"],
                       capture_output=True, text=True, cwd=str(ROOT))
  if rep.returncode != 0:
    raise SystemExit(rep.stderr[-800:])
  body = rep.stdout.strip()
  text = Path(a.report).read_text()
  text = replace_between(text, f"<!-- {a.marker}_START -->", f"<!-- {a.marker}_END -->", body)

  LABEL = {"none": "no centering", "value": "value centering", "vn": "VN (value normalization)",
           "sc": "SC (simple centering)", "rs": "RS (return-std scaling)"}
  verdicts = []
  jf = ROOT / a.json
  if jf.exists():
    d = json.loads(jf.read_text())
    for v in d.get("verdicts", []):
      verdicts.append(f"| {LABEL.get(v['arm'], v['arm'])} | {v['offset']:g} | n={v['n']}, walks {v['walks']}/{v['n']} | {v['reading']} |")
  if verdicts:
    block = ("| arm | c | n and walks | pre-registered verdict (d157) |\n|---|---|---|---|\n" + "\n".join(verdicts))
    text = replace_between(text, "<!-- VERDICTS_START -->", "<!-- VERDICTS_END -->", block)
  Path(a.report).write_text(text)
  print(f"injected into {a.report}: {len(body.splitlines())} table lines, {len(verdicts)} verdicts")


if __name__ == "__main__":
  main()
