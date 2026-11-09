"""Finish the round unattended: wait for the claim set, then produce the readout and stop the pods.

    python analysis/r7_finish.py --campaign r7vnsc-a565805a --launch-campaign r7pilot-63bd51f3 \\
        --manifest manifests/r7vnsc-a565805a.json --deadline-h 7

Runs by itself (systemd user unit `r7finish`). It

  1. waits until every job has a `DONE` marker, or the deadline passes;
  2. runs the claim readout (`r7_report.py claim --timing --fig --json`) and writes its output to
     `results/r7_final.md`, next to the spend line from `r7_spend.py` and the build histogram;
  3. publishes `TERMINATE` for the campaign, so the pods delete themselves on their next poll instead of idling warm
     (d165's keep-alive would otherwise keep three pods billing for up to eight hours);
  4. prints a one-screen summary.

If the deadline passes first it still writes what exists, labelled with the count, and terminates the pods — a partial
set is reported rather than silently left running.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from mjlab_centering import r2  # noqa: E402


def count_done(campaign: str) -> int:
  return sum(1 for k in r2.list_keys(f"{campaign}/runs") if k.endswith("/DONE"))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r7vnsc-a565805a")
  ap.add_argument("--launch-campaign", default="r7pilot-63bd51f3")
  ap.add_argument("--manifest", default="manifests/r7vnsc-a565805a.json")
  ap.add_argument("--deadline-h", type=float, default=7.0)
  ap.add_argument("--poll", type=float, default=300.0)
  ap.add_argument("--out", default="results/r7_final.md")
  ap.add_argument("--json", default="results/r7_claim.json",
                  help="where the claim readout json goes; give a per-campaign name so a later round cannot overwrite it")
  ap.add_argument("--marker", default="CLAIM_TABLES", help="report marker family this campaign's tables belong to")
  ap.add_argument("--fig", action="store_true",
                  help="also redraw the claim figure; off by default because the figure belongs to the first claim set")
  ap.add_argument("--terminate", action="store_true",
                  help="publish a TERMINATE marker so the pods stop billing; off by default so a test run cannot kill work")
  a = ap.parse_args()
  total = len(json.loads((ROOT / a.manifest).read_text())["jobs"])
  deadline = time.time() + a.deadline_h * 3600
  while True:
    n = count_done(a.campaign)
    print(f"[finish] {time.strftime('%H:%M:%SZ', time.gmtime())} DONE={n}/{total}", flush=True)
    if n >= total or time.time() > deadline:
      break
    time.sleep(a.poll)

  lines = [f"# Round 7 - claim readout ({a.campaign}, {n}/{total} runs complete)\n"]
  cmd = [sys.executable, str(HERE / "r7_report.py"), "claim", "--campaigns", a.campaign,
         "--timing", "--json", a.json]
  if a.fig:
    cmd.append("--fig")
  rep = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
  lines += ["```", rep.stdout.strip()[-20000:], "```", ""]
  sp = subprocess.run([sys.executable, str(HERE / "r7_spend.py")], capture_output=True, text=True, cwd=str(ROOT))
  lines += ["## spend", "", "```", sp.stdout.strip()[:4000], "```", ""]
  pr = subprocess.run([sys.executable, str(HERE / "r7_progress.py")], capture_output=True, text=True, cwd=str(ROOT))
  lines += ["## progress", "", "```", pr.stdout.strip()[:4000], "```", ""]
  (ROOT / a.out).write_text("\n".join(lines))
  # and write the same tables into the report itself, between its markers
  inj = subprocess.run([sys.executable, str(HERE / "r7_inject_report.py"), "--campaign", a.campaign,
                        "--json", a.json, "--marker", a.marker],
                       capture_output=True, text=True, cwd=str(ROOT))
  print(f"[finish] report injection: {inj.stdout.strip() or inj.stderr.strip()[:300]}", flush=True)
  print(f"[finish] wrote {ROOT / a.out}", flush=True)

  if a.terminate:
    r2.put_json(f"{a.campaign}/TERMINATE", {"at": time.time(), "done": n, "of": total})
    print(f"[finish] TERMINATE published for {a.campaign}: the pods stop billing within a minute", flush=True)
  else:  # a test run must never stop the campaign; say so explicitly
    print("[finish] --terminate not given: pods left running (safe for tests)", flush=True)


if __name__ == "__main__":
  main()
