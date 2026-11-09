"""Decision-84 gate: decide whether G1 gets centering cells, from the corrected campaign's own results.

    python analysis/g1_gate.py --campaign fix1-b7c40a [--launch]

Reads the three corrected G1 seeds (`fix_g1_raw_none_o0_s1000/1001/1002`, gamma .99, offset 0, penalty 0) and the
penalty-10 arm (`fix_g1_pen10_none_o0_s1000`), applies the rule fixed in decision 84, writes the verdict to
`results/g1_gate_decision.json`, and - only with `--launch` and only if the gate passes - generates
`manifests/g1c.json` and starts the next campaign detached with a hard cap.

Rule:
  walks = number of the three seeds whose last-5-eval `track_lin_frac` >= 0.5.
  gate passes iff walks >= 2 AND the penalty-10 arm walks.
  G1 penalty for the next campaign = 10 if the penalty-10 arm walks, else 5.7 (the 130x standing-reward rule).
A gate that does not pass is recorded, not retried: G1 is then reported as "walked on N of 3 seeds".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from mjlab_centering import r2, waves

ROOT = Path(__file__).resolve().parents[1]
WALK = 0.5
G1_SEEDS = ("fix_g1_raw_none_o0_s1000", "fix_g1_raw_none_o0_s1001", "fix_g1_raw_none_o0_s1002")
PEN10 = "fix_g1_pen10_none_o0_s1000"


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def last5(campaign: str, jid: str) -> float:
  ev = rows(f"{campaign}/runs/{jid}/eval.jsonl")
  if not ev:
    return float("nan")
  return float(np.mean([e["track_lin_frac"] for e in ev[-5:]]))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--launch", action="store_true")
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--hard-cap", type=float, default=6.0)
  ap.add_argument("--wait-min", type=float, default=0.0,
                  help="poll until every gated job is DONE (or this many minutes pass) before deciding")
  a = ap.parse_args()

  if a.wait_min > 0:
    t0 = time.time()
    while True:
      missing = [j for j in list(G1_SEEDS) + [PEN10] if not r2.exists(f"{a.campaign}/runs/{j}/DONE")]
      if not missing or time.time() - t0 > a.wait_min * 60:
        break
      print(f"[gate] waiting on {len(missing)} job(s): {', '.join(missing)}", flush=True)
      time.sleep(120)

  gates = {j: last5(a.campaign, j) for j in list(G1_SEEDS) + [PEN10]}
  done = {j: r2.exists(f"{a.campaign}/runs/{j}/DONE") for j in gates}
  walks = sum(1 for j in G1_SEEDS if gates[j] >= WALK)
  pen_ok = gates[PEN10] >= WALK
  pending = [j for j in gates if not done[j]]
  verdict = {
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    "campaign": a.campaign,
    "track_last5": gates,
    "done": done,
    "walks_of_3": walks,
    "penalty10_walks": pen_ok,
    "g1_penalty": 10.0 if pen_ok else 5.7,
    "gate_passes": bool(walks >= 2 and pen_ok and not pending),
    "pending": pending,
  }
  out = ROOT / "results" / "g1_gate_decision.json"
  out.write_text(json.dumps(verdict, indent=1))
  print(json.dumps(verdict, indent=1))
  if not verdict["gate_passes"]:
    print("[gate] not passing (yet)" + (" - still running: " + ", ".join(pending) if pending else ""))
    return
  if not a.launch:
    print("[gate] would launch; rerun with --launch")
    return
  jobs = waves.g1_centering_pilots(a.iters, penalty=verdict["g1_penalty"])
  (ROOT / "manifests" / "g1c.json").write_text(json.dumps(waves.to_manifest(jobs), indent=1))
  campaign = "g1c-" + time.strftime("%m%d%H")
  cmd = [str(ROOT / ".venv" / "bin" / "python"), "orchestrator.py", "--campaign", campaign,
         "--manifest", "manifests/g1c.json", "--launch", "2", "--poll", "120",
         "--hard-cap", str(a.hard_cap), "--soft-cap", str(a.hard_cap * 0.85)]
  log = open(ROOT / "results" / f"orch_{campaign}.log", "a")
  subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                   start_new_session=True, stdin=subprocess.DEVNULL)
  verdict["launched_campaign"] = campaign
  verdict["launched_jobs"] = len(jobs)
  out.write_text(json.dumps(verdict, indent=1))
  print(f"[gate] launched {campaign} with {len(jobs)} jobs (G1 penalty {verdict['g1_penalty']})")


if __name__ == "__main__":
  main()
