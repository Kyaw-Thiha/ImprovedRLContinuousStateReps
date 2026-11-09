"""Wait for a campaign to drain, restore its runs locally, and write a summary a human can read cold.

    python analysis/finalize_campaign.py --campaign night1-3e71b4 --wait-min 600

Output: `results/<campaign>_summary.md` plus a copy of the per-cell table on stdout. Safe to run repeatedly; it always
re-derives from the R2 mirrors (authoritative) and trims each run to its newest attempt (decision 80).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mjlab_centering import r2

ROOT = Path(__file__).resolve().parents[1]
WALK = 0.5


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def newest(rows_: list[dict]) -> list[dict]:
  starts = [i for i, r in enumerate(rows_) if r.get("iter") == 0]
  return rows_[starts[-1]:] if len(starts) > 1 else rows_


def last5(run_rows: list[dict], key: str) -> float:
  v = [r[key] for r in run_rows[-5:] if key in r]
  return float(np.mean(v)) if v else float("nan")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--wait-min", type=float, default=0.0)
  ap.add_argument("--restore", action="store_true", default=True)
  ap.add_argument("--no-restore", dest="restore", action="store_false",
                  help="skip the local mirror restore (the summary itself always reads R2)")
  a = ap.parse_args()

  if a.wait_min > 0:
    t0 = time.time()
    while time.time() - t0 < a.wait_min * 60:
      man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
      pending = [j["job_id"] for j in man["jobs"] if j["job_id"] not in r2.done_set(a.campaign)]
      print(f"[finalize] {time.strftime('%H:%M:%S')} pending={len(pending)}", flush=True)
      if not pending:
        break
      time.sleep(180)

  if a.restore:
    subprocess.run([sys.executable, str(ROOT / "analysis" / "restore_runs.py"), "--out", "results/merged_r2"],
                   cwd=str(ROOT), check=False, capture_output=True, text=True)

  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  cells: dict[tuple, list[dict]] = defaultdict(list)
  for job in man["jobs"]:
    jid = job["job_id"]
    ev = newest(rows(f"{a.campaign}/runs/{jid}/eval.jsonl"))
    it = newest(rows(f"{a.campaign}/runs/{jid}/iter.jsonl"))
    cells[(job["block"], job["robot"], job["center"], job["gamma"], job["offset"], job["penalty"])].append({
      "seed": job["seed"], "done": r2.exists(f"{a.campaign}/runs/{jid}/DONE"),
      "track": last5(ev, "track_lin_frac"), "J": last5(ev, "J_task"),
      "std": float(it[-1].get("action_std", float("nan"))) if it else float("nan"),
      "iters": len(it),
    })

  lines = [f"# Campaign `{a.campaign}` summary", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the R2 mirrors. "
           "Walk = last-5 mean `track_lin_frac` >= 0.5 (standing is 0.267 / 0.28).", "",
           "| block | robot | center | gamma | offset | penalty | n | done | track | J_task | std | walks |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
  for (blk, robot, center, gam, off, pen), recs in sorted(cells.items()):
    tr = np.nanmean([r["track"] for r in recs])
    jt = np.nanmean([r["J"] for r in recs])
    sd = np.nanmean([r["std"] for r in recs])
    wk = sum(1 for r in recs if r["track"] >= WALK)
    lines.append(f"| {blk} | {robot} | {center} | {gam:g} | {off:g} | {pen:g} | {len(recs)} | "
                 f"{sum(r['done'] for r in recs)} | {tr:.3f} | {jt:.4f} | {sd:.2f} | {wk}/{len(recs)} |")

  lines += ["", "## Paired centering gaps (value - none), same seeds", ""]
  seen = {(k[0], k[1], k[3], k[4], k[5]) for k in cells}
  for blk, robot, gam, off, pen in sorted(seen):
    v = {r["seed"]: r for r in cells[(blk, robot, "value", gam, off, pen)]}
    n0 = {r["seed"]: r for r in cells[(blk, robot, "none", gam, off, pen)]}
    shared = sorted(set(v) & set(n0))
    if not shared:
      continue
    d = [v[s]["J"] - n0[s]["J"] for s in shared]
    dt = [v[s]["track"] - n0[s]["track"] for s in shared]
    lines.append(f"- {blk} {robot} gamma {gam:g} offset {off:g}: n={len(d)} "
                 f"dJ={np.mean(d):+.4f} [+-{np.std(d, ddof=1) / max(1, np.sqrt(len(d))):.4f} se] "
                 f"dtrack={np.mean(dt):+.3f}; per-seed dJ {[round(x, 4) for x in d]}")

  out = ROOT / "results" / f"{a.campaign}_summary.md"
  out.write_text("\n".join(lines) + "\n")
  print("\n".join(lines))
  print(f"\n[finalize] wrote {out}")


if __name__ == "__main__":
  main()
