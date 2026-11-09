"""Merge recovered pod snapshots into one canonical results tree, one directory per job id.

Pods can run the same job more than once (retasked queues, duplicated attempts), so the same job id appears in several
snapshots. Precedence is documented, not implicit: a run that reached `final.json` beats one that did not, and among
finished runs the most recently written one wins. The chosen source is recorded in `merge_report.json` so any table can
be traced back to a pod and a time.

    python analysis/merge_snapshots.py --snapshots results/from_r2 --out results/merged
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def find_runs(snapshots: Path) -> list[tuple[str, Path, bool, float]]:
  out = []
  for final in snapshots.rglob("final.json"):
    run_dir = final.parent
    mtime = os.path.getmtime(final)
    out.append((run_dir.name, run_dir, True, mtime))
  for iterf in snapshots.rglob("iter.jsonl"):
    run_dir = iterf.parent
    if (run_dir / "final.json").exists():
      continue
    out.append((run_dir.name, run_dir, False, os.path.getmtime(iterf)))
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--snapshots", default="results/from_r2")
  ap.add_argument("--out", default="results/merged")
  a = ap.parse_args()
  snaps, out = Path(a.snapshots), Path(a.out)
  best: dict[str, tuple[Path, bool, float]] = {}
  for jid, d, done, mt in find_runs(snaps):
    key = (done, mt)
    if jid not in best or key > (best[jid][1], best[jid][2]):
      best[jid] = (d, done, mt)
  if out.exists():
    shutil.rmtree(out)
  out.mkdir(parents=True)
  report = {}
  for jid, (d, done, mt) in sorted(best.items()):
    dst = out / jid
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("iter.jsonl", "eval.jsonl", "final.json"):
      if (d / f).exists():
        shutil.copy2(d / f, dst / f)
    report[jid] = {"source": str(d), "finished": done, "mtime": mt}
  (out / "merge_report.json").write_text(json.dumps(report, indent=1))
  n_done = sum(1 for r in report.values() if r["finished"])
  print(f"{len(report)} job ids merged ({n_done} finished) -> {out}")


if __name__ == "__main__":
  main()
