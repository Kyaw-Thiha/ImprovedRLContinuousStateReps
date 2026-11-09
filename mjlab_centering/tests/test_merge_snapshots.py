"""Merge precedence: a finished run beats a partial one, and the newest finished run wins."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import merge_snapshots as ms  # noqa: E402


def _mkrun(base: Path, job: str, finished: bool, mtime: float, j_task: float):
  d = base / job
  d.mkdir(parents=True, exist_ok=True)
  (d / "iter.jsonl").write_text('{"iter": 0}\n')
  (d / "eval.jsonl").write_text(json.dumps({"iter": 0, "J_task": j_task}) + "\n")
  f = d / "final.json" if finished else d / "iter.jsonl"
  if finished:
    (d / "final.json").write_text(json.dumps({"eval": [{"J_task": j_task}]}))
  os.utime(f, (mtime, mtime))


def test_finished_and_newest_wins(tmp_path):
  snaps = tmp_path / "snaps"
  _mkrun(snaps / "pod1", "job", finished=True, mtime=time.time() - 100, j_task=0.1)
  _mkrun(snaps / "pod2", "job", finished=True, mtime=time.time(), j_task=0.2)
  _mkrun(snaps / "pod3", "job2", finished=False, mtime=time.time() + 50, j_task=0.3)
  snaps_argv = ["--snapshots", str(snaps), "--out", str(tmp_path / "merged")]
  import sys as _s
  _s.argv = ["merge_snapshots.py", *snaps_argv]
  ms.main()
  merged = tmp_path / "merged"
  assert json.loads((merged / "job" / "final.json").read_text())["eval"][0]["J_task"] == 0.2
  rep = json.loads((merged / "merge_report.json").read_text())
  assert rep["job"]["source"].endswith("pod2/job")
  assert rep["job2"]["finished"] is False
