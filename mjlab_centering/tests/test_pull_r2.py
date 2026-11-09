"""The R2 puller must pick the newest snapshot per pod tag (a final snapshot must beat periodic ones)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import pull_r2  # noqa: E402


class _Client:
  def __init__(self, objs):
    self.objs = objs

  def list_objects_v2(self, **kw):
    prefix = kw["Prefix"]
    return {"Contents": [o for o in self.objs if o["Key"].startswith(prefix)], "IsTruncated": False}


def test_newest_snapshot_wins_per_tag():
  t0 = datetime(2026, 11, 1, 3, 0)
  objs = [
    {"Key": "mjlab_ssp_centering/pods/pod1/_project_results.tgz", "LastModified": t0, "Size": 1},
    {"Key": "mjlab_ssp_centering/pods/pod1/_FINAL_results.tgz", "LastModified": t0 + timedelta(hours=1), "Size": 2},
    {"Key": "mjlab_ssp_centering/pods/pod2/_project_results.tgz", "LastModified": t0, "Size": 3},
    {"Key": "mjlab_ssp_centering/w9/core_x/final.json", "LastModified": t0, "Size": 4},
  ]
  best = pull_r2.latest_per_tag(_Client(objs), "bucket", "pods/")
  assert set(best) == {("pod1", "_project_results.tgz"), ("pod1", "_FINAL_results.tgz"), ("pod2", "_project_results.tgz")}
  assert "linear" not in str(best)


def test_several_tarballs_from_one_pod_do_not_shadow_each_other():
  t0 = datetime(2026, 11, 1, 3, 0)
  objs = [
    {"Key": "mjlab_ssp_centering/pods/pod1/_project_results.tgz", "LastModified": t0, "Size": 1},
    {"Key": "mjlab_ssp_centering/pods/pod1/_project_linear_slice_out.tgz", "LastModified": t0 + timedelta(minutes=5), "Size": 2},
  ]
  best = pull_r2.latest_per_tag(_Client(objs), "bucket", "pods/")
  assert len(best) == 2
