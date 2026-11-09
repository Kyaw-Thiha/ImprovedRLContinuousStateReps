"""d151: video retrains clone the claim cells exactly; claim blocks upload checkpoints, SSP/search blocks do not."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jobs_r4  # noqa: E402

IGNORE = {"job_id", "block", "upload_ckpt", "notes", "priority"}


def _claim(jid, man):
  jobs = json.loads((ROOT / "manifests" / man).read_text())["jobs"]
  return next(j for j in jobs if j["job_id"] == jid)


def _same(a, b):
  assert {k: v for k, v in a.items() if k not in IGNORE} == {k: v for k, v in b.items() if k not in IGNORE}


def test_video_runs_equal_the_claim_cells_except_checkpoint_upload():
  jobs = {j["job_id"]: j for j in jobs_r4.video_runs()}
  jobs_r4.check(list(jobs.values()))
  assert len(jobs) == 4 and all(j["upload_ckpt"] for j in jobs.values())
  for c in ("none", "value"):
    _same(jobs[f"r7vid_go1_g0.999_o1_{c}_s3002"], _claim(f"w2main_go1_raw_{c}_g0.999_o1_s3002", "night1.json"))
    _same(jobs[f"r7vid_g1_g0.99_o1_{c}_s3012"], _claim(f"r6g1_g1_g0.99_o1_{c}_s3012", "r6g1queue-0ede5636.json"))


def test_claim_blocks_upload_and_ssp_blocks_do_not():
  assert all(j["upload_ckpt"] for j in jobs_r4.g1_claim(2500))
  assert all(j["upload_ckpt"] for j in jobs_r4.off0_20() + jobs_r4.r5bias() + jobs_r4.ssp_confirm("ssp_low", 2, 2))
  assert not any(j["upload_ckpt"] for j in jobs_r4.ssp_pilot(2, 0.3) + jobs_r4.ssp_ls_sweep(2) + jobs_r4.g1_probe() + jobs_r4.width())
