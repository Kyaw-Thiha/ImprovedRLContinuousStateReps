"""REPORT_004 plumbing: manifests clone the E11 protocol, the offset shock is absolute, RFF draws follow the run seed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jobs_r4  # noqa: E402

from mjlab_centering.ssp_torch import HexSSP  # noqa: E402
from mjlab_centering.train import RunSpec, build_spec, make_env  # noqa: E402


def test_manifests_pass_protocol_checks_and_have_expected_sizes():
  assert len(jobs_r4.sweep()) == 40
  sh = jobs_r4.shock()
  assert len([j for j in sh if j["block"] == "r4mother"]) == 20 and len(sh) == 100
  assert len(jobs_r4.width()) == 15
  for jobs in (jobs_r4.sweep(), sh, jobs_r4.width(), jobs_r4.g1_20(), jobs_r4.clip20(), jobs_r4.off0_20()):
    jobs_r4.check(jobs)


def test_branches_resume_from_the_mother_checkpoint_name_and_controls_do_not_shock():
  sh = {j["job_id"]: j for j in jobs_r4.shock()}
  for j in sh.values():
    if j["block"] != "r4branch":
      continue
    m = sh[j["extra"]["after"]]
    assert m["save_at"] == j["shock_at"] == 1200 and m["center"] == j["center"] and m["seed"] == j["seed"]
    assert m["upload_ckpt"] is True and j["iters"] == 1600 and j["offset"] == 0.0
  ctrl = sh["r4branch_go1_g0.999_o0_ctrl_none_s3000"]
  assert ctrl["shock_kind"] == "none" and ctrl["shock_at"] == 1200


def test_cloned_jobs_differ_from_e11_only_in_the_named_fields():
  night = {j["job_id"]: j for j in json.loads(jobs_r4.NIGHT1.read_text())["jobs"]}
  ref = night["w2main_go1_raw_value_g0.999_o1_s3000"]
  new = jobs_r4.sweep()[1]
  assert {k for k in ref if ref[k] != new[k]} - {"upload_ckpt"}  # d151 == {"job_id", "block", "offset", "notes"}


def test_absolute_offset_shock_is_applied_even_when_the_base_offset_is_zero():
  spec = build_spec({**jobs_r4.shock(seeds=(3000,))[-2], "num_envs": 4, "eval_envs": 0})  # a branch with an offset shock
  for j in jobs_r4.shock(seeds=(3000,)):
    if j["job_id"] == "r4branch_go1_g0.999_o0_off+0.1_value_s3000":
      spec = build_spec({**j, "num_envs": 4})
  assert spec.shock_kind == "offset" and spec.offset == 0.0
  env, vec = make_env(spec, 4)
  try:
    assert vec.shock_delta == 0.1
    vec.iter = 1199
    assert vec.offset_now == 0.0
    vec.iter = 1200
    assert abs(vec.offset_now - 0.1) < 1e-9
  finally:
    env.close()


def test_legacy_multiplicative_offset_shock_is_unchanged():
  spec = RunSpec(job_id="x", offset=1.0, shock_kind="offset", shock_at=5, shock_factor=1.1, num_envs=4)
  env, vec = make_env(spec, 4)
  try:
    assert abs(vec.shock_delta - 0.1) < 1e-9
  finally:
    env.close()


def test_rff_code_follows_the_run_seed_and_ssp_is_deterministic():
  a0, a1 = HexSSP(3, rff=True, rff_dim=33, seed=0).A, HexSSP(3, rff=True, rff_dim=33, seed=1).A
  assert not torch.equal(a0, a1) and a0.shape == a1.shape == (33, 3)
  assert torch.equal(HexSSP(3, n_rotates=2, n_scales=2).A, HexSSP(3, n_rotates=2, n_scales=2).A)
  assert HexSSP(3, n_rotates=2, n_scales=2).out_dim == 33 and HexSSP(3, n_rotates=4, n_scales=4).out_dim == 129
