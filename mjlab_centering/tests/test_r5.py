"""REPORT_005 plumbing (d113-d121): bias init, branch-only manifests, top-up manifests, shards, community-only launching."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import jobs_r4  # noqa: E402
import orchestrator  # noqa: E402
import podrun  # noqa: E402

from mjlab_centering.algo import _NormCritic, init_critic_output_bias  # noqa: E402


def _round4_jobs() -> dict[str, dict]:
  out = {}
  for f in glob.glob(str(ROOT / "manifests" / "r4*.json")):
    for j in json.loads(Path(f).read_text())["jobs"]:
      out.setdefault(j["job_id"], j)
  return out


def _diff(a: dict, b: dict) -> set[str]:
  return {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"upload_ckpt"}  # d151: claim blocks upload checkpoints


# ---------------------------------------------------------------- critic bias
def test_critic_bias_init_sets_only_the_last_layer_bias_and_works_through_the_wrapper():
  net = nn.Sequential(nn.Linear(4, 8), nn.ELU(), nn.Linear(8, 1))
  before = [p.clone() for p in net.parameters()]
  assert init_critic_output_bias(net, 1000.0) == 1000.0
  after = list(net.parameters())
  assert torch.all(after[3] == 1000.0)
  assert all(torch.equal(b, a) for b, a in zip(before[:3], after[:3]))  # first layer and last weights untouched
  out = net(torch.zeros(5, 4))  # with zero input the output is bias + w . ELU(b1): within a few units of 1000
  assert float((out - 1000.0).abs().max()) < 5.0
  wrapped = _NormCritic(nn.Sequential(nn.Linear(4, 8), nn.ELU(), nn.Linear(8, 1)))
  init_critic_output_bias(wrapped, -3.5)
  assert float(wrapped.critic[2].bias) == -3.5


def test_bias_pilot_manifest_is_the_e11_uncentered_cell_plus_the_bias_flag():
  night = {j["job_id"]: j for j in json.loads(jobs_r4.NIGHT1.read_text())["jobs"]}
  jobs = jobs_r4.r5bias()
  assert [j["seed"] for j in jobs] == [3000, 3001, 3002]
  jobs_r4.check(jobs)
  for j in jobs:
    ref = night[f"w2main_go1_raw_none_g0.999_o1_s{j['seed']}"]
    assert _diff(ref, j) == {"job_id", "block", "notes", "extra"}, _diff(ref, j)
    assert j["extra"]["critic_bias_init"] == "offset" and j["center"] == "none" and j["offset"] == 1.0
    assert abs(j["offset"] / (1.0 - j["gamma"]) - 1000.0) < 1e-6


# ---------------------------------------------------------------- top-up manifests
def test_sweep_top20_is_43_jobs_identical_to_round4_cells_apart_from_seed_and_id():
  jobs = jobs_r4.sweep_top20()
  assert len(jobs) == 43 and len({j["job_id"] for j in jobs}) == 43
  jobs_r4.check(jobs)
  r4 = _round4_jobs()
  ref = next(j for j in r4.values() if j["block"] == "r4sweep" and j["offset"] == 0.1 and j["center"] == "none")
  for j in jobs:
    peer = next(x for x in r4.values() if x["block"] == "r4sweep" and x["offset"] == j["offset"] and x["center"] == j["center"])
    assert _diff(peer, j) <= {"job_id", "seed"}, _diff(peer, j)
  assert _diff(ref, jobs[3]) <= {"job_id", "seed", "offset", "center"}
  seeds = {(j["offset"], j["center"]): [] for j in jobs}
  for j in jobs:
    seeds[(j["offset"], j["center"])].append(j["seed"])
  # seeds 3010-3019 for every cell, plus seed 3009 where the cap cut it
  for (o, c), ss in seeds.items():
    assert set(range(3010, 3020)) <= set(ss)
  assert sorted(j["job_id"] for j in jobs if j["seed"] == 3009) == sorted(jobs_r4.SWEEP_CUT)


def test_offset0_topup_matches_the_existing_offset0_cells():
  night = {j["job_id"]: j for j in json.loads(jobs_r4.NIGHT1.read_text())["jobs"]}
  jobs = jobs_r4.off0_20()
  assert len(jobs) == 20 and {j["seed"] for j in jobs} == set(range(3010, 3020))
  for j in jobs:
    ref = night[f"w2off0_go1_raw_{j['center']}_g0.999_o0_s3000"]
    assert _diff(ref, j) <= {"job_id", "seed", "block", "notes", "priority"}, _diff(ref, j)


def test_width_rest_and_g1_splits_are_complete_and_disjoint():
  assert len(jobs_r4.width_rest()) == 12
  assert not {j["job_id"] for j in jobs_r4.width_rest()} & set(jobs_r4.WIDTH_RAW_DONE)
  t, r = jobs_r4.g1_timing(), jobs_r4.g1_rest()
  assert len(t) == 6 and len(r) == 34
  assert {j["job_id"] for j in t} | {j["job_id"] for j in r} == {j["job_id"] for j in jobs_r4.g1_20()}
  assert {j["seed"] for j in t} == {3000, 3001, 3002}


# ---------------------------------------------------------------- branch-only shock manifest
def test_shock_branches_are_branch_only_and_identical_to_round4_branches():
  jobs = jobs_r4.shock_branches()
  assert len(jobs) == 80 and all(j["block"] == "r4branch" for j in jobs)
  jobs_r4.check(jobs)  # no mother in the manifest, and that must be allowed
  assert {(j["seed"], j["center"]) for j in jobs} == {(s, c) for s in range(3010, 3020) for c in ("none", "value")}
  assert {j["job_id"].split("_o0_")[1].rsplit("_", 2)[0] for j in jobs} == {"off+0.1", "off+1.0", "actx5", "ctrl"}
  r4 = _round4_jobs()
  for j in jobs:
    kind = j["job_id"].split("_o0_")[1].rsplit("_", 2)[0]
    peer = r4[f"r4branch_go1_g0.999_o0_{kind}_{j['center']}_s3000"]
    assert _diff(peer, j) == {"job_id", "seed", "extra"}, _diff(peer, j)
    ex_a, ex_b = dict(peer["extra"]), dict(j["extra"])
    assert ex_b.pop("after_campaign") == jobs_r4.MOTHER_CAMPAIGN
    assert ex_b.pop("after") == jobs_r4.mother_id(j["center"], j["seed"])
    ex_a.pop("after")
    assert ex_a == ex_b
    assert j["shock_at"] == 1200 and j["iters"] == 1600 and j["eval_every"] == 25


def test_podrun_reads_the_mother_from_after_campaign_when_given():
  assert podrun.mother_prefix({"after": "m", "after_campaign": "other-1"}, "mine-2") == "other-1"
  assert podrun.mother_prefix({"after": "m"}, "mine-2") == "mine-2"
  assert podrun.mother_prefix({}, None) is None


# ---------------------------------------------------------------- shards
def test_shards_are_disjoint_complete_and_keep_a_mothers_branches_together():
  jobs = jobs_r4.shock_branches()
  parts = [jobs_r4.shard(jobs, i, 3) for i in range(3)]
  ids = [j["job_id"] for p in parts for j in p]
  assert sorted(ids) == sorted(j["job_id"] for j in jobs) and len(set(ids)) == len(ids)
  where = {}
  for i, p in enumerate(parts):
    for j in p:
      where.setdefault((j["seed"], j["center"]), set()).add(i)
  assert all(len(v) == 1 for v in where.values())
  sw = jobs_r4.sweep_top20()
  assert sorted(j["job_id"] for i in range(4) for j in jobs_r4.shard(sw, i, 4)) == sorted(j["job_id"] for j in sw)


# ---------------------------------------------------------------- community-only launching
GPUS = orchestrator._parse_gpus(
  "NVIDIA GeForce RTX 4090:0.34:COMMUNITY,NVIDIA GeForce RTX 3090:0.22:COMMUNITY," + orchestrator.SECURE_GPUS)


def test_secure_is_never_chosen_without_permission():
  for rounds, stock_s in [(0, 0.0), (3, 0.0), (9, 0.0), (0, 7200.0), (5, 7200.0)]:
    cands, needs = orchestrator.choose_candidates(GPUS, rounds, 3, stock_s, 3600.0, allow_secure=False)
    assert all(g[2] == "COMMUNITY" for g in cands)
    assert needs == (rounds >= 3 or stock_s >= 3600.0)


def test_secure_needs_rounds_or_the_no_stock_window_and_the_explicit_flag():
  cands, needs = orchestrator.choose_candidates(GPUS, 2, 3, 1200.0, 3600.0, allow_secure=True)
  assert not needs and all(g[2] == "COMMUNITY" for g in cands)
  cands, needs = orchestrator.choose_candidates(GPUS, 3, 3, 0.0, 3600.0, allow_secure=True)
  assert needs and all(g[2] == "SECURE" for g in cands)
  cands, needs = orchestrator.choose_candidates(GPUS, 0, 3, 3600.0, 3600.0, allow_secure=True)
  assert needs and all(g[2] == "SECURE" for g in cands)


def test_the_default_gpu_list_has_no_secure_entries():
  import subprocess
  base = [sys.executable, str(ROOT / "orchestrator.py"), "--campaign", "x", "--manifest", str(ROOT / "manifests" / "clip1.json")]
  out = subprocess.run(base, capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}).stdout
  assert "gpu preference" in out and "SECURE" not in out
  out2 = subprocess.run(base + ["--allow-secure"], capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}).stdout
  assert "SECURE" in out2


# ---------------------------------------------------------------- stock-first
def test_stock_candidates_keep_only_available_community_skus_pinned_to_a_dc_with_stock():
  gpus = orchestrator._parse_gpus("NVIDIA GeForce RTX 4090:0.34:COMMUNITY,NVIDIA L40:0.69:COMMUNITY,"
                                  "NVIDIA GeForce RTX 4090:0.89:SECURE")
  stock = [{"gpuId": "NVIDIA L40", "price": 0.69, "dcs": ["EU-SE-1", "US-TX-3"]},
           {"gpuId": "NVIDIA H200", "price": 3.59, "dcs": ["CA-MTL-3"]}]  # not configured: skipped, never silently used
  out = orchestrator.stock_candidates(gpus, stock)
  assert out == [("NVIDIA L40", 0.69, "COMMUNITY", "EU-SE-1")]
  assert orchestrator.stock_candidates(gpus, []) == []
  # a price ceiling keeps an opportunistic expensive SKU out of a cheap campaign
  assert orchestrator.stock_candidates(gpus, stock, max_price=0.5) == []
  priced = [{"gpuId": "NVIDIA L40", "price": 0.69, "dcs": ["EU-SE-1"]},
            {"gpuId": "NVIDIA GeForce RTX 4090", "price": 0.34, "dcs": ["EU-CZ-1"]}]


def test_stock_first_off_leaves_the_configured_list_alone():
  gpus = orchestrator._parse_gpus("NVIDIA GeForce RTX 4090:0.34:COMMUNITY")
  cands, needs = orchestrator.choose_candidates(gpus, 0, 3, 0.0, 3600.0, allow_secure=False)
  assert cands == [("NVIDIA GeForce RTX 4090", 0.34, "COMMUNITY")] and not needs
