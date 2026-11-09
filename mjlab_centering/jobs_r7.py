"""Manifests for REPORT_007 (d155 onward): value normalisation, return-std scaling and simple centering on Go1.

    python jobs_r7.py pilot           # Stage 1: search seeds 1000-1002, c = 1, one run per arm per seed (12 runs)
    python jobs_r7.py vnsc            # Stage 2: VN and SC, seeds 3000-3019, c in {0.1, 1} (80 runs)
    python jobs_r7.py rs              # Stage 3: RS, same seeds and offsets (40 runs), only if the budget gate allows

Jobs are ordered **seed-major** (seed -> offset -> arm), so that if a pod is lost part-way through a stage the
indicator for which jobs ran on which pod cannot line up with an arm.

Every job is cloned from the corrected-protocol E11 cells (`w2main_*`, night1), exactly like jobs_r4.make, so settings
cannot drift from the cells these runs are paired with. Only these fields differ from the paired cell: the arm's own
knob and `log_rew_std = True` (d155: per-step reward std, task and raw, every iteration).
Go1, gamma .999, fall penalty 10, 2000 iterations, 2048 envs, evals every 50 (the paired cells' cadence).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jobs_r4 as J4

ROOT = Path(__file__).resolve().parent
CLAIM_SEEDS = J4.CLAIM_SEEDS          # 3000-3019
SEARCH_SEEDS = J4.SEARCH_SEEDS        # 1000-1002
OFFSETS = (0.1, 1.0)
ITERS = 2000

# arm -> the only fields that differ from the paired uncentered/value cell (all arms are uncentered PPO)
ARMS = {
  "vn": dict(ret_norm="rlgames"),                                             # rl_games value normalisation
  "vn_noclamp": dict(ret_norm="rlgames", vn_clamp=0.0),                       # diagnostic: VN without the +-5 clamp
  "rs": dict(reward_scale="ret_std"),                                        # SB3 VecNormalize reward scaling
  "sc": dict(center="simple_raw", beta=0.03, r_bar_init="first_batch"),      # simple centering, eta 0.03
}
PILOT_ARMS = ("vn", "vn_noclamp", "rs", "sc")
CLAIM_ARMS = ("vn", "sc")
RS_ARMS = ("rs",)


def job(block: str, arm: str, seed: int, offset: float) -> dict:
  kw = dict(ARMS[arm])
  center = kw.pop("center", "none")  # every arm is built from the uncentered E11 cell and then changed
  j = J4.make(block, "none", seed, f"go1_{arm}_g0.999_o{offset:g}", offset=float(offset), log_rew_std=True, **kw)
  j["center"] = center
  j["job_id"] = f"{block}_go1_{arm}_g0.999_o{offset:g}_s{seed}"
  j["notes"] = f"round7:{arm}"
  return j


def pilot(seeds=SEARCH_SEEDS, arms=PILOT_ARMS, offset=1.0) -> list[dict]:
  """Stage 1: sanity only. One run per arm per seed at c = 1; the two reference arms give the curves a baseline."""
  return [job("r7pilot", a, s, offset) for s in seeds for a in arms]


def vnsc(seeds=CLAIM_SEEDS, offsets=OFFSETS, arms=CLAIM_ARMS) -> list[dict]:
  """Stage 2: VN and SC against the existing uncentered and value cells, paired by seed, at both offsets."""
  return [job("r7vnsc", a, s, o) for s in seeds for o in offsets for a in arms]


def rs(seeds=CLAIM_SEEDS, offsets=OFFSETS, arms=RS_ARMS) -> list[dict]:
  """Stage 3: RS, only if the budget gate allows it after Stage 2."""
  return [job("r7rs", a, s, o) for s in seeds for o in offsets for a in arms]


def check(jobs: list[dict]) -> None:
  ids = [j["job_id"] for j in jobs]
  assert len(ids) == len(set(ids)), "duplicate job ids"
  for j in jobs:
    assert j["robot"] == "go1" and j["gamma"] == 0.999 and j["penalty"] == 10.0 and j["iters"] == ITERS, j["job_id"]
    assert j["time_outs"] and j["falls_reset"], j["job_id"]     # corrected protocol
    assert j["log_rew_std"] is True, j["job_id"]                # d155: every new run logs the per-step reward std
    assert j["num_envs"] == 2048 and j["eval_every"] == 50, j["job_id"]
    if j["center"] in ("value", "simple_raw"):
      assert j["r_bar_init"] == "first_batch", j["job_id"]
    if j["center"] == "simple_raw":
      assert j["beta"] == 0.03, j["job_id"]  # same eta as value centering; only the driver of r_bar differs
    if j["center"] == "value":
      assert j["eta"] == 0.03, j["job_id"]
    if j["ret_norm"] == "rlgames":
      assert j["center"] == "none", j["job_id"]


def campaign_id(tag: str, jobs: list[dict]) -> str:
  h = hashlib.sha1(json.dumps([j["job_id"] for j in jobs]).encode()).hexdigest()[:8]
  return f"r7{tag}-{h}"


def main(argv: list[str]) -> None:
  what = argv[0] if argv else "pilot"
  jobs = {"pilot": pilot, "vnsc": vnsc, "rs": rs}[what]()
  check(jobs)
  camp = campaign_id({"pilot": "pilot", "vnsc": "vnsc", "rs": "rs"}[what], jobs)
  out = ROOT / "manifests" / f"{camp}.json"
  out.write_text(json.dumps({"campaign": camp, "jobs": jobs}, indent=1))
  print(json.dumps({"campaign": camp, "manifest": str(out), "jobs": len(jobs),
                    "iterations": sum(j["iters"] for j in jobs)}, indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
