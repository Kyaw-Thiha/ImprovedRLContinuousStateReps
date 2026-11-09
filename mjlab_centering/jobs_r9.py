"""Manifests for REPORT_009 (round 9a): the three "race" arms, plus the clip-off control.

    python jobs_r9.py pilot          # 57 jobs: A1 at two bias rates, A2 warm-up (clip on/off), A3, clip-off top-up

Every job is cloned from the corrected-protocol E11 cell (`w2main_*` at c = 1, night1) exactly like jobs_r4.make, so
nothing can drift from the cells these runs are paired with. Only the arm's own knob, `log_rew_std = True` (d155) and
`save_every = 250` (d216: a preempted pod resumes from its rolling checkpoint) differ.

Go1, gamma .999, fall penalty 10, offset +1, 2000 iterations, mjlab defaults, 2048 envs, evals every 50.
Seeds 3000-3009 (pilot grade: 10 seeds is a pilot under the study's rule, never a claim).

  a1_005    A1 fast output bias, bias lr 0.05, value clipping OFF
  a1_05     A1 fast output bias, bias lr 0.5,  value clipping OFF
  clipoff   control: value clipping OFF and nothing else (seeds 3003-3009; 3000-3002 already exist as `clip1_*`)
  a2_on     A2 critic warm-up, value clipping ON (the default)
  a2_off    A2 critic warm-up, value clipping OFF
  a3        A3 slow policy commitment: actor lr x0.1 for the first 500 iterations, then normal

Jobs are ordered seed-major (seed -> arm) so a partial set is balanced across arms.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jobs_r4 as J4

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(3000, 3010))
ITERS = 2000

ARMS = {
  "a1_005": dict(bias_lr=0.05, value_clip=False),
  "a1_05": dict(bias_lr=0.5, value_clip=False),
  "clipoff": dict(value_clip=False),
  "a2_on": dict(warmup=True),
  "a2_off": dict(warmup=True, value_clip=False),
  "a3": dict(actor_lr_scale=0.1, actor_lr_iters=500),
}
ORDER = ("a1_05", "a1_005", "clipoff", "a2_off", "a2_on", "a3")
# the clip-off control's first three seeds are the round-4 pilot (`clip1_go1_raw_none_clipoff_g0.999_o1_s3000-3002`)
EXISTING = {"clipoff": tuple(range(3000, 3003))}


def job(arm: str, seed: int, offset: float = 1.0) -> dict:
  kw = dict(ARMS[arm])
  clip = kw.pop("value_clip", None)
  tag = f"go1_{arm}_g0.999_o{offset:g}"
  j = J4.make("r9a", "none", seed, tag, offset=float(offset), log_rew_std=True, save_every=250, **kw)
  if clip is not None:
    j["value_clip"] = bool(clip)
  j["job_id"] = f"r9a_{tag}_s{seed}"
  j["notes"] = f"round9a:{arm}"
  return j


def pilot(seeds=SEEDS) -> list[dict]:
  out = []
  for s in seeds:
    for arm in ORDER:
      if s in EXISTING.get(arm, ()):
        continue
      out.append(job(arm, s))
  return out


def a4(seeds=SEEDS, offset=0.1) -> list[dict]:
  """A4 (optional, budget-gated): A1 at c = 0.1, testing whether the effect scales with the level (10x smaller)."""
  return [dict(job("a1_05", s, offset)) for s in seeds]


def escalation(seed_lo: int, seed_hi: int, arms) -> list[dict]:
  """Seeds 3010-3019 for whichever arm rescues uncentered PPO (the pre-registered escalation rule)."""
  return [job(a, s) for s in range(seed_lo, seed_hi + 1) for a in arms]


def check(jobs: list[dict]) -> None:
  ids = [j["job_id"] for j in jobs]
  assert len(ids) == len(set(ids)), "duplicate job ids"
  for j in jobs:
    assert j["robot"] == "go1" and j["gamma"] == 0.999 and j["penalty"] == 10.0, j["job_id"]
    assert j["offset"] in (0.1, 1.0) and j["iters"] == ITERS, j["job_id"]
    assert j["time_outs"] and j["falls_reset"], j["job_id"]          # corrected protocol
    assert j["log_rew_std"] is True, j["job_id"]                     # per-step reward std in every run
    assert j["save_every"] == 250, j["job_id"]                       # mid-run checkpointing
    assert j["num_envs"] == 2048 and j["eval_every"] == 50, j["job_id"]
    assert j["center"] == "none" and j["ret_norm"] == "none" and j["reward_scale"] == "none", j["job_id"]
    assert not j.get("bias_lr") or j["value_clip"] is False, "A1 runs with value clipping off"
    if j.get("warmup"):
      assert j["warmup_max_iters"] if "warmup_max_iters" in j else True


def campaign_id(tag: str, jobs: list[dict]) -> str:
  h = hashlib.sha1(json.dumps([j["job_id"] for j in jobs]).encode()).hexdigest()[:8]
  return f"r9{tag}-{h}"


def main(argv: list[str]) -> None:
  what = argv[0] if argv else "pilot"
  if what == "pilot":
    jobs, tag = pilot(), "pilot"
  elif what == "a4":
    jobs, tag = a4(), "a4"
  elif what == "leak":
    jobs, tag = leak(), "aleak"
  elif what == "optional":
    jobs, tag = leak_optional(argv[1]), "aopt"
  elif what == "escalate":
    jobs, tag = escalation(int(argv[1]), int(argv[2]), argv[3].split(",")), "esc"
  else:
    raise SystemExit(f"unknown: {what}")
  (check_leak if what in ("leak", "optional") else check)(jobs)
  camp = campaign_id(tag, jobs)
  out = ROOT / "manifests" / f"{camp}.json"
  out.write_text(json.dumps({"campaign": camp, "jobs": jobs}, indent=1))
  print(json.dumps({"campaign": camp, "manifest": str(out), "jobs": len(jobs),
                    "iterations": sum(j["iters"] for j in jobs)}, indent=1))




# --------------------------------------------------------------------------------- round 9b: the truncated-GAE leak
# Campaigns are `r9aleak-*` / `r9adiag-*` so they keep the round's `r9a` prefix (isolation and spend attribution both key
# on it) and pods stay `mjc-r9aleak-*`.
LEAK_SEEDS = tuple(range(3000, 3010))
LEAK_ITERS = 2000
DIAG_ITERS = 600
DIAG_SEEDS = (3000, 3001, 3002)

# arm -> fields that differ from its template (`none` = the E11 uncentered cell, `value` = the centred one)
LEAK_ARMS = {
  "posnorm": dict(template="none", adv_mode="pos_norm"),
  "poscenter": dict(template="none", adv_mode="pos_center"),
  "lam0_none": dict(template="none", lam=0.0),
  "lam0_value": dict(template="value", lam=0.0),
  "len96": dict(template="none", adv_mode="pos_norm"),  # optional: the same fix at a 96-step rollout
}


def leak_job(arm: str, seed: int, block: str = "r9leak", iters: int = LEAK_ITERS, offset: float = 1.0,
             num_steps_per_env: int | None = None, num_envs: int | None = None) -> dict:
  spec = dict(LEAK_ARMS[arm])
  template = spec.pop("template")
  # every 9b job states its own lambda and advantage mode, so `config.json` is self-describing and the wiring check has
  # something to assert (the RunSpec default lam=None would record as null and read as "unknown").
  spec.setdefault("lam", 0.95)
  spec.setdefault("adv_mode", "global")
  tag = f"go1_{arm}_g0.999_o{offset:g}"
  over = dict(offset=float(offset), iters=int(iters), log_rew_std=True, save_every=250, log_adv_pos=True, **spec)
  if num_steps_per_env is not None:
    over["num_steps_per_env"] = int(num_steps_per_env)
  if num_envs is not None:
    over["num_envs"] = int(num_envs)
  j = J4.make(block, template, seed, tag, **over)
  j["job_id"] = f"{block}_{tag}_s{seed}"
  j["notes"] = f"round9b:{arm}"
  return j


def diag_job(arm: str, seed: int) -> dict:
  j = J4.make("r9diag", arm, seed, f"go1_diag_{arm}_g0.999_o1", offset=1.0, iters=DIAG_ITERS,
              log_rew_std=True, save_every=250, log_adv_pos=True, lam=0.95, adv_mode="global")
  j["job_id"] = f"r9diag_go1_diag_{arm}_g0.999_o1_s{seed}"
  j["notes"] = f"round9b:diag_{arm}"
  return j


def leak(seeds=LEAK_SEEDS) -> list[dict]:
  """The pre-registered round-9b set: the per-position diagnostic first, then the four arms, seed-major."""
  out = [diag_job(a, s) for s in DIAG_SEEDS for a in ("none", "value")]
  out += [leak_job(a, s) for s in seeds for a in ("posnorm", "poscenter", "lam0_none", "lam0_value")]
  return out


def leak_optional(what: str) -> list[dict]:
  """Budget-gated extras (d1104): `posnorm01` = the per-position fix at c = 0.1, 5 seeds; `len96` = a 96-step rollout."""
  if what == "posnorm01":
    return [leak_job("posnorm", s, iters=LEAK_ITERS, offset=0.1) for s in range(3000, 3005)]
  if what == "len96":
    # same batch (24 x 2048 = 96 x 512) so only the rollout length changes
    return [leak_job("len96", s, iters=LEAK_ITERS, num_steps_per_env=96, num_envs=512) for s in range(3000, 3010)]
  raise SystemExit(f"unknown: {what}")


def check_leak(jobs: list[dict]) -> None:
  ids = [j["job_id"] for j in jobs]
  assert len(ids) == len(set(ids)), "duplicate job ids"
  for j in jobs:
    diag = j["block"] == "r9diag"
    assert j["robot"] == "go1" and abs(j["gamma"] - 0.999) < 1e-12 and j["penalty"] == 10.0, j["job_id"]
    assert j["time_outs"] and j["falls_reset"] and j["log_rew_std"] is True, j["job_id"]
    assert j["save_every"] == 250 and j["eval_every"] == 50, j["job_id"]
    assert j["iters"] == (DIAG_ITERS if diag else LEAK_ITERS), j["job_id"]
    assert j["log_adv_pos"] is True, j["job_id"]  # every 9b run logs the position profile
    assert j["ret_norm"] == "none" and j["reward_scale"] == "none", j["job_id"]
    assert j["adv_mode"] in ("global", "pos_norm", "pos_center"), j["job_id"]
    assert j["lam"] in (0.0, 0.95), j["job_id"]
    if j["adv_mode"] != "global":
      assert j["center"] == "none", j["job_id"]
    if j["lam"] == 0.0:
      assert j["center"] in ("none", "value"), j["job_id"]
      if j["center"] == "value":  # the validity gate for the lambda arm must be a real centred arm
        assert j["eta"] == 0.03 and j["r_bar_init"] == "first_batch", j["job_id"]
    assert j["num_envs"] in (512, 2048), j["job_id"]
    if j["num_steps_per_env"] == 96:
      assert j["num_envs"] == 512, "the 96-step arm keeps the batch size: 96 x 512 == 24 x 2048"


if __name__ == "__main__":
  main(sys.argv[1:])
