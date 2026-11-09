"""Round 9b: the wiring checker's expectations, verified before any pod result exists.

`r9_wiring.check` is the thing that decides whether a run was wired as its job id claims. A checker that accepts
everything would let a mis-wired arm through silently, so each 9b arm is exercised twice: with the config its job id
implies (must PASS) and with a neighbour's config (must FAIL). The mapping itself is pinned against the manifest, so an
arm whose id and config disagree cannot be papered over by editing either side alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import r9_wiring as W  # noqa: E402

MANIFEST = ROOT / "manifests" / "r9aleak-faa5c3d5.json"


def _cfg(job: dict, **over) -> dict:
  cfg = {"adv_mode": job["adv_mode"], "lam": job["lam"], "log_adv_pos": job["log_adv_pos"],
         "center": job["center"], "eta": job["eta"], "num_steps_per_env": job["num_steps_per_env"],
         "num_envs": job["num_envs"]}
  cfg.update(over)
  return cfg


def _row(job: dict) -> dict:
  return {"iter": 0, "adv_pos_mean": [0.0] * job["num_steps_per_env"],
          "adv_mode": job["adv_mode"], "lam": job["lam"]}


@pytest.fixture(scope="module")
def jobs() -> dict[str, dict]:
  man = json.loads(MANIFEST.read_text())
  out = {}
  for j in man["jobs"]:
    jid = j["job_id"]
    if "_go1_diag_" in jid:
      out["diag_" + jid.split("_go1_diag_")[1].split("_g0.999")[0]] = j
    else:
      out[jid.split("_go1_")[1].rsplit("_g0.999", 1)[0]] = j
  return out


def test_every_pre_registered_arm_is_covered(jobs):
  assert set(jobs) == {"posnorm", "poscenter", "lam0_none", "lam0_value",
                       "diag_none", "diag_value"}, sorted(jobs)
  assert set(W.LEAK_EXPECT) >= set(jobs)


@pytest.mark.parametrize("arm", ["posnorm", "poscenter", "lam0_none", "lam0_value"])
def test_each_arm_passes_with_its_own_config(jobs, arm):
  job = jobs[arm]
  res = W.check(_cfg(job), [_row(job)], arm)
  fails = [r for r in res if r[1] == "FAIL"]
  assert not fails, fails
  assert any(r[1] == "PASS" for r in res)


def test_a_swapped_advantage_mode_is_caught(jobs):
  """posnorm's config submitted as poscenter is the exact mis-wiring the checker exists for."""
  job = jobs["posnorm"]
  res = W.check(_cfg(job, adv_mode="pos_center"), [_row(job)], "posnorm")
  assert any(r[1] == "FAIL" and "adv_mode" in r[2] for r in res), res


def test_a_wrong_lambda_is_caught(jobs):
  job = jobs["lam0_none"]
  res = W.check(_cfg(job, lam=0.95), [_row(job)], "lam0_none")
  assert any(r[1] == "FAIL" and r[2].startswith("lam ") for r in res), res


def test_the_centred_template_of_the_lambda_gate_is_checked(jobs):
  job = jobs["lam0_value"]
  res = W.check(_cfg(job, center="none", eta=0.0), [_row(job)], "lam0_value")
  assert any(r[1] == "FAIL" and "centred template" in r[2] for r in res), res
  ok = W.check(_cfg(job), [_row(job)], "lam0_value")
  assert not [r for r in ok if r[1] == "FAIL"], ok


def test_a_short_position_vector_is_caught(jobs):
  job = jobs["posnorm"]
  row = {"iter": 0, "adv_pos_mean": [0.0] * 23, "adv_mode": "pos_norm", "lam": 0.95}
  res = W.check(_cfg(job), [row], "posnorm")
  assert any(r[1] == "FAIL" and "adv_pos_mean" in r[2] for r in res), res


def test_a_96_step_run_must_show_96_positions(jobs):
  """The optional len96 arm: its position vector is 96 long, so a 24-long one is a real failure, not a format detail."""
  job = jobs.get("len96")
  if job is None:  # the optional manifest is separate; synthesise the same expectation
    job = {"adv_mode": "pos_norm", "lam": 0.95, "log_adv_pos": True, "center": "none", "eta": 0.0,
           "num_steps_per_env": 96, "num_envs": 512}
  res = W.check(_cfg(job), [{"iter": 0, "adv_pos_mean": [0.0] * 96, "adv_mode": "pos_norm", "lam": 0.95}], "len96")
  assert not [r for r in res if r[1] == "FAIL"], res
  res = W.check(_cfg(job), [{"iter": 0, "adv_pos_mean": [0.0] * 24, "adv_mode": "pos_norm", "lam": 0.95}], "len96")
  assert any(r[1] == "FAIL" for r in res), res
