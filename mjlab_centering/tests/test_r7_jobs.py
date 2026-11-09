"""Round-7 manifests (d155): the three stages, and that each arm differs from the paired cell only where intended."""

from __future__ import annotations

import json

import numpy as np
import pytest

import jobs_r4 as J4
import jobs_r7 as J7


def _paired(offset: float = 1.0, seed: int = 1000) -> dict:
  return [j for j in json.loads(J4.NIGHT1.read_text())["jobs"]
          if j["job_id"] == "w2main_go1_raw_none_g0.999_o1_s3000"][0] | {"seed": seed, "offset": offset}


def test_stage1_is_the_three_arms_plus_the_vn_diagnostic_on_search_seeds():
  jobs = J7.pilot()
  assert len(jobs) == 12  # 4 arms x 3 search seeds, one run per arm per seed
  assert {j["seed"] for j in jobs} == {1000, 1001, 1002}
  arms = {j["job_id"].split("_go1_")[1].split("_g0.999")[0] for j in jobs}
  assert arms == {"vn", "vn_noclamp", "rs", "sc"}
  assert all(j["offset"] == 1.0 for j in jobs)
  assert all(j["block"] == "r7pilot" for j in jobs)


def test_stage2_is_eighty_runs_of_vn_and_sc_and_stage3_forty_of_rs_on_claim_seeds():
  a = J7.vnsc()
  assert len(a) == 80 and {j["seed"] for j in a} == set(range(3000, 3020))
  assert {j["offset"] for j in a} == {0.1, 1.0}
  assert {j["job_id"].split("_go1_")[1].split("_g0.999")[0] for j in a} == {"vn", "sc"}
  b = J7.rs()
  assert len(b) == 40 and {j["seed"] for j in b} == set(range(3000, 3020))
  assert {j["job_id"].split("_go1_")[1].split("_g0.999")[0] for j in b} == {"rs"}


def test_search_and_claim_seeds_are_never_pooled():
  assert not {j["seed"] for j in J7.pilot()} & {j["seed"] for j in J7.vnsc()}


def test_each_arm_changes_only_its_own_knob_relative_to_the_paired_cell():
  base = _paired()
  want = {
    "vn": dict(ret_norm="rlgames"),
    "vn_noclamp": dict(ret_norm="rlgames", vn_clamp=0.0),
    "rs": dict(reward_scale="ret_std"),
    "sc": dict(center="simple_raw", beta=0.03, r_bar_init="first_batch"),
  }
  jobs = {j["job_id"].split("_go1_")[1].split("_g0.999")[0]: j for j in J7.pilot() if j["seed"] == 1000}
  for arm, changed in want.items():
    diff = {k for k in set(base) | set(jobs[arm]) if base.get(k) != jobs[arm].get(k)}
    allowed = set(changed) | {"job_id", "block", "notes", "log_rew_std", "seed"}
    assert diff <= allowed, (arm, diff - allowed)
    for k, v in changed.items():
      assert jobs[arm][k] == v, (arm, k)
    assert jobs[arm]["log_rew_std"] is True  # d155: every new run logs the per-step reward std


def test_corrected_protocol_and_run_settings_are_asserted_by_check():
  J7.check(J7.pilot())
  J7.check(J7.vnsc())
  J7.check(J7.rs())
  bad = [j for j in J7.vnsc()][:1]
  bad[0] = dict(bad[0], log_rew_std=False)
  try:
    J7.check(bad)
  except AssertionError:
    pass
  else:  # pragma: no cover
    raise AssertionError("check() must reject a run with the reward-std logging off")


def _cells(off, arm_vals):
  return {(off, arm): {s: {"track": t, "J": j, "walks": j > 0.09, "n_evals": 40}
                       for s, (t, j) in enumerate(vals)} for arm, vals in arm_vals.items()}


def _with_jitter(mean: float, jmean: float, n: int = 20, sd: float = 0.05, jsd: float = 0.002):
  rng = np.random.default_rng(0)
  return [(mean + sd * z, jmean + jsd * z) for z in rng.standard_normal(n)]


def test_reading_rule_fixes_partial_worse_and_no_help():
  import sys as _s
  from pathlib import Path as _P
  _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "analysis"))
  import r7_report as RP
  none = _with_jitter(0.645, 0.082)
  value = _with_jitter(0.884, 0.0999)
  # equivalent to value centering: within the d117 margins on all three readings
  assert RP.verdict(_cells(0.1, {"none": none, "value": value, "vn": _with_jitter(0.884, 0.0999)}),
                    0.1, "vn")["reading"].startswith("fixes")
  # beats uncentered but sits below value: partial
  assert RP.verdict(_cells(0.1, {"none": none, "value": value, "vn": _with_jitter(0.790, 0.0940)}),
                    0.1, "vn")["reading"].startswith("partial")
  # no gain over uncentered: doesn't help (the interval spans 0)
  assert RP.verdict(_cells(1.0, {"none": none, "value": value, "sc": _with_jitter(0.645, 0.082)}),
                    1.0, "sc")["reading"].startswith("doesn't help")
  # significantly worse than uncentered is reported as such, not folded into "partial"
  assert "is worse" in RP.verdict(_cells(1.0, {"none": none, "value": value, "rs": _with_jitter(0.500, 0.070)}),
                                  1.0, "rs")["reading"]


def test_gate_arithmetic_matches_the_pre_registered_rule():
  import sys as _s
  from pathlib import Path as _P
  _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "analysis"))
  import r7_gate as G
  assert G.rate_per_1000it(2.4, 24000) == pytest.approx(0.1)
  assert G.gate(2.0, 10.0)["gate_need"] == pytest.approx(15.0)
  assert G.gate(2.0, 10.0)["pass"]            # 2 + 1.3*10 = 15 <= 18
  assert not G.gate(6.0, 10.0)["pass"]        # 6 + 13 = 19 > 18


def test_equivalence_parts_report_which_d117_margin_failed():
  import sys as _s
  from pathlib import Path as _P
  _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "analysis"))
  import r7_report as RP
  none = _with_jitter(0.645, 0.082)
  value = _with_jitter(0.884, 0.0999)
  ok = RP.verdict(_cells(0.1, {"none": none, "value": value, "vn": _with_jitter(0.884, 0.0999)}), 0.1, "vn")
  assert all(ok["equiv_parts"].values())
  # walks identical and tracking equivalent, but J_task 4% below: only the J margin fails
  bad = RP.verdict(_cells(0.1, {"none": none, "value": value, "vn": _with_jitter(0.884, 0.0960)}), 0.1, "vn")
  assert bad["equiv_parts"]["walks"] and bad["equiv_parts"]["track"] and not bad["equiv_parts"]["J"]


def test_publish_queue_payload_carries_the_jobs_and_refuses_an_empty_queue():
  import sys as _s
  from pathlib import Path as _P
  _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "analysis"))
  import r7_publish_queue as Q
  man = {"campaign": "r7vnsc-x", "jobs": [{"job_id": "a"}, {"job_id": "b"}]}
  p = Q.payload(man, "gate passed")
  assert len(p["jobs"]) == 2 and p["campaign"] == "r7vnsc-x" and p["note"] == "gate passed" and p["published"] > 0
  with pytest.raises(AssertionError):
    Q.payload({"jobs": []})


def test_publishing_a_queue_also_publishes_the_target_manifest(monkeypatch):
  """The runs are written under the target campaign; the readouts find job ids in <campaign>/manifest.json (d174)."""
  import sys as _s
  from pathlib import Path as _P
  _s.path.insert(0, str(_P(__file__).resolve().parents[1] / "analysis"))
  import r7_publish_queue as Q
  wrote = {}
  monkeypatch.setattr(Q.r2, "put_json", lambda key, obj: wrote.__setitem__(key, obj))
  man = {"campaign": "r7vnsc-x", "jobs": [{"job_id": "a"}]}
  Q.publish("r7pilot-y", _P(__file__).with_name("_nonexistent_") if False else _write_tmp(man))
  assert "r7pilot-y/queue.json" in wrote and "r7vnsc-x/manifest.json" in wrote
  assert wrote["r7vnsc-x/manifest.json"]["jobs"] == man["jobs"]


def _write_tmp(man):
  import json, tempfile
  from pathlib import Path
  p = Path(tempfile.mkdtemp()) / "m.json"
  p.write_text(json.dumps(man))
  return p
