"""Wave manifests: counts, unique ids, hold-out seeds, and shock branches that really descend from their mother."""

from __future__ import annotations

import collections

from mjlab_centering import waves


def test_wave1_counts_and_seeds():
  jobs = waves.wave1(400)
  by = collections.Counter(j.block for j in jobs)
  assert by == {"eta": 15, "sslen": 9, "g1eta": 9}
  assert {j.seed for j in jobs} == set(waves.SEARCH_SEEDS)
  assert len({j.job_id for j in jobs}) == len(jobs)


def test_wave2_counts_seeds_and_branch_lineage():
  jobs = waves.wave2(iters=1000, shock_at=600, eta=0.03, ssp_ls=0.3)
  by = collections.Counter(j.block for j in jobs)
  assert by == {"core": 80, "rival": 20, "shock": 60}
  assert {j.seed for j in jobs} == set(waves.CLAIM_SEEDS)
  assert not {j.seed for j in jobs} & set(waves.SEARCH_SEEDS)
  ids = [j.job_id for j in jobs]
  assert len(set(ids)) == len(ids)
  pos = {j.job_id: i for i, j in enumerate(jobs)}
  for j in jobs:
    if j.block != "shock":
      continue
    m = next(x for x in jobs if x.job_id == j.extra["after"])
    assert pos[m.job_id] < pos[j.job_id]  # mothers first
    assert (m.seed, m.center, m.gamma, m.offset, m.eta) == (j.seed, j.center, j.gamma, j.offset, j.eta)
    assert m.save_at == j.shock_at == 600
    assert m.extra.get("encoder") is None  # raw mothers only
  assert sum(1 for j in jobs if j.save_at) == 20  # only raw gamma=0.999 core runs write a checkpoint


def test_wave3_wave4_counts():
  w3 = collections.Counter(j.block for j in waves.wave3(1000, 0.03, 0.3))
  assert w3 == {"core_g1": 40, "tie_g1": 20}
  w4 = waves.wave4(1000, 0.03, 0.03, 0.3)
  assert len(w4) == 10 * 2 * 4
  assert {j.seed for j in w4} == set(waves.TOPUP_SEEDS)
  assert not set(waves.TOPUP_SEEDS) & set(waves.CLAIM_SEEDS)


def test_centered_arms_carry_eta_and_uncentered_do_not_need_it():
  for j in waves.wave2(1000, 600, 0.03, 0.3):
    if j.center == "value":
      assert j.eta == 0.03


def test_diag_width_arm_has_the_equal_width_control():
  jobs = waves.diag_width(800, 0.03, 0.3)
  assert len(jobs) == 9
  assert {j.extra.get("encoder", "raw") for j in jobs} == {"raw", "rff_low", "ssp_low"}
  assert {j.seed for j in jobs} == set(waves.SEARCH_SEEDS)


def test_centered_wave2_wave3_cells_use_the_frozen_r_bar_initialisation():
  for j in waves.wave2(800, 600, 0.03, 0.3) + waves.wave3(2500, 0.03, 0.3):
    assert j.r_bar_init == ("first_batch" if j.center == "value" else "zero"), j.job_id


def test_wave2_without_ssp_keeps_every_raw_cell_and_all_shock_branches():
  full = waves.wave2(800, 600, 0.03, 0.3)
  raw = waves.wave2(800, 600, 0.03, 0.3, include_ssp=False)
  assert len(full) == 160 and len(raw) == 120
  assert sum(1 for j in raw if j.block == "shock") == 60 == sum(1 for j in full if j.block == "shock")
  assert all(j.extra.get("encoder", "raw") == "raw" for j in raw)
  raw_ids = {j.job_id for j in raw}
  assert raw_ids <= {j.job_id for j in full}


def test_wave2_can_be_trimmed_to_one_shock_type_and_keeps_its_controls():
  jobs = waves.wave2(800, 600, 0.03, 0.3, shock_kinds=("none", "offset"))
  shr = [j for j in jobs if j.block == "shock"]
  assert len(shr) == 40 and {j.shock_kind for j in shr} == {"none", "offset"}
  assert all(any(x.job_id == j.extra["after"] for x in jobs) for j in shr)


def test_podrun_clears_stale_attempts_before_restarting(tmp_path):
  import json as _json
  import sys
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  import podrun  # the queue runner lives at the package root, not inside src/

  out = tmp_path / "w" / "job1"
  out.mkdir(parents=True)
  (out / "iter.jsonl").write_text('{"iter": 0}\n')
  (out / "eval.jsonl").write_text('{"iter": 0}\n')
  job = {"job_id": "job1", "iters": 1, "num_envs": 2, "eval_envs": 0}
  manifest = tmp_path / "m.json"
  manifest.write_text(_json.dumps({"jobs": [job]}))
  # run_job with a missing interpreter fails twice, but the stale files must already be gone
  res = podrun.run_job(job, tmp_path / "w", "/bin/false", {}, __import__("threading").Lock())
  assert res["status"] == "failed"
  assert not (out / "iter.jsonl").exists() and not (out / "eval.jsonl").exists()


def test_g1_diag_is_two_single_change_runs():
  """Decision 73: one run isolates the offset (offset 0, uncentered), one isolates centering (offset 1, value-centered)."""
  jobs = waves.g1_diag(2000)
  assert len(jobs) == 2
  a, b = jobs
  assert {j.job_id for j in jobs} == {
    "g1diag_raw_none_g0.99_o0_s1000", "g1diag_raw_value_g0.99_o1_s1000"}
  assert a.center == "none" and a.offset == 0.0
  assert b.center == "value" and b.offset == 1.0 and b.eta == 0.03 and b.r_bar_init == "first_batch"
  for j in jobs:
    assert j.robot == "g1" and j.gamma == 0.99 and j.iters == 2000 and j.seed == 1000
    assert j.episodic is False and j.shock_kind == "none" and j.resume is None and j.save_at is None
    assert j.extra.get("encoder", "raw") == "raw"


def test_g1_episodic_control_is_the_plain_env():
  (j,) = waves.g1_episodic(2000)
  assert j.job_id == "g1diag_episodic_none_g0.99_o0_s1000"
  assert j.episodic is True and j.center == "none" and j.offset == 0.0
  assert j.robot == "g1" and j.iters == 2000 and j.shock_kind == "none"
  assert len(waves.g1_diag(2000) + waves.g1_episodic(2000)) == 3


def test_g1_timeouts_no_falls_arm_keeps_episodes():
  (j,) = waves.g1_timeouts_no_falls(2000)
  assert j.job_id == "g1diag_timeouts_nofalls_g0.99_s1000"
  assert j.episodic is True and j.falls_reset is False and j.center == "none" and j.offset == 0.0


def test_pilot_wave_is_the_decision_77_protocol():
  jobs = waves.pilot_timeouts(2000) + waves.g1_wrapper_replication(2000)
  assert len(jobs) == 24
  ids = [j.job_id for j in jobs]
  assert len(set(ids)) == len(ids)
  pilots = [j for j in jobs if j.block == "pilot"]
  assert sum(j.robot == "go1" for j in pilots) == 8 and sum(j.robot == "g1" for j in pilots) == 12
  for j in pilots:  # new protocol by default: time-outs kept, falls non-terminal, no fall penalty
    assert j.time_outs is True and j.falls_reset is True and j.penalty == 0.0
    assert j.gamma == 0.999 and j.episodic is False
    assert (j.eta == 0.03 and j.r_bar_init == "first_batch") if j.center == "value" else (j.eta == 0.0)
  reps = [j for j in jobs if j.block == "g1rep"]
  assert sorted({j.job_id for j in reps}) == [
    "g1rep_legacy_none_o0_s1001", "g1rep_legacy_none_o0_s1002",
    "g1rep_timeouts_none_o0_s1001", "g1rep_timeouts_none_o0_s1002"]
  legacy = [j for j in reps if "legacy" in j.job_id]
  assert all(j.time_outs is False and j.falls_reset is True and j.penalty == 10.0 for j in legacy)


def test_g1_gamma99_gap_cells():
  jobs = waves.g1_gamma99_gap(2000)
  assert [j.job_id for j in jobs] == [
    "pilot99_g1_raw_none_o0_s1000", "pilot99_g1_raw_none_o0_s1003", "pilot99_g1_raw_none_o0_s1004",
    "pilot99_g1_raw_value_o0_s1000", "pilot99_g1_raw_value_o0_s1003"]
  assert all(j.gamma == 0.99 and j.offset == 0.0 and j.penalty == 0.0 for j in jobs)
  assert all(j.time_outs and j.falls_reset for j in jobs)
  assert [j.center for j in jobs] == ["none"] * 3 + ["value"] * 2


def test_g1_episodic_replication_cells():
  jobs = waves.g1_episodic_rep(2000)
  assert [j.job_id for j in jobs] == ["g1rep_episodic_none_o0_s1003", "g1rep_episodic_none_o0_s1004"]
  assert all(j.episodic and j.center == "none" and j.offset == 0.0 and j.gamma == 0.99 for j in jobs)


def test_g1_long_probe_pair():
  jobs = waves.g1_long_probe()
  assert [j.job_id for j in jobs] == [
    "g1long_timeouts_none_o0_s1001_i4000", "g1long_legacy_none_o0_s1001_i4000"]
  assert all(j.iters == 4000 and j.priority == 10 and j.seed == 1001 for j in jobs)
  assert jobs[0].time_outs is True and jobs[1].time_outs is False


def test_worker_clears_a_stale_attempt(tmp_path):
  """A re-claimed job restarts at iteration 0; the mirror must not interleave two attempts (decision 80)."""
  import json as _json
  import sys
  from pathlib import Path

  from mjlab_centering import worker  # pod-side queue module (the podrun script at the root is a different file)

  d = tmp_path / "job1"
  d.mkdir()
  (d / "iter.jsonl").write_text('{"iter": 0}\n')
  (d / "eval.jsonl").write_text('{"iter": 0}\n')
  (d / "final.json").write_text("{}")
  (d / "ckpt_it600.pt").write_bytes(b"x")
  (d / "keep.txt").write_text("keep me")
  removed = worker.clear_stale(str(d))
  assert set(removed) == {"iter.jsonl", "eval.jsonl", "final.json", "ckpt_it600.pt"}
  assert (d / "keep.txt").exists()
  assert worker.clear_stale(str(tmp_path / "missing")) == []


def test_corrected_pilots_protocol():
  jobs = waves.corrected_pilots(2000)
  assert len(jobs) == 11
  g1 = [j for j in jobs if j.robot == "g1"]
  go1 = [j for j in jobs if j.robot == "go1"]
  assert len(g1) == 3 and len(go1) == 8
  assert all(j.time_outs and j.falls_reset and j.penalty == 0.0 and j.gamma == 0.99 for j in g1)
  assert all(j.time_outs and j.falls_reset and j.penalty == 10.0 and j.gamma == 0.999 for j in go1)
  assert [j.priority for j in g1] == [0, 0, 0]


def test_g1_penalty_check_is_the_conditional_run():
    jobs = waves.g1_penalty_check(2000)
    assert [j.job_id for j in jobs] == ["fix_g1_pen10_none_o0_s1000"]
    (j,) = jobs
    assert j.penalty == 10.0 and j.time_outs and j.falls_reset and j.gamma == 0.99


def test_g1_centering_pilots_shape():
    jobs = waves.g1_centering_pilots(2000, penalty=5.7)
    assert len(jobs) == 12
    assert {(j.gamma, j.offset) for j in jobs} == {(0.99, 1.0), (0.999, 0.0)}
    assert all(j.penalty == 5.7 and j.robot == "g1" and j.time_outs and j.falls_reset for j in jobs)
    assert sum(j.center == "value" for j in jobs) == 6 and sum(j.center == "none" for j in jobs) == 6


def test_wave2_shape_and_priorities():
    jobs = waves.overnight(2000, g1_penalty=5.7)
    assert len(jobs) == 40 + 12 + 20 + 6 + 10
    by = collections.Counter(j.block for j in jobs)
    assert by["w2main"] == 40 and by["w2g1c"] == 6 and by["w2g1c999"] == 6
    assert by["w2off0"] == 20 and by["w2off0g99"] == 6 and by["w2shock"] == 10
    shock = [j for j in jobs if j.block == "w2shock"]
    assert {j.shock_kind for j in shock} == {"offset", "action_rate"}
    assert all(j.shock_at == 1200 and j.center == "value" for j in shock)
    assert all(j.time_outs and j.falls_reset for j in jobs)
    assert {j.seed for j in jobs if j.block == "w2main"} == set(range(3000, 3020))
    assert all(j.penalty == 5.7 for j in jobs if j.robot == "g1")
    assert all(j.penalty == 10.0 for j in jobs if j.robot == "go1")


def test_clipoff_pilot_matches_the_main_effect_cell():
    jobs = waves.clipoff_pilot()
    assert [j.job_id for j in jobs] == [f"clip1_go1_raw_none_clipoff_g0.999_o1_s{s}" for s in (3000, 3001, 3002)]
    assert all(j.value_clip is False and j.center == "none" and j.gamma == 0.999 and j.offset == 1.0 for j in jobs)
    assert all(j.penalty == 10.0 and j.iters == 2000 and j.time_outs and j.falls_reset for j in jobs)
