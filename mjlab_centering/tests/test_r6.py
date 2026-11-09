"""REPORT_006 manifests (d126-d130): G1 probe and claim set, SSP length-scale check and Stage B pilot, slope rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
import jobs_r4  # noqa: E402
import g1_slope  # noqa: E402


def _diff(a, b):
  return {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - {"upload_ckpt"}  # d151: claim blocks upload checkpoints


def test_probe_is_six_search_seed_jobs_on_the_g1_cell_with_25_iteration_evals():
  jobs = jobs_r4.g1_probe()
  jobs_r4.check(jobs)
  assert len(jobs) == 6 and {j["seed"] for j in jobs} == {1000, 1001, 1002}
  assert all(j["robot"] == "g1" and j["gamma"] == 0.99 and j["offset"] == 1.0 and j["penalty"] == 5.7 for j in jobs)
  assert all(j["iters"] == 3000 and j["eval_every"] == 25 for j in jobs)
  assert {j["seed"] for j in jobs}.isdisjoint(jobs_r4.CLAIM_SEEDS)
  bias = [j for j in jobs if (j["extra"] or {}).get("critic_bias_init") == "offset"]
  cent = [j for j in jobs if j["center"] == "value"]
  assert len(bias) == 3 and len(cent) == 3 and all(j["center"] == "none" and j["eta"] == 0.0 for j in bias)


def test_claim_set_is_sixty_jobs_three_arms_on_fresh_seeds_and_timing_batch_is_six():
  jobs = jobs_r4.g1_claim(2000)
  jobs_r4.check(jobs)
  assert len(jobs) == 60 and {j["seed"] for j in jobs} == set(range(3000, 3020))
  arms = {}
  for j in jobs:
    key = j["center"] + ("+bias" if (j["extra"] or {}).get("critic_bias_init") else "")
    arms[key] = arms.get(key, 0) + 1
  assert arms == {"none": 20, "value": 20, "none+bias": 20}
  t, r = jobs_r4.g1_claim_timing(2000), jobs_r4.g1_claim_rest(2000)
  assert len(t) == 6 and len(r) == 54 and not ({j["job_id"] for j in t} & {j["job_id"] for j in r})
  assert all(j["iters"] == 3000 for j in jobs_r4.g1_claim(3000))


def test_probe_and_claim_cells_differ_from_the_d120_g1_cell_only_in_what_was_intended():
  base = {(j["seed"], j["center"]): j for j in jobs_r4.g1_20()}
  for j in jobs_r4.g1_claim(2000):
    if (j["extra"] or {}).get("critic_bias_init"):
      continue
    assert _diff(j, base[(j["seed"], j["center"])]) <= {"block", "job_id"}


def test_ssp_pilot_is_twenty_jobs_and_matches_the_raw_cells_except_for_the_encoder():
  jobs = jobs_r4.ssp_pilot(4, 0.3)
  jobs_r4.check(jobs)
  assert len(jobs) == 20 and {j["seed"] for j in jobs} == set(range(3000, 3005))
  assert {j["offset"] for j in jobs} == {0.1, 1.0} and {j["center"] for j in jobs} == {"none", "value"}
  assert all(j["gamma"] == 0.999 and j["iters"] == 2000 and j["robot"] == "go1" for j in jobs)
  raw = jobs_r4.make("x", "value", 3000, "t")
  for j in jobs:
    ref = jobs_r4.make("x", j["center"], j["seed"], "t", offset=j["offset"])
    assert _diff(j, ref) <= {"block", "job_id", "extra"}, j["job_id"]
    assert j["extra"]["encoder"] == "ssp_low" and j["extra"]["ssp_n_rotates"] == 4
  assert raw["eta"] == 0.03


def test_escalation_builder_uses_the_remaining_fifteen_seeds_only():
  jobs = jobs_r4.ssp_pilot(4, 0.3, seeds=tuple(range(3005, 3020)), offsets=(0.1,), centers=("none",), block="r6sspe")
  assert len(jobs) == 15 and {j["seed"] for j in jobs} == set(range(3005, 3020))
  pilot = {j["job_id"] for j in jobs_r4.ssp_pilot(4, 0.3)}
  assert not pilot & {j["job_id"] for j in jobs}


def test_ls_check_is_six_jobs_at_two_new_length_scales():
  jobs = jobs_r4.ssp_ls_check(4)
  jobs_r4.check(jobs)
  assert len(jobs) == 6 and {j["extra"]["ssp_length_scale"] for j in jobs} == {0.15, 0.6}
  assert {j["seed"] for j in jobs} == set(jobs_r4.SEARCH_SEEDS) and all(j["center"] == "value" and j["offset"] == 1.0 for j in jobs)


def test_first_queue_has_the_probe_first_then_the_twelve_remaining_width_jobs():
  jobs = jobs_r4.r6a()
  jobs_r4.check(jobs)
  assert len(jobs) == 18 and [j["block"] for j in jobs[:6]] == ["r6probe"] * 6
  assert min(j["priority"] for j in jobs) == 0 and all(j["priority"] == 0 for j in jobs[:6])
  assert all(j["priority"] >= 1 for j in jobs[6:])
  assert not any(j["job_id"] in jobs_r4.WIDTH_RAW_DONE for j in jobs)


# ------------------------------------------------------------------ the slope rule
def _curve(f, t_max=3000, step=25):
  it = np.arange(step, t_max + 1, step, dtype=float)
  return it, np.array([f(t) for t in it])


def test_slope_is_per_hundred_iterations_and_matches_a_known_line():
  it, y = _curve(lambda t: 0.5 + 0.0002 * t)  # 0.02 per 100 iterations
  b, se, n = g1_slope.slope_per100(it, y, 2000)
  assert abs(b - 0.02) < 1e-9 and n == 11 and se < 1e-9


def test_a_seed_that_has_not_walked_is_not_flat_even_if_its_slope_is_zero():
  it, y = _curve(lambda t: 0.30)
  assert not g1_slope.seed_state(it, y, 2000)["flat"]
  it, y = _curve(lambda t: 0.80)
  assert g1_slope.seed_state(it, y, 2000)["flat"]


def test_decision_2000_when_both_arms_flat_and_extended_when_one_arm_still_rises():
  flat = _curve(lambda t: 0.8)
  rising_then_flat = _curve(lambda t: 0.8 if t >= 2300 else 0.8 - 0.0004 * (2300 - t))  # rising before 2300
  arms = {"centered": [flat, flat, flat], "bias": [rising_then_flat] * 3}
  assert g1_slope.decide({"centered": [flat] * 3, "bias": [flat] * 3})["length"] == 2000
  d = g1_slope.decide(arms)
  assert d["length"] is not None and d["length"] > 2000 and d["first_flat_t"] >= 2300
  assert d["length"] == min(3000, int(np.ceil((d["first_flat_t"] + 250) / 250) * 250))


def test_two_of_three_flat_seeds_is_enough_and_one_is_not():
  flat = _curve(lambda t: 0.8)
  rising = _curve(lambda t: 0.3 + 0.0003 * t)
  assert g1_slope.decide({"c": [flat, flat, rising], "b": [flat] * 3})["length"] == 2000
  assert g1_slope.decide({"c": [flat, rising, rising], "b": [flat] * 3})["length"] == 3000


# ------------------------------------------------------------------ the SSP reading rules
import json  # noqa: E402

import ssp_readout  # noqa: E402


def _put(root: Path, job: str, level: float) -> None:
  d = root / job
  d.mkdir(parents=True, exist_ok=True)
  (d / "eval.jsonl").write_text("\n".join(json.dumps({"iter": i, "track_lin_frac": level}) for i in range(0, 2000, 50)))
  (d / "DONE").write_text("{}")  # fixtures stand for finished runs; the readout requires the marker


def _width_world(root: Path, raw: float, ssp: dict, rff: float = 0.4) -> None:
  for s in ssp_readout.SEARCH:
    _put(root, f"r4width_go1_raw_value_s{s}", raw)
    for n in (2, 4):
      _put(root, f"r4width_go1_ssp_low{n}x{n}_value_s{s}", ssp[n][s - 1000] if isinstance(ssp[n], list) else ssp[n])
      _put(root, f"r4width_go1_rff_low{n}x{n}_value_s{s}", rff)


def _sweep(root: Path, n: int, levels: dict) -> None:
  for s in ssp_readout.SEARCH:
    for ls, lvl in levels.items():
      _put(root, f"r6lss_go1_ssp_low{n}x{n}_ls{ls:g}_value_s{s}", lvl)


def test_width_stage_picks_the_best_width_then_asks_for_the_unconditional_sweep(tmp_path):
  _width_world(tmp_path, raw=0.88, ssp={2: 0.70, 4: 0.86})
  r = ssp_readout.width_report(tmp_path)
  assert r["best_at_ls0.3"] == "ssp4x4@0.3" and r["closes_gap_d105i"] and r["run_8x8_pilot_d105iv"] is True
  assert not r["ls_sweep"]["done"] and "length-scale sweep at 4x4" in r["verdict"]
  assert "stage_b_gate_passes" not in r
  _sweep(tmp_path, 4, {0.03: 0.31, 0.1: 0.87, 1.0: 0.60})
  r = ssp_readout.width_report(tmp_path)
  assert r["ls_sweep"]["done"] and r["best_ls"] == 0.1 and r["best_overall"] == "ssp4x4@0.1"
  assert r["stage_b_gate_passes"] and "Stage B runs on ssp4x4@0.1" in r["verdict"]


def test_sweep_is_unconditional_even_when_the_gap_to_raw_is_small(tmp_path):
  _width_world(tmp_path, raw=0.88, ssp={2: 0.70, 4: 0.87})
  r = ssp_readout.width_report(tmp_path)
  assert r["closes_gap_d105i"] is True
  assert not r["ls_sweep"]["done"] and "length-scale sweep" in r["verdict"]


def test_gate_fails_when_the_best_walked_length_scale_walks_on_fewer_than_two_of_three(tmp_path):
  _width_world(tmp_path, raw=0.88, ssp={2: 0.45, 4: 0.30})
  _sweep(tmp_path, 2, {0.03: 0.30, 0.1: 0.30, 1.0: 0.30})
  r = ssp_readout.width_report(tmp_path)
  assert r["best_overall"] == "ssp2x2@0.3" and r["best_walks"] == "0 of 3"
  assert not r["stage_b_gate_passes"] and "skip Stage B" in r["verdict"]


def test_sweep_ties_go_to_the_length_scale_nearest_the_default(tmp_path):
  _width_world(tmp_path, raw=0.88, ssp={2: 0.70, 4: 0.86})
  _sweep(tmp_path, 4, {0.03: 0.90, 0.1: 0.90, 1.0: 0.40})  # 0.3 stays at the width-stage 0.86
  r = ssp_readout.width_report(tmp_path)
  assert r["best_ls"] == 0.1  # 0.03 and 0.1 tie above 0.3; 0.1 is closer to the default in log space


def _pilot_world(root: Path, ssp: dict, raw: dict) -> None:
  for o in (0.1, 1.0):
    for c in ("none", "value"):
      for s in ssp_readout.PILOT_SEEDS:
        a, b = ssp_readout.pilot_ids(4, 0.3, o, c, s)
        _put(root, a, ssp[(o, c)][s - 3000])
        _put(root, b, raw[(o, c)][s - 3000])


def test_stage_b_null_when_ssp_tracks_raw_and_escalates_on_a_two_of_five_reliability_gap(tmp_path):
  same = {(o, c): [0.88] * 5 for o in (0.1, 1.0) for c in ("none", "value")}
  _pilot_world(tmp_path, same, same)
  assert ssp_readout.pilot_report(4, 0.3, tmp_path)["verdict"] == "pilot null"
  raw = dict(same)
  raw[(0.1, "none")] = [0.88, 0.88, 0.88, 0.3, 0.3]  # raw walks 3 of 5
  ssp = dict(same)
  ssp[(0.1, "none")] = [0.88, 0.3, 0.3, 0.3, 0.3]  # SSP walks 1 of 5: delta -2, and an interaction of +0.23 as a side effect
  _pilot_world(tmp_path, ssp, raw)
  r = ssp_readout.pilot_report(4, 0.3, tmp_path)
  assert {(e["offset"], e["center"]) for e in r["escalate"]} == {(0.1, "none"), (0.1, "value")}
  why = {e["center"]: e["why"] for e in r["escalate"]}
  assert "reliability" in why["none"] and "interaction" in why["none"] and "reliability" not in why["value"]
  # both arms lose the same seeds: no interaction, but each cell trips the reliability rule by itself
  both = dict(same)
  both[(0.1, "none")] = both[(0.1, "value")] = [0.88, 0.3, 0.3, 0.3, 0.3]
  _pilot_world(tmp_path, both, same)
  r = ssp_readout.pilot_report(4, 0.3, tmp_path)
  assert abs(r["offsets"]["0.1"]["interaction"]["mean"]) < 1e-9
  assert all("reliability" in e["why"] and "interaction" not in e["why"] for e in r["escalate"]) and len(r["escalate"]) == 2


def test_stage_b_escalates_both_arms_on_an_interaction_over_a_tenth_and_not_below_it(tmp_path):
  raw = {(o, c): [0.88] * 5 for o in (0.1, 1.0) for c in ("none", "value")}
  ssp = dict(raw)
  ssp[(1.0, "value")] = [0.76] * 5  # SSP loses 0.12 under centering only: DiD -0.12
  _pilot_world(tmp_path, ssp, raw)
  r = ssp_readout.pilot_report(4, 0.3, tmp_path)
  assert {(e["offset"], e["center"]) for e in r["escalate"]} == {(1.0, "none"), (1.0, "value")}
  ssp[(1.0, "value")] = [0.80] * 5  # DiD -0.08: below the 0.10 rule
  _pilot_world(tmp_path, ssp, raw)
  assert ssp_readout.pilot_report(4, 0.3, tmp_path)["verdict"] == "pilot null"


# ------------------------------------------------------------------ H2 reading rules
import h2_report  # noqa: E402


def _summary(scores, ls=None):
  return {"final_eval": {"n_seeds": len(scores), "per_seed_scores": list(scores), "score_mean": float(np.mean(scores)),
                         "score_std": float(np.std(scores)), "score_min": float(np.min(scores)),
                         "score_max": float(np.max(scores)), "share_solved_lastk": float(np.mean(np.add(scores, 0) >= 475))},
          "winner_search_params": {"length_scale": ls}, "filter_winner": {"mean": float(np.mean(scores))}}


def test_h2_r1_when_ssp_matches_tile_and_r2_when_it_clearly_trails():
  tile = [494, 496, 499, 500, 498, 495, 497, 493, 499, 500, 496, 494, 497, 498, 495, 499, 496, 493, 498, 500]
  near = [t - 4 for t in tile]
  r = h2_report.read({"ssp_h2": _summary(near), "tile_h2": _summary(tile)})
  assert "R1 supported" in r["rules"]["R1"] and r["vs_tile"]["lo"] > -r["vs_tile"]["margin"]
  far = [t - 60 for t in tile]
  r = h2_report.read({"ssp_h2": _summary(far), "tile_h2": _summary(tile)})
  assert "R2 supported" in r["rules"]["R1"] and r["vs_tile"]["hi"] < -r["vs_tile"]["margin"]


def test_h2_r3_separates_the_wider_range_from_normalisation():
  tile = [490 + i % 7 for i in range(20)]
  norm = [t - 5 for t in tile]
  r = h2_report.read({"ssp_h2": _summary(norm, 0.04), "ssp_h2raw": _summary([t - 8 for t in tile], 0.05),
                      "tile_h2": _summary(tile)})
  assert "wider length-scale range" in r["rules"]["R3"]
  r = h2_report.read({"ssp_h2": _summary(norm, 0.04), "ssp_h2raw": _summary([t - 80 for t in tile], 0.3),
                      "tile_h2": _summary(tile)})
  assert "normalisation is what fixed it" in r["rules"]["R3"]


def test_h2_reports_incomplete_instead_of_guessing():
  assert "incomplete" in h2_report.read({"ssp_h2": _summary([1, 2])})["verdict"]


def test_stage_b_says_incomplete_before_any_run_instead_of_pilot_null(tmp_path):
  r = ssp_readout.pilot_report(4, 0.3, tmp_path)
  assert r["verdict"].startswith("incomplete") and not r["escalate"]


def test_probe_mode_names_the_six_jobs_the_manifest_created():
  import jobs_r4
  ids = {j["job_id"] for j in jobs_r4.g1_probe()}
  assert ids == {"r6probe_g1_g0.99_o1_value_s1000", "r6probe_g1_g0.99_o1_value_s1001",
                 "r6probe_g1_g0.99_o1_value_s1002", "r6probe_g1_g0.99_o1_bias_none_s1000",
                 "r6probe_g1_g0.99_o1_bias_none_s1001", "r6probe_g1_g0.99_o1_bias_none_s1002"}


def test_claim_queue_runs_the_six_timing_jobs_first_then_the_rest():
  q = jobs_r4.g1_queue(2500)
  jobs_r4.check(q)
  assert len(q) == 60 and [j["priority"] for j in q[:6]] == [0] * 6 and all(j["priority"] == 1 for j in q[6:])
  assert {j["seed"] for j in q[:6]} == {3000, 3001} and all(j["iters"] == 2500 for j in q)


def test_h2_guard_estimates_spend_from_rate_and_uptime_not_the_billing_label():
  import h2_guard
  ps = [{"name": "dqn-v2-ssp-h2-none-linear", "costPerHr": 0.06, "runtime": {"uptimeInSeconds": 3600}},
        {"name": "dqn-v2-tile-h2-none-linear", "costPerHr": 0.06, "uptimeSeconds": 7200}]
  assert abs(h2_guard.h2_estimate(ps) - 0.18) < 1e-9
  assert h2_guard.h2_estimate([]) == 0.0


def test_guard_spend_accumulates_across_container_restarts(tmp_path, monkeypatch):
  import h2_guard
  monkeypatch.setattr(h2_guard, "STATE", tmp_path / "state.json")
  p = [{"id": "x", "costPerHr": 0.06, "runtime": {"uptimeInSeconds": 1800}}]
  assert abs(h2_guard._accumulate(p)[0] - 0.03) < 1e-9
  p[0]["runtime"]["uptimeInSeconds"] = 3600  # half an hour later
  assert abs(h2_guard._accumulate(p)[0] - 0.06) < 1e-9
  p[0]["runtime"]["uptimeInSeconds"] = 60  # container restarted: uptime resets, spend must not
  assert abs(h2_guard._accumulate(p)[0] - 0.06) < 1e-9
  assert abs(h2_guard._accumulate(p)[0] - 0.06) < 1e-9


# ------------------------------------------------------------------ G1 claim-set readout
import g1_claim_report  # noqa: E402


def _arm(n, track, j=0.09, walks=None):
  walks = track if walks is None else walks
  return {3000 + i: {"track": track[i], "J": j, "walks": track[i] >= 0.5, "done": True} for i in range(n)}


def test_claim_readout_pairs_by_seed_and_flags_equivalence():
  none = _arm(4, [0.30, 0.31, 0.30, 0.88])
  value = _arm(4, [0.80, 0.82, 0.81, 0.84], j=0.0995)
  bias = _arm(4, [0.805, 0.815, 0.812, 0.845], j=0.0996)
  out = g1_claim_report.summarise({"none": none, "value": value, "bias": bias})
  assert out["arms"]["none"]["walks"] == 1 and out["arms"]["value"]["walks"] == 4
  c = out["contrasts"]["value - none"]
  assert c["n_pairs"] == 4 and c["d_walks"] == 3 and c["dtrack"] > 0.3
  assert out["bias_vs_value_equivalent_d117"] is True


def test_claim_readout_drops_unpaired_seeds_and_needs_two_pairs():
  none = _arm(4, [0.30, 0.31, 0.30, 0.88])
  value = {k: v for k, v in _arm(4, [0.80, 0.82, 0.81, 0.84]).items() if k != 3003}
  out = g1_claim_report.summarise({"none": none, "value": value})
  assert out["contrasts"]["value - none"]["n_pairs"] == 3
  out1 = g1_claim_report.summarise({"none": {3000: none[3000]}, "value": {3000: value[3000]}})
  assert "value - none" not in out1["contrasts"]


def test_job_claims_expire_so_a_dead_pod_cannot_wedge_the_queue():
  import podrun
  now = 1_000_000.0
  assert podrun.claim_is_free(None, now)
  assert not podrun.claim_is_free(now - 60, now)          # another pod claimed a minute ago
  assert not podrun.claim_is_free(now - 3 * 3600, now)    # still inside the 4 h lease
  assert podrun.claim_is_free(now - 4 * 3600 - 1, now)    # lease expired: safe to take over


def test_stage_b_launcher_parses_the_chosen_config():
  import ssp_pilot_launch
  assert ssp_pilot_launch.parse_best("ssp4x4@0.1") == (4, 0.1)
  assert ssp_pilot_launch.parse_best("ssp2x2@0.03") == (2, 0.03)
  assert ssp_pilot_launch.parse_best("ssp8x8@1") == (8, 1.0)


def test_claim_readout_ignores_runs_that_have_not_finished(tmp_path, monkeypatch):
  import g1_claim_report as g

  class _C:
    def __init__(self, finished): self.finished = finished

    def get_object(self, Bucket, Key):
      job = Key.split("/")[-2]
      kind = Key.split("/")[-1]
      if kind == "final.json" and job not in self.finished:
        raise KeyError("NoSuchKey")
      body = "".join(json.dumps({"iter": i, "track_lin_frac": 0.8, "J_task": 0.1}) + chr(10) for i in range(0, 500, 50))
      return {"Body": type("B", (), {"read": staticmethod(lambda: body.encode())})()}

  class _R:
    @staticmethod
    def client(): return _C({"r6g1_g1_g0.99_o1_value_s3000"})

    @staticmethod
    def bucket(): return "b"

    @staticmethod
    def key(p): return p

  import mjlab_centering.r2 as real
  monkeypatch.setattr(real, "client", _R.client)
  monkeypatch.setattr(real, "bucket", _R.bucket)
  monkeypatch.setattr(real, "key", _R.key)
  out = g.read_arm("camp", "r6g1_g1_g0.99_o1_value_s{}")
  assert list(out) == [3000]


def test_width_queue_carries_the_sweep_for_both_widths_in_one_campaign():
  jobs = jobs_r4.ssp_width_queue()
  jobs_r4.check(jobs)
  assert len(jobs) == 30
  blocks = {j["block"] for j in jobs}
  assert blocks == {"r4width", "r6lss"}
  cells = {j["job_id"] for j in jobs}
  assert not cells & set(jobs_r4.WIDTH_RAW_DONE)          # the three completed raw cells are reused
  assert sum(1 for j in jobs if j["block"] == "r6lss") == 18
  assert {j["extra"]["ssp_n_rotates"] for j in jobs if j["block"] == "r6lss"} == {2, 4}
  assert min(j["priority"] for j in jobs) == 1 and max(j["priority"] for j in jobs) == 2


def test_time_to_threshold_interpolates_between_evals_and_reports_never():
  import g1_claim_report as g
  rows = [{"iter": 0, "track_lin_frac": 0.2}, {"iter": 100, "track_lin_frac": 0.4},
          {"iter": 200, "track_lin_frac": 0.6}, {"iter": 300, "track_lin_frac": 0.95}]
  # rows carry completed iterations (iter+1), so the 0.4 -> 0.6 step spans 101..201
  assert abs(g.crossing(rows, 0.5) - 151.0) < 1e-6
  assert abs(g.crossing(rows, 0.9) - (201 + (0.3 / 0.35) * 100)) < 1e-6
  assert g.crossing(rows, 0.99) is None      # never reached


def test_speed_summary_counts_seeds_that_reached_the_threshold():
  rec = {1: {"t50": 100.0, "t90": None, "track": .8, "J": .1, "walks": True},
         2: {"t50": 300.0, "t90": None, "track": .8, "J": .1, "walks": True}}
  s = g1_claim_report._speed(rec, "t50")
  assert s["reached"] == 2 and s["median"] == 200.0 and s["of"] == 2
  assert g1_claim_report._speed(rec, "t90")["reached"] == 0


def test_sweep_selector_uses_the_winning_width_even_when_the_other_width_sweeps_higher(tmp_path):
  # 2x2 wins the width stage, so its sweep decides the length scale even though 4x4's sweep looks better
  _width_world(tmp_path, raw=0.88, ssp={2: 0.90, 4: 0.86})
  _sweep(tmp_path, 2, {0.03: 0.91, 0.1: 0.60, 1.0: 0.40})
  _sweep(tmp_path, 4, {0.03: 0.50, 0.1: 0.99, 1.0: 0.50})
  r = ssp_readout.width_report(tmp_path)
  assert r["best_at_ls0.3"] == "ssp2x2@0.3"
  assert r["best_overall"] == "ssp2x2@0.03" and r["best_ls"] == 0.03
  assert r["stage_b_gate_passes"] and "ssp2x2@0.03" in r["verdict"]


def test_pod_preflight_rejects_hosts_whose_driver_cannot_run_torch_cu130():
  import orchestrator
  assert orchestrator.cuda_ok("13.0") and orchestrator.cuda_ok("13.7") and orchestrator.cuda_ok("14.1")
  assert not orchestrator.cuda_ok("12.8") and not orchestrator.cuda_ok("") and not orchestrator.cuda_ok("n/a")
  b = orchestrator.BOOTSTRAP
  assert "CUDA Version" in b and "__CUDA_OK__" not in b
  assert "TORCH_CHANNEL" in b and "torch==2.14.1" in b and "torch==2.11.0" in b
  assert "--override /tmp/torch_override.txt" in b
  assert b.index("torch==2.14.1") < b.index("uv install (override")


def test_report_table_renders_missing_and_present_cells(tmp_path):
  import ssp_report_table as rt
  _width_world(tmp_path, raw=0.88, ssp={2: 0.70, 4: 0.86})
  _sweep(tmp_path, 4, {0.03: 0.31, 0.1: 0.87, 1.0: 0.60})
  rep = ssp_readout.width_report(tmp_path)
  table = rt.width_table(rep)
  assert table.startswith("| cell |") and "| raw | 3/3 | 0.880 | 3/3 |" in table
  assert "| ssp4x4@0.1 | 3/3 | 0.870 | 3/3 |" in table
  lines = "\n".join(rt.summary_lines(rep))
  assert "best overall: **ssp4x4@0.1**" in lines and "PASSES" in lines


def test_stage_b_launcher_end_to_end_from_width_and_sweep_results(tmp_path):
  import ssp_pilot_launch as pl
  root = tmp_path / "runs"
  root.mkdir()
  _width_world(root, raw=0.88, ssp={2: 0.70, 4: 0.86})
  _sweep(root, 4, {0.03: 0.31, 0.1: 0.87, 1.0: 0.60})
  out = tmp_path / "man"; out.mkdir()
  b = pl.build(root, out)
  assert b["best"] == "ssp4x4@0.1" and b["jobs"] == 20 and b["iters"] == 40000
  man = json.loads((out / f"{b['campaign']}.json").read_text())
  assert man["campaign"] == b["campaign"] and len(man["jobs"]) == 20
  assert {j["extra"]["ssp_length_scale"] for j in man["jobs"]} == {0.1}
  assert {j["extra"]["ssp_n_rotates"] for j in man["jobs"]} == {4}


def test_stage_b_launcher_refuses_when_the_gate_fails(tmp_path):
  import ssp_pilot_launch as pl
  root = tmp_path / "runs"; root.mkdir()
  _width_world(root, raw=0.88, ssp={2: 0.45, 4: 0.30})
  _sweep(root, 2, {0.03: 0.30, 0.1: 0.30, 1.0: 0.30})
  out = tmp_path / "man"; out.mkdir()
  try:
    pl.build(root, out)
    raise AssertionError("should have refused")
  except SystemExit as e:
    assert "gate fails" in str(e) and not list(out.iterdir())


def test_readout_ignores_runs_that_have_not_finished(tmp_path):
  # an in-flight run: eval rows exist, no DONE marker
  d = tmp_path / "r4width_go1_ssp_low2x2_value_s1000"
  d.mkdir(parents=True)
  (d / "eval.jsonl").write_text("\n".join(json.dumps({"iter": i, "track_lin_frac": 0.9}) for i in range(0, 2000, 50)))
  assert ssp_readout.track("r4width_go1_ssp_low2x2_value_s1000", tmp_path) is None
  (d / "DONE").write_text("{}")
  assert ssp_readout.track("r4width_go1_ssp_low2x2_value_s1000", tmp_path) == 0.9


def test_pilot_table_renders_cells_and_interaction(tmp_path):
  import ssp_pilot_table as pt
  same = {(o, c): [0.88] * 5 for o in (0.1, 1.0) for c in ("none", "value")}
  ssp = dict(same); raw = dict(same)
  ssp[(1.0, "value")] = [0.76] * 5  # interaction -0.12 at offset 1
  _pilot_world(tmp_path, ssp, raw)
  rep = ssp_readout.pilot_report(4, 0.3, tmp_path)
  body = pt.table(rep)
  assert body.startswith("| offset |") and "| 1.0 | value |" in body
  assert "interaction (DiD) -0.120" in pt.summary(rep)


def test_round7_queue_has_the_escalation_and_the_boundary_check():
  jobs = jobs_r4.r7_queue()
  jobs_r4.check(jobs)
  assert len(jobs) == 21
  esc = [j for j in jobs if j["block"] == "r6sspe"]
  bnd = [j for j in jobs if j["block"] == "r7bnd"]
  assert len(esc) == 15 and {j["seed"] for j in esc} == set(range(3005, 3020))
  assert all(j["center"] == "none" and j["offset"] == 0.1 and j["extra"]["ssp_length_scale"] == 1.0 for j in esc)
  assert len(bnd) == 6 and {j["extra"]["ssp_length_scale"] for j in bnd} == {2.0, 4.0}
  assert {j["seed"] for j in bnd} == {1000, 1001, 1002} and all(j["center"] == "value" for j in bnd)
  assert all(j["save_every"] == 250 for j in jobs)


def test_round7_advantage_rule_needs_both_a_positive_interval_and_three_more_walks():
  import r7_readout
  assert r7_readout.resolved(0.02, 15, 12) is True
  assert r7_readout.resolved(0.02, 14, 12) is False   # only 2 more walks
  assert r7_readout.resolved(-0.01, 16, 12) is False  # interval includes zero
  assert r7_readout.resolved(0.02, 15, 12, seeds=10) is False  # not claim grade


def test_podrun_resumes_from_the_mirrored_checkpoint_and_keeps_the_curve(tmp_path, monkeypatch):
  import threading
  import podrun

  job = {"job_id": "j1", "iters": 2000}
  out = tmp_path / "j1"          # run_job writes to out_root/<job_id>
  out.mkdir()
  (out / "iter.jsonl").write_text('{"iter": 0}\n')          # rows from before the preemption
  fetched = []

  def fake_fetch(rel, dest):
    fetched.append(rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.name != "iter.jsonl":  # the iter curve already exists locally; the fetch must not clobber it in this stub
      dest.write_text("x")
    return True

  monkeypatch.setattr(podrun, "_r2_exists", lambda rel: False)
  monkeypatch.setattr(podrun, "_r2_claim", lambda rel, jid: True)
  monkeypatch.setattr(podrun, "_r2_fetch", fake_fetch)

  seen = {}

  class R:  # stand-in for subprocess.run: records the command, then certifies the run
    returncode = 0
    STDOUT = -2
    @staticmethod
    def run(cmd, **kw):
      seen["cmd"] = cmd
      (out / "final.json").write_text("{}")
      return R

  monkeypatch.setattr(podrun, "subprocess", R)
  rec = podrun.run_job(job, tmp_path, "python", {}, threading.Lock(), r2="camp")
  assert rec["status"] == "ok"
  assert any("ckpt_latest.pt" in r for r in fetched)          # the checkpoint was fetched
  assert '"resume"' in " ".join(seen["cmd"])                   # and handed to the trainer inside the job spec
  assert (out / "iter.jsonl").read_text().strip() == '{"iter": 0}'  # the old curve was NOT cleared
