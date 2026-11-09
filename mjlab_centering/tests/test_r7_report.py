"""The round-7 readout engine (analysis/r7_report.py): pairing, the mechanical reading, and the figure path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import r7_report as RP  # noqa: E402


def _jit(mean: float, n: int = 20, sd: float = 0.02, seed: int = 0) -> list[float]:
  return list(mean + sd * np.random.default_rng(seed).standard_normal(n))


def test_cells_are_paired_by_seed_and_the_reference_cells_come_from_their_own_ids(monkeypatch):
  jobs = {"r7vnsc_go1_vn_g0.999_o0.1_s3000": {}, "r7vnsc_go1_sc_g0.999_o0.1_s3001": {}}
  monkeypatch.setattr(RP, "new_jobs", lambda camps: jobs)
  calls = {}

  def fake_cell(jid):
    calls[jid] = True
    if jid.startswith("r7"):
      return {"job_id": jid, "track": 0.88, "J": 0.0999, "walks": True, "n_evals": 40}
    return {"job_id": jid, "track": 0.66, "J": 0.0821, "walks": False, "n_evals": 40}

  monkeypatch.setattr(RP, "cell", fake_cell)
  cells = RP.collect(["x"])
  assert set(cells[(0.1, "vn")]) == {3000} and set(cells[(0.1, "sc")]) == {3001}
  # the reference cells are looked up on the same seeds, from the existing campaigns' job ids
  assert "r4sweep_go1_g0.999_o0.1_none_s3000" in calls and "r4sweep_go1_g0.999_o0.1_value_s3001" in calls
  assert RP.verdict(cells, 0.1, "vn")["pilot"] is True  # one pair is a pilot, never a claim


def test_summary_counts_walks_and_means_last5():
  cells = {1: {"track": 0.9, "J": 0.1, "walks": True, "n_evals": 40},
           2: {"track": 0.3, "J": 0.05, "walks": False, "n_evals": 40}}
  s = RP.summary(cells)
  assert s["n"] == 2 and s["walks"] == 1 and s["track"] == pytest.approx(0.6)


def test_curves_intersect_eval_iterations_across_seeds(monkeypatch):
  rows = {0: [{"iter": 0, "track_lin_frac": 0.3}, {"iter": 50, "track_lin_frac": 0.5}],
          1: [{"iter": 50, "track_lin_frac": 0.7}, {"iter": 100, "track_lin_frac": 0.9}]}
  order = iter([rows[0], rows[1]])
  monkeypatch.setattr(RP, "evals", lambda jid: next(order))
  its, Y = RP.curves({"r7pilot_go1_vn_g0.999_o1_s1000": {}, "r7pilot_go1_vn_g0.999_o1_s1001": {}}, "vn", 1.0)
  assert its == [50] and Y.shape == (2, 1) and Y[1, 0] == pytest.approx(0.7)


def test_figure_writes_png_and_pdf(tmp_path, monkeypatch):
  monkeypatch.setattr(RP, "OUT", tmp_path)
  events = {0: [{"iter": 0, "track_lin_frac": 0.2}, {"iter": 50, "track_lin_frac": 0.4}, {"iter": 100, "track_lin_frac": 0.5}],
            1: [{"iter": 0, "track_lin_frac": 0.2}, {"iter": 50, "track_lin_frac": 0.45}, {"iter": 100, "track_lin_frac": 0.55}]}
  monkeypatch.setattr(RP, "evals", lambda jid: events[0] if jid.endswith("s3000") else events[1])
  jobs = {f"r7vnsc_go1_{a}_g0.999_o1_s{s}": {} for a in ("none", "vn") for s in (3000, 3001)}
  cells = {(1.0, a): {s: {"track": 0.5, "J": 0.09, "walks": True, "n_evals": 3} for s in (3000, 3001)}
           for a in ("none", "value", "vn")}
  RP.figure(cells, jobs, [1.0], "t")
  assert (tmp_path / "t.png").exists() and (tmp_path / "t.pdf").exists()


def test_table_prints_one_row_per_arm_and_offset():
  cells = {(0.1, "none"): {s: {"track": 0.66, "J": 0.0821, "walks": False, "n_evals": 40} for s in range(20)},
           (0.1, "value"): {s: {"track": 0.885, "J": 0.0999, "walks": True, "n_evals": 40} for s in range(20)},
           (0.1, "vn"): {s: {"track": 0.886, "J": 0.0998, "walks": True, "n_evals": 40} for s in range(20)}}
  lines = RP.table(cells, [0.1], ["none", "value", "vn"], "t")
  assert len([l for l in lines if l.startswith("| ") and "@ c=" in l]) == 3
  assert "fixes the offset failure" in "\n".join(lines)


def test_diagnostics_reports_the_last_iteration_and_sc_r_bar_minus_task(tmp_path, monkeypatch):
  d = tmp_path / "r7pilot_go1_sc_g0.999_o1_s1000"
  d.mkdir()
  (d / "iter.jsonl").write_text("\n".join(json.dumps(r) for r in [
    {"iter": 0, "r_bar": 0.0, "r_task_rate": 0.0}, {"iter": 1, "r_bar": 1.0, "r_task_rate": 0.0},
    {"iter": 0, "r_bar": 0.0, "r_task_rate": 0.0}, {"iter": 1, "r_bar": 1.02, "r_task_rate": 0.02, "ev": 0.5}]))
  monkeypatch.setattr(RP, "MERGED", tmp_path)
  out = RP.diagnostics("r7pilot_go1_sc_g0.999_o1_s1000", "sc")
  assert out["r_bar"] == pytest.approx(1.02)          # the newest attempt wins (the second iter==0 restarts the log)
  assert out["r_bar_minus_task"] == pytest.approx(1.0)  # SC tracks the raw reward, i.e. task + offset
  assert out["ev"] == pytest.approx(0.5)


def test_reference_curves_are_read_from_the_old_mirror(tmp_path, monkeypatch):
  """The c = 1 reference cells live in the REPORT_003 mirror (merged_r2), the new runs in merged_r4."""
  old = tmp_path / "results" / "merged_r2"
  new = tmp_path / "merged_r4"
  for base in (old, new):
    (base / "w2main_go1_raw_none_g0.999_o1_s3000").mkdir(parents=True)
  (new / "w2main_go1_raw_none_g0.999_o1_s3000" / "eval.jsonl").write_text("")  # empty, so the fallback is exercised
  (old / "w2main_go1_raw_none_g0.999_o1_s3000" / "eval.jsonl").write_text(
    "\n".join(json.dumps({"iter": i, "track_lin_frac": 0.3}) for i in (0, 50, 99)))
  monkeypatch.setattr(RP, "MERGED", new)
  monkeypatch.setattr(RP, "MJROOT", tmp_path)
  its, Y = RP.curves({}, "none", 1.0, seeds=[3000])
  assert its == [0, 50, 99] and Y.shape == (1, 3)


def test_builds_are_read_from_each_runs_config(tmp_path, monkeypatch):
  for jid, ver in (("a_s3000", "2.11.0+cu128"), ("b_s3001", "2.11.0+cu128"), ("c_s3002", None)):
    d = tmp_path / jid
    d.mkdir()
    if ver:
      (d / "config.json").write_text(json.dumps({"torch": ver}))
  monkeypatch.setattr(RP, "MERGED", tmp_path)
  out = RP.builds({"a_s3000": {}, "b_s3001": {}, "c_s3002": {}})
  assert out["torch 2.11.0+cu128"] == 2
  assert any(k.startswith("build unknown") for k in out)


def test_figure_survives_an_arm_with_no_curves(tmp_path, monkeypatch):
  """An arm with no run data must be skipped, not crash the figure on a stale `its`."""
  monkeypatch.setattr(RP, "OUT", tmp_path)
  rows = [{"iter": 0, "track_lin_frac": 0.2}, {"iter": 50, "track_lin_frac": 0.6}]
  monkeypatch.setattr(RP, "evals", lambda jid: rows if jid.endswith("s3000") else [])
  jobs = {"r7x_go1_vn_g0.999_o1_s3000": {}, "r7x_go1_sc_g0.999_o1_s3001": {}}
  cells = {(1.0, "none"): {}, (1.0, "value"): {},
           (1.0, "vn"): {3000: {"track": 0.6, "J": 0.1, "walks": True, "n_evals": 2}},
           (1.0, "sc"): {3001: {"track": 0.0, "J": 0.0, "walks": False, "n_evals": 0}}}
  monkeypatch.setattr(RP, "existing", lambda arm, off, seed: f"r7x_go1_{arm}_g0.999_o1_s{seed}")
  RP.figure(cells, jobs, [1.0], "t2")
  assert (tmp_path / "t2.png").exists()


def test_timing_reports_take_off_for_a_walking_run_and_nothing_for_a_flat_one(tmp_path, monkeypatch):
  d = tmp_path / "a_s3000"; d.mkdir()
  (d / "eval.jsonl").write_text("\n".join(json.dumps({"iter": i * 50, "track_lin_frac": min(0.9, i * 0.1)})
                                          for i in range(1, 21)))
  f = tmp_path / "b_s3001"; f.mkdir()
  (f / "eval.jsonl").write_text("\n".join(json.dumps({"iter": i * 50, "track_lin_frac": 0.3}) for i in range(1, 21)))
  monkeypatch.setattr(RP, "MERGED", tmp_path)
  t = RP.timing("a_s3000")
  assert t["t50"] == 250 and t["t90rel"] == pytest.approx(450) and t["final"] == pytest.approx(0.9)
  assert RP.timing("b_s3001")["t50"] == float("inf") or RP.timing("b_s3001")["t50"] != RP.timing("b_s3001")["t50"]


def test_unfinished_manifest_jobs_do_not_poison_the_paired_contrasts(monkeypatch, tmp_path):
  """The manifest lists every job; a job with no evals yet must not enter `cells` as a nan (it made every CI nan)."""
  monkeypatch.setattr(RP, "new_jobs", lambda camps: {"r7x_go1_vn_g0.999_o1_s3000": {}, "r7x_go1_vn_g0.999_o1_s3001": {}})
  monkeypatch.setattr(RP, "evals", lambda jid: [{"iter": i, "track_lin_frac": 0.9} for i in (0, 50, 99)] if jid.endswith("s3000") else [])
  monkeypatch.setattr(RP, "cell", lambda jid: ({"job_id": jid, "track": 0.9, "J": 0.1, "walks": True, "n_evals": 40}
                                               if jid.endswith("s3000") else
                                               {"job_id": jid, "track": float("nan"), "J": float("nan"),
                                                "walks": False, "n_evals": 0}))
  monkeypatch.setattr(RP, "existing", lambda arm, off, seed: f"ref_{arm}_{off}_{seed}")
  cells = RP.collect(["x"])
  assert sorted(cells[(1.0, "vn")]) == [3000]          # only the finished run is a cell
  assert RP.summary(cells[(1.0, "vn")])["n"] == 1
