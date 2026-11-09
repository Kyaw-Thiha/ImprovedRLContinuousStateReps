"""Analysis arithmetic on hand-built runs with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import analyze_waves as aw  # noqa: E402


def _run(block, seed, center, rep="raw", gamma=0.999, robot="go1", curve=None, kind="none", extra=None, shock_at=0):
  curve = curve if curve is not None else [0.1] * 8
  ev = [{"iter": 50 * i + 49, "J_task": v, "track_lin_frac": v * 9} for i, v in enumerate(curve)]
  ex = dict(extra or {})
  if rep != "raw":
    ex["encoder"] = rep
  job = {"job_id": f"{block}_{robot}_{rep}_{center}_{gamma}_{kind}_{seed}", "block": block, "seed": seed, "center": center,
         "gamma": gamma, "robot": robot, "shock_kind": kind, "extra": ex, "eta": 0.03, "r_bar_init": "zero",
         "shock_at": shock_at}
  return job["job_id"], {"job": job, "eval": ev, "iter": [], "done": True}


def test_interaction_is_zero_when_effects_are_additive_and_detects_a_real_one():
  rng = np.random.default_rng(0)
  runs = {}
  for s in range(2000, 2010):
    noise = rng.normal(0, 0.001)
    for rep, c, base in (("raw", "none", 0.10), ("raw", "value", 0.11), ("ssp_low", "none", 0.095), ("ssp_low", "value", 0.105)):
      k, r = _run("core", s, c, rep, curve=[base + noise] * 8)
      runs[k] = r
  out = aw.core_contrasts(runs, "go1", 0.999, 0.03)
  assert abs(out["interaction"]["mean"]) < 0.002
  assert out["interaction"]["equivalent"]
  assert abs(out["centering_main_raw"]["mean"] - 0.01) < 0.002 and not out["centering_main_raw"]["equivalent"]
  runs2 = {}
  for s in range(2000, 2010):
    for rep, c, base in (("raw", "none", 0.10), ("raw", "value", 0.10), ("ssp_low", "none", 0.10), ("ssp_low", "value", 0.12)):
      k, r = _run("core", s, c, rep, curve=[base + 0.0003 * (s % 3)] * 8)
      runs2[k] = r
  inter = aw.core_contrasts(runs2, "go1", 0.999, 0.03)["interaction"]
  assert abs(inter["mean"] - 0.02) < 1e-6 and not inter["equivalent"]


def test_time_to_reads_the_curve():
  _, r = _run("core", 1, "none", curve=[0.0, 0.0, 0.02, 0.05, 0.1, 0.1, 0.1, 0.1])
  assert aw.time_to(r, 0.5) == 199  # smoothed curve first reaches 50% of final at the 4th eval


def test_shock_recovery_and_branch_validation():
  runs = {}
  for s in (1, 2, 3):
    pre = [0.10] * 12  # mother evals at it 49..599; the shock lands at 600
    mk, mr = _run("core", s, "value", curve=pre + [0.10] * 4)
    mr["job"]["job_id"] = f"mother_{s}"
    runs[f"mother_{s}"] = mr
    for kind, post in (("none", [0.10] * 4), ("offset", [0.06, 0.08, 0.096, 0.10])):
      # a branch is RESUMED at 600, so its own evals start after the shock: the baseline must come from the mother
      k, r = _run("shock", s, "value", curve=post, kind=kind, extra={"after": f"mother_{s}"}, shock_at=600)
      r["eval"] = [dict(e, iter=649 + 50 * i) for i, e in enumerate(r["eval"])]
      runs[k] = r
  sm = aw.shock_metrics(runs, 600)
  assert abs(sm["value/offset"]["dip_mean"] - 0.04) < 1e-9
  assert sm["value/offset"]["recovery50_mean"] <= sm["value/offset"]["recovery_mean"]
  assert sm["value/offset"]["recovery_mean"] == 149.0  # first eval >= 0.95 x 0.10 is at it 749; shock at 600
  assert abs(sm["value/none"]["dip_mean"]) < 1e-9  # the resumed-unshocked control loses nothing
  bv = aw.branch_validation(runs, 600)
  assert bv["n"] == 3 and abs(bv["mean_diff"]) < 1e-9


def test_boot_ci_contains_the_mean_and_is_ordered():
  m, lo, hi = aw.boot_ci(np.array([0.1, 0.2, 0.15, 0.12, 0.18]))
  assert lo <= m <= hi


def test_margin_scales_with_the_task_not_with_a_broken_comparator():
  rng = np.random.default_rng(1)
  runs = {}
  for s in range(2000, 2010):
    for rep, c, base in (("raw", "none", 0.03), ("raw", "value", 0.09), ("ssp_low", "none", 0.03), ("ssp_low", "value", 0.07)):
      k, r = _run("core", s, c, rep, curve=[base + 0.0005 * rng.standard_normal()] * 8)
      runs[k] = r
  out = aw.core_contrasts(runs, "go1", 0.999, 0.03)
  assert 0.03 * 0.088 < out["margin"] < 0.03 * 0.092  # 3% of the best cell (~0.09), not of the failing arm (~0.03)
  assert out["centering_main_raw"]["mean"] > 0.05 and not out["centering_main_raw"]["equivalent"]
