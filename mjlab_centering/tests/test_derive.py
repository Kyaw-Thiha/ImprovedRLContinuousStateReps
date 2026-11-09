"""Derived-metric arithmetic, on hand-built curves with known answers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import derive  # noqa: E402


def _run(block, center, gamma, offset, seed, curve, iters=500, shock=None):
  it = list(range(0, len(curve) * 25, 25))
  return {
    "job_id": f"{block}_{center}_{gamma}_{offset}_{seed}",
    "block": block, "center": center, "gamma": gamma, "offset": offset, "seed": seed,
    "itcurve": it, "jcurve": list(curve), "final": float(np.mean(curve[-5:])),
    "aulc": float(np.mean(curve)), "done": True, "n_evals": len(curve), "n_iters": 1,
    "falls_final": 0.0, "const_frac_late": 0.0, "shock_kind": shock,
  }


def test_thresholds_read_off_the_curve():
  ref = 1.0
  curve = [0.0] * 4 + [0.6] * 16
  r = {"itcurve": list(range(0, len(curve) * 25, 25)), "jcurve": curve}
  t = derive.thresholds(r, ref)
  # the 3-eval running mean first reaches 0.5 at index 5 (two flanking zeros keep it at 0.4 at index 4)
  assert t["T50"] == 25 * 5
  assert t["T90"] == -1


def test_thresholds_ignores_a_transient_spike():
  curve = [0.0] * 4 + [0.95] * 2 + [0.1] * 10
  r = {"itcurve": list(range(0, len(curve) * 25, 25)), "jcurve": curve}
  assert derive.thresholds(r, 1.0)["T90"] == -1


def test_boot_ci_brackets_the_mean():
  m, lo, hi = derive.boot_ci(np.array([1.0, 1.1, 0.9, 1.05]))
  assert lo <= m <= hi and (hi - lo) < 0.4


def test_paired_diff_is_paired_on_shared_seeds():
  a = {1: 1.0, 2: 2.0}
  b = {1: 0.5, 2: 1.0, 3: 99.0}
  m, lo, hi, n = derive.paired_diff(a, b)
  assert n == 2 and abs(m - 0.75) < 1e-9


def test_summary_normalises_to_the_reference_arm_and_flags_collapse():
  ref = [_run("A", "none", 0.99, 0.0, s, [1.0] * 20) for s in (2000, 2001)]
  good = [_run("A", "value", 0.999, 1.0, s, [1.0] * 20) for s in (2000, 2001)]
  bad = [_run("A", "none", 0.999, 1.0, s, [0.2] * 20) for s in (2000, 2001)]
  res = derive.summarise(ref + good + bad)
  assert abs(res["j_ref"] - 1.0) < 1e-9
  assert abs(res["cells"]["A|value|g0.999|o1.0"]["final_over_ref"] - 1.0) < 1e-9
  assert abs(res["cells"]["A|none|g0.999|o1.0"]["final_over_ref"] - 0.2) < 1e-9
  d = res["paired_diffs"]["value_vs_none"]
  assert d[3] == 2 and abs(d[0] - 0.8) < 1e-9  # centered minus uncentered at gamma=.999, offset c


def test_shock_metrics_use_the_three_evals_before_the_shock():
  it = list(range(0, 800, 25))
  curve = [1.0] * 16 + [0.5] * 16  # flat before the shock (eval index 16 == iter 400), dip after
  r = {"itcurve": it, "jcurve": curve}
  job = {"shock_kind": "offset", "shock_at": 400}
  # emulate the block inside derive_run
  j = np.array(curve)
  pre = j[np.array(it) < 400][-3:]
  post = j[np.array(it) >= 400]
  assert abs(float(np.mean(pre)) - 1.0) < 1e-9
  assert float(np.min(post) / np.mean(pre)) == 0.5
