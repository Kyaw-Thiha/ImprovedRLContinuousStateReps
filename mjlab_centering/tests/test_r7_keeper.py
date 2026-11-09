"""The unattended gate (analysis/r7_keeper.py): it may publish Stage 2 only if the harness checks and the budget pass."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import r7_keeper as K  # noqa: E402

JOBS = [{"job_id": f"r7pilot_go1_{a}_g0.999_o1_s{1000 + s}"} for a in ("vn", "sc", "rs") for s in range(2)]


def _wire(monkeypatch, *, done=True, drop: str | None = None, spent=1.2):
  monkeypatch.setattr(K.r2, "get_json", lambda key, default=None: {"jobs": JOBS})
  monkeypatch.setattr(K.r2, "done_set", lambda camp: {j["job_id"] for j in JOBS} if done else set())
  monkeypatch.setattr(K, "spend_now", lambda: {"round7_total": spent})

  def rows(campaign, jid, name):
    if name == "eval.jsonl":
      return [{"iter": i} for i in range(40)]
    arm = K.arm_of(jid)
    row = {"iter": 1999, "r_task_rate": 0.03, "r_bar": 1.0, "vn_mean": 90.0, "vn_std": 60.0, "rs_ret_std": 190.0}
    if drop and arm == drop[0]:
      row.pop(drop[1], None)
    return [{"iter": 0}, row]

  monkeypatch.setattr(K, "_rows", rows)


def test_publishes_when_the_pilot_is_complete_and_the_gate_passes(monkeypatch):
  _wire(monkeypatch)
  d = K.decide("r7pilot-x", Path("q.json"), 24000, 160000)
  assert d["fails"] == [] and d["gate"]["pass"] and d["publish"]
  assert d["gate"]["projected"] == pytest.approx(1.2 / 24 * 160, rel=1e-6)


def test_does_not_publish_while_a_run_is_missing(monkeypatch):
  _wire(monkeypatch, done=False)
  d = K.decide("r7pilot-x", Path("q.json"), 24000, 160000)
  assert not d["publish"] and d["fails"]


def test_does_not_publish_when_an_arms_diagnostic_is_missing(monkeypatch):
  _wire(monkeypatch, drop=("sc", "r_bar"))
  d = K.decide("r7pilot-x", Path("q.json"), 24000, 160000)
  assert not d["publish"] and any("r_bar" in f for f in d["fails"])


def test_does_not_publish_when_the_budget_gate_fails(monkeypatch):
  _wire(monkeypatch, spent=16.0)  # marginal 16/24 = 0.667/1000it -> 16 + 1.3*106 = 154 > 18
  d = K.decide("r7pilot-x", Path("q.json"), 24000, 160000, sunk=0.0)
  assert d["fails"] == [] and not d["gate"]["pass"] and not d["publish"]


def test_sunk_pods_are_counted_once_not_twice(monkeypatch):
  """d168: lost pods belong in `spent`, not in the marginal rate; otherwise a stage is blocked for money already paid."""
  _wire(monkeypatch, spent=2.46)
  d = K.decide("r7pilot-x", Path("q.json"), 24000, 160000, sunk=0.87)
  assert d["marginal_cost_of_pilot"] == pytest.approx(1.59)
  assert d["rate_per_1000it"] == pytest.approx(1.59 / 24)
  assert d["gate"]["projected"] == pytest.approx(1.59 / 24 * 160)
  assert d["gate"]["pass"] and d["publish"]
  # and without the correction the same data blocks the stage, which is the bug being fixed
  d0 = K.decide("r7pilot-x", Path("q.json"), 24000, 160000, sunk=0.0)
  assert not d0["gate"]["pass"]
