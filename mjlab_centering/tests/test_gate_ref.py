"""The gate must pass a learning curve and fail the flat 'stands still' curve seen in e1 (J_task 0.077, tracking 0.018)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import gate_ref  # noqa: E402


def _rows(track, falls=0.5, n=40):
  it = np.arange(24, 24 + 25 * n, 25)
  return [{"iter": int(i), "J_task": 0.07, "track_lin_frac": float(t), "fall_rate_per_1k": falls} for i, t in zip(it, track)]


def _learning(n=40, top=0.7):
  return _rows(0.018 + (top - 0.018) * (1 - np.exp(-np.arange(n) / 8)), n=n)


def _flat(n=40):
  return _rows(np.full(n, 0.018), n=n)


def test_learning_curves_pass():
  w = [gate_ref.run_summary(_learning()), gate_ref.run_summary(_learning(top=0.65))]
  e = [gate_ref.run_summary(_learning(top=0.7))]
  v = gate_ref.verdict(w, e)
  assert v["pass"], v["checks"]
  assert v["plateau_iter_track_lin"] > 0


def test_flat_wrapped_fails_even_if_episodic_learns():
  v = gate_ref.verdict([gate_ref.run_summary(_flat())], [gate_ref.run_summary(_learning())])
  assert not v["pass"]
  assert not v["checks"]["wrapped_learns_all_seeds"]


def test_wrapped_learning_but_far_below_episodic_fails_match():
  v = gate_ref.verdict([gate_ref.run_summary(_learning(top=0.3))], [gate_ref.run_summary(_learning(top=0.8))])
  assert v["checks"]["wrapped_learns_all_seeds"] and not v["checks"]["wrapped_matches_episodic"]
  assert not v["pass"]


def test_falling_robot_fails_even_with_tracking():
  rows = _rows(0.018 + 0.6 * (1 - np.exp(-np.arange(40) / 8)), falls=50.0)
  v = gate_ref.verdict([gate_ref.run_summary(rows)], [gate_ref.run_summary(_learning())])
  assert not v["checks"]["wrapped_learns_all_seeds"]
