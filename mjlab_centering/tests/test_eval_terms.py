"""Regression: evaluate() must report per-step term means, not means divided by num_envs again (decision 43)."""

from __future__ import annotations

import types

import torch

from mjlab_centering.train import evaluate


class _FakeVec:
  def __init__(self, n_envs=8, f=0.5, w=2.0):
    self.num_envs, self.device, self.f, self.w = n_envs, torch.device("cpu"), f, w
    cfg = types.SimpleNamespace(weight=w)
    rm = types.SimpleNamespace(active_terms=["track_linear_velocity"], _term_cfgs=[cfg])
    self.unwrapped = types.SimpleNamespace(reward_manager=rm, step_dt=0.02)

  def get_observations(self):
    return torch.zeros(self.num_envs, 1)

  def step(self, act):
    n = self.num_envs
    cc = {"task": torch.full((n,), 0.04), "seen": torch.full((n,), 0.04), "fell": torch.zeros(n),
          "terms": torch.tensor([self.w * self.f * 0.02]), "term_names": ["track_linear_velocity"]}
    return torch.zeros(n, 1), torch.zeros(n), torch.zeros(n), {"cc": cc}


def test_term_means_and_tracking_fraction_do_not_depend_on_num_envs():
  for n in (4, 128):
    out = evaluate(_FakeVec(n_envs=n), lambda obs, stochastic_output=False: torch.zeros(n, 1), steps=5)
    assert abs(out["term/track_linear_velocity"] - 0.02) < 1e-6  # w x f x dt = 2.0 x 0.5 x 0.02
    assert abs(out["track_lin_frac"] - 0.5) < 1e-6
    assert abs(out["J_task"] - 0.04) < 1e-6
