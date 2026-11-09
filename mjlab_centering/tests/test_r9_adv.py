"""Round 9b: the per-position advantage modes (algo.adv_by_mode) and the position statistics.

What must hold:
  * `pos_norm` gives every rollout position mean 0 and std 1;
  * `pos_center` gives every position mean 0 and the whole batch std 1 (positions keep their relative spread);
  * both are **invariant** to adding a position-dependent constant `b * w_t` (the leak), while `global` is not -- that
    invariance is the entire causal claim of the arm;
  * `pos_center` and `pos_norm` leave the *ordering within a position* untouched (they are affine per position), so a
    position-standardised advantage cannot change which action is favoured inside a position;
  * `global` reproduces rsl-rl's own formula exactly, so selecting the default cannot change any existing run.
"""
from __future__ import annotations

import numpy as np
import torch

from mjlab_centering.algo import adv_by_mode, adv_position_stats


def _raw(T: int = 24, N: int = 4096, seed: int = 0) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  return torch.randn(T, N, generator=g, dtype=torch.float64)


def _w(gamma: float = 0.999, lam: float = 0.95, T: int = 24) -> torch.Tensor:
  return torch.tensor([sum((gamma * lam) ** k for k in range(T - t)) for t in range(T)], dtype=torch.float64)


def test_pos_norm_standardises_every_position():
  raw = _raw()
  out = adv_by_mode(raw, "pos_norm")
  assert torch.allclose(out.mean(dim=1), torch.zeros(24, dtype=torch.float64), atol=1e-12)
  assert torch.allclose(out.std(dim=1), torch.ones(24, dtype=torch.float64), atol=1e-6)


def test_pos_center_removes_the_mean_and_keeps_one_global_scale():
  raw = _raw()
  out = adv_by_mode(raw, "pos_center")
  assert torch.allclose(out.mean(dim=1), torch.zeros(24, dtype=torch.float64), atol=1e-12)
  assert abs(float(out.std()) - 1.0) < 1e-6
  # the relative spread of positions is preserved (unlike pos_norm, which equalises it)
  assert not torch.allclose(out.std(dim=1), torch.ones(24, dtype=torch.float64), atol=1e-3)


def test_both_position_modes_are_invariant_to_the_leak_and_global_is_not():
  raw = _raw()
  w = _w()
  leaked = raw + 1.5 * w[:, None]  # a position-dependent constant: exactly the theory's bias shape
  for mode in ("pos_norm", "pos_center"):
    a, b = adv_by_mode(raw, mode), adv_by_mode(leaked, mode)
    assert torch.allclose(a, b, atol=1e-9), mode
  g0, g1 = adv_by_mode(raw, "global"), adv_by_mode(leaked, "global")
  assert not torch.allclose(g0, g1, atol=1e-3), "global normalisation must NOT absorb the position bias"


def test_position_modes_are_affine_within_a_position():
  """They may not reorder the signals inside a position: that is what makes the arm a test of the leak, not of ranking."""
  raw = _raw(N=64)
  for mode in ("pos_norm", "pos_center"):
    out = adv_by_mode(raw, mode)
    for t in (0, 11, 23):
      a, b = raw[t], out[t]
      assert torch.equal(a.argsort(), b.argsort()), (mode, t)


def test_global_reproduces_rsl_rl_formula_bit_for_bit():
  raw = _raw()
  ref = (raw - raw.mean()) / (raw.std() + 1e-8)
  assert torch.equal(adv_by_mode(raw, "global"), ref)


def test_position_stats_measure_the_position_share_of_variance():
  raw = _raw()
  w = _w()
  flat = adv_by_mode(raw, "pos_center")  # position means removed
  s0 = adv_position_stats(flat)
  assert abs(s0["adv_pos_r2"]) < 0.02  # almost no variance left in position
  s1 = adv_position_stats(flat + 3.0 * w[:, None])
  assert s1["adv_pos_r2"] > 0.95  # a leak of this size is almost all of the variance
  assert len(s0["adv_pos_mean"]) == 24 and len(s0["adv_pos_std"]) == 24
  # and the profile has the theory's shape: the first position carries the largest bias, the last the smallest,
  # with the ratio w_0 / w_23 = 14.03
  prof = np.array(s1["adv_pos_mean"])
  assert prof[0] == prof.max() and prof[-1] == prof.min()
  assert 12 < prof[0] / prof[-1] < 16
