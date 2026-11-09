"""The leak-ratio arithmetic of analysis/r9_leak_retro.py, checked on advantages whose bias is injected by hand."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import r9_leak_retro as L  # noqa: E402


def test_weights_match_the_closed_form():
  g = 0.999 * 0.95
  w = L.gae_weights(0.999)
  assert len(w) == 24
  assert w[-1] == 1.0  # last position: a single delta
  assert abs(w[0] - (1 - g ** 24) / (1 - g)) < 1e-12
  assert 14.0 < w[0] < 14.3  # "about 14" in the theory's statement


def test_rho_recovers_an_injected_position_bias_exactly():
  """Advantages = zero-mean signal + delta_bar * w_t. rho must equal |delta_bar|*std(w) / std(advantage)."""
  rng = np.random.default_rng(0)
  gamma, T, N = 0.999, 24, 200000
  w = L.gae_weights(gamma)
  delta_bar = 0.37
  signal = rng.normal(0.0, 0.8, size=(T, N))
  adv = signal + delta_bar * w[:, None]
  err_mean, err_mse = adv.mean(), (adv ** 2).mean()
  rho = L.leak_ratio(np.array([delta_bar]), np.array([err_mean]), np.array([err_mse]), gamma)[0]
  # exact value from the construction (sample statistics of the signal enter at the 1e-3 level with N = 200k)
  exact = abs(delta_bar) * np.std(w) / adv.std()
  assert abs(rho - exact) < 1e-9, (rho, exact)
  expected = abs(delta_bar) * np.std(w) / np.sqrt(0.8 ** 2 + (delta_bar * np.std(w)) ** 2)
  assert abs(rho - expected) < 2e-3


def test_rho_is_zero_without_bias_and_saturates_with_a_huge_one():
  gamma = 0.999
  assert L.leak_ratio(np.array([0.0]), np.array([0.0]), np.array([1.0]), gamma)[0] == 0.0
  big = L.leak_ratio(np.array([50.0]), np.array([50.0 * L.gae_weights(gamma).mean()]),
                     np.array([(50.0 * L.gae_weights(gamma).mean()) ** 2 + (50.0 * np.std(L.gae_weights(gamma))) ** 2 + 1e-4]),
                     gamma)[0]
  assert big > 0.999
