"""Round 9b (REPORT_009): does a constant critic error e really become -(1-gamma)*e*w_t in the advantages?

This drives **rsl-rl's own `PPO.compute_returns`** on synthetic storage, shifting every stored value and the bootstrap
value by the same constant e, and compares the difference against the closed form

    A_t(e) - A_t(0) = -(1-gamma) * e * w_t,     w_t = sum_{k=0}^{T-t-1} (gamma*lambda)^k

for a rollout with no episode ends. Two further cases are checked because the protocol depends on them:

  * one done at step k inside the rollout: the sum is truncated there, and the step itself keeps a shift of -e (the
    `1 - done` factor removes the gamma*V term), so the closed form is only valid up to k;
  * the time-out path: rsl-rl folds `gamma*V(s_t)` into the stored reward at a time-out step, so shifting V by e shifts
    that reward by gamma*e too and the two terms cancel: the leak at the time-out step is (gamma-1)*e, *identical* to the
    ordinary case. (Checked because the first draft of this test asserted (2*gamma-1)*e and the run refused it; the
    continuing Go1 protocol never times out anyway: 2000 iters x 24 steps << 1e6 s / dt.)

**A numerical property found while writing this**: rsl-rl computes `next_is_not_terminal * gamma * next_values` with
`dones.float()`, so `gamma*V` is evaluated in float32 before being added to the float64 reward/values. The bootstrap term
therefore carries a relative error of ~1e-5 (gamma .999 becomes 0.9990000128), which is what the expected values below
use: the algebra is exact, the arithmetic is not.

Also asserted: with `normalize_advantage_per_mini_batch=True` (the study's setting is verified separately) the stored
advantages are the *raw* GAE ones, which is what makes the leak visible at all; and `err = returns - values` in algo.py
is that same raw advantage, so `const_frac`/`err_mean` are statistics of it.
"""
from __future__ import annotations

import math
import types

import numpy as np
import torch

from rsl_rl.algorithms import PPO


class _Critic:
  """Returns a constant (V_T + e) for the bootstrap, with the two calls compute_returns makes."""

  def __init__(self, v_last: float) -> None:
    self.v_last = float(v_last)

  def get_hidden_state(self):
    return None

  def reset(self, *a, **k):
    return None

  def __call__(self, obs):
    return torch.full((obs.shape[0],), self.v_last, dtype=torch.float64)


def _storage(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray | None = None):
  T, N = rewards.shape
  dones = np.zeros_like(rewards) if dones is None else dones
  st = types.SimpleNamespace()
  st.num_transitions_per_env = T
  st.rewards = torch.tensor(rewards, dtype=torch.float64)
  st.dones = torch.tensor(dones, dtype=torch.float64)
  st.values = torch.tensor(values, dtype=torch.float64)
  st.returns = torch.zeros_like(st.values)
  st.advantages = torch.zeros_like(st.values)
  return st


def _ppo(st, gamma: float, lam: float) -> PPO:
  """A PPO whose only real parts are compute_returns' inputs (the method itself is rsl-rl's)."""
  fake = types.SimpleNamespace(storage=st, gamma=gamma, lam=lam, normalize_advantage_per_mini_batch=True,
                               critic=_Critic(0.0), is_multi_gpu=False, gpu_global_rank=0, gpu_world_size=1)
  return fake


def _advantages(rewards, values, v_last, gamma, lam, dones=None, shift=0.0):
  st = _storage(rewards, values + shift, dones)
  fake = _ppo(st, gamma, lam)
  fake.critic = _Critic(v_last + shift)
  PPO.compute_returns(fake, torch.zeros((rewards.shape[1], 1)))
  return st.advantages


def _g32(gamma: float) -> float:
  """The gamma rsl-rl actually multiplies by: `1.0 - dones.float()` makes the product float32."""
  return float(np.float32(np.float32(1.0) * gamma))


def _m32(gamma: float, lam: float) -> float:
  """The decay factor the GAE recursion actually uses: (float32 gamma) * lam, rounded to float32."""
  return float(np.float32(np.float32(np.float32(1.0) * gamma) * lam))


def _w(gamma: float, lam: float, T: int) -> np.ndarray:
  m = _m32(gamma, lam)
  return np.array([sum(m ** k for k in range(T - t)) for t in range(T)])


def test_constant_value_error_becomes_minus_one_minus_gamma_e_times_the_gae_weights():
  rng = np.random.default_rng(0)
  T, N, gamma, lam, e = 24, 8, 0.999, 0.95, 500.0
  rewards = rng.normal(1.0, 0.1, size=(T, N))
  values = 1000.0 + rng.normal(0.0, 2.0, size=(T, N))
  v_last = 1000.0
  a0 = _advantages(rewards, values, v_last, gamma, lam)
  a1 = _advantages(rewards, values, v_last, gamma, lam, shift=e)
  got = (a1 - a0).numpy()
  want = ((_g32(gamma) - 1.0) * e) * _w(gamma, lam, T)[:, None]
  assert np.allclose(got, want, atol=1e-5), (got[:, 0], want[:, 0])
  # the size the theory quotes: about 0.5 at the last position, about 7 at the first (e = 500, gamma .999)
  assert abs(want[-1, 0] - (-0.5)) < 1e-4
  assert abs(want[0, 0] - (-0.5 * 14.03)) < 0.02
  # and it is position-dependent, which is why batch normalisation cannot remove it
  assert np.std(got[:, 0]) > 1.0


def test_zero_error_changes_nothing():
  rng = np.random.default_rng(1)
  rewards, values = rng.normal(1.0, 0.1, (24, 4)), rng.normal(0.0, 1.0, (24, 4))
  a0 = _advantages(rewards, values, 0.0, 0.999, 0.95)
  a1 = _advantages(rewards, values, 0.0, 0.999, 0.95, shift=0.0)
  assert torch.equal(a0, a1)


def test_a_done_truncates_the_sum_and_the_step_itself_shifts_by_minus_e():
  rng = np.random.default_rng(2)
  T, N, gamma, lam, e, k = 24, 2, 0.999, 0.95, 500.0, 16
  rewards = rng.normal(1.0, 0.1, (T, N))
  values = 1000.0 + rng.normal(0.0, 2.0, (T, N))
  dones = np.zeros((T, N))
  dones[k, :] = 1.0
  a0 = _advantages(rewards, values, 1000.0, gamma, lam, dones)
  a1 = _advantages(rewards, values, 1000.0, gamma, lam, dones, shift=e)
  got = (a1 - a0).numpy()[:, 0]
  # recursion, backwards over the whole rollout: shift_k = -e at the done (the 1-done factor removes the gamma*V term);
  # elsewhere shift_t = (gamma-1)e + gamma*lam*shift_{t+1}, and the recursion starts from zero past the last step
  c, m = (_g32(gamma) - 1.0) * e, _m32(gamma, lam)
  want = np.zeros(T)
  for t in range(T - 1, -1, -1):
    if t == k:
      want[t] = -e
    else:
      want[t] = c + (m * want[t + 1] if t < T - 1 else 0.0)
  assert np.allclose(got, want, atol=1e-4), (got, want)
  assert abs(got[k] + e) < 1e-4  # the done step shifts by -e, not by (gamma-1)e
  # positions after the done are unaffected by it (the recursion only truncates at k)
  assert abs(got[k + 1] - (c + m * want[k + 2] if k + 2 < T else c)) < 1e-4


def test_a_timeout_bootstrap_gives_the_same_leak_as_the_ordinary_case():
  """At a time-out rsl-rl folds gamma*V(s_t) into the stored reward, and that term shifts by gamma*e with V, so the step's
  leak is (gamma-1)*e -- the same as any other step, not (2*gamma-1)*e."""
  gamma, e = 0.999, 500.0
  rewards, values = np.zeros((2, 1)), np.zeros((2, 1))
  dones = np.array([[1.0], [0.0]])  # time-out encoded as a done with a bootstrapped reward
  a0 = _advantages(rewards, values, 0.0, gamma, 0.95, dones)
  r1 = rewards.copy()
  r1[0, 0] = _g32(gamma) * e  # collection folded gamma*V into the reward, so the shift carries gamma*e as well
  a1 = _advantages(r1, values, 0.0, gamma, 0.95, dones, shift=e)
  got = float((a1 - a0)[0, 0])
  assert abs(got - ((_g32(gamma) - 1.0) * e)) < 1e-6, got
  # and the continuing Go1 protocol cannot reach a time-out: 2000 iters x 24 steps << episode_length 1e6 s / dt
  assert 2000 * 24 < 1e6 / 0.02
