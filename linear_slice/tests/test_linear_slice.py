from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import algos  # noqa: E402
import features  # noqa: E402
from catch import Catch, exact_policy_quantities  # noqa: E402


def test_exact_quantities_are_consistent():
  env = Catch()
  pi = env.chase_policy()
  q = exact_policy_quantities(env, pi, 0.99, offset=1.0)
  assert abs(q["d"].sum() - 1) < 1e-9 and (q["d"] >= -1e-12).all()
  assert np.allclose(q["d"] @ q["Ppi"], q["d"], atol=1e-9)  # stationary
  # r(pi) from the stationary distribution matches simulation
  env_rng = np.random.default_rng(0)
  s, tot, n = env.reset(env_rng), 0.0, 60_000
  cum = np.cumsum(pi, 1)
  for _ in range(n):
    a = min(int(np.searchsorted(cum[s], env_rng.random())), 2)
    s, r = env.step(s, a, env_rng)
    tot += r
  assert abs(tot / n - q["rate"]) < 0.02
  # centred value = discounted value with the true r_bar removed
  assert np.allclose(q["v_centred"], q["v_gamma"] - q["rbar"] / (1 - 0.99))


def test_td0_with_tabular_features_recovers_true_values():
  env = Catch()
  pi = env.chase_policy()
  gamma = 0.9
  Phi = np.eye(env.S)
  w, _, _ = algos.td0_fixed_policy(env, Phi, pi, gamma, 0.0, "none", steps=400_000, alpha0=0.05, eta=0.0, seed=0)
  q = exact_policy_quantities(env, pi, gamma)
  st = algos.critic_error_stats(Phi @ w, q["v_gamma"], q["d"])
  assert st["mse"] < 0.02, st


def test_centred_r_bar_tracks_the_true_average_reward_up_to_a_gamma_dependent_bias():
  """With gamma < 1 the TD-based r_bar fixed point is r(pi) - (1-gamma) * mean(v_hat), not exactly r(pi) (for tabular
  values every r_bar is consistent with its own centred values). The bias is O(1-gamma): <15% at 0.99."""
  env = Catch()
  pi = env.chase_policy()
  Phi = features.make("raw_bias", env.state_x())
  q = exact_policy_quantities(env, pi, 0.99, offset=1.0)
  _, r_bar, _ = algos.td0_fixed_policy(env, Phi, pi, 0.99, 1.0, "value", steps=300_000, alpha0=0.1, eta=0.1, seed=1)
  assert abs(r_bar - q["rbar"]) < 0.15 * abs(q["rbar"]), (r_bar, q["rbar"])


def test_offset_without_a_bias_unit_leaves_a_large_constant_error_that_centering_removes():
  env = Catch()
  pi = env.chase_policy()
  gamma = 0.99
  Phi = features.make("raw", env.state_x())  # 3 coordinates in [0,1], no bias: cannot represent a constant
  q = exact_policy_quantities(env, pi, gamma, offset=1.0)
  kw = dict(steps=300_000, alpha0=0.1, seed=2)
  wu, _, _ = algos.td0_fixed_policy(env, Phi, pi, gamma, 1.0, "none", eta=0.0, **kw)
  wc, rb, _ = algos.td0_fixed_policy(env, Phi, pi, gamma, 1.0, "value", eta=0.1, **kw)
  su = algos.critic_error_stats(Phi @ wu, q["v_gamma"], q["d"])
  # the centred critic fits the value of (r - r_bar_hat): its own learned r_bar defines its target
  sc = algos.critic_error_stats(Phi @ wc, q["v_gamma"] - rb / (1 - gamma), q["d"])
  assert su["mse"] > 10 * sc["mse"], (su, sc)
  assert su["const_frac"] > 0.5


def test_error_decomposition_identifies_a_leftover_constant():
  """The centred critic's fixed point can carry a constant k/(1-gamma) (Naik eq. 6). It must land in const_part, not in
  the decision-relevant part, otherwise a harmless offset reads as "centering is worse"."""
  import numpy as np
  d = np.array([1.0])
  target = np.array([5.0])
  assert algos.critic_error_stats(np.array([5.0 + 220.0]), target, d)["const_frac"] > 0.999
  st = algos.critic_error_stats(np.array([5.0 + 220.0]), target, d)
  assert st["state_part"] < 1e-9 and abs(st["const_part"] - 220.0**2) < 1e-6
  st2 = algos.critic_error_stats(np.array([5.0 + 220.0 + 1e-3]), target, np.array([1.0]))
  assert st2["state_part"] < 1e-5


def test_actor_critic_first_window_init_keeps_the_policy_alive_at_an_offset():
  env = Catch()
  Phi = features.make("tile", env.state_x())
  a = algos.actor_critic(env, Phi, 0.99, 1.0, "value", 60_000, 0.1, 0.02, 0.1, 0, r_bar_init="zero")
  b = algos.actor_critic(env, Phi, 0.99, 1.0, "value", 60_000, 0.1, 0.02, 0.1, 0, r_bar_init="first_window")
  assert abs(b["r_bar"][0] - 1.0) < 0.6  # starts near the true rate instead of 0
  assert np.mean(b["reward"][-2:]) >= np.mean(a["reward"][-2:]) - 1e-6
