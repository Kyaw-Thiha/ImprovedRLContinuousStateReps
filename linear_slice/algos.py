"""Linear TD(0) with a fixed policy (primary mechanism test) and linear one-step actor-critic (control).

Centering follows Naik et al. (value-based): delta = r - r_bar + gamma w.phi' - w.phi ; w += alpha delta phi ;
r_bar += eta * alpha * delta. `center="none"` is plain discounted TD. A constant `offset` is added to every reward.
Step sizes are normalised by the mean squared feature norm so feature sets of different scale are comparable.
"""

from __future__ import annotations

import numpy as np

from catch import Catch


def td0_fixed_policy(env: Catch, Phi: np.ndarray, pi: np.ndarray, gamma: float, offset: float, center: str,
                     steps: int, alpha0: float, eta: float, seed: int, log_every: int = 10_000):
  """Returns final weights, r_bar, and a log of (step, w) snapshots reduced to value vectors."""
  rng = np.random.default_rng(seed)
  alpha = alpha0 / float(np.mean(np.sum(Phi * Phi, axis=1)))
  w = np.zeros(Phi.shape[1])
  r_bar = 0.0
  s = env.reset(rng)
  snaps = []
  cum = np.cumsum(pi, axis=1)
  for t in range(1, steps + 1):
    a = int(np.searchsorted(cum[s], rng.random()))
    a = min(a, env.A - 1)
    s2, r = env.step(s, a, rng)
    r = r + offset
    delta = r - (r_bar if center == "value" else 0.0) + gamma * (Phi[s2] @ w) - Phi[s] @ w
    w += alpha * delta * Phi[s]
    if center == "value":
      r_bar += eta * alpha * delta
    s = s2
    if t % log_every == 0:
      snaps.append((t, Phi @ w, r_bar))
  return w, r_bar, snaps


def critic_error_stats(v_hat: np.ndarray, target: np.ndarray, d: np.ndarray) -> dict:
  """Error split under the stationary distribution: the constant part (mean error) and the state-dependent rest."""
  e = v_hat - target
  mean = float(d @ e)
  mse = float(d @ (e * e))
  return {
    "mse": mse,
    "const_part": mean * mean,
    "state_part": max(mse - mean * mean, 0.0),
    "const_frac": (mean * mean) / (mse + 1e-12),
  }


def softmax(z):
  z = z - z.max()
  e = np.exp(z)
  return e / e.sum()


def actor_critic(env: Catch, Phi: np.ndarray, gamma: float, offset: float, center: str, steps: int,
                 alpha_c: float, alpha_a: float, eta: float, seed: int, window: int = 20_000,
                 r_bar_init: str = "zero"):
  """One-step linear actor-critic (A2C-style, online) with a linear softmax actor. Returns the per-window average
  task reward (offset excluded), mean |critic value| and mean TD error."""
  rng = np.random.default_rng(seed)
  norm = float(np.mean(np.sum(Phi * Phi, axis=1)))
  ac, aa = alpha_c / norm, alpha_a / norm
  w = np.zeros(Phi.shape[1])
  theta = np.zeros((Phi.shape[1], env.A))
  r_bar = 0.0
  s = env.reset(rng)
  if r_bar_init == "first_window" and center == "value":
    # initialise r_bar from a short pre-roll of the initial policy: starting at 0 with offset 1 means every early
    # delta is ~1, which drives the actor to a saturated bad policy before r_bar catches up (the Go1 eta finding).
    pre, ps = 0.0, env.reset(rng)
    for _ in range(2000):
      p = softmax(Phi[ps] @ theta)
      a = int(rng.choice(env.A, p=p))
      ps, r = env.step(ps, a, rng)
      pre += r
    r_bar = (pre + 2000 * offset) / 2000
    s = env.reset(rng)
  out = {"reward": [], "abs_v": [], "delta": [], "r_bar": []}
  acc_r = acc_v = acc_d = 0.0
  for t in range(1, steps + 1):
    p = softmax(Phi[s] @ theta)
    a = int(rng.choice(env.A, p=p))
    s2, r_task = env.step(s, a, rng)
    r = r_task + offset
    v, v2 = Phi[s] @ w, Phi[s2] @ w
    delta = r - (r_bar if center == "value" else 0.0) + gamma * v2 - v
    w += ac * delta * Phi[s]
    g = -p
    g[a] += 1.0
    theta += aa * delta * np.outer(Phi[s], g)
    if center == "value":
      r_bar += eta * ac * delta
    acc_r += r_task
    acc_v += abs(v)
    acc_d += delta
    s = s2
    if t % window == 0:
      out["reward"].append(acc_r / window)
      out["abs_v"].append(acc_v / window)
      out["delta"].append(acc_d / window)
      out["r_bar"].append(r_bar)
      acc_r = acc_v = acc_d = 0.0
  return out
