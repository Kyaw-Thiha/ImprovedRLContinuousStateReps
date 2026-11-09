"""Continuing Catch (CSuite-style) as an exactly solvable finite MDP.

A ball falls one row per step in an R x C grid; the paddle (bottom) moves left / stay / right. When the ball
reaches the last row the step pays +1 (paddle under the ball) or -1, and a new ball appears in row 0 in a uniformly
random column. There is no terminal state, so it is a continuing task. State = (ball row, ball col, paddle col),
with the ball row in 0..R-2 (225 states for R=10, C=5), small enough to compute the stationary distribution, the
average reward and the discounted values of any fixed policy exactly. Truth, not estimates, is what the mechanism
test needs: the constant part of a critic's error is only defined against the true value.
"""

from __future__ import annotations

import numpy as np


class Catch:
  def __init__(self, rows: int = 10, cols: int = 5):
    self.R, self.C = rows, cols
    self.nr = rows - 1  # ball-row values 0..R-2 after a respawn / before the last row
    self.S = self.nr * cols * cols
    self.A = 3

  def idx(self, r, bc, pc):
    return (r * self.C + bc) * self.C + pc

  def unidx(self, s):
    pc = s % self.C
    bc = (s // self.C) % self.C
    r = s // (self.C * self.C)
    return r, bc, pc

  def state_x(self) -> np.ndarray:
    """Continuous coordinates in [0, 1]^3 for every state (what the feature maps see)."""
    X = np.zeros((self.S, 3))
    for s in range(self.S):
      r, bc, pc = self.unidx(s)
      X[s] = (r / max(self.nr - 1, 1), bc / (self.C - 1), pc / (self.C - 1))
    return X

  def step_dist(self, s: int, a: int):
    """Exact transition: list of (prob, next_state, reward)."""
    r, bc, pc = self.unidx(s)
    pc2 = int(np.clip(pc + (a - 1), 0, self.C - 1))
    if r + 1 < self.nr:
      return [(1.0, self.idx(r + 1, bc, pc2), 0.0)]
    rew = 1.0 if pc2 == bc else -1.0
    return [(1.0 / self.C, self.idx(0, nb, pc2), rew) for nb in range(self.C)]

  def tables(self):
    """P[a, s, s'] and expected reward R[a, s]."""
    P = np.zeros((self.A, self.S, self.S))
    Rw = np.zeros((self.A, self.S))
    for s in range(self.S):
      for a in range(self.A):
        for p, s2, rew in self.step_dist(s, a):
          P[a, s, s2] += p
          Rw[a, s] += p * rew
    return P, Rw

  def reset(self, rng: np.random.Generator) -> int:
    return self.idx(0, int(rng.integers(self.C)), int(rng.integers(self.C)))

  def step(self, s: int, a: int, rng: np.random.Generator):
    r, bc, pc = self.unidx(s)
    pc2 = int(np.clip(pc + (a - 1), 0, self.C - 1))
    if r + 1 < self.nr:
      return self.idx(r + 1, bc, pc2), 0.0
    rew = 1.0 if pc2 == bc else -1.0
    return self.idx(0, int(rng.integers(self.C)), pc2), rew

  def chase_policy(self, p_good: float = 0.7) -> np.ndarray:
    """pi[s, a]: with prob p_good move toward the ball column, else uniform. A fixed, reasonably good policy."""
    pi = np.full((self.S, self.A), (1 - p_good) / self.A)
    for s in range(self.S):
      _, bc, pc = self.unidx(s)
      pi[s, 1 + int(np.sign(bc - pc))] += p_good
    return pi


def exact_policy_quantities(env: Catch, pi: np.ndarray, gamma: float, offset: float = 0.0):
  """Stationary distribution d, average reward r(pi) (without offset), and values of the offset-shifted reward.

  Returns dict with: d, rbar (= r(pi) + offset), v_gamma (discounted value of r + offset), v_centred
  (= v_gamma - rbar / (1 - gamma), the fixed point of centred TD with the true r_bar).
  """
  P, Rw = env.tables()
  Ppi = np.einsum("sa,asz->sz", pi, P)
  rpi = np.einsum("sa,as->s", pi, Rw)
  w, v = np.linalg.eig(Ppi.T)
  d = np.real(v[:, np.argmin(np.abs(w - 1))])
  d = np.abs(d) / np.abs(d).sum()
  rate = float(d @ rpi)
  v_g = np.linalg.solve(np.eye(env.S) - gamma * Ppi, rpi + offset)
  rbar = rate + offset
  return {"d": d, "rate": rate, "rbar": rbar, "v_gamma": v_g, "v_centred": v_g - rbar / (1.0 - gamma), "Ppi": Ppi}
