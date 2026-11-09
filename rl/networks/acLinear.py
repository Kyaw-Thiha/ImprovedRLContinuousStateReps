"""Linear TD(0) actor-critic, NumPy port of the legacy Nengo A2C (no Nengo, no PyTorch).

Faithful port of `rl/networks/acBasic_nengo.py` + `rl/rules/td0_nengo.py` / `td0centering_nengo.py`
for the configuration the A2C grid used (`state_neurons=None`, `force_rho_one=True`). This is NOT a
textbook policy-gradient A2C; the quirks below are the legacy rule and are kept on purpose:

* One weight matrix W of shape (n_actions + 1, dim): row 0 is the state value V, rows 1.. are linear
  action PREFERENCES. The policy is epsilon-greedy over the preferences (argmax = the greedy policy).
* Normalised step: every update is divided by ||phi(s)||^2 (`scale`), so `lr` 0.4-0.8 is a fraction of an
  NLMS step, not a plain SGD rate. For a binary tile vector that is 1 / n_active.
* The value row moves by lr * td * phi * scale; the TAKEN action's preference row moves by
  act_dis * td * phi * scale (the same TD error as the critic; no pi(1-pi) term).
* The TD target uses a STALE bootstrap: V(s) was computed one step earlier (before that step's update),
  and the preferences returned for s' (used to pick the next action) are computed from the weights
  BEFORE this step's update.
* Terminal handling is done by the caller as an extra step on the same state with reward 0 (n_done=1),
  not by bootstrapping to zero.
* Centering: `simple`: td uses r - R; R += beta * (r - R_old). `value`: td uses r - R; R += eta * lr * td.
  td is computed with the OLD R in both modes.
* W starts at zero, so the policy is deterministic given the data; the seed only affects the env and
  the exploration draws.

`phi` is either a dense float vector (sparse=False) or an int index array of unique active features
(sparse=True, features are binary 1.0). Both give the same numbers up to float summation order.
"""

import numpy as np


class LinearActorCritic:
    def __init__(
        self,
        dim: int,
        n_actions: int = 2,
        lr: float = 0.1,
        act_dis: float = 0.9,
        state_dis: float = 0.99,
        centering: str = "none",
        beta: float = 0.001,
        eta: float = 1.0,
        avg_reward_init: float = 0.0,
        sparse: bool = False,
    ):
        if centering not in ("none", "simple", "value"):
            raise ValueError(f"unknown centering mode {centering!r}")
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.act_dis = float(act_dis)
        self.state_dis = float(state_dis)
        self.centering = centering
        self.beta = float(beta)
        self.eta = float(eta)
        self.avg_reward = float(avg_reward_init)
        self.sparse = bool(sparse)
        self.W = np.zeros((self.n_actions + 1, self.dim))
        self._prev_phi = None
        self._prev_value = 0.0
        self._prev_sq = 0.0

    # -- helpers ------------------------------------------------------------------------------
    def _forward(self, phi):
        if self.sparse:
            return self.W[:, phi].sum(axis=1)
        return self.W @ phi

    def _remember(self, phi, value):
        self._prev_phi = phi
        self._prev_value = float(value)
        # ||phi||^2 (dense) / number of active unique binary features (sparse)
        self._prev_sq = float(len(phi)) if self.sparse else float(np.sum(phi**2))  # np.sum: same pairwise rounding as the legacy rule

    def preferences(self, phi):
        """Action preferences of the ACTOR rows only, for any state (no side effects). Same arithmetic as the
        saved policy (rl/policy_io_a2c.py), so greedy evaluation during training == the reloaded policy."""
        A = self.W[1:]
        return A[:, phi].sum(axis=1) if self.sparse else A @ phi

    def value(self, phi) -> float:
        return float(self.W[0, phi].sum()) if self.sparse else float(self.W[0] @ phi)

    # -- API ----------------------------------------------------------------------------------
    def reset(self, phi):
        """Episode start: remember the first state, do no update. Returns its action preferences."""
        res = self._forward(phi)
        self._remember(phi, res[0])
        return res[1:]

    def step(self, phi, action: int, reward: float):
        """One transition (prev state --action--> phi, reward). Returns the preferences for `phi`
        (computed from the weights before this update, as in the legacy code)."""
        res = self._forward(phi)
        value = res[0]
        reward_c = reward - self.avg_reward if self.centering != "none" else reward
        td = reward_c + self.state_dis * value - self._prev_value

        if self.centering == "simple":
            self.avg_reward = self.avg_reward + self.beta * (reward - self.avg_reward)
        elif self.centering == "value":
            self.avg_reward = self.avg_reward + self.eta * self.lr * td

        scale = 1.0 / self._prev_sq if self._prev_sq != 0 else 0.0
        W = self.W
        prev = self._prev_phi
        row = 1 + int(action)
        if self.sparse:
            W[0, prev] += self.lr * td * scale
            W[row, prev] += self.act_dis * td * scale
        else:
            # same left-to-right operation order as the legacy rule, so dense results match it bit for bit
            W[0] += self.lr * td * prev * scale
            W[row] += self.act_dis * td * prev * scale

        self._remember(phi, value)
        return res[1:]
