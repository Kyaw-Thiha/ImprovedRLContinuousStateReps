"""Vectorised POPGym wrapper: N envs in lockstep (these tasks have fixed-length episodes, so no truncation logic).

The observation is exposed as an integer tensor (B, F) plus a per-env step index, which is what ``models.Policy``
consumes and what makes the positional baseline well defined.
"""

import numpy as np
import torch

from tasks import make_env, spec_from_env


class Vec:
    def __init__(self, fam, diff, n_envs, seed0=0, **env_kw):
        self.fam, self.diff = fam, diff
        self.envs = [make_env(fam, diff, **env_kw) for _ in range(n_envs)]
        self.spec = spec_from_env(fam, diff, self.envs[0])
        self.n = n_envs
        self.ep_len = self.spec["ep_len"]
        self.obs = None
        self.t = None
        self.done_ever = 0
        self.ep_returns = []
        self.reset(seed0)

    def reset(self, seed0=0):
        self.obs = np.stack([np.atleast_1d(e.reset(seed=seed0 + i)[0]) for i, e in enumerate(self.envs)])
        self.t = np.zeros(self.n, dtype=np.int64)
        self.ep_return = np.zeros(self.n, dtype=np.float64)
        self.all_done = False
        return self.obs

    def obs_tensor(self):
        return torch.as_tensor(self.obs).long()

    def t_tensor(self):
        return torch.as_tensor(self.t).long()

    def step(self, actions):
        rew = np.zeros(self.n, dtype=np.float64)
        done = np.zeros(self.n, dtype=bool)
        for i, e in enumerate(self.envs):
            o, r, te, tr, _ = e.step(int(actions[i]))
            rew[i] = r
            self.ep_return[i] += r
            done[i] = te or tr
            if done[i]:
                self.ep_returns.append(self.ep_return[i])
                o, _ = e.reset()
                self.ep_return[i] = 0.0
                self.t[i] = -1  # incremented to 0 below
            self.obs[i] = np.atleast_1d(o)
        self.t = self.t + 1
        self.all_done = bool(done.all())
        return self.obs, rew, done
