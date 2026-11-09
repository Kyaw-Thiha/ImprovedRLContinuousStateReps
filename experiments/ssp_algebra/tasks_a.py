"""Variant A: path-integration map (docs/11 section 3.1), as a supervised proxy for the 2-D simulator.

A kinematic point agent moves by small random steps in a 2-D arena.  It observes the *identity* of a landmark when
one is within radius (else a "none" token).  The query at each step is a landmark that has been seen before, and the
target is the displacement from the agent's current position **to where that landmark was seen**
(``x_{t_seen} - x_t``) -- i.e. "return to where you saw the red landmark", which is the relative-coordinate question
the algebra answers exactly.

Why it is a proxy: the doc's simulator adds goals and a policy.  The supervised version isolates the representation
question (can the arm compute a remembered relative vector?) with no RL in the way, which is the same
kill-before-spending pattern as the rest of the plan.  The task is translation-dominated by construction, which is
where docs/10 section 10.2 says the algebra is exact; SO(2) heading is deferred.

Batches (same keys as variants B and C):
    obs     (B, T) long    landmark id currently visible, or ``n_landmarks`` for none
    action  (B, T, 2)      the step taken at that time (must be integrated to know the pose)
    query   (B, T, L)      one-hot of the queried landmark (zero when the step is masked)
    target  (B, T, 2)      displacement to where the queried landmark was last seen
    mask    (B, T) bool    True where the query is defined
"""

import numpy as np
import torch

import feats


class LandmarkTask:
    name = "path"
    loss = "mse"

    def __init__(self, n_landmarks=6, ep_len=32, arena=4.0, radius=1.0, step_scale=0.5, min_age=8, seed=0):
        self.L = int(n_landmarks)
        self.ep_len = int(ep_len)
        self.arena = float(arena)
        self.radius = float(radius)
        self.step_scale = float(step_scale)
        self.min_age = int(min_age)
        self.seed = int(seed)

    @property
    def out_dim(self):
        return 2

    @property
    def query_dim(self):
        return self.L

    @property
    def in_dim(self):
        # one-hot(obs) + action(2) + query one-hot(L) + absolute-time features (fixed scale, not t/T)
        return (self.L + 1) + 2 + self.L + feats.TIME_DIM

    def batch(self, B, T=None, device="cpu", rng=None, step_scale=None, arena=None, **kw):
        rng = rng if rng is not None else np.random.default_rng(0)
        T = self.ep_len if T is None else T
        s = self.step_scale if step_scale is None else float(step_scale)
        A = self.arena if arena is None else float(arena)
        L = self.L
        pos0 = rng.uniform(-A, A, size=(B, 2))
        act = rng.uniform(-s, s, size=(B, T, 2))
        pos = pos0[:, None, :] + np.cumsum(act, axis=1)                      # (B, T, 2) agent positions
        land = rng.uniform(-A, A, size=(B, L, 2))                            # per row
        # landmark 0 sits next to the start, so every row has at least one remembered landmark from t=1 on
        land[:, 0, :] = pos0 + rng.uniform(-0.3, 0.3, size=(B, 2))
        d = np.linalg.norm(pos[:, :, None, :] - land[:, None, :, :], axis=-1)  # (B, T, L)
        vis = d <= self.radius
        obs = np.full((B, T), L, dtype=np.int64)
        any_vis = vis.any(-1)
        nearest = np.where(vis, d, np.inf).argmin(-1)
        obs[any_vis] = nearest[any_vis]
        # last time each landmark was seen strictly before t
        last_seen = np.full((B, L), -1, dtype=np.int64)
        query = np.zeros((B, T, L), dtype=np.float32)
        target = np.zeros((B, T, 2), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        rows = np.arange(B)
        for t in range(T):
            # A query is only asked about a landmark last seen at least ``min_age`` steps ago.  Without that
            # constraint the answer is usually just minus the previous action (the landmark was seen at t-1), which
            # is why RUN_007's proxy was uninformative (no-memory floor 0.024 vs best 0.013).  With an age floor the
            # target is a *displacement integrated over several steps*, which is the capability variant A exists to
            # test (docs/11 3.1; handoff 09 section 15.4 step 6).
            live = (last_seen >= 0) & (last_seen <= t - self.min_age)
            if live.any():
                u = np.where(live, rng.random((B, L)), -np.inf)
                choice = u.argmax(-1)
                seen_t = last_seen[rows, choice]
                ok = live[rows, choice]
                query[rows[ok], t, choice[ok]] = 1.0
                target[rows[ok], t] = pos[rows[ok], seen_t[ok]] - pos[rows[ok], t]
                mask[rows[ok], t] = True
            for l in range(L):     # update with what is visible at t (after the query, so "seen before")
                b_hit = vis[:, t, l]
                last_seen[b_hit, l] = t
        return {
            "obs": torch.as_tensor(obs, dtype=torch.long).to(device),
            "action": torch.as_tensor(act, dtype=torch.float32).to(device),
            "t": torch.arange(T).expand(B, T).to(device),
            "query": torch.as_tensor(query).to(device),
            "target": torch.as_tensor(target).to(device),
            "mask": torch.as_tensor(mask).to(device),
            "loss": self.loss,
            "meta": {"ep_len": T, "n_landmarks": L, "arena": A, "step_scale": s, "min_age": self.min_age},
        }

    def features(self, batch):
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"].clamp(max=self.L), self.L + 1).float()
        tfeat = feats.abs_time_feats(batch["t"])
        return torch.cat([oh, batch["action"], batch["query"], tfeat], dim=-1)
