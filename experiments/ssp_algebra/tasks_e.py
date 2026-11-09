"""Spatial recall: landmarks written once, motion applied to the whole memory, read in the current frame.

Handoff 11 step 3 / `docs/11` section 14.5 -- **the gate for the robotics direction**.  Variant A (`tasks_a`) was the
first attempt and it failed (the hand-wired path arm's MSE 0.147 against the GRU's 0.049), but that test had two
weaknesses this one removes: the landmark identity had to be re-recognised from intermittent observations, and the
answer was the displacement to *where the landmark was seen* rather than the landmark's position in the current frame,
so a plain integrator could pass without ever moving a stored item.

Construction (all translation, no rotation -- `docs/10` section 10.2 says the algebra is exact there):

  * ``K`` landmarks are shown once each, in the first ``K`` steps, as a **current-frame position** (what a sensor
    gives).  Their world positions never appear;
  * every step carries a 2-D egocentric **motion increment** ``d_t``, and the pose is ``p_t = sum_{s<=t} d_s``;
  * from step ``K`` on, a query names one landmark and the target is that landmark **in the current frame**,
    ``x_seen - (p_t - p_seen)``.

So the memory must hold ``K`` items and apply one accumulated rigid motion to all of them at read time -- which is
exactly the capability ``docs/11`` section 14.4 lists as potentially unique to SSP ("motion update by binding:
translate the whole memory by ``phi(-d)`` instead of rewriting slots"), and the one no testbed in this repo covered.
An SSP arm does it with a single element-wise multiplication per step, independent of how many landmarks are held;
the honest competitor is any recurrent arm, which must integrate the motion and re-express the answer.

Batches (keys, shapes and layout chosen to match variants A/B/C so the same dump/analysis path works):
    lag     (B, T) long      steps back to the queried landmark's write step (0 where unqueried); the positional
                             form of the query, for memories whose keys are time features
    obs     (B, T) long      landmark shown at this step, or ``K`` for "none"
    action  (B, T, 2)        egocentric motion increment of this step (float)
    lpos    (B, T, 2)        the shown landmark's *current-frame* position (0 where nothing is shown)
    query   (B, T, K)        one-hot of the queried landmark (zero at unqueried steps)
    target  (B, T, 2)        the queried landmark in the current frame
    mask    (B, T) bool      True where the query is defined
"""

import numpy as np
import torch

import feats


class SpatialRecallTask:
    name = "spatial"
    loss = "mse"

    def __init__(self, n_landmarks=3, ep_len=32, arena=2.0, step_scale=0.15, n_queries=None, seed=0):
        self.K = int(n_landmarks)
        self.ep_len = int(ep_len)
        self.arena = float(arena)
        self.step_scale = float(step_scale)
        # queries fill the steps after the write region
        self.n_queries = int(n_queries) if n_queries else max(self.ep_len - self.K, 1)
        self.seed = int(seed)

    @property
    def out_dim(self):
        return 2

    @property
    def query_dim(self):
        return self.K

    @property
    def in_dim(self):
        # one-hot(obs, K+1) + action(2) + lpos(2) + query one-hot(K) + absolute-time features
        return (self.K + 1) + 2 + 2 + self.K + feats.TIME_DIM

    def batch(self, B, T=None, device="cpu", rng=None, step_scale=None, arena=None, obs_noise=0.0, **kw):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.seed)
        T = int(T if T is not None else self.ep_len)
        K = self.K
        scale = float(step_scale) if step_scale is not None else self.step_scale
        arena = float(arena) if arena is not None else self.arena

        obs = np.full((B, T), K, dtype=np.int64)                 # K = "nothing shown"
        action = np.zeros((B, T, 2), dtype=np.float32)
        lpos = np.zeros((B, T, 2), dtype=np.float32)
        query = np.zeros((B, T, K), dtype=np.float32)
        target = np.zeros((B, T, 2), dtype=np.float32)
        mask = np.zeros((B, T), dtype=bool)
        lag = np.zeros((B, T), dtype=np.int64)                   # steps back to the queried landmark's write step

        for b in range(B):
            # landmark positions in the *world* frame (never observed directly) and the shown current-frame positions
            Lw = rng.uniform(-arena, arena, size=(K, 2))
            d = rng.normal(0.0, scale, size=(T, 2)).astype(np.float32)
            d[0] = 0.0                                            # step 0 defines the initial frame
            p = np.cumsum(d, axis=0)                              # pose in the world frame at each step
            action[b] = d
            for k in range(min(K, T)):
                obs[b, k] = k
                lpos[b, k] = (Lw[k] - p[k]).astype(np.float32)     # the sensor gives it in the current frame
            for j in range(self.n_queries):
                t = K + j
                if t >= T:
                    break
                k = int(rng.integers(0, K))
                query[b, t, k] = 1.0
                # the same landmark, re-expressed in the frame at t: undo the motion accumulated since it was shown
                target[b, t] = (lpos[b, k] - (p[t] - p[k])).astype(np.float32)
                mask[b, t] = True
                # For a *positional* memory (`FWPArm`, whose keys are time features) the query must be a lag, not a
                # role: reading at the absolute time `t - lag` is reading at the step the landmark was written.  This
                # is the same convention `delay` and `pointer` use, and it gives each memory the query in the form
                # that memory natively addresses by -- matched aid, not a thumb on the scale.
                lag[b, t] = t - k
        if obs_noise:                                             # corrupt the shown *positions* only
            flip = rng.random((B, T, 1)) < float(obs_noise)
            lpos = np.where(flip, rng.uniform(-arena, arena, size=(B, T, 2)), lpos).astype(np.float32)
        t = np.arange(T)[None, :].repeat(B, 0)
        return {"obs": torch.as_tensor(obs).to(device), "t": torch.as_tensor(t).to(device),
                "action": torch.as_tensor(action).to(device),
                "lpos": torch.as_tensor(lpos).to(device),
                "query": torch.as_tensor(query).to(device),
                "target": torch.as_tensor(target).to(device),
                "mask": torch.as_tensor(mask).to(device),
                "lag": torch.as_tensor(lag).to(device),
                "loss": self.loss, "meta": {"ep_len": T, "n_landmarks": K, "n_queries": self.n_queries}}

    def features(self, batch):
        """Identical for every non-SSP arm: one-hot(obs) || motion || shown position || query one-hot || time."""
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], self.K + 1).float()
        return torch.cat([oh, batch["action"], batch["lpos"], batch["query"],
                          feats.abs_time_feats(batch["t"])], dim=-1)


TASKS = {"spatial": SpatialRecallTask}
