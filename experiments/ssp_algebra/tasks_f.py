"""Delay recall with distractors: selective memory (RUN_012 step 2; docs/11 section 14.5).

The delay task without distractors is solved by any recency mechanism, so it cannot separate a memory that *chooses
what to store* from one that stores everything.  Here a fraction of steps are marked **distractors**: their symbols
are drawn from the same alphabet, they are flagged in the features, and they are **never** an answer.  The target at
step ``t`` is the symbol at the ``lag``-th *content* (non-distractor) step back, so the distractors sit between the
query and the answer and must be ignored.

Why this task: the two supervised winners of RUN_012 are input-dependent gates (1.000 on index) and multi-timescale
banks (2x fewer tokens), and `docs/11` section 14.3 says gates should matter when the memory must *not* store
everything.  No candidate has been measured on a task that asks for that.

Alphabet discipline: every split keeps `obs` in ``[0, n_items)`` and `out_dim = n_items``.  A split that changed the
alphabet broke the content family twice (RUN_012 section 4), so the difficulty axes here are the distractor rate and
the stream length, never the symbol set.

Interface (same as `tasks_c.DelayTask`):
    obs (B,T) long, t (B,T) long, target (B,T) long, mask (B,T) bool, query (B,T) float (the lag, normalised),
    lag (B,) long, is_dist (B,T) bool, loss "ce", meta
"""

import numpy as np
import torch

import feats

IGNORE = -100


class DelayDistractTask:
    """Predict the symbol ``lag`` *content* steps back; marked distractor steps are never answers."""

    name = "delay_distract"
    loss = "ce"

    def __init__(self, n_items=8, ep_len=32, train_lags=(1, 2, 3, 4), p_dist=0.25, seed=0):
        self.n_items = int(n_items)
        self.ep_len = int(ep_len)
        self.train_lags = tuple(int(l) for l in train_lags)
        self.p_dist = float(p_dist)
        self.seed = int(seed)

    @property
    def out_dim(self):
        return self.n_items

    @property
    def query_dim(self):
        return 1

    def batch(self, B, T, device="cpu", rng=None, lags=None, p_dist=None, cue_drop=0.0, obs_noise=0.0, **kw):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.seed)
        lags = self.train_lags if lags is None else tuple(int(l) for l in lags)
        p = self.p_dist if p_dist is None else float(p_dist)
        T = int(T)
        lag = torch.as_tensor(rng.choice(list(lags), size=B), dtype=torch.long)
        obs = rng.integers(0, self.n_items, size=(B, T))
        is_dist = rng.random((B, T)) < p
        is_dist[:, :1] = False                      # the first step is always content: it may be an answer later
        target = np.full((B, T), IGNORE, dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        for b in range(B):
            content = []                            # positions of content steps seen so far
            for t in range(T):
                if not is_dist[b, t]:
                    l = int(lag[b])
                    if len(content) >= l:
                        ans = content[-l]           # the lag-th content step back, skipping distractors
                        target[b, t] = obs[b, ans]
                        mask[b, t] = True
                    content.append(t)
        if cue_drop:                                # corrupt the *observation* of the answer steps only
            drop = rng.random((B, T)) < float(cue_drop)
            drop &= ~is_dist
            alt = rng.integers(0, self.n_items, size=(B, T))
            obs = np.where(drop, alt, obs)
        if obs_noise:
            flip = rng.random((B, T)) < float(obs_noise)
            obs = np.where(flip, rng.integers(0, self.n_items, size=(B, T)), obs)
        t = np.arange(T)[None, :].repeat(B, 0)
        valid = (lag[:, None].float() <= torch.as_tensor(
            np.cumsum(~is_dist, axis=1) - 1, dtype=torch.float)).numpy()   # mask is authoritative; kept for parity
        return {"obs": torch.as_tensor(obs).to(device), "t": torch.as_tensor(t).to(device),
                "target": torch.as_tensor(target).to(device), "mask": torch.as_tensor(mask).to(device),
                "query": (lag[:, None].float() / feats.NORM).expand(B, T).contiguous().to(device),
                "lag": lag.to(device), "is_dist": torch.as_tensor(is_dist).to(device),
                "loss": self.loss, "meta": {"ep_len": T, "lags": lags, "p_dist": p}}

    def features(self, batch):
        """Identical for every arm: one-hot(obs) || lag (normalised) || is_distractor || absolute-time features."""
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], self.n_items).float()
        return torch.cat([oh, batch["query"][..., None], batch["is_dist"].float()[..., None],
                          feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return self.n_items + 2 + feats.TIME_DIM


class DelayNoLagTask(DelayDistractTask):
    """The same delay recall, with the **lag feature removed from the input**.

    Why (handoffs/11 step 2b): every supervised delay/index cell in this project hands the arm the lag (or index) as
    an input feature, so the arm never has to *discover which offset to read at* -- it only has to learn a map from a
    given number to an address.  RUN_012's RL decomposition (section 6) showed that is exactly the half of the
    discovery problem the supervised tasks were blind to: on `RepeatPrevious` no lag is given, tying alone left the
    arm at -0.133, and *setting* the offset reached +0.832.  This variant makes the supervised setting reproduce
    that condition, cheaply and with 10 seeds, so a candidate for the lag-selection problem (a bank of read offsets
    with a learned selector) can be judged before spending RL compute.

    Deliberately a **separate variant**, not a split: removing a feature changes ``in_dim``, and a split that changes
    the input width is the same class of breakage as a split that changes the alphabet (RUN_012 section 4).  A run is
    trained and evaluated lag-free end to end, so the arm is built with the lag-free width.

    ``query`` is still present in the batch (zeros) because the harness interface requires the key; the one-hot
    observation, the distractor flag and the absolute-time features are unchanged, so the only missing information
    is "which lag".
    """

    name = "delay_nolag"

    def __init__(self, n_items=8, ep_len=32, train_lags=(1, 2, 3, 4), p_dist=0.0, seed=0):
        super().__init__(n_items=n_items, ep_len=ep_len, train_lags=train_lags, p_dist=p_dist, seed=seed)

    def batch(self, B, T, device="cpu", **kw):
        b = super().batch(B, T, device=device, **kw)
        # The arm reads the lag from ``batch["lag"]`` (arms_learned.py ``_read_addr``), NOT from the feature column;
        # zeroing only ``query`` leaves the lag fully visible and the task is not lag-free at all (caught by
        # test_delay_nolag_gives_the_arm_no_lag_channel, which asks the arm itself).  Targets and mask are already
        # materialised by super(), so the true lag is no longer needed.
        b["lag"] = torch.zeros_like(b["lag"])
        b["query"] = torch.zeros_like(b["query"])
        return b

    def features(self, batch):
        """one-hot(obs) || is_distractor || absolute-time features -- no lag column."""
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], self.n_items).float()
        return torch.cat([oh, batch["is_dist"].float()[..., None],
                          feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return self.n_items + 1 + feats.TIME_DIM


TASKS = {"delay_distract": DelayDistractTask, "delay_nolag": DelayNoLagTask}
