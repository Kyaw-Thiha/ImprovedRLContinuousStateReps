"""Variant C tasks: learned-delta read and index-addressed recall (docs/11 section 3.3).

Both are supervised sequence tasks with a *query per step* and a teacher-forced target per step, so all arms get
the same interface (see ``harness.py``):

    delay : output the symbol seen ``lag`` steps ago.  ``lag`` is a per-episode scalar that every arm receives as
            an input feature, so "which lag?" is answerable by all arms and the SSP arm's job is to turn it into a
            continuous read offset with one gradient-trained parameter.  Train on short lags, test on long ones.
    index : output the symbol that was observed at the given absolute index ``j``.  No learned offset: the read is
            ``M (*) conj(pos(j))``.

This is deliberately the *substitutable* variant (docs/11 section 1.1): a diagonal linear recurrence and a
kernel-attention read with a learned delta are both expected to match it.  Its role is to validate the harness.

The real POPGym RepeatPrevious is a special case (single fixed lag = k-1, no query feature -- see
``popgym_vsa/tasks.py``).  The synthetic generator is used instead of the env because the train-short/test-long axis
needs a lag the caller chooses, and because the env's per-episode ``k`` is fixed at construction.

Layout of a batch (all tensors on ``device``):
    obs     (B, T) long      symbol observed at each step
    query   (B, T) float     the requested lag / index, normalised by ``T`` (0 before the first valid step)
    target  (B, T) long      class index to predict; ``IGNORE`` where ``mask`` is False
    mask    (B, T) bool      steps contributing to the loss
    t       (B, T) long      step index
    index   (B, T) long      index task only: the absolute index ``j`` (so the SSP arm reads at ``j``, not ``j*T``)

The query and the clock are at a *fixed* scale (``feats.NORM`` and ``feats.abs_time_feats``), so a feature means the
same thing at ``T = 16`` and at ``T = 64`` (handoff 09 section 5.3-5.4); RUN_007's ``/T`` versions were not fair
between arms and were not reusable across episode lengths.
"""

import torch

import feats

IGNORE = -100


class DelayTask:
    """Repeat-previous: predict the symbol from ``lag`` steps back; ``lag`` is given as an input feature."""

    name = "delay"
    loss = "ce"

    def __init__(self, n_items=8, ep_len=32, train_lags=(1, 2, 3, 4), seed=0):
        self.n_items = int(n_items)
        self.ep_len = int(ep_len)
        self.train_lags = tuple(int(l) for l in train_lags)
        self.seed = int(seed)

    # --- interface used by harness ---
    @property
    def out_dim(self):
        return self.n_items

    @property
    def query_dim(self):
        return 1

    def batch(self, B, T, device="cpu", rng=None, lags=None, **kw):
        rng = rng if rng is not None else np.random.default_rng(0)
        lags = self.train_lags if lags is None else tuple(lags)
        lag = torch.as_tensor(rng.choice(list(lags), size=B), dtype=torch.long)
        obs = torch.as_tensor(rng.integers(0, self.n_items, size=(B, T)), dtype=torch.long)
        t = torch.arange(T).expand(B, T)
        idx = t - lag[:, None]
        valid = idx >= 0
        target = torch.where(valid, obs.gather(1, idx.clamp(min=0)), torch.full_like(obs, IGNORE))
        query = (lag[:, None].float() / feats.NORM).expand(B, T) * valid.float()
        return {"obs": obs.to(device), "t": t.to(device), "target": target.to(device),
                "mask": valid.to(device), "query": query.to(device), "lag": lag.to(device),
                "loss": self.loss, "meta": {"ep_len": T, "lags": lags}}

    def features(self, batch):
        """Standard input featurisation shared by every non-SSP arm: one-hot(obs) || normalised query."""
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], self.n_items).float()
        return torch.cat([oh, batch["query"][..., None], feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return self.n_items + self.query_dim + feats.TIME_DIM


class IndexTask:
    """Index-addressed recall: predict the symbol observed at the given absolute index ``j``."""

    name = "index"
    loss = "ce"

    def __init__(self, n_items=8, ep_len=32, seed=0):
        self.n_items = int(n_items)
        self.ep_len = int(ep_len)
        self.seed = int(seed)

    @property
    def out_dim(self):
        return self.n_items

    @property
    def query_dim(self):
        return 1

    def batch(self, B, T, device="cpu", rng=None, lags=None, jmax=None, **kw):
        rng = rng if rng is not None else torch.Generator().manual_seed(0)
        obs = torch.as_tensor(rng.integers(0, self.n_items, size=(B, T)), dtype=torch.long)
        t = torch.arange(T).expand(B, T)
        # Strictly in the past.  `j = min(j, t)` (an earlier version) asked for the *current* observation
        # whenever the sampled index exceeded t, which a memoryless model answers for free: the no-memory floor
        # was 0.597 instead of chance.  A row-constant j in [0, T-2] with `valid = t > j` removes that leak.
        hi = max(T - 1, 1) if jmax is None else max(min(int(jmax) + 1, T - 1), 1)   # curriculum: cap the index
        j = torch.as_tensor(rng.integers(0, hi, size=(B,)), dtype=torch.long)[:, None].expand(B, T)
        valid = t > j
        target = torch.where(valid, obs.gather(1, j.clamp(min=0)), torch.full_like(obs, IGNORE))
        return {"obs": obs.to(device), "t": t.to(device), "target": target.to(device),
                "mask": valid.to(device), "query": (j.float() / feats.NORM).to(device),
                "index": j.to(device), "lag": j[:, 0].to(device),
                "loss": self.loss, "meta": {"ep_len": T}}

    def features(self, batch):
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], self.n_items).float()
        return torch.cat([oh, batch["query"][..., None], feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return self.n_items + self.query_dim + feats.TIME_DIM


TASKS = {"delay": DelayTask, "index": IndexTask}


def make_task(name, **kw):
    return TASKS[name](**kw)


def to_rng_numpy(seed):
    """The harness uses numpy RNGs for generation and torch generators for weights; be explicit about which."""
    import numpy as np

    return np.random.default_rng(seed)
