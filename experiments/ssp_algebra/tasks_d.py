"""Content-keyed associative recall: the *non-positional* family (handoff 11 step 1, docs/11 section 14.3).

Why this task exists.  `index` and `delay` are both addressed by *position*: the write address is the time step and
the query asks for an absolute index or a lag, so a design that happens to suit position is never challenged
(docs/11 section 1.1 lists key-value association as *substitutable* and this is where that has to be settled).  Here
nothing is addressed by position:

    pairs   : ``n_pairs`` key->value pairs are presented in a random order, adjacent (even step = key, odd = value)
    queries : every remaining step asks for the value of a *specific key* (the key is the step's observation)
    target  : the value bound to that key; only the query steps contribute to the loss

An arm that succeeds must store an association and retrieve it *by content*.  A positional memory has no advantage,
and the SSP arm here uses the classical VSA construction (bundle of ``bind(key, value)``, read by unbinding the
query key) rather than a phase ramp, so this task also asks whether the *algebra* is load-bearing: the
``AssocSSPArm(bind=False)`` ablation bundles key and value additively instead of binding them.

Interface is the same as ``tasks_c`` (see its module docstring):
    obs (B,T) long (keys 0..P-1, values P..2P-1), target (B,T) long, mask (B,T) bool, t (B,T) long,
    query (B,T) float (1.0 on query steps), is_query (B,T) bool, and ``key`` (B,T) long for the content-addressed arm.
"""

import numpy as np
import torch

import feats

IGNORE = -100


class AssocTask:
    """Content-keyed recall with a single alphabet: keys are ``0..P-1``, values are ``P..2P-1``."""

    name = "assoc"
    loss = "ce"

    def __init__(self, n_pairs=4, ep_len=16, seed=0, n_queries=None):
        self.n_pairs = int(n_pairs)
        self.ep_len = int(ep_len)
        self.seed = int(seed)
        # queries fill whatever is left after the pair region; keep at least 1
        self.n_queries = int(n_queries) if n_queries else max(self.ep_len - 2 * self.n_pairs, 1)

    @property
    def out_dim(self):
        """The value alphabet only: the model reports a value, never a key."""
        return self.n_pairs

    @property
    def query_dim(self):
        return 1

    def _value_offset(self):
        return self.n_pairs

    def batch(self, B, T, device="cpu", rng=None, lags=None, n_pairs=None, n_distract=0, cue_drop=0.0, obs_noise=0.0,
              **kw):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.seed)
        P = int(n_pairs) if n_pairs else self.n_pairs
        T = int(T)
        nd = max(int(n_distract), 0)              # extra *distractor* pairs: same alphabets, never queried
        nq = max(T - 2 * (P + nd), 1)
        obs = np.zeros((B, T), dtype=np.int64)
        target = np.full((B, T), IGNORE, dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        is_q = np.zeros((B, T), dtype=bool)
        key = np.zeros((B, T), dtype=np.int64)
        for b in range(B):
            vals = rng.permutation(P)                      # each key gets a distinct value
            order = rng.permutation(P)                     # presentation order is irrelevant to the answer
            for i, k in enumerate(order):
                obs[b, 2 * i] = k
                obs[b, 2 * i + 1] = P + vals[k]
            for i in range(nd):                    # distractors: irrelevant associations from the same alphabets
                obs[b, 2 * (P + i)] = rng.integers(0, P)
                obs[b, 2 * (P + i) + 1] = P + rng.integers(0, P)
            # queries: keys in a random order, positions after the pair region
            qkeys = rng.permutation(P)
            for j in range(nq):
                k = qkeys[j % P]
                if cue_drop and rng.random() < float(cue_drop):
                    k = rng.integers(0, P)                 # corrupt the *query key* only (targets stay clean)
                st = 2 * (P + nd) + j
                obs[b, st] = k
                key[b, st] = k
                target[b, st] = P + vals[k]                # a value id in [P, 2P); out_dim maps it to [0, P)
                mask[b, st] = True
                is_q[b, st] = True
        if obs_noise:
            flip = rng.random((B, T)) < float(obs_noise)
            obs = np.where(flip, rng.integers(0, 2 * P, size=(B, T)), obs)
        t = np.arange(T)[None, :].repeat(B, 0)
        tgt = target - P                                    # values are reported in [0, P)
        tgt[~mask] = IGNORE
        tgt = np.where(mask, tgt, IGNORE)
        return {"obs": torch.as_tensor(obs).to(device), "t": torch.as_tensor(t).to(device),
                "target": torch.as_tensor(tgt).to(device), "mask": torch.as_tensor(mask).to(device),
                "query": torch.as_tensor(is_q.astype(np.float32)).to(device),
                "is_query": torch.as_tensor(is_q).to(device),
                "key": torch.as_tensor(key).to(device), "n_pairs": P, "n_pair_steps": int(2 * (P + nd)),
                "loss": self.loss, "meta": {"ep_len": T, "n_pairs": P, "n_distract": nd, "n_queries": nq}}

    def features(self, batch):
        """Identical for every non-SSP arm: one-hot(obs) || is_query || the shared absolute-time features.

        The time features are still handed over (every variant in this folder does), but they carry no addressing
        information here -- the pairs arrive in a random order -- so an arm cannot use them as an address.
        """
        B, T = batch["obs"].shape
        P = batch["n_pairs"]
        oh = torch.nn.functional.one_hot(batch["obs"], 2 * P).float()
        return torch.cat([oh, batch["query"][..., None], feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return 2 * self.n_pairs + 1 + feats.TIME_DIM


class AssocMixedTask(AssocTask):
    """Content-plus-position keys: each query asks *either* by content *or* by index (handoff 11 step 6).

    Why.  `assoc` is content-only and `index`/`delay` are position-only, so every family so far lets an arm pick one
    addressing convention and never have to hold both.  A deployable memory in a real task is rarely purely one: "the
    value I stored under *this object*" and "the value I stored *third*" are different questions about the same store.
    This variant asks both, in one episode, against the same bundle.

    Construction.  The pair region is written as in `assoc` (position ``i`` holds key ``key_of_pos[i]``, key ``k``
    holds value ``val_of_key[k]``).  Query steps alternate:
      * **content mode** (``mode = +1``): ``obs`` is the *key* symbol; the arm must use the key.
      * **index mode** (``mode = -1``): ``obs`` is the *pair position* ``j``; the arm must use position ``j``.
    Both ask the same question about the same pair, so the target is identical and the *only* signal that distinguishes
    the two meanings of ``obs`` is the mode column.  That matters: ``obs`` for an index query is a symbol in ``[0, P)``,
    which is also the key alphabet, so an arm that ignores ``mode`` is provably using the wrong addressing for half the
    queries.

    ``features`` gains the ``mode`` column over `assoc` (``in_dim = 2P + 2 + TIME_DIM``), which is why this is a
    separate variant rather than a split.
    """

    name = "assoc_mixed"
    loss = "ce"

    def batch(self, B, T, device="cpu", rng=None, n_pairs=None, n_distract=0, cue_drop=0.0, obs_noise=0.0, **kw):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.seed)
        P = int(n_pairs) if n_pairs else self.n_pairs
        T = int(T)
        nd = max(int(n_distract), 0)
        nq = max(T - 2 * (P + nd), 1)
        obs = np.zeros((B, T), dtype=np.int64)
        target = np.full((B, T), IGNORE, dtype=np.int64)
        mask = np.zeros((B, T), dtype=bool)
        is_q = np.zeros((B, T), dtype=bool)
        key = np.zeros((B, T), dtype=np.int64)
        mode = np.ones((B, T), dtype=np.float32)                 # +1 content, -1 index; non-query steps stay +1
        for b in range(B):
            key_of_pos = rng.permutation(P)                      # position i holds this key
            val_of_key = rng.permutation(P)                      # key k holds this value
            prev_pos = -1
            for i, k in enumerate(key_of_pos):
                obs[b, 2 * i] = k
                obs[b, 2 * i + 1] = P + val_of_key[k]
            for i in range(nd):
                obs[b, 2 * (P + i)] = rng.integers(0, P)
                obs[b, 2 * (P + i) + 1] = P + rng.integers(0, P)
            for j in range(nq):
                # Each query's pair is drawn **independently**: pairing the two modes on the same pair would let an
                # index-mode query be answered from the content-mode query about the same pair on the previous step
                # (measured: the `pos=False` ablation still scored 0.736 on index queries when they were paired).
                pos = int(rng.integers(0, P))
                if j:
                    while pos == prev_pos:                # never ask about the same pair twice in a row
                        pos = int(rng.integers(0, P))
                prev_pos = pos
                k = int(key_of_pos[pos])
                st = 2 * (P + nd) + j
                if cue_drop and rng.random() < float(cue_drop):
                    # corrupt the *query* only: a content query gets a wrong key, an index query a wrong position
                    pos = int(rng.integers(0, P))
                    k = int(key_of_pos[pos])
                if j % 2 == 0:                                    # content query
                    obs[b, st] = k
                    mode[b, st] = 1.0
                else:                                             # index query: the position, not the key
                    obs[b, st] = pos
                    mode[b, st] = -1.0
                key[b, st] = k
                target[b, st] = val_of_key[k]
                mask[b, st] = True
                is_q[b, st] = True
        if obs_noise:
            flip = rng.random((B, T)) < float(obs_noise)
            obs = np.where(flip, rng.integers(0, 2 * P, size=(B, T)), obs)
        t = np.arange(T)[None, :].repeat(B, 0)
        tgt = np.where(mask, target, IGNORE)
        return {"obs": torch.as_tensor(obs).to(device), "t": torch.as_tensor(t).to(device),
                "target": torch.as_tensor(tgt).to(device), "mask": torch.as_tensor(mask).to(device),
                "query": torch.as_tensor(is_q.astype(np.float32)).to(device),
                "is_query": torch.as_tensor(is_q).to(device),
                "key": torch.as_tensor(key).to(device), "mode": torch.as_tensor(mode).to(device),
                "n_pairs": P, "n_pair_steps": int(2 * (P + nd)),
                "loss": self.loss, "meta": {"ep_len": T, "n_pairs": P, "n_distract": nd, "n_queries": nq}}

    def features(self, batch):
        """one-hot(obs) || is_query || mode (+1 content / -1 index) || the shared absolute-time features."""
        B, T = batch["obs"].shape
        oh = torch.nn.functional.one_hot(batch["obs"], 2 * self.n_pairs).float()
        mode = batch.get("mode")
        if mode is None:
            mode = torch.ones(B, T, device=oh.device)
        return torch.cat([oh, batch["query"][..., None], mode[..., None],
                          feats.abs_time_feats(batch["t"])], dim=-1)

    @property
    def in_dim(self):
        return 2 * self.n_pairs + 2 + feats.TIME_DIM


TASKS = {"assoc": AssocTask, "assoc_mixed": AssocMixedTask}
