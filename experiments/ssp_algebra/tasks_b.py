"""Variant B: iterated query / pointer chasing (docs/11 section 3.2).

A random function ``f`` over ``n_nodes`` nodes is presented as ``n_nodes`` bound pairs ``key -> value`` in a random
order, as a *stream* of tokens: ``[k_0, f(k_0), k_1, f(k_1), ...]`` followed by query tokens.  The query gives a
start node ``s`` and a hop count ``h``; the target is ``f^h(s)``.

Why this variant (docs/11 section 1.2, unique #2): the algebraic step is *identical at every hop*, so an arm that
learns "unbind once, clean up" should extrapolate to more hops for free.  Train at 1-2 hops, test at 4-8; a
cue-corruption axis zeroes a fraction of the query dimensions.

Batches (same keys as variants C and A, so the harness and every arm work unchanged):
    obs     (B, T) long    token stream: 2*n_nodes pair tokens then n_query query tokens (token id ``n_nodes``)
    query   (B, T) float   [start_node / n_nodes, hops / max_hops, visibility] -- zero at non-query steps
    target  (B, T) long    node index, only at query steps
    mask    (B, T) bool    True only at query steps
    t       (B, T) long    step index
"""

import numpy as np
import torch

import feats

IGNORE = -100

QUERY_TOKEN = -1  # set per instance to n_nodes


class PointerTask:
    name = "pointer"
    loss = "ce"

    def __init__(self, n_nodes=8, n_query=8, train_hops=(1, 2), max_hops=8, seed=0, chain=False):
        self.n_nodes = int(n_nodes)
        self.n_query = int(n_query)
        self.train_hops = tuple(int(h) for h in train_hops)
        self.max_hops = int(max_hops)
        self.seed = int(seed)
        # ``chain`` is the *equal-aid* protocol for the baseline arms (handoff 09 section 15.4 step 4): every query is
        # expanded into ``max_hops`` sub-steps with a step marker, and each sub-step carries the supervised target
        # f^k(s).  The SSP arm's aid is per-hop supervision of its own iteration; a baseline can only be compared
        # against it if it receives an equivalent aid, and this is the shape that gives it one without handing it the
        # true intermediate nodes (the arm sees the step marker, not the answer to the previous hop).
        self.chain = bool(chain)
        self.query_token = self.n_nodes
        # index of the queried hop (``hop / max_hops``) inside ``features``: [one-hot tokens (n+1), query, time].
        # A halting mechanism that must read the hop *from the observation* (rung 3) needs to name that channel; the
        # scale is then fixed, which matters because with hops 1-2 trained the learned decode's slope is not
        # identifiable from the task loss (see arms_learned.SSPPointerLearned, halting='counter').
        self.hop_feat_index = (self.n_nodes + 1) + 1

    @property
    def out_dim(self):
        return self.n_nodes

    @property
    def query_dim(self):
        # chain mode adds a per-sub-step marker (which hop of the chain this step is)
        return 4 if self.chain else 3

    @property
    def ep_len(self):
        return 2 * self.n_nodes + self.n_query * (self.max_hops if self.chain else 1)

    @property
    def in_dim(self):
        # tokens = nodes + query marker
        return self.n_nodes + 1 + self.query_dim + feats.TIME_DIM

    def batch(self, B, T=None, device="cpu", rng=None, hops=None, cue_drop=0.0, obs_noise=0.0, **kw):
        rng = rng if rng is not None else np.random.default_rng(0)
        T = self.ep_len if T is None else T
        n, nq = self.n_nodes, self.n_query
        # one random permutation per episode, and its composition powers f^1..f^max_hops
        perms = np.stack([rng.permutation(n) for _ in range(B)])                    # perms[b, k] = f(k)
        powers = [perms]                                                            # powers[h-1]
        for _ in range(self.max_hops - 1):
            powers.append(np.take_along_axis(perms, powers[-1], axis=1))
        powers = np.stack(powers)                                                   # (max_hops, B, n)
        # pair stream: (key_0, val_0, key_1, val_1, ...)
        order = np.stack([rng.permutation(n) for _ in range(B)])                    # the order pairs are shown in
        keys = order                                                               # (B, n)
        vals = np.take_along_axis(perms, keys, axis=1)                              # (B, n) = f(key)
        stream = np.empty((B, 2 * n), dtype=np.int64)
        stream[:, 0::2], stream[:, 1::2] = keys, vals
        start = np.stack([rng.integers(0, n, size=nq) for _ in range(B)])           # (B, nq)
        hop = np.stack([rng.choice(list(hops if hops is not None else self.train_hops), size=nq)
                        for _ in range(B)])                                         # (B, nq)
        tgt = np.zeros((B, nq), dtype=np.int64)
        for h in range(1, self.max_hops + 1):
            rows, cols = np.nonzero(hop == h)                      # (row, query) pairs asking for h hops
            if rows.size:
                tgt[rows, cols] = powers[h - 1][rows, start[rows, cols]]
        obs = np.concatenate([stream, np.full((B, nq), self.query_token, dtype=np.int64)], axis=1)
        T_eff = obs.shape[1]
        assert T_eff <= T, f"PointerTask ep_len {T_eff} does not fit requested T={T}"
        obs = obs[:, :T_eff]
        t = np.arange(T_eff)[None, :].repeat(B, 0)
        mask = np.zeros((B, T_eff), dtype=bool)
        mask[:, 2 * n:] = True
        target = np.full((B, T_eff), IGNORE, dtype=np.int64)
        target[:, 2 * n:2 * n + nq] = tgt
        query = np.zeros((B, T_eff, self.query_dim), dtype=np.float32)
        query[:, 2 * n:2 * n + nq, 0] = start / max(n - 1, 1)
        query[:, 2 * n:2 * n + nq, 1] = hop / self.max_hops
        query[:, 2 * n:2 * n + nq, 2] = 1.0                                          # query flag
        if cue_drop > 0:                                                             # corrupt the query dims
            keep = rng.random((B, nq, self.query_dim)) >= cue_drop
            query[:, 2 * n:2 * n + nq, :] *= keep
        ans_mask = mask.copy()
        if self.chain:
            obs, t, query, target, mask, ans_mask = self._expand_chain(
                obs, t, query, target, mask, powers, start, hop, nq, n, B)
        # true intermediate nodes f^1(s) .. f^h(s) per query (IGNORE past h), for per-hop supervision of a learned
        # rung: `inter[b, q, k] = f^(k+1)(s)`, valid while k < hop.  Rung 0 ignores this key.
        inter = np.full((B, nq, self.max_hops), IGNORE, dtype=np.int64)
        for k in range(self.max_hops):
            rows, cols = np.nonzero(hop > k)
            if rows.size:
                inter[rows, cols, k] = powers[k][rows, start[rows, cols]]
        if obs_noise > 0:
            # corrupt *input tokens only*, after every target has been computed from clean data (so the label of a
            # corrupted step is still what the clean stream implies: this measures robustness, not label noise)
            corrupt = rng.random((B, 2 * n)) < obs_noise
            obs[:, :2 * n] = np.where(corrupt, rng.integers(0, n, size=(B, 2 * n)), obs[:, :2 * n])
        out = {
            "obs": torch.as_tensor(obs, dtype=torch.long).to(device),
            "t": torch.as_tensor(t, dtype=torch.long).to(device),
            "target": torch.as_tensor(target, dtype=torch.long).to(device),
            "mask": torch.as_tensor(mask).to(device),
            "ans_mask": torch.as_tensor(ans_mask).to(device),
            "query": torch.as_tensor(query).to(device),
            "perms": torch.as_tensor(perms, dtype=torch.long).to(device),
            "start": torch.as_tensor(start, dtype=torch.long).to(device),
            "hops": torch.as_tensor(hop, dtype=torch.long).to(device),
            "inter": torch.as_tensor(inter, dtype=torch.long).to(device),
            "loss": self.loss,
            "meta": {"ep_len": T_eff, "n_pairs": n, "chain": self.chain, "n_nodes": n},
        }
        return out

    def _expand_chain(self, obs, t, query, target, mask, powers, start, hop, nq, n, B):
        """Chain expansion: query ``q`` becomes ``max_hops`` sub-steps, sub-step ``k`` labelled ``f^k(s)``.

        ``mask`` is True at every sub-step up to and including the query's own hop (so the *loss* sees the whole
        chain of supervised prefixes), while ``ans_mask`` is True only at the final sub-step: accuracy must be
        measured on the answer only, or an arm could look good by solving the easy first hops.  Sub-steps beyond a
        query's hop are masked out entirely (label IGNORE), so the padding carries no supervision.
        """
        H = self.max_hops
        obs_c = np.concatenate([obs[:, :2 * n], np.tile(obs[:, 2 * n:2 * n + nq, None], (1, 1, H)).reshape(B, nq * H)],
                               axis=1)
        T2 = obs_c.shape[1]
        t_c = np.arange(T2)[None, :].repeat(B, 0)
        q_c = np.zeros((B, T2, self.query_dim), dtype=np.float32)
        tg_c = np.full((B, T2), IGNORE, dtype=np.int64)
        m_c = np.zeros((B, T2), dtype=bool)
        ans_c = np.zeros((B, T2), dtype=bool)
        for q in range(nq):
            base = 2 * n + q * H
            src = 2 * n + q
            h = hop[:, q]
            for k in range(H):
                col = base + k
                valid = h > k
                if not valid.any():
                    continue
                q_c[valid, col, :3] = query[valid, src, :3]                        # start, hop, query flag
                q_c[valid, col, 3] = (k + 1) / H                                   # the step marker
                m_c[valid, col] = True
                tg_c[valid, col] = powers[k][np.nonzero(valid)[0], start[valid, q]]
                ans_c[valid & (h == k + 1), col] = True
        return obs_c, t_c, q_c, tg_c, m_c, ans_c

    def features(self, batch):
        B, T = batch["obs"].shape
        tok = batch["obs"].clamp(0, self.n_nodes)          # query marker -> last slot
        oh = torch.nn.functional.one_hot(tok, self.n_nodes + 1).float()
        return torch.cat([oh, batch["query"], feats.abs_time_feats(batch["t"])], dim=-1)
