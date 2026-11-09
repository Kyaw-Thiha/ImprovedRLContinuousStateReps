"""Policies for the POPGym memory tasks, all behind one recurrent interface:

    state = model.initial_state(B, device)
    logits, value, state = model.step(obs_t, t_vec, state)
    state = model.reset_state(state, done_mask)     # episode boundaries
    state = model.detach_state(state)               # bounds BPTT memory to a fixed window

The baselines are the field's own, from POPGym's published comparison set: a plain MLP (the non-memory control), a
**positional MLP** (the mandatory baseline -- POPGym found it nearly solves masked stateless CartPole), and a GRU
(their strongest reported model).  The VSA arms use ``memory.TorchMemory``: ``vsa_count`` reads the count directly
out of the running bundle, ``vsa_delay`` decodes the item from k steps ago by unbinding the position code.
"""

import torch
import torch.nn as nn

from memory import TorchMemory, obs_onehot, time_feats

MODELS = ("mlp", "posmlp", "gru", "vsa_count", "vsa_delay")


class Policy(nn.Module):
    def __init__(self, name, spec, dim=256, hidden=64, k_max=256, code="random", cleanup=False, seed=0):
        super().__init__()
        assert name in MODELS, f"unknown model {name}"
        if name == "vsa_count":
            assert len(spec["cards"]) == 2, "vsa_count needs a (value, query) observation (CountRecall)"
        if name == "vsa_delay":
            assert spec["lag"] > 0 or spec["k"] > 1, "vsa_delay needs a nonzero lag (RepeatPrevious)"
        self.name = name
        self.spec = spec
        self.cards = spec["cards"]
        self.ep_len = spec["ep_len"]
        self.act_dim = spec["act_dim"]
        self.onehot_dim = sum(self.cards)
        self.mem = None

        if name == "mlp":
            feat_dim = self.onehot_dim
        elif name == "posmlp":
            feat_dim = self.onehot_dim + (1 + 8)  # 4 harmonics -> 1 + 2*4
        elif name == "gru":
            self.rnn = nn.GRU(self.onehot_dim, hidden, batch_first=True)
            feat_dim = hidden
        elif name == "vsa_count":
            self.mem = TorchMemory(spec["n_items"], dim, k_max, lag=spec["lag"], code=code, seed=seed)
            # The raw readout is one scalar; turning it into one of `act_dim` counts is a rounding staircase, which
            # a small head learns slowly.  `cleanup=True` adds the standard VSA nearest-neighbour decode
            # exp(-(est-k)^2/2) over candidate counts, so the answer is (nearly) linear in the features.
            self.cleanup = cleanup
            feat_dim = 1 + (self.act_dim if cleanup else 0) + self.onehot_dim
        else:  # vsa_delay
            self.mem = TorchMemory(spec["n_items"], dim, k_max, lag=spec["lag"], code=code, seed=seed)
            feat_dim = spec["n_items"] + self.onehot_dim  # cleanup scores + current one-hot

        self.trunk = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, self.act_dim)
        self.v = nn.Linear(hidden, 1)

    # --- interface ---------------------------------------------------------------------------------------
    def initial_state(self, batch, device="cpu"):
        if self.name == "gru":
            return torch.zeros(1, batch, self.rnn.hidden_size, device=device)
        if self.name in ("vsa_count", "vsa_delay"):
            return self.mem.zero_state(batch)
        return None

    def reset_state(self, state, done):
        """Zero the state rows of terminated environments (episodes are handled in lockstep, so this is per-row).

        The GRU keeps its state as (layers, batch, hidden) while the VSA memories use (batch, dim), so the mask has
        to be shaped per model -- broadcasting a (B,1,1) mask into a (1,B,H) state silently produces (B,B,H).
        """
        if state is None:
            return None
        keep = (1.0 - done.float())
        if self.name == "gru":
            return state * keep.view(1, -1, 1)
        return state * keep.view(-1, *([1] * (state.dim() - 1)))

    def detach_state(self, state):
        return state.detach() if state is not None else None

    def step(self, obs_t, t_vec, state):
        if self.name == "mlp":
            feats, new_state = obs_onehot(obs_t, self.cards), None
        elif self.name == "posmlp":
            feats = torch.cat([obs_onehot(obs_t, self.cards), time_feats(t_vec, self.ep_len)], dim=-1)
            new_state = None
        elif self.name == "gru":
            x = obs_onehot(obs_t, self.cards).unsqueeze(1)  # (B, 1, F)
            out, h = self.rnn(x, state)
            feats, new_state = out[:, 0], h
        elif self.name == "vsa_count":
            est, new_state = self.mem.count_step(state, obs_t)
            parts = [est[:, None]]
            if self.cleanup:
                ks = torch.arange(self.act_dim, dtype=est.dtype, device=est.device)
                parts.append(torch.exp(-((est[:, None] - ks[None, :]) ** 2) / 2.0))
            parts.append(obs_onehot(obs_t, self.cards))
            feats = torch.cat(parts, dim=-1)
        else:
            scores, new_state = self.mem.delay_step(state, obs_t, t_vec)
            feats = torch.cat([scores, obs_onehot(obs_t, self.cards)], dim=-1)

        h = self.trunk(feats)
        return self.pi(h), self.v(h).squeeze(-1), new_state


def build_model(name, spec, **kw):
    return Policy(name, spec, **kw)
