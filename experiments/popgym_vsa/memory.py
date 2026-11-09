"""VSA (FHRR) memory primitives for discrete POPGym observations.

The encoder thread (continuous SSP) is not what these tasks test: their observations are discrete symbols.  What is
tested is the *algebra* -- bundle as storage, unbind as retrieval:

  * CountRecall  : the running bundle of item codes IS the count vector; one inner product with the query code
                   yields the count.  No learned memory parameters.
  * RepeatPrevious: bind each item to its position code (fractional binding), so unbinding at ``t - k`` recovers the
                   item from k steps ago.  Capacity follows RUN_005 T4 (m ~ D/4), which is why ``dim`` is 256 here
                   against a task that holds at most ~64 live positions.

Item codes are random unit-modulus phasors.  This choice is *deliberately* design-independent: RUN_005 section 5.6
showed capacity is the same for the legacy, current and rff frequency designs, so the frequency-design comparison
carries no information for these tests and the module stays self-contained (numpy + torch only, no import of
``vsa_memory_stage0/vsa.py``).
"""

import numpy as np
import torch


class TorchMemory:
    """FHRR item memory.

    ``code="random"``: unit-modulus phasors.  Crosstalk between items is ``rho_ij = Re(e_i conj(e_j))/D`` with
    ``sd ~ 1/sqrt(2D)``, so a *count* readout carries error ``sum_{i!=q} counts_i * rho_iq`` -- it grows with the
    total number of items ever bundled and only vanishes as ``1/sqrt(D)``.

    ``code="block"``: each item gets a disjoint block of ``D/n_items`` dimensions, so ``rho = 0`` exactly and the
    bundle *is* the count vector.  Use this when the item count is small (CountRecall: 2-13) and the question is
    counting rather than capacity; the random codes are for the capacity question (RUN_005 T4/T7, ``m ~ D/4``).

    ``lag`` is the unbinding lag in *observation-index* terms.  POPGym's RepeatPrevious ``k`` counts the card from k
    back *including the current one*, so the algebra needs ``lag = k - 1`` (verified: hit 0.25 at lag=k, 0.99 at
    lag=k-1 for k=4).
    """

    def __init__(self, n_items, dim, k_max, lag=0, code="random", seed=0):
        rng = np.random.default_rng(seed)
        self.n_items = int(n_items)
        self.lag = int(lag)
        self.code = code
        if code == "block":
            block = int(np.ceil(dim / n_items))
            dim = block * n_items  # pad the requested dim so every item gets an equal disjoint block
            item = np.zeros((n_items, dim), dtype=np.complex64)
            for i in range(n_items):
                item[i, i * block : (i + 1) * block] = np.exp(1j * rng.uniform(-np.pi, np.pi, block))
            self.norm = float(block)  # self-overlap: sum_j Re(e conj(e)) = block
        else:
            item = np.exp(1j * rng.uniform(-np.pi, np.pi, (n_items, dim)))
            self.norm = float(dim)
        self.dim = int(dim)
        self.item = torch.tensor(item, dtype=torch.complex64)
        self.pos = torch.tensor(np.exp(1j * rng.uniform(-np.pi, np.pi, (k_max + 1, dim))), dtype=torch.complex64)

    def zero_state(self, batch):
        return torch.zeros(batch, self.dim, dtype=torch.complex64, device=self.item.device)

    def count_step(self, state, obs_t):
        """state: running bundle M (B, d).  obs_t: (B, 2) long [value, query].  Returns estimate (B,), new state."""
        m = state + self.item[obs_t[:, 0]]
        q = self.item[obs_t[:, 1]]
        est = (m * q.conj()).real.sum(-1) / self.norm
        return est, m

    def delay_step(self, state, obs_t, t_vec):
        """state: running bundle of (position (*) item) (B, d).  Returns cleanup scores (B, n_items), new state.

        ``t_vec`` is per-environment so that state reset on episode boundaries is handled by the caller.  For t < lag
        the readout uses a zero position code, so the scores are exactly zero -- the same "no information yet" state
        the oracle is in.
        """
        s = state + self.item[obs_t[:, 0]] * self.pos[t_vec]
        j = t_vec - self.lag
        pj = self.pos[j.clamp(min=0)]
        p = torch.where((j >= 0)[:, None], pj, torch.zeros_like(pj))
        r = s * p.conj()
        scores = (r[:, None, :] * self.item.conj()[None]).real.sum(-1) / self.norm
        return scores, s


def one_hot(field, card):
    """field: (B,) long -> (B, card) float."""
    return torch.nn.functional.one_hot(field, card).float()


def obs_onehot(obs_t, cards):
    """obs_t: (B, F) long -> (B, sum(cards)) float."""
    return torch.cat([one_hot(obs_t[:, i], c) for i, c in enumerate(cards)], dim=-1)


def time_feats(t, ep_len, n_harm=4):
    """t: (B,) long -> (B, 1 + 2*n_harm) Fourier positional features (the mandatory positional baseline's clock)."""
    x = (t.float() / max(ep_len - 1, 1)).clamp(0, 1)[:, None]
    feats = [x]
    for h in range(1, n_harm + 1):
        feats += [torch.sin(np.pi * h * x), torch.cos(np.pi * h * x)]
    return torch.cat(feats, dim=-1)
