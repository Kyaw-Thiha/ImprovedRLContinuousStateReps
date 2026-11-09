"""SSP memory as an RL policy component: interface + stub (handoff 10 phase 5.2).

**Design (no RL runs yet; this file is the interface and its contract.)**

The supervised study writes at a *given* address (``pos(t)``) and reads at an address that comes from a *given*
feature (the raw index or lag).  A policy has neither: it must decide where to write from its own state and time,
where to read from its own state, and it must keep learning while it does.  So the module exposes:

    reset(batch)                     -> zero the streaming memory (episode boundary / state budget)
    write(token, t)                  -> M <- lam * M + code(token) * pos(phi_t(t)); returns the write address
    read(address)                    -> cleanup scores of M unbound at ``address``  (address in index units)
    read_from(hidden, t)             -> the same, with the address produced by addr_head(hidden, time_feats(t))
    aux_loss(token, hidden, t)       -> label-free reconstruction of the stream's own token at time t
    freeze_interface()               -> stop gradients into (w_time, addr_head, omega)

Four design decisions, and why:

1. **The read address comes from the policy's hidden state plus time features, not from a task feature.**  The
   supervised arms read at a *feature-derived* address (``batch["index"]``/``batch["lag"]``); a policy has no such
   feature, so the address must be a function of (hidden state, t).  This is the one genuinely new design question in
   the port, and the reason the port is not a copy of the supervised arm.
2. **The write phase is the same learned time ramp**, so the memory is indexed by *time*, and the policy chooses the
   read address relative to now (``address`` is an absolute index; ``t - address`` is the lag it is asking for).
3. **Discrete observations only.**  ``token`` indexes the item codebook.  Continuous observations need an item
   encoder, which is a different claim and is explicitly out of scope (docs/11: "never claimed").
4. **The interface is separable from the readout** (``freeze_interface``), so the two-phase protocol survives the port:
   pretrain the interface on cheap synthetic streams (or keep it aligned with ``aux_loss`` in the loop), then train
   the policy head.  If the policy still has to learn the address from scratch, the supervised result does not carry
   over and that is the finding.

Token accounting: ``aux_loss`` needs no extra samples -- its labels are the stream's own observations -- so it costs
no tokens, only compute.  That is the property that makes it usable inside RL, where there are no separate
"pretraining tokens".
"""

import torch
import torch.nn as nn

import feats
import ssp_ops as S

COMPLEX = torch.complex64


def _time_scale_init(vec_width):
    """A weight vector over ``feats.abs_time_feats`` whose first component recovers raw time (that feature is t/32)."""
    w = torch.zeros(vec_width)
    w[0] = feats.RAMP_SCALE
    return w


class SSPMemory(nn.Module):
    def __init__(self, n_items, dim=128, hidden=32, seed=0, lam_init=0.99, length_scale=1.0,
                 omega_kind="uniform", ls_hi=None):
        super().__init__()
        torch.manual_seed(seed)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.register_buffer("codes", S.random_codes(self.n_items, dim, seed=seed + 1))
        omega = S.make_omega(dim, 1, length_scale=length_scale, kind=omega_kind, seed=seed + 2, ls_hi=ls_hi)
        self.register_buffer("omega", omega.reshape(-1))
        self.w_time = nn.Parameter(_time_scale_init(feats.TIME_DIM))
        # the policy speaks through a hidden state; the address head is a function of (hidden, time features)
        self.addr_head = nn.Linear(hidden + feats.TIME_DIM, 1)
        with torch.no_grad():                      # start reading at "now" (address = t), a neutral choice
            self.addr_head.weight.zero_()
            self.addr_head.bias.fill_(0.0)
        self.lam_raw = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(lam_init / (1 - lam_init))))))
        self.register_buffer("mem", torch.zeros(0, dim, dtype=COMPLEX))
        self._last_address = None
        self._last_scores = None

    # --- lifecycle -------------------------------------------------------------------------------------
    def reset(self, batch_size, device=None):
        dev = device or self.w_time.device
        self.register_buffer("mem", torch.zeros(batch_size, self.dim, dtype=COMPLEX, device=dev))
        return self

    def _mem(self, device):
        if self.mem.shape[0] == 0:
            raise RuntimeError("call reset(batch_size) before write/read")
        return self.mem.to(device)

    # --- interface -------------------------------------------------------------------------------------
    def write(self, token, t):
        """``M <- lam * M + code(token) * pos(phase(t))``.  ``token``: (B,) long; ``t``: (B,) float (time index)."""
        M = self._mem(token.device)
        lam = torch.sigmoid(self.lam_raw)
        phase = torch.as_tensor(feats.abs_time_feats(t), device=token.device, dtype=self.w_time.dtype) @ self.w_time
        pos = S.pos(phase, self.omega)
        self.register_buffer("mem", lam * M + self.codes[token] * pos)
        self._last_address = phase.detach()
        return phase

    def read(self, address):
        """Cleanup scores of the memory unbound at ``address`` (absolute index, (B,) float)."""
        M = self._mem(address.device)
        r = M * S.pos(address, self.omega).conj()
        self._last_scores = S.cleanup_scores(r, self.codes)
        return self._last_scores

    def read_from(self, hidden, t):
        """Read at an address produced by the policy's hidden state and the current time."""
        tf = torch.as_tensor(feats.abs_time_feats(t), device=hidden.device, dtype=hidden.dtype)
        addr = self.addr_head(torch.cat([hidden, tf], -1))[..., 0]
        self.read(addr)
        return self._last_scores, addr

    # --- training aids ---------------------------------------------------------------------------------
    def aux_loss(self, token, hidden, t, tau=0.1):
        """Label-free: can the module recover the token it wrote *now* when read at its own address?

        ``token`` is the observation the policy just saw, so the target costs nothing.  Reading at the address the
        module currently believes is what makes this an alignment signal for the address head rather than a decoder
        for a fixed address.
        """
        scores, _ = self.read_from(hidden, t)
        return torch.nn.functional.cross_entropy(scores / tau, token.reshape(-1))

    def interface_params(self):
        return [self.w_time, self.addr_head.weight, self.addr_head.bias, self.omega]

    def freeze_interface(self):
        for p in self.interface_params():
            p.requires_grad_(False)
        return self

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_floats(self):
        """Carried state in floats.  The codebook is a constant table, not carried state (docs/11 4.3 accounting)."""
        return 2 * self.dim
