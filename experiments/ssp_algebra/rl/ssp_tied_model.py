"""Tied relative addressing (and an untied control with the same lag parameterisation) for the POPGym RL port.

Why (runs/RUN_012 section 5): in the supervised arm, tying made the read address a value *the write function also
produces*: the wanted item's own time ``t - lag`` pushed through the **same** ``w_time`` that produced its write phase, so
the read has no origin of its own.  In the supervised delay task the lag is a *given* input feature.  A policy is given
no lag, so here the lag is the one thing that must be produced:

    lag_t  = LAG_SCALE * lag_head([hid_proj(z_t), abs_time_feats(t)])            # index units, a learned scalar per step
    read   = pos( w_time . abs_time_feats(t - lag_t) )                           # the write phase function at t - lag

``lag_head`` has zero weights; its bias starts at ``lag_init / LAG_SCALE`` (0 = "read now", no task constant; the control
sets the task's lag).  ``SSPMemory.addr_head`` is unused here (frozen), so no second address origin exists.

Parameterisation note (a measured finding, REPORT_TIED.md): PPO's Adam at lr 5e-5 moves a parameter by at most about
lr per SGD step, i.e. about 0.18 in 3 600 steps (1M env steps in this protocol).  A bias in raw index units therefore
cannot travel 4 index units whatever the gradient says.  ``LAG_SCALE = feats.NORM = 16`` (the repo's fixed
de-normalisation for lags) is the single default; it is a unit, not a tuned constant, and it is applied to both new
arms and to the untied control so the comparison isolates tying.

Aux (``tied_aux_lag``, default on when ``aux_coeff > 0``): label-free recall of the stream's own token **at the lag the
head currently points to**, ``token[t - round(lag)]`` (valid where that step is inside the chunk), instead of the literal
``SSPMemory.aux_loss`` target (the current token), which pulls the read towards "now" -- the opposite of any lag.
"""

import torch
import torch.nn as nn

import feats
import ssp_ops as S
from ssp_model import SSPModel, TAU

LAG_SCALE = float(feats.NORM)


def _extra_cfg():
    return {"lag_init": 0.0, "lag_scale": LAG_SCALE, "tied_aux_lag": True}


class SSPTiedModel(SSPModel):
    MODEL_CONFIG = {**SSPModel.MODEL_CONFIG, **_extra_cfg()}

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        cfg = self.cfg
        self.lag_scale = float(cfg["lag_scale"])
        self.tied_aux_lag = bool(cfg["tied_aux_lag"])
        hid = self.mem.hidden
        self.lag_head = nn.Linear(hid + feats.TIME_DIM, 1)
        with torch.no_grad():
            self.lag_head.weight.zero_()
            self.lag_head.bias.fill_(float(cfg["lag_init"]) / self.lag_scale)
        for p in self.mem.addr_head.parameters():          # tied: the untied address head must not exist as a second origin
            p.requires_grad_(False)
        self._lag = None

    def read_address(self, h, tf, t):
        lag = self.lag_scale * self.lag_head(torch.cat([h, tf], -1))[..., 0]                         # [B,T]
        self._lag = lag
        tf_read = feats.abs_time_feats(t - lag).to(h.dtype)
        return tf_read @ self.mem.w_time                                                              # the write function at t - lag

    def aux_term(self, scores, tok, seq_lens):
        if not self.tied_aux_lag:
            return super().aux_term(scores, tok, seq_lens)
        B, T = tok.shape
        idx = torch.arange(T, device=tok.device).reshape(1, -1)
        src = idx - torch.round(self._lag.detach()).long()                                           # local source step
        valid = ((src >= 0) & (src < T) & (idx < seq_lens.reshape(-1, 1))).float()
        target = tok.gather(1, src.clamp(0, T - 1))
        ce = nn.functional.cross_entropy((scores / TAU).reshape(B * T, -1), target.reshape(-1), reduction="none")
        return (ce * valid.reshape(-1)).sum() / valid.sum().clamp_min(1.0)

    def address_summary(self):
        w = self.lag_head.weight.detach()
        hid = self.mem.hidden
        tf = feats.abs_time_feats(torch.tensor(25.0))
        probe = self.lag_head.bias.detach()[0] + (w[0, hid:] * tf).sum()
        return {"lag_bias": self.lag_scale * float(self.lag_head.bias.detach()[0]),
                "lag_probe_t25_h0": self.lag_scale * float(probe),
                "lag_w_norm": self.lag_scale * float(w.norm()), "w_time0": float(self.mem.w_time.detach()[0]),
                "lam": float(torch.sigmoid(self.mem.lam_raw.detach()))}


class SSPTiedLagBankModel(SSPTiedModel):
    """Read at *every* lag in a fixed bank and let a learned selector combine the reads.

    Why (handoffs/11 step 4; RUN_012 section 6).  In RL the discovery problem split in two: tying removed the read
    *origin* (the only part the supervised delay task tested, because there the lag is a given input feature), but
    nothing removed the *lag-selection* problem -- `RepeatPrevious` hands the arm no lag, so it must find the read
    offset itself and tying alone left it at -0.133, while *setting* the offset reached +0.832.  The continuous lag
    head cannot travel there: PPO's Adam moves a parameter by about `lr` per SGD step, so a raw index-unit bias
    cannot cross 4 index units inside this protocol (REPORT_TIED.md).

    This arm replaces the continuous search by **selection**, which has no dead zone: every bank lag produces an
    address that is *exactly right for that lag*, the reads are scored independently and a softmax selector
    ``sel_head([h_t, tf_t])`` mixes them.  At init the selector is uniform (zero weights and bias), so no unit of
    the bank is preferred and no task constant is injected -- the two things the practitioner must set are the bank
    itself (`lag_bank`, a range) and nothing else; `K` and the range are counted on the constants scoreboard, as
    handoffs/11 decision 1 requires.

    State is unchanged from ``SSPTiedModel`` (the bundle only); the selector adds parameters, not carried floats.
    ``lag_head`` is inherited but unused (documented, not deleted, so subclasses stay comparable); ``read_address``
    raises rather than silently returning a wrong single-lag address.
    """

    MODEL_CONFIG = {**SSPTiedModel.MODEL_CONFIG, "lag_bank": [1, 2, 3, 4, 5, 6, 7, 8], "bank_include_now": False}

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        cfg = self.cfg
        lags = [float(x) for x in cfg["lag_bank"]]
        if cfg["bank_include_now"]:
            lags = [0.0] + lags
        assert lags and all(l >= 0.0 for l in lags), "lag_bank must be a non-empty list of non-negative lags"
        self.register_buffer("bank_lags", torch.tensor(sorted(set(lags)), dtype=torch.float32))
        self.sel_head = nn.Linear(self.mem.hidden + feats.TIME_DIM, int(self.bank_lags.numel()))
        with torch.no_grad():                      # uniform selection at init: the bank is covered, not directed
            self.sel_head.weight.zero_()
            self.sel_head.bias.zero_()
        self._sel = None

    def read_address(self, h, tf, t):
        raise NotImplementedError("SSPTiedLagBankModel scores every bank lag; use memory_core")

    def memory_core(self, z, tok, state_vec, t_starts, seq_lens):
        """Identical to the parent's write; the read is one address per bank lag, mixed by the learned selector."""
        m = self.mem
        B, T, _ = z.shape
        M, tf, t = self._write_bundle(z, tok, state_vec, t_starts, seq_lens)
        h = self.hid_proj(z)
        lags = self.bank_lags.to(t.dtype).reshape(1, 1, -1)                            # [1,1,L]
        tf_read = feats.abs_time_feats(t.unsqueeze(-1) - lags).to(z.dtype)             # [B,T,L,TIME_DIM]
        addr = tf_read @ m.w_time                                                      # [B,T,L] index units
        r = M.unsqueeze(2) * S.pos(addr, m.omega).conj()                               # [B,T,L,D]
        sc = S.cleanup_scores(r, m.codes)                                              # [B,T,L,n]
        sel = torch.softmax(self.sel_head(torch.cat([h, tf], -1)), -1)                 # [B,T,L]
        self._sel = sel
        self._lag = (sel * self.bank_lags.to(sel.dtype).reshape(1, 1, -1)).sum(-1)     # effective lag, for the aux
        scores = (sel.unsqueeze(-1) * sc).sum(2)                                       # [B,T,n]
        last = (seq_lens.long() - 1).clamp(0, T - 1)
        Mlast = M[torch.arange(B, device=z.device), last]
        new_state = torch.cat([Mlast.real, Mlast.imag], -1)
        return scores, M, addr, new_state

    def address_summary(self):
        """The mechanism check: the effective lag the selector points at, and how sharp the selection is."""
        s = super().address_summary()
        if self._sel is not None:
            sel = self._sel.detach()
            lags = self.bank_lags.to(sel.dtype)
            s["lag_eff"] = float((sel * lags.reshape(1, 1, -1)).sum(-1).mean())
            s["sel_entropy"] = float(-(sel.clamp_min(1e-9).log() * sel).sum(-1).mean())
            s["sel_max"] = float(sel.max(-1).values.mean())
            s["sel_argmax_lag"] = float(lags[int(sel.mean(dim=(0, 1)).argmax())])
        return s


class SSPScaledUntiedModel(SSPModel):
    """The untied control with the *same* lag parameterisation: read = base address head (ramp with slope error, as in
    the first pilot) minus ``LAG_SCALE * off_head(...)``.  Differs from ``SSPTiedModel`` by having its own read origin
    (the slope-error ramp) instead of the shared ``w_time`` function."""

    MODEL_CONFIG = {**SSPModel.MODEL_CONFIG, **_extra_cfg()}

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        cfg = self.cfg
        self.lag_scale = float(cfg["lag_scale"])
        hid = self.mem.hidden
        self.off_head = nn.Linear(hid + feats.TIME_DIM, 1)
        with torch.no_grad():
            self.off_head.weight.zero_()
            self.off_head.bias.fill_(float(cfg["lag_init"]) / self.lag_scale)

    def read_address(self, h, tf, t):
        base = self.mem.addr_head(torch.cat([h, tf], -1))[..., 0]
        return base - self.lag_scale * self.off_head(torch.cat([h, tf], -1))[..., 0]

    def address_summary(self):
        s = super().address_summary()
        s["lag_bias"] = self.lag_scale * float(self.off_head.bias.detach()[0])
        s["lag_w_norm"] = self.lag_scale * float(self.off_head.weight.detach().norm())
        return s
