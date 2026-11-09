"""Learned-structure SSP arms (handoff 09 step E / docs/11 section 3.4, rungs 1-2) and the non-SSP control.

Kept in a separate module so that `arms.py` (rung 0) is untouched: rung 0 is the *reference upper bound* and its
numbers must stay comparable.  Contract is identical to `arms.py` (``arm(batch, feat) -> logits``,
``state_floats(T)``, ``n_params()``), so the harness trains them unchanged.

What is learned per rung (docs/11 section 3.4):

    rung 1  the write phase and the read phase come from learned heads; ``Omega`` and the item codes stay fixed
    rung 2  ``Omega`` and the codes become parameters as well (initialised from the SSP prior)

The algebra's own operations (the product that composes phases, the conjugate that unbinds) stay fixed in every
rung: the claim under test is that a controller learns *what to feed them*, not that it re-derives multiplication.

``init="identity"`` sets the heads so that rung 1 *is* rung 0 exactly (write phase ``t``, read address ``j`` or
``lag``): that proves rung 1 can express rung 0 and is the near-identity aid of docs/11 section 3.4.  It is a
capability check, not a thesis result.
"""

import math

import torch
import torch.nn as nn

import feats
import ssp_ops as S

COMPLEX = torch.complex64


def _time_scale_init(vec_width):
    """A weight vector over ``feats.abs_time_feats`` whose first component recovers raw time (that feature is t/32)."""
    w = torch.zeros(vec_width)
    w[0] = feats.RAMP_SCALE          # t/32 * 32 = t
    return w


class SSPIndexLearned(nn.Module):
    """Variant C with learned addresses.

    Write:  ``M_t = lam * M_{t-1} + code(obs_t) * phi(s_t)``       ``s_t = <w_time, abs_time_feats(t)>``
    Read:   ``r_t = M_t * conj(phi(s_q))``                          ``s_q`` from a learned head on the raw address
    Cleanup: nearest neighbour in the item codebook (softmax, straight-through), then a readout over the scores.

    ``is_index`` picks the read semantics: absolute index ``j`` (so ``s_q`` must recover ``j``) or, for the delay
    task, a per-episode offset ``delta`` (``s_q`` must recover ``lag``, and the read is at ``t - delta``).
    """

    def __init__(self, task, dim=128, hidden=64, rung=1, init="random", length_scale=1.0, ls_start=None,
                 ls_mode="fixed", freeze_write=False, lam_init=0.99, affine_eps=0.0, affine_abs=0.0,
                 affine_slope_err=0.0, affine_offset=0.0, affine_offset_zero=False,
                 decay=True, temp=0.1, seed=0, omega_kind="uniform", ls_hi=None, n_bands=6, fine_frac=0.0,
                 addr_mode="learned", read_residual=False, read_heads=1, write_rule="plain", beta_init=1.0,
                 read_feat_scale=1.0, read_feat_shift=0.0, gate="none", banks=1, **kw):
        super().__init__()
        torch.manual_seed(seed)                     # before any submodule: the determinism gate depends on it
        self.task = task
        self.dim = dim
        self.rung = int(rung)
        self.temp = float(temp)
        self.temp0 = float(temp)
        self.is_index = getattr(task, "name", "") == "index"
        self.out_dim = task.out_dim
        # the address span R the diagnostics scale with (docs/11 3.5: index addresses are ``j in [0, T-2]``, delay
        # addresses are the lag); used by ``init="affine"`` only
        if self.is_index:
            R_addr = max(int(getattr(task, "ep_len", 16)) - 2, 1)
        else:
            R_addr = max(int(max(getattr(task, "train_lags", (4,)))), 1)
        # index of the cue's hop channel within ``task.features`` (-1 when the task has none): one assignment, read by
        # the start decoder's mask and by the cue-read halting
        self.hop_feat_idx = int(getattr(task, "hop_feat_index", -1))

        codes = S.random_codes(task.out_dim, dim, seed=seed + 1)
        if self.rung >= 2:
            self.codes = nn.Parameter(codes.clone())
        else:
            self.register_buffer("codes", codes)
        omega = S.make_omega(dim, 1, length_scale=length_scale, seed=seed + 2, kind=omega_kind, ls_hi=ls_hi,
                             n_bands=n_bands, fine_frac=fine_frac)
        self.omega_kind = omega_kind
        if self.rung >= 2:
            self.omega = nn.Parameter(omega.reshape(-1).clone())
        else:
            self.register_buffer("omega", omega.reshape(-1))

        # Kernel width (coarse-to-fine, docs: the similarity is sinc(pi d / ls), informative only within ~ls/2 of a
        # match).  The effective frequencies are omega / mult with mult = ls_now / length_scale >= 1 at the start:
        #   ls_mode="fixed"   mult = 1 (rung-0 behaviour, unchanged)
        #   ls_mode="sched"   mult decays linearly from ls_start/length_scale to 1 over `width_steps` (harness hook
        #                     arm.set_width); a *curriculum* on the kernel, not on the data
        #   ls_mode="learned" log(mult) is a trained scalar starting at log(ls_start/length_scale); no schedule
        self.ls_mode = ls_mode
        self.m0 = 1.0 if ls_start is None else max(float(ls_start) / float(length_scale), 1.0)
        if ls_mode == "learned":
            self.log_mult = nn.Parameter(torch.tensor(math.log(self.m0)))
        else:
            self.register_buffer("log_mult", torch.tensor(math.log(self.m0)))
        # learned write phase: a scalar function of absolute time; learned read phase: a scalar from the raw address
        self.w_time = nn.Parameter(_time_scale_init(feats.TIME_DIM))    # SSP prior: write phase = the time ramp
        if init == "scrambled":     # nothing handed over: both the write phase and the read head start random
            with torch.no_grad():
                self.w_time.copy_(torch.randn(feats.TIME_DIM) * 0.5)
        self.freeze_write = bool(freeze_write)
        if self.freeze_write:                       # diagnostic: one fewer moving part (the write phase stays at its init)
            self.w_time.requires_grad_(False)
        self.read_head = nn.Linear(feats.TIME_DIM, 1)
        # --- docs/11 14.3 axis 2 item 1: ONE address function for writing and reading.  With ``addr_mode="tied"`` the
        # read scalar is produced by the *same* weights that produce the write phase, so a query equal to a key gets
        # the same address by construction: the read address has no origin of its own to discover, which is what the
        # measured dead zone is about (RUN_011 3.2).  ``read_residual`` adds a zero-initialised correction so the
        # model can still learn a unit/origin mismatch in the address feature.
        self.addr_mode = str(addr_mode)
        self.read_feat_scale = float(read_feat_scale)
        self.read_feat_shift = float(read_feat_shift)
        if self.addr_mode not in ("learned", "tied"):
            raise ValueError(f"addr_mode must be 'learned' or 'tied', got {addr_mode!r}")
        self.read_residual = None
        if self.addr_mode == "tied":
            # One weight vector, both roles: the read scalar is `feats(addr) @ w_time`, the *same* function that
            # produced the write phase.  `read_head` is removed so no second origin can exist; `init` is therefore
            # irrelevant here (the shared address function starts at the SSP time-ramp prior, `_time_scale_init`).
            self.read_head = None
            if read_residual:
                self.read_residual = nn.Linear(feats.TIME_DIM, 1)
                with torch.no_grad():
                    self.read_residual.weight.zero_(); self.read_residual.bias.zero_()
        self.read_heads = max(int(read_heads), 1)      # K parallel read heads, offsets spread by half an index
        self.head_offsets = None
        if self.read_heads > 1:
            half = (self.read_heads - 1) / 2.0
            self.head_offsets = [round((k - half) * 0.5, 4) for k in range(self.read_heads)]
        self.write_rule = str(write_rule)
        if self.write_rule not in ("plain", "delta"):
            raise ValueError(f"write_rule must be 'plain' or 'delta', got {write_rule!r}")
        if self.write_rule == "delta":
            self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
        # --- docs/11 14.3 axis 1 item 3: input-dependent write strength and forgetting.  The memory decides what to
        # store and what to drop from the *content*, instead of one scalar for the whole task (the bitter-lesson
        # argument for gates).  Both outputs start at the identity of the scalar rule (bias set so sigmoid ~= 1 and
        # ~= lam_init; the weight is zero), so a gate cell starts exactly where the ungated arm does.
        self.gate = str(gate)
        if self.gate not in ("none", "input"):
            raise ValueError(f"gate must be 'none' or 'input', got {gate!r}")
        self.gate_mlp = None
        if self.gate == "input":
            gs = max(int(kw.get("gate_hidden", 32)), 4)
            self.gate_mlp = nn.Sequential(nn.Linear(task.in_dim, gs), nn.Tanh(), nn.Linear(gs, 2))
            with torch.no_grad():
                self.gate_mlp[2].weight.zero_()
                self.gate_mlp[2].bias.copy_(torch.tensor([4.0, math.log(lam_init / (1 - lam_init))]))
        # --- docs/11 14.3 axis 1 item 4 (the architecture half): several memory banks with different horizons, so no
        # single `lam_init` has to be right for the task.  `dim` is split across the banks, so the carried state is
        # unchanged (a bank cell is not allowed to buy state).
        self.banks = max(int(banks), 1)
        if self.dim % self.banks != 0:
            raise ValueError(f"dim={self.dim} is not divisible by banks={self.banks}")
        if self.banks > 1:
            # horizons spread around lam_init: fast and slow, decided by the data at read time
            hs = [lam_init ** (2.0 ** (k - (self.banks - 1) / 2.0)) for k in range(self.banks)]
            self.bank_lams = nn.Parameter(torch.tensor([math.log(max(min(h, 0.999), 0.05) / (1 - max(min(h, 0.999), 0.05)))
                                                       for h in hs]))
        if self.addr_mode == "tied":
            pass                                    # nothing to initialise: the address function is `w_time`, already set
        elif init == "identity":                    # rung 0 exactly: write at t, read at j (or lag)
            with torch.no_grad():
                self.read_head.weight.copy_(_time_scale_init(feats.TIME_DIM)[None, :])
                self.read_head.bias.zero_()
        elif init == "affine":
            # Phase 0.3 diagnostic (docs/11 3.6 item 4) and, with ``affine_abs``, RUN_011's recipe: how much *prior*
            # does a learned arm need?  Three parameterisations, and the differences matter:
            #   ``affine_eps``: error *relative to R* -- slope 1 + eps*N, offset eps*R/2*N.  Measured: recovers from
            #     eps = 0.4 at T=16 (a 6.8-index error) but FAILS on longer streams (conv 0.35-0.46 at T=24/32/48),
            #     because the *absolute* error grows with R while the kernel basin stays ~1 index at length_scale 1.
            #   ``affine_abs``: error in *index units*, ``slope = 1 + abs*N/R`` and ``offset = abs*N``, so the
            #     absolute error is the same at every stream length -- the scale-free form a recipe needs (4.7: a rule
            #     that sets its constants from the task descriptor, here R).  Measured: consistent but never reliable
            #     (0.81-0.96, at most 9 of 10 seeds), because its *offset* draw can land in a sinc dead zone.
            #   ``affine_offset_zero``: keep the drawn slope, force the offset to exactly 0.  This is the fix the audit
            #     in section 3.2 points at: a dead-zone offset gives every position the same uninformative read so
            #     those seeds never move, while a slope error always leaves some positions near-correct.
            # eps = 0 is exactly ``init="identity"`` (rung 0's address); larger eps approaches ``init="random"``.
            # Reported as an *aid*, with its own cost, whenever it is used to get a run started.
            g = torch.Generator().manual_seed(int(seed) + 11)
            a_abs = float(affine_abs)
            if float(affine_slope_err) != 0.0 or float(affine_offset) != 0.0:
                # *deterministic signed* error channels, no draw: the basin map (RUN_011 section 4.4 open question) has
                # to separate "failure is set by the absolute error" from "by the slope error" from "by the stream
                # length", and the random forms confound all three.  Defaults (0, 0) mean "unused", so every earlier
                # result is bit-for-bit unchanged.
                slope = 1.0 + float(affine_slope_err)
                offset = float(affine_offset)
            elif a_abs > 0:
                slope = 1.0 + a_abs * float(torch.randn(1, generator=g)) / max(float(R_addr), 1.0)
                offset = a_abs * float(torch.randn(1, generator=g))
            else:
                e = float(affine_eps)
                slope = 1.0 + e * float(torch.randn(1, generator=g))
                offset = e * 0.5 * R_addr * float(torch.randn(1, generator=g))
            with torch.no_grad():
                self.read_head.weight.copy_((_time_scale_init(feats.TIME_DIM) * slope)[None, :])
                self.read_head.bias.fill_(0.0 if affine_offset_zero else offset)
            self.affine_eps = float(affine_eps)
            self.affine_abs = a_abs
            self.affine_slope_err = float(affine_slope_err)
            self.affine_offset = float(affine_offset)
            self.affine_offset_zero = bool(affine_offset_zero)

        if decay:
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.register_buffer("lam_raw", torch.tensor(60.0))
        read_in = task.out_dim * self.read_heads * self.banks + task.in_dim
        self.mix = nn.Sequential(nn.Linear(read_in, hidden), nn.ReLU(), nn.Linear(hidden, task.out_dim))
        self._aux = None                            # last auxiliary address loss, for the harness aid
        # ls_mode="adaptive": narrow the kernel when the training loss plateaus, never slower than the linear ceiling
        self._adapt = {"hist": [], "mult": self.m0, "tol": 0.05, "shrink": 0.75}
        self._scores = None

    # --- kernel width ----------------------------------------------------------------------------------
    def _mult(self):
        return torch.exp(self.log_mult.clamp(math.log(0.25), math.log(64.0)))

    def set_width(self, step, width_steps, feedback=None):
        """Coarse-to-fine kernel width.  ``sched``: linear from ``ls_start`` to the final width over ``width_steps``.
        ``adaptive``: the width also shrinks by 0.75x whenever the training loss improved by less than 5% over the last
        25 updates (``feedback`` = the previous update's loss), and is capped by the linear schedule so that it always
        finishes by ``width_steps``.  A no-op for the other modes."""
        if not width_steps or self.ls_mode not in ("sched", "adaptive"):
            return
        linear = self.m0 + (1.0 - self.m0) * min(1.0, step / float(width_steps))
        mult = linear
        if self.ls_mode == "adaptive":
            st = self._adapt
            if step == 0:
                st["hist"], st["mult"] = [], self.m0
            if feedback is not None:
                st["hist"].append(float(feedback))
            if step > 0 and step % 25 == 0 and len(st["hist"]) >= 50:
                h = st["hist"]
                prev, cur = sum(h[-50:-25]) / 25.0, sum(h[-25:]) / 25.0
                if (prev - cur) / max(abs(prev), 1e-9) < st["tol"]:
                    st["mult"] = max(1.0, st["mult"] * st["shrink"])
            mult = min(st["mult"], linear)
        with torch.no_grad():
            self.log_mult.fill_(math.log(max(mult, 1.0)))

    # --- diagnostics hooks (address parameters, label-free retrieval loss) -----------------------------
    def address_params(self):
        """The parameters that define *where* things are written and read (not the readout, not the leak)."""
        cand = [self.w_time, self.log_mult]
        if self.read_head is not None:
            cand += [self.read_head.weight, self.read_head.bias]
        if self.read_residual is not None:
            cand += list(self.read_residual.parameters())
        return [p for p in cand if isinstance(p, nn.Parameter) and p.requires_grad]

    def freeze_address(self):
        for p in self.address_params():
            p.requires_grad_(False)
            p.grad = None

    def retrieval_loss(self, batch, tau=0.1):
        """Cross-entropy of the cleanup *scores* against the token that was written at the read address.  It uses only
        the memory (no readout) and the stream's own tokens, so it can train the address heads on unlimited synthetic
        streams without any task-specific supervision of the readout."""
        sc = self._scores
        tgt = batch["target"].clone()
        tgt[~batch["mask"]] = -100
        return torch.nn.functional.cross_entropy(sc.reshape(-1, sc.shape[-1]) / tau, tgt.reshape(-1), ignore_index=-100)

    # --- addresses -------------------------------------------------------------------------------------
    def _codes_unit(self):
        return self.codes / self.codes.abs().clamp_min(1e-6)

    def _write_phase(self, batch):
        s_t = feats.abs_time_feats(batch["t"]) @ self.w_time          # (B, T)
        return S.pos(s_t, self.omega / self._mult())

    def _read_addr(self, batch):
        """The raw address the read phase must recover: the absolute index ``j``, or the requested ``lag``."""
        if self.is_index:
            return batch["index"].float()
        return batch["lag"].float()[:, None].expand_as(batch["t"].float()) * batch["mask"].float()

    def _read_phase(self, batch):
        """Returns the read phase and the learned offset.

        For the index task the read happens at the *absolute* position ``j``, so the phase is ``pos(s_q)``.  For the
        delay task the item asked for sits ``lag`` steps back, so the phase is ``pos(t - s_q)`` -- exactly rung 0's
        ``M * conj(pos(t - delta))``; using ``pos(s_q)`` there would read the item at position ``lag`` instead.
        """
        addr = self._read_addr(batch)
        if self.read_feat_scale != 1.0 or self.read_feat_shift != 0.0:
            # A *deployment* stress test, not a prior: a practitioner who hands the query to the memory in different
            # units or with a different origin than the stream's own time feature.  Defaults (1, 0) = no transform.
            addr = addr * float(self.read_feat_scale) + float(self.read_feat_shift)
        if self.read_head is None and not self.is_index:
            # Tied *relative* addressing (docs/11 14.3 axis 2 item 1): the query names the item it wants, so the
            # address is the shared write-phase function evaluated at the wanted item's own time, ``t - lag``.  The
            # first tied form (apply the shared function to the lag itself) cannot work on delay: it produces an
            # address that no write ever used, which is why the delay tied cells failed (RUN_012 p8: 0.352).
            addr_abs = batch["t"].float() - self._read_addr(batch)
            if self.read_feat_scale != 1.0 or self.read_feat_shift != 0.0:
                addr_abs = addr_abs * float(self.read_feat_scale) + float(self.read_feat_shift)
            feats_addr = feats.abs_time_feats(addr_abs)
            s_q = feats_addr @ self.w_time
            if self.read_residual is not None:
                s_q = s_q + self.read_residual(feats_addr)[..., 0]
            read_at = s_q
        else:
            feats_addr = feats.abs_time_feats(addr)
            if self.read_head is None:              # addr_mode="tied": the shared write-phase weights
                s_q = feats_addr @ self.w_time
            else:
                s_q = self.read_head(feats_addr)[..., 0]                  # (B, T)
            if self.read_residual is not None:
                s_q = s_q + self.read_residual(feats_addr)[..., 0]
            read_at = s_q if self.is_index else (batch["t"].float() - s_q)
        om = self.omega / self._mult()
        if self.head_offsets is not None:
            return [S.pos(read_at + off, om) for off in self.head_offsets], s_q, addr
        return S.pos(read_at, om), s_q, addr

    # --- forward ---------------------------------------------------------------------------------------
    def _memory_pass(self, batch, feat):
        """One teacher-forced pass over the stream: write, unbind, cleanup scores (+ the auxiliary address loss)."""
        B, T = batch["obs"].shape
        codes = self._codes_unit()
        lam = torch.sigmoid(self.lam_raw)
        write_codes = torch.einsum(
            "bti,id->btd", torch.nn.functional.one_hot(batch["obs"], self.task.out_dim).to(COMPLEX), codes)
        pos_w = self._write_phase(batch)
        pos_q, s_q, addr = self._read_phase(batch)
        dims = self.dim // self.banks
        M = torch.zeros(B, self.banks, dims, dtype=COMPLEX, device=feat.device)
        heads = pos_q if isinstance(pos_q, list) else [pos_q]
        reads = [[] for _ in heads]
        wc = write_codes.reshape(B, T, self.banks, dims)          # each bank sees its own slice of the code
        pwb = pos_w.reshape(B, T, self.banks, dims)               # ... and the phase is sliced identically, so each
                                                                  # bank binds and unbinds with its own frequencies
        if self.banks > 1:
            lams = torch.sigmoid(self.bank_lams)                   # (banks,)
        else:
            lams = lam.reshape(1)
        beta = torch.sigmoid(self.beta_raw) if self.write_rule == "delta" else None
        simple = (self.banks == 1 and self.gate_mlp is None)
        if simple:
            # The historical path, kept verbatim so that every arm configuration that existed before this round is
            # bit-for-bit reproducible (a broadcast view changes the last bit; RUN_011 section 0b is about how far
            # such differences can matter, and the standard here is exactness).
            M1 = torch.zeros(B, self.dim, dtype=COMPLEX, device=feat.device)
            reads1 = [[] for _ in heads]
            for t in range(T):
                if beta is None:
                    M1 = lam * M1 + write_codes[:, t] * pos_w[:, t]
                else:
                    r_w = M1 * pos_w[:, t].conj()
                    M1 = lam * M1 + beta * (write_codes[:, t] - r_w) * pos_w[:, t]
                for hi, pq in enumerate(heads):
                    reads1[hi].append(M1 * pq[:, t].conj())
            scores_each = [S.cleanup_scores(torch.stack(rd, dim=1), codes) for rd in reads1]
            scores = scores_each[0] if len(scores_each) == 1 else torch.cat(scores_each, dim=-1)
            self._scores = scores_each[0]
            self._aux = None
            if batch["mask"].any():
                self._aux = ((s_q - addr) ** 2 * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
            return scores
        for t in range(T):
            if self.gate_mlp is not None:
                g = self.gate_mlp(feat[:, t])                      # (B, 2): write strength, decay
                lam_t = torch.sigmoid(g[:, 1])[:, None]            # (B, 1)
                beta_t = torch.sigmoid(g[:, 0])[:, None]
            else:
                lam_t, beta_t = lams[None, :], (beta if beta is not None else 1.0)
            if beta is None:
                if self.gate_mlp is not None:
                    M = lam_t[:, :, None] * M + beta_t[:, :, None] * wc[:, t] * pwb[:, t]
                else:
                    M = lams[None, :, None] * M + wc[:, t] * pwb[:, t]
            else:
                # delta-rule write (docs/11 14.3 axis 2 item 3): correct the slot instead of piling onto it, which is
                # what limits capacity at a fixed state (the measured T=64-at-512-floats ceiling).  The retrieved code
                # is the memory unbound at the *write* address; only the residual is written.
                pw = pwb[:, t]
                r_w = M * pw.conj()
                M = lam_t[:, :, None] * M + beta_t[:, :, None] * (wc[:, t] - r_w) * pw
            for hi, pq in enumerate(heads):
                pqb = pq.reshape(B, T, self.banks, dims)      # `pos` returns (B, T, dim)
                reads[hi].append(M * pqb[:, t].conj())
        # per-bank cleanup, concatenated: the banks are separate memories, not one bundle
        scores_each = []
        for rd in reads:
            st = torch.stack(rd, dim=1)                            # (B, T, banks, dims)
            if self.banks > 1:
                cs = self._codes_unit()
                per = [S.cleanup_scores(st[:, :, k], cs[:, k * dims:(k + 1) * dims]) for k in range(self.banks)]
                scores_each_k = torch.cat(per, dim=-1)
            else:
                scores_each_k = S.cleanup_scores(st[:, :, 0], self._codes_unit())
            scores_each.append(scores_each_k)
        scores = scores_each[0] if len(scores_each) == 1 else torch.cat(scores_each, dim=-1)
        self._scores = scores_each[0]        # the auxiliary retrieval loss stays on the centred head
        self._aux = None
        if batch["mask"].any():
            # auxiliary address loss: the learned read phase must reproduce the raw address (an *aid*, only used
            # when the harness passes a non-zero aux weight; it never changes the task target)
            self._aux = ((s_q - addr) ** 2 * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
        return scores

    def memory_features(self, batch, feat):
        """The vector the readout consumes (cleanup scores + shared features).

        Exposed so a *throwaway* probe readout can be trained on the arm's memory during interface pretraining
        (handoff 09 step 3), which is a cleaner signal than the arm's own untrained readout.
        """
        return torch.cat([self._memory_pass(batch, feat), feat], -1)

    def forward(self, batch, feat):
        return self.mix(torch.cat([self._memory_pass(batch, feat), feat], -1))

    def load_address_params(self, src):
        """Copy the *interface* (write phase, read head, kernel width) from another arm.

        This is what makes interface pretraining amortised rather than just more training data (docs/11 4.7): the
        same interface, learned once on cheap synthetic streams, is reused on a different stream length.  Shapes are
        checked by ``copy_``, so a mismatch raises instead of silently copying nothing.
        """
        with torch.no_grad():
            self.w_time.copy_(src.w_time)
            if self.read_head is not None and src.read_head is not None:
                self.read_head.weight.copy_(src.read_head.weight)
                self.read_head.bias.copy_(src.read_head.bias)
            self.log_mult.copy_(src.log_mult)
        return self

    def aux_loss(self, batch):
        return self._aux

    def set_temp(self, step, temp_min=0.02, decay_steps=1000):
        self.temp = max(temp_min, self.temp0 * (1.0 - min(1.0, step / max(decay_steps, 1))))

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class FWPArm(nn.Module):
    """Outer-product (fast-weight / linear-attention) memory: the **non-SSP control** (docs/11 section 4.2).

    Binds item key/value vectors by an outer product into a ``d x d`` matrix (``d = sqrt(budget)``, so ``d = 16`` at
    a 256-float budget) and reads with a learned query key derived from the *same* raw address the SSP arm gets.
    This is the competing explanation that must be ruled out (handoff 09 section 1): "an addressable key-value
    memory with a given index" is what a fast-weight memory also provides, and RUN_007 already showed the SSP *time
    parameterisation* is not what wins on index recall (random position table 0.998 vs SSP 0.999).
    """

    def __init__(self, task, budget=256, hidden=64, lam_init=0.99, decay=True, seed=0, rule="plain", tied=False,
                 read_abs=False, q_init_offset=0.0, q_copy_init=False, beta_init=1.0, gate="none", banks=1,
                 gate_hidden=32, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        # Equal *aid* for the baseline (RUN_012 step 1): the SSP arm has `gate` and `banks`, so the fast-weight arm
        # gets the same two knobs here.  Both default OFF, and `d` is computed the same way at banks=1, so the plain
        # control stays bit-exact (test_default_path_hash guards it).  `banks` splits the carried state, never grows it.
        self.banks = max(int(banks), 1)
        self.d = max(int(round(math.sqrt(max(budget // self.banks, 16)))), 4)
        self.is_index = getattr(task, "name", "") == "index"
        self.is_pointer = getattr(task, "name", "") == "pointer"
        self.out_dim = task.out_dim
        # The write KEY must come from *position* (the same fixed absolute-time features every arm receives): index recall
        # and delay recall address by where an item was written, so a key computed from the item alone cannot be looked
        # up by index at all.  (The first version keyed on one_hot(obs) only; it scored ~0.3 because it could not
        # represent the task, i.e. it was a strawman control -- see runs/RUN_008.)
        self.k_head = nn.Linear(feats.TIME_DIM, self.d)
        self.v_head = nn.Linear(task.out_dim, self.d)
        self.q_head = nn.Linear(feats.TIME_DIM, self.d)       # raw address -> query key
        # --- docs/11 section 14.3: the practitioner's fast-weight form.  Every knob is OFF by default, so the plain
        # control (the arm every earlier number used) is bit-for-bit unchanged: `rule="plain"`, `tied=False`,
        # `read_abs=False`, no offset, no copy-init. ------------------------------------------------------------------
        if rule not in ("plain", "delta"):
            raise ValueError(f"rule must be 'plain' or 'delta', got {rule!r}")
        if rule == "delta" and self.is_pointer:
            raise NotImplementedError("the delta rule is implemented for index and delay only")
        self.rule, self.tied, self.read_abs = rule, bool(tied), bool(read_abs)
        self.q_init_offset = float(q_init_offset)
        if self.tied:
            # ONE encoder for the written item and the query (docs/11 14.3 axis 2 item 1): the read address has no
            # origin of its own to discover -- a query equal to a key gets the same address by construction.
            self.q_head = self.k_head
        elif q_copy_init or self.q_init_offset != 0.0:
            with torch.no_grad():          # identity address first, so `q_init_offset` is the *only* initial error
                self.q_head.weight.copy_(self.k_head.weight)
                self.q_head.bias.copy_(self.k_head.bias)
        if rule == "delta":
            self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))    # write strength: sigmoid(beta_raw)
        # input-dependent gates (outputs: write strength, decay) with an identity init: the output layer's weight is
        # zero and its bias gives sigmoid ~= 1 for the write and ~= `lam_init` for the decay, so a gated arm starts at
        # the ungated rule and can only move away from it if the data says so.
        self.gate = str(gate)
        if self.gate not in ("none", "input"):
            raise ValueError(f"gate must be 'none' or 'input', got {gate!r}")
        self.gate_mlp = None
        if self.gate == "input":
            if self.is_pointer:
                raise NotImplementedError("input-dependent gates are implemented for index and delay only")
            gs = max(int(gate_hidden), 4)
            self.gate_mlp = nn.Sequential(nn.Linear(task.in_dim, gs), nn.Tanh(), nn.Linear(gs, 2))
            with torch.no_grad():
                self.gate_mlp[2].weight.zero_()
                self.gate_mlp[2].bias.copy_(torch.tensor([4.0, math.log(lam_init / (1 - lam_init))]))
        # multi-timescale banks: one d x d matrix each, its own learned decay, reads concatenated.  State is
        # banks * d^2 (reported exactly); the horizons are spread around `lam_init` so no single horizon has to be right.
        self.bank_lams = None
        if self.banks > 1:
            if self.is_pointer:
                raise NotImplementedError("banks are implemented for index and delay only")
            hs = [lam_init ** (2.0 ** (k - (self.banks - 1) / 2.0)) for k in range(self.banks)]
            self.bank_lams = nn.Parameter(torch.tensor(
                [math.log(min(max(h, 0.05), 0.999) / (1 - min(max(h, 0.05), 0.999))) for h in hs]))
        if decay:
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.register_buffer("lam_raw", torch.tensor(60.0))
        read_dim = self.d
        if self.is_pointer:
            # iterated read (handoff 09 step 4): the same non-SSP outer-product memory, but the query key is fed back
            # from the retrieved node for `hops` iterations -- the control for "is iteration itself doing the work?".
            # Roles stay hand-wired by parity, exactly as rung 0's, so the only thing under test is the iteration.
            self.n = task.n_nodes
            self.max_hops = task.max_hops
            self.qmap = nn.Linear(self.n, self.d, bias=False)
            self.node_key = nn.Embedding(self.n, self.d)
            self.read_head = nn.Linear(self.d, self.n)
            read_dim = self.n
        self.mix = nn.Sequential(nn.Linear(read_dim * self.banks + task.in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, task.out_dim))

    def _addr(self, batch):
        if self.is_index:
            return batch["index"].float()
        lag = batch["lag"].float()
        if lag.dim() == 1:                       # delay/pointer: one lag per row, broadcast over the stream (unchanged)
            lag = lag[:, None].expand_as(batch["t"].float())
        # ... or a per-step lag, for tasks whose query changes from step to step (`tasks_e.SpatialRecallTask`).
        # `read_abs`: read at the *absolute* time of the wanted item, `t - lag`, which is the write key's own address
        # for that item; the default keeps the historical lag-only address (the plain control is unchanged).
        return batch["t"].float() - lag if self.read_abs else lag

    # --- interface hooks (Phase 4: give the non-SSP control the *same* aid as the SSP arm) --------------
    def address_params(self):
        """The control's interface: the heads that decide *where* to write and *what* to read with.

        This is the counterpart of the SSP arm's ``w_time`` / ``read_head`` / ``log_mult``: those are exactly the
        parameters that its interface pretraining touches.  ``v_head`` (what is written) stays out, as the codebook
        does for the SSP arm.
        """
        cand = [self.k_head.weight, self.k_head.bias, self.q_head.weight, self.q_head.bias]
        if self.is_pointer:
            cand += [self.qmap.weight, self.node_key.weight]
        return [p for p in cand if isinstance(p, nn.Parameter) and p.requires_grad]

    def freeze_address(self):
        for p in self.address_params():
            p.requires_grad_(False)
            p.grad = None

    def _memory_pass(self, batch, feat):
        """The vector the readout consumes, before ``mix``: the same shape the probe pretraining needs."""
        B, T = batch["obs"].shape
        lam = torch.sigmoid(self.lam_raw)
        oh = torch.nn.functional.one_hot(batch["obs"].clamp(0, self.task.out_dim - 1), self.task.out_dim).float()
        if self.is_pointer:
            return self._pointer_features(batch, feat, oh, B, T, lam)
        q = self.q_head(feats.abs_time_feats(self._addr(batch) + self.q_init_offset))
        kt = self.k_head(feats.abs_time_feats(batch["t"]))
        if self.banks > 1 or self.gate_mlp is not None:
            return self._memory_pass_aided(batch, feat, oh, B, T, lam, q, kt)
        M = torch.zeros(B, self.d, self.d, device=feat.device)
        reads = []
        if self.rule == "delta":
            beta = torch.sigmoid(self.beta_raw)
            kn = torch.nn.functional.normalize(kt, dim=-1)
            qn = torch.nn.functional.normalize(q, dim=-1)
            for t in range(T):
                k, v = kn[:, t], self.v_head(oh[:, t])
                r = torch.bmm(M, k[:, :, None])[..., 0]                     # what this key already holds
                M = lam * M + beta * (v - r)[:, :, None] * k[:, None, :]    # delta rule: correct the slot, do not pile on
                reads.append(torch.bmm(M, qn[:, t, :, None])[..., 0])
            return torch.cat([torch.stack(reads, dim=1), feat], -1)
        for t in range(T):
            k, v = kt[:, t], self.v_head(oh[:, t])
            M = lam * M + v[:, :, None] * k[:, None, :]
            reads.append(torch.bmm(M, q[:, t, :, None])[..., 0])
        return torch.cat([torch.stack(reads, dim=1), feat], -1)

    def _memory_pass_aided(self, batch, feat, oh, B, T, lam, q, kt):
        """The `gate`/`banks` path (RUN_012 step 1).  Identical memory semantics to `_memory_pass`, with per-step
        decay and write strength and one matrix per bank; reads are concatenated per bank."""
        d = self.d
        lams = torch.sigmoid(self.bank_lams) if self.banks > 1 else lam.reshape(1)     # (banks,)
        beta = torch.sigmoid(self.beta_raw) if self.rule == "delta" else None
        # match the ungated path's read semantics exactly: the delta rule reads with the *normalised* query, the plain
        # rule with the raw one (`_memory_pass`).  A mismatch here is what the identity check caught.
        qn = torch.nn.functional.normalize(q, dim=-1) if self.rule == "delta" else q
        M = torch.zeros(B, self.banks, d, d, device=feat.device)
        reads = [[] for _ in range(self.banks)]
        for t in range(T):
            if self.gate_mlp is not None:
                g = self.gate_mlp(feat[:, t])                     # (B, 2): write-strength factor, decay
                # the decomposed variants: the gate *modulates* the decay and the write strength rather than
                # replacing them, so at the identity init (sigmoid ~= 1) a gated arm is its ungated self
                lam_t = torch.sigmoid(g[:, 1])[:, None, None, None]        # (B, 1, 1, 1): broadcast over the banks
                beta_t = (beta if beta is not None else 1.0) * torch.sigmoid(g[:, 0])[:, None, None, None]
            else:
                lam_t = lams[None, :, None, None]
                beta_t = (beta if beta is not None else 1.0)
            v = self.v_head(oh[:, t])                             # (B, d)
            k = kt[:, t]                                          # (B, d)
            if self.rule == "delta":
                kn = torch.nn.functional.normalize(k, dim=-1)
                r = torch.einsum("bkij,bkj->bki", M, kn[:, None, :])          # broadcast the key over the banks
                M = lam_t * M + beta_t * (v[:, None, :] - r)[:, :, :, None] * kn[:, None, None, :]
            else:
                M = lam_t * M + v[:, None, :, None] * k[:, None, None, :]
            for b in range(self.banks):
                reads[b].append(torch.bmm(M[:, b], qn[:, t, :, None])[..., 0])
        out = torch.cat([torch.stack(rd, dim=1) for rd in reads], -1)
        return torch.cat([out, feat], -1)

    def memory_features(self, batch, feat):
        """What ``mix`` consumes, exposed so ``pretrain_mode='probe'`` can train the interface through a throwaway
        readout (the aid the SSP arm already receives in Phase 2)."""
        return self._memory_pass(batch, feat)

    def forward(self, batch, feat):
        return self.mix(self._memory_pass(batch, feat))

    def _forward_pointer(self, batch, feat, oh, B, T, lam):
        """Pointer chasing by an iterated fast-weight read: ``cur <- key(retrieved node)``, once per hop.

        The memory is written as ``value ⊗ current_key`` for value tokens (roles by parity, as rung 0), and the read
        is repeated ``hops`` times, feeding the retrieved node's key back as the next query -- the generic-memory
        analogue of the SSP arm's "unbind, clean up, unbind again".  Nothing algebraic is used: this is the control
        for "iteration plus supervision works, and the algebra is irrelevant" (handoff 09 step 4).
        """
        return self.mix(self._pointer_features(batch, feat, oh, B, T, lam))

    def _pointer_features(self, batch, feat, oh, B, T, lam):
        n = self.n
        kt = self.k_head(feats.abs_time_feats(batch["t"]))                # (B, T, d) keys from write positions
        is_val = (batch["t"] % 2 == 1).float()[..., None]                 # parity roles, hand-wired as in rung 0
        M = torch.zeros(B, self.d, self.d, device=feat.device)
        cur = torch.zeros(B, self.d, device=feat.device)
        for t in range(T):
            v = self.v_head(oh[:, t])
            M = lam * M + is_val[:, t, :, None] * (v[:, :, None] @ cur[:, None, :])
            cur = (1 - is_val[:, t]) * kt[:, t] + is_val[:, t] * cur
        nq = self.task.n_query
        start = batch["start"]
        c = self.qmap(torch.nn.functional.one_hot(start, n).float())     # (B, nq, d)
        hops = batch["hops"]
        sc_last = None
        for h in range(1, self.max_hops + 1):
            rd = torch.einsum("bij,bqj->bqi", M, c)                      # (B, nq, d): memory read per query
            sc = self.read_head(rd)                                      # (B, nq, n)
            if sc_last is None:
                sc_last = sc
            step = (hops >= h)[:, :, None]
            sc_last = torch.where(step, sc, sc_last)
            nxt = self.node_key.weight[sc.argmax(-1)]
            c = torch.where(step, nxt, c)                                # feed the retrieved key back
        self._q_scores = sc_last
        qsteps = torch.arange(T - nq, T, device=feat.device)
        out_feat = torch.zeros(B, T, n + feat.shape[-1], device=feat.device)
        out_feat[:, qsteps, :n] = sc_last
        out_feat[:, qsteps, n:] = feat[:, qsteps]
        return out_feat

    def query_scores(self):
        return getattr(self, "_q_scores", None)

    def aux_loss(self, batch):
        return None

    def set_temp(self, step, **kw):
        return None

    def state_floats(self, T):
        return self.banks * self.d * self.d

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class SSPPointerLearned(nn.Module):
    """Variant B with learned structure (handoff 09 step F, docs/11 section 3.4 rung 1; rung 2 also learns the codes).

    Rung 0 hard-wired three things; this arm learns them and keeps only the algebra fixed:

      * **role of each stream token** -- rung 0 used ``t % 2`` (even = key, odd = value).  Here a gate
        ``r_t = sigmoid(w . feat_t + b)`` says "this token is a value".  Soft key tracking:
        ``cur_key <- (1-r_t) * code_t + r_t * cur_key`` and a soft write ``M += r_t * cur_key * P(code_t)``.
        With ``r_t = parity`` this is rung 0 exactly.
      * **query code** -- rung 0 used ``codes[start]``.  Here ``c0 = softmax(W onehot(start)) @ codes``.
      * the hop loop is unchanged (hop count still given; removing it is rung 3, not built here).

    The asymmetric bind ``key * P(value)`` (fixed permutation ``P``) stays: it is part of the algebra, and RUN_007 section
    10 shows a commutative bind makes every hop direction-ambiguous.

    ``init="identity"`` sets the role gate to the parity feature and ``W`` to a scaled identity, so the arm *is* rung 0
    (a capability check, not a thesis result).  ``aux_loss`` is the per-hop cross-entropy against the true
    intermediate nodes ``batch["inter"]`` -- an *aid* whose sample cost must be reported with the result.
    """

    def __init__(self, task, dim=256, hidden=64, rung=1, init="random", temp=0.1, decay=True, lam_init=0.99,
                 seed=0, cue=None, halting=None, mask_hop_in_start=False, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task, self.dim, self.rung = task, dim, int(rung)
        self.n = task.n_nodes
        self.out_dim = task.out_dim
        self.max_hops = task.max_hops
        # index of the cue's hop channel within ``task.features`` (-1 if the task has none): one assignment, read by the
        # start decoder's mask and by the cue-read halting
        self.hop_feat_idx = int(getattr(task, "hop_feat_index", -1))
        # Rung 3 removes the *task-visible hints* (docs/11 section 3.4): no hop count (the arm must learn to halt) and
        # the query decoded from the corruptible features rather than from ``batch["start"]`` / ``batch["hops"]``.
        # ``rung=3`` turns both on; either can also be set directly, and the defaults are rung 1's behaviour so every
        # earlier result stays reproducible.
        self.cue = cue or ("features" if self.rung >= 3 else "batch")
        self.halting = halting or ("learned" if self.rung >= 3 else "given")
        if self.cue not in ("batch", "features") or self.halting not in ("given", "learned", "counter", "cue_hop"):
            raise ValueError(f"cue={self.cue!r} halting={self.halting!r}")
        self.temp = self.temp0 = float(temp)
        codes = S.random_codes(self.n, dim, seed=seed + 1)
        if self.rung >= 2:
            self.codes = nn.Parameter(codes.clone())
        else:
            self.register_buffer("codes", codes)
        g = torch.Generator().manual_seed(int(seed) + 7)
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
        self.role = nn.Linear(task.in_dim, 1)
        self.qmap = nn.Linear(self.n, self.n, bias=False)
        if self.cue == "features":
            # decode the start node from the query-step *features* (rung 3): a small MLP so it can place a bump
            # anywhere in the node range, which a single linear layer over a scalar address cannot do
            self.start_head = nn.Sequential(nn.Linear(task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n))
            # ``mask_hop_in_start``: zero the hop channel in the start decoder's input.  RUN_011 section 6.0 measured
            # why: trained with the hop channel pinned at one value it saw several at test, the decode drifted, and
            # held-out hop 8 fell from 1.000 to 0.802 -- with the halting mechanism exonerated (both halting modes
            # scored identically).  The hop count is still read from the cue for the halting; it is only removed from
            # the *start* decode.  Default False keeps every earlier result reproducible.
            self.mask_hop_in_start = bool(mask_hop_in_start)
            if self.mask_hop_in_start and self.hop_feat_idx < 0:
                raise ValueError("mask_hop_in_start needs task.hop_feat_index")
        if self.halting == "learned":
            # a free embedding row per hop (RUN_010: rows for untrained hops never move, so the arm's *answer* is the
            # halt-weighted ensemble and its peak stays on a trained hop)
            self.hop_emb = nn.Embedding(self.max_hops, 8)
            self.stop_head = nn.Sequential(nn.Linear(task.in_dim + 8, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.halt_temp = 1.0
        elif self.halting == "counter":
            # handoff 10 phase 0.2.  Three designs were tried and measured before this one:
            # (1) a free MLP on [cue, k/max_hops] halt logits -- a *broad* monotone distribution (peak 0.36 on its own
            #     hop at h=4);
            # (2) gating the iteration with sigmoid((hop_hat - h)/tau), tau = 0.1 -- vanishing gradient at the initial
            #     hop_hat = 1 (the decode never moved);
            # (3) a *soft* quadratic score -c*(hop_hat - k)^2 in index units -- gradients are fine but the in-
            #     distribution loss is nearly flat in the decode's *slope*, so SGD stops in a compressed region:
            #     measured hop_hat = 1.05 + 0.30*h (weight 2.42 on the cue's hop channel where the exact value is 8),
            #     giving ~0.4 accuracy on held-out hops.
            # This design removes (3)'s slack: the gate is *hard* (``hop_hat >= h``) with a straight-through surrogate
            # for gradients, so a compressed decode costs in-distribution loss.  ``hop_hat`` is the queried hop
            # *index*, decoded from the corruptible cue (no task hint; the cue carries ``hop/max_hops`` linearly, so
            # the correct decode is affine with slope max_hops and extrapolates).  Init: zero weights, bias 1.
            self.hop_dec = nn.Linear(task.in_dim, 1)
            self.gate_tau = 0.5                     # surrogate width in index units (gradient only)
            with torch.no_grad():
                self.hop_dec.weight.zero_()
                self.hop_dec.bias.fill_(1.0)
        elif self.halting == "cue_hop":
            # The same hard gate as 'counter', but the hop scale is *fixed* by reading the cue's own hop channel
            # (``hop / max_hops``) instead of learning a slope.  This is the honest fallback: with hops 1-2 trained the
            # learned slope is not identifiable (measured: 0.15 with the straight-through gate, 0.30 with a soft
            # quadratic), because a whole slab of (intercept, slope) pairs is loss-free in distribution; the *channel*
            # is part of the observation, and the cue-dropout split still corrupts it, so this is not a free hint.
            # Like the counter, this keeps ``hop_hat`` in hop-*index* units.
            self.gate_tau = 0.5
            if self.hop_feat_idx < 0:
                raise ValueError("halting='cue_hop' needs task.hop_feat_index (the cue's hop channel)")
        self._halt_w = None
        self._q_scores = None
        if init == "identity":
            with torch.no_grad():
                self.role.weight.zero_()
                # parity feature: cos(2 pi t / 2) = +1 at even (key) steps, -1 at odd (value) steps; feature layout is
                # [one-hot nodes+marker (n+1), query (3), time (TIME_DIM)] and cos(period 2) is time index 2
                self.role.weight[0, self.n + 1 + 3 + 2] = -8.0
                self.role.bias.zero_()
                self.qmap.weight.copy_(torch.eye(self.n) * 10.0)
        if decay:
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.register_buffer("lam_raw", torch.tensor(60.0))
        self.mix = nn.Sequential(nn.Linear(self.n + task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, task.out_dim))
        self._aux = None

    def _codes_unit(self):
        return self.codes / self.codes.abs().clamp_min(1e-6)

    def forward(self, batch, feat):
        B, T = batch["obs"].shape
        dev = feat.device
        codes = self._codes_unit()
        lam = torch.sigmoid(self.lam_raw)
        tok = batch["obs"]
        real = (tok < self.n).float()[..., None]
        code_t = codes[tok.clamp(0, self.n - 1)] * real                         # (B, T, D)
        r = torch.sigmoid(self.role(feat))[..., 0]                              # (B, T): P(token is a value)
        M = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        cur = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        for t in range(T):
            rt = r[:, t, None]
            M = lam * M + rt * cur * code_t[:, t][:, self.perm]                 # soft asymmetric write
            cur = (1 - rt) * code_t[:, t] + rt * cur                            # soft key tracking
        nq = self.task.n_query
        hops = batch["hops"]
        qsteps = torch.arange(T - nq, T, device=dev)
        if self.cue == "features":
            # rung 3: the query comes from the corruptible features, not from batch["start"]
            qf = feat[:, qsteps]
            if self.mask_hop_in_start and self.hop_feat_idx >= 0:
                qf = qf.clone()
                qf[..., self.hop_feat_idx] = 0.0
            c = torch.softmax(self.start_head(qf), -1).to(COMPLEX) @ codes        # (B, nq, D)
        else:
            start = batch["start"]                                                # (B, nq)
            q_logits = self.qmap(torch.nn.functional.one_hot(start, self.n).float())
            c = torch.softmax(q_logits, -1).to(COMPLEX) @ codes                    # learned query code (B, nq, D)
        sc_last = S.cleanup_scores(c.reshape(-1, self.dim), codes).reshape(B, nq, self.n)
        hop_hat = None
        if self.halting == "counter":
            hop_hat = self.hop_dec(feat[:, qsteps])[..., 0]                      # (B, nq) hop *index*
        elif self.halting == "cue_hop":
            hop_hat = self.max_hops * feat[:, qsteps, self.hop_feat_idx]         # (B, nq) hop index, fixed scale
        hop_scores = []
        gates = []
        for h in range(1, self.max_hops + 1):
            rd = (M[:, None, :] * c.conj())[..., self.inv_perm]
            sc = S.cleanup_scores(rd.reshape(-1, self.dim), codes).reshape(B, nq, self.n)
            soft = torch.softmax(sc / self.temp, -1).to(COMPLEX) @ codes
            hard = codes[sc.argmax(-1)]
            c_new = soft + (hard - soft).detach()
            # with a given hop count the iteration stops when it reaches it; with learned halting every hop is
            # computed and the *answer* is the halt-weighted combination below (the arm is not told when to stop)
            if self.halting == "given":
                step = (hops >= h)[:, :, None]
                c = torch.where(step, c_new, c)
                sc_last = torch.where(step, sc, sc_last)
            else:
                # the chain always advances, so ``_hop_scores`` stays the clean per-hop reading (the iteration's own
                # extrapolation is then measurable); only the *answer* is gated at the decoded hop
                c = c_new
                if self.halting in ("counter", "cue_hop"):
                    g_hard = (hop_hat.clamp(min=1.0) >= h).to(feat.dtype)[:, :, None]
                    g_surr = torch.sigmoid((hop_hat - (h - 0.5)) / self.gate_tau)[:, :, None]
                    # straight-through: hard value forward, surrogate gradient backward.  ``g_surr - g_surr.detach()``
                    # is 0 in the forward pass but *keeps* the path to ``hop_hat`` (writing
                    # ``g_hard + (g_surr - g_hard).detach()`` detaches the surrogate too and the decode then never
                    # moves -- measured: hop_hat pinned at 1.00 for every queried hop).
                    gate = g_hard + g_surr - g_surr.detach()
                    sc_last = sc_last + gate * (sc - sc_last)
                    gates.append(g_hard)
            hop_scores.append(sc)
        self._hop_scores = torch.stack(hop_scores, dim=2)                       # (B, nq, max_hops, n)
        self._halt_w = None
        if self.halting == "learned":
            qf = feat[:, qsteps]                                               # (B, nq, in_dim)
            hop_feat = self.hop_emb(torch.arange(self.max_hops, device=dev))[None, None]
            hop_feat = hop_feat.expand(B, nq, -1, -1)                       # (B, nq, H, 8)
            pair = torch.cat([qf[:, :, None, :].expand(-1, -1, self.max_hops, -1),
                              hop_feat], -1)                                    # (B, nq, H, in+8)
            self._halt_w = torch.softmax(self.stop_head(pair)[..., 0] / self.halt_temp, -1)
        elif self.halting in ("counter", "cue_hop"):
            # P(hop = k) = 1[hop_hat >= k] - 1[hop_hat >= k+1]: the decoded halting distribution, for the diagnostics
            g = torch.cat(gates, dim=2)                                         # (B, nq, H), hard
            z = torch.zeros_like(g[:, :, :1])
            self._halt_w = g - torch.cat([g[:, :, 1:], z], dim=2)
        if self._halt_w is not None and self.halting == "learned":
            # the *learned* arm's answer is an ensemble over the unrolled hops, weighted by its halt distribution.
            # (The given-hop and counter arms have already selected their hop above.)
            dist = torch.softmax(self._hop_scores / self.temp, -1)
            sc_last = (self._halt_w[..., None] * dist).sum(2)
        self._q_scores = sc_last
        out_feat = torch.zeros(B, T, self.n + feat.shape[-1], device=dev)
        out_feat[:, qsteps, : self.n] = sc_last
        out_feat[:, qsteps, self.n:] = feat[:, qsteps]
        self._aux = None
        if "inter" in batch:
            tgt = batch["inter"]                                                # (B, nq, max_hops), IGNORE past hop
            self._aux = torch.nn.functional.cross_entropy(
                (self._hop_scores / self.temp).reshape(-1, self.n), tgt.reshape(-1), ignore_index=-100)
        return self.mix(out_feat)

    def aux_loss(self, batch):
        return self._aux

    def query_scores(self):
        """The retrieval scores the readout consumed at the query steps (B, nq, n) -- used by the cue tests."""
        return self._q_scores

    def halt_weights(self):
        """The learned halting distribution over hops (B, nq, max_hops), or None when the hop count is given."""
        return self._halt_w

    def set_temp(self, step, temp_min=0.02, decay_steps=1000):
        self.temp = max(temp_min, self.temp0 * (1.0 - min(1.0, step / max(decay_steps, 1))))

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


LEARNED_ARMS = ("ssp_index_learned", "fwp", "ssp_pointer_learned", "fwp_delta", "assoc_ssp", "assoc_fwp",
                "assoc_pos_ssp", "ssp_spatial")


def build_learned(name, task, budget=256, seed=0, **cfg):
    """Same return contract as ``arms.build_arm``, for the learned-structure arms and the FWP control."""
    if name == "ssp_index_learned":
        arm = SSPIndexLearned(task, **{k: v for k, v in cfg.items() if k != "hidden"}, dim=max(budget // 2, 8),
                              seed=seed)
        knob = f"SSPIndexLearned(dim={arm.dim}, rung={arm.rung}, init={'identity' if cfg.get('init') == 'identity' else 'random'})"
    elif name == "ssp_pointer_learned":
        arm = SSPPointerLearned(task, dim=max(budget // 2, 8), seed=seed, **cfg)
        knob = f"SSPPointerLearned(dim={arm.dim}, rung={arm.rung}, init={cfg.get('init', 'random')})"
    elif name in ("assoc_ssp", "assoc_fwp", "assoc_pos_ssp"):
        import arms_assoc as AA
        keep = ("bind", "delta", "beta_init", "learned_codes", "hidden", "lam_init", "pos")
        kw = {k: v for k, v in cfg.items() if k in keep}
        kw.pop("hidden", None)
        cls = {"assoc_ssp": AA.AssocSSPArm, "assoc_fwp": AA.AssocFWPArm,
               "assoc_pos_ssp": AA.AssocPosSSPArm}[name]
        arm = cls(task, dim=max(budget // 2, 8), seed=seed, **kw)
        dim = getattr(arm, "dim", getattr(arm, "d", 0))
        knob = f"{cls.__name__}(dim={dim}, bind={int(getattr(arm, 'bind', True))}, delta={int(getattr(arm, 'delta', False))})"
    elif name == "ssp_spatial":
        import arms_spatial as ASP
        keep = ("shift", "roles", "decay", "lam_init", "hidden", "learned_roles")
        kw = {k: v for k, v in cfg.items() if k in keep}
        arm = ASP.SpatialSSPArm(task, dim=max(budget // 2, 8), seed=seed, **kw)
        dim = getattr(arm, "dim", 0)
        knob = (f"SpatialSSPArm(dim={dim}, shift={int(arm.shift)}, "
                f"roles={'distinct' if arm.roles_distinct else 'shared'})")
    elif name in ("fwp", "fwp_delta"):
        keep = ("hidden", "lam_init", "decay", "q_init_offset", "q_copy_init", "beta_init", "read_abs", "tied",
                "rule", "gate", "banks", "gate_hidden")
        kw = {k: v for k, v in cfg.items() if k in keep}
        if name == "fwp_delta":
            kw.setdefault("rule", "delta")     # the incumbent's form: delta rule + tied encoder + read at the item's own time
            kw.setdefault("tied", True)
            kw.setdefault("read_abs", True)
        arm = FWPArm(task, budget=budget, seed=seed, **kw)
        knob = f"FWPArm(d={arm.d}, rule={arm.rule}, tied={int(arm.tied)}, read_abs={int(arm.read_abs)})"
    else:
        raise ValueError(f"unknown learned arm {name}")
    info = {"name": name, "budget": budget, "knob": knob, "state_floats": int(arm.state_floats(task.ep_len)),
            "params": int(arm.n_params())}
    info["achieved"] = info["state_floats"]
    return arm, info
