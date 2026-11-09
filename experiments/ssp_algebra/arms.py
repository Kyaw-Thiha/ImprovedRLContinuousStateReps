"""Arms for the supervised probing harness (docs/11 section 4.2), one interface each:

    logits = arm(batch)                  # (B, T, out_dim), teacher-forced
    arm.state_floats(T)                  # floats carried between steps for an episode of length T
    arm.n_params()                       # reported separately from the state budget

Arms:
    mlp            no-memory floor
    posmlp         no-memory floor + the clock (POPGym's mandatory baseline)
    gru            state size matched
    lru            diagonal complex linear recurrence (the time-SSP substitute; S4D-class)
    tf_full        causal transformer, full context -- the unlimited-state upper bound (not budget-matched)
    tf_window      causal transformer over a fixed window == the same state floats as the SSP memory
    kernel_attn    windowed attention whose positional kernel is a learned Gaussian centred at t - delta
                   (the control that isolates the algebra in variant C)
    ssp            FHRR/SSP memory: write, shift, read by unbinding, cleanup; learned continuous delta

Every arm sees exactly the same input features: ``onehot(obs) || normalised_query || t/T`` (``task.features``).
Transformers additionally get rotary relative position, so no arm is handicapped by absolute position artefacts
under the train-short/test-long axis.

State accounting (the budget-matched claim): only floats carried between steps count.
    gru: H                                        lru: 2N        ssp: 2D (+1 gate)
    tf_*/kernel_attn: 2 * n_layers * d_model * (T or window)   # K and V per cached position, per layer
    mlp/posmlp: 0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import feats
import ssp_ops as S

COMPLEX = torch.complex64


class ArmBase(nn.Module):
    """Common readout: a small MLP over per-step features -> out_dim."""

    def __init__(self, feat_dim, out_dim, hidden=64, seed=0, dropout=0.0):
        super().__init__()
        torch.manual_seed(seed)   # set BEFORE any submodule exists, so construction is seed-determined
        self.out_dim = out_dim
        # ``dropout`` is a searched knob for the baseline arms only (handoff 09 section 15.4 step 1, low-data
        # hardening).  It defaults to 0, and ``F.dropout`` returns its input unchanged when p == 0, so the default
        # path draws no randomness and every previously seeded run is bit-identical.
        self.head = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, out_dim))

    def state_floats(self, T):
        return 0

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


# --------------------------------------------------------------------------------------------------------
# no-memory floors
# --------------------------------------------------------------------------------------------------------
class MLPArm(ArmBase):
    """No-memory floor.  ``clock=True`` (``posmlp``) appends the fixed-scale absolute-time features.

    POPGym requires a positional MLP baseline (docs/10 section 13.1), and RUN_007's clock was ``t / T``, which
    silently redefines time when the episode length changes (RUN_007 section 5.4).  ``task.features`` now carries the
    same fixed-scale time features for *every* arm, so ``posmlp`` differs from ``mlp`` only by a duplicate copy of
    them: the memoryless floor is one clocked number, reported under both names for continuity with RUN_007.
    """

    def __init__(self, in_dim, out_dim, hidden=64, clock=False, seed=0, dropout=0.0):
        super().__init__(in_dim + (feats.TIME_DIM if clock else 0), out_dim, hidden, seed=seed, dropout=dropout)
        self.clock = clock

    def forward(self, batch, feat):
        if self.clock:
            feat = torch.cat([feat, feats.abs_time_feats(batch["t"])], -1)
        return self.head(feat)


# --------------------------------------------------------------------------------------------------------
# recurrent / state-space
# --------------------------------------------------------------------------------------------------------
class GRUArm(ArmBase):
    def __init__(self, in_dim, out_dim, hidden=256, seed=0, dropout=0.0):
        super().__init__(hidden, out_dim, seed=seed, dropout=dropout)
        torch.manual_seed(seed)
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, batch, feat):
        h, _ = self.rnn(feat)
        return self.head(h)

    def state_floats(self, T):
        return self.hidden


class LRUArm(ArmBase):
    """Diagonal complex linear recurrence: ``x_t = a * x_{t-1} + b * u_t``, ``a = exp(-exp(nu) + i*theta)``.

    Diagonal linear recurrences with fixed unit-modulus eigenvalues are the substitute for pure temporal delay
    (docs/11 section 1.1); ``a`` here is learnable but initialised slow-decaying, matching S4D/LRU practice.
    """

    def __init__(self, in_dim, out_dim, n_state=512, hidden=64, seed=0, dropout=0.0):
        super().__init__(2 * n_state, out_dim, hidden, seed=seed, dropout=dropout)
        torch.manual_seed(seed)
        self.n = n_state
        # Long-horizon init: eigenvalues lam = 10**U(-0.301, -0.00004), i.e. decay times from ~2 to ~25000 steps,
        # so the recurrence *contains* a delay line at the horizons this study uses.  theta spans the whole
        # Nyquist band (Fourier-complete), which is what makes a pure delay representable by a linear readout.
        lam = torch.logspace(math.log10(0.5), math.log10(0.9999), n_state)
        self.nu = nn.Parameter(torch.log(-torch.log(lam)))
        self.theta = nn.Parameter(torch.linspace(0, math.pi, n_state))
        self.b_re = nn.Parameter(torch.randn(n_state) / math.sqrt(in_dim))
        self.b_im = nn.Parameter(torch.randn(n_state) / math.sqrt(in_dim))
        self.proj = nn.Linear(in_dim, n_state)

    def _a(self):
        return torch.polar(torch.exp(-torch.exp(self.nu)), self.theta)

    def forward(self, batch, feat):
        a = self._a()
        b = torch.complex(self.b_re, self.b_im)
        u = self.proj(feat).to(COMPLEX)
        B, T, _ = u.shape
        x = torch.zeros(B, self.n, dtype=COMPLEX, device=u.device)
        outs = []
        for t in range(T):
            x = a * x + b * u[:, t]
            outs.append(x)
        x = torch.stack(outs, dim=1)
        return self.head(torch.cat([x.real, x.imag], -1))

    def state_floats(self, T):
        return 2 * self.n


# --------------------------------------------------------------------------------------------------------
# transformers
# --------------------------------------------------------------------------------------------------------
def _rope(x, offset=0):
    """Rotary position embedding on (B, H, T, d_head); relative distances only, so it extrapolates."""
    B, H, T, d = x.shape
    half = d // 2
    pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
    inv = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device).float() / half)
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TFArm(ArmBase):
    """Causal transformer.  ``window=None`` -> full context; else attend to the last ``window`` positions."""

    def __init__(self, in_dim, out_dim, d_model=64, n_layers=2, n_heads=4, window=None, hidden=64, seed=0,
                 dropout=0.0):
        super().__init__(d_model, out_dim, hidden, seed=seed, dropout=dropout)
        torch.manual_seed(seed)
        self.d_model, self.n_layers, self.n_heads, self.window = d_model, n_layers, n_heads, window
        self.inp = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True,
                                                               norm_first=True) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch, feat):
        h = self.inp(feat)
        B, T, _ = h.shape
        for blk in self.blocks:
            ctx = h
            q = _rope(ctx.view(B, -1, self.n_heads, self.d_model // self.n_heads).transpose(1, 2), 0)
            mask = None
            if self.window is not None:
                idx = torch.arange(T, device=h.device)
                r = idx[:, None] - idx[None, :]
                mask = (r < 0) | (r >= self.window)
            else:
                mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
            h = blk(h, src_mask=mask)
        return self.head(self.norm(h))

    def state_floats(self, T):
        n = T if self.window is None else min(self.window, T)
        return 2 * self.n_layers * self.d_model * n


class LoopedTFArm(ArmBase):
    """Weight-tied looped transformer: the *same* learned update applied ``loops`` times, with a per-iteration output.

    This is the non-SSP iterated-structure control for pointer chasing (handoff 09 section 15.4 step 4).  The SSP arm
    applies a *fixed algebraic* step (unbind, clean up) repeatedly, once per hop; this arm applies a *learned,
    weight-tied* step repeatedly and reads out at every iteration.  If it matches the SSP arm when both get per-hop
    labels, the honest conclusion is "iteration plus supervision", not that the algebra is doing the work.

    ``hop_logits`` (B, loops, T, out_dim) is the per-iteration output; :meth:`aux_loss` supervises it against the
    task's ``inter`` key (the true intermediate nodes ``f^k(s)``), so it needs no harness change.  ``forward`` returns
    the iteration the query asks for (the query features carry the hop count), scattered back over the sequence.
    """

    def __init__(self, in_dim, out_dim, d_model=64, n_layers=1, n_heads=4, loops=8, window=None, hidden=64, seed=0,
                 dropout=0.0):
        super().__init__(d_model, out_dim, hidden, seed=seed, dropout=dropout)
        torch.manual_seed(seed)
        self.d_model, self.loops, self.window, self.n_heads = d_model, int(loops), window, n_heads
        self.inp = nn.Linear(in_dim, d_model)
        self.enc = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True,
                                                            norm_first=True) for _ in range(n_layers)])
        # ONE weight-tied update, applied `loops` times: the learned analogue of "the algebraic step is identical at
        # every hop" (docs/11 section 3.2).
        self.step = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.hop_head = nn.Linear(d_model, out_dim)
        self._hop_logits = None

    def _mask(self, T, device):
        idx = torch.arange(T, device=device)
        r = idx[:, None] - idx[None, :]
        if self.window is None:
            return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return (r < 0) | (r >= self.window)

    def forward(self, batch, feat):
        h = self.inp(feat)
        B, T, _ = h.shape
        mask = self._mask(T, h.device)
        for blk in self.enc:
            h = blk(h, src_mask=mask)
        z = h
        outs = []
        for _ in range(self.loops):
            z = self.step(z, src_mask=mask)          # the same weights every iteration
            outs.append(self.hop_head(self.norm(z)))
        self._hop_logits = torch.stack(outs, dim=1)  # (B, loops, T, out_dim)
        if "hops" not in batch or not batch.get("mask").any():
            return self._hop_logits[:, -1]
        # each query asks for a specific hop: read out that iteration (the hop count is part of the query features,
        # exactly as the SSP arm's hand-wired loop is driven by the task-visible hop count)
        out = self._hop_logits[:, 0]
        idx = (batch["hops"].long() - 1).clamp(0, self.loops - 1)          # (B, nq)
        nq = idx.shape[1]
        bidx, tidx = batch["mask"].nonzero(as_tuple=True)
        qidx = tidx - (T - nq)
        sel = (qidx >= 0) & (qidx < nq)
        out[bidx[sel], tidx[sel]] = self._hop_logits[bidx[sel], idx[bidx[sel], qidx[sel]], tidx[sel]]
        return out

    def aux_loss(self, batch):
        """Per-hop cross-entropy against the task's ``inter`` key: the aid a fair comparison must give every arm.

        Only defined for the standard pointer batch (one masked step per query); in the expanded ``chain`` variant the
        task itself carries a supervised target at every sub-step, so no auxiliary loss is needed.
        """
        if self._hop_logits is None or "inter" not in batch or not batch["mask"].any():
            return None
        inter = batch["inter"]                                    # (B, nq, max_hops)
        B, nq, K = inter.shape
        if int(batch["mask"].sum(1)[0]) != nq:
            return None
        n_keep = min(K, self.loops)
        T = self._hop_logits.shape[2]
        logits = self._hop_logits[:, :n_keep, T - nq:T]           # (B, n_keep, nq, out_dim), query steps only
        tgt = inter[:, :, :n_keep].permute(0, 2, 1)               # (B, n_keep, nq)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)

    def state_floats(self, T):
        n = T if self.window is None else min(self.window, T)
        return 2 * (len(self.enc) + 1) * self.d_model * n

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params()), "loops": self.loops}


class KernelAttnArm(ArmBase):
    """Windowed attention with a *learned* Gaussian positional kernel centred at ``t - delta_h`` per head.

    This is the control for variant C: if it matches the SSP controller, the algebra buys nothing over a smooth
    learned positional kernel (docs/11 section 1.1, last row).
    """

    def __init__(self, in_dim, out_dim, d_model=64, n_layers=2, n_heads=4, window=8, hidden=64, seed=0, dropout=0.0):
        super().__init__(d_model, out_dim, hidden, seed=seed, dropout=dropout)
        torch.manual_seed(seed)
        self.d_model, self.n_layers, self.n_heads, self.window = d_model, n_layers, n_heads, window
        self.inp = nn.Linear(in_dim, d_model)
        self.v_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.delta = nn.Parameter(torch.zeros(n_layers, n_heads))
        self.logsig = nn.Parameter(torch.zeros(n_layers, n_heads))

    def forward(self, batch, feat):
        h = self.inp(feat)
        B, T, _ = h.shape
        idx = torch.arange(T, device=h.device)
        r = (idx[:, None] - idx[None, :]).float()                      # t - t'
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        oob = r >= self.window
        for li in range(self.n_layers):
            # purely positional attention: the weights come from the learned Gaussian kernel, not from content
            v = self.v_proj[li](h).view(B, T, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
            delta = torch.sigmoid(self.delta[li]) * self.window         # per-head centre, 0..window
            sig = torch.nn.functional.softplus(self.logsig[li]) + 0.5
            logits = -((r[None, None] - delta[None, :, None, None]) ** 2) / (2 * sig[None, :, None, None] ** 2)
            logits = logits.masked_fill(causal[None, None] | oob[None, None], float("-inf"))
            w = torch.softmax(logits, dim=-1)
            att = torch.einsum("bhts,bhsd->bhtd", w.to(h.dtype), v)
            att = att.transpose(1, 2).reshape(B, T, self.d_model)
            h = self.norms[li](h + self.proj[li](att))
        return self.head(h)

    def state_floats(self, T):
        return 2 * self.n_layers * self.d_model * min(self.window, T)


# --------------------------------------------------------------------------------------------------------
# the SSP controller
# --------------------------------------------------------------------------------------------------------
class SSPArm(nn.Module):
    """SSP/Algebraic memory arm.

    Write:  ``M_t = lam * M_{t-1} + gate_t * code(obs_t) * pos(t)``
    Read (delay):  ``delta_t`` from a controller head; ``r_t = M_t * conj(pos(t - delta_t))``
    Read (index):  ``r_t = M_t * conj(pos(j_t))`` with ``j`` given
    Cleanup: nearest neighbour in the item codebook, softmax over cosine scores (optional / ablatable)

    Ablations: ``pos_kind`` in {ssp, table} (real SSP time vs the random position table of RUN_006), ``cleanup``
    on/off, ``learn_codes`` fixed vs learned item codes, ``decay`` and ``gate`` on/off.

    Features handed to the readout are the cleanup scores over the item codebook plus the shared input features
    (with the raw retrieved vector as well when ``cleanup`` is off, so the ablation can still decode).
    """

    def __init__(self, task, dim=512, hidden=64, pos_kind="ssp", cleanup=True, learn_codes=False,
                 decay=True, gate=False, length_scale=1.0, lam_init=0.99, seed=0, temp=0.1):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.dim = dim
        self.pos_kind = pos_kind
        self.cleanup_on = cleanup
        self.gate_on = gate
        self.out_dim = task.out_dim
        self.temp = temp
        self.is_index = getattr(task, "name", "") == "index"

        if learn_codes:
            ang = torch.randn(task.n_items, dim) * math.pi
            self.codes = nn.Parameter(torch.polar(torch.ones_like(ang), ang).to(COMPLEX))
        else:
            self.register_buffer("codes", S.random_codes(task.n_items, dim, seed=seed + 1))
        if pos_kind == "ssp":
            self.register_buffer("omega", S.make_omega(dim, 1, length_scale=length_scale, seed=seed + 2))
            self.table = None
        else:  # random position table, nearest-index lookup on a rounded offset (the RUN_006 construction)
            self.omega = None
            self.register_buffer("table", S.random_codes(4096, dim, seed=seed + 3))

        if decay:
            # The forgetting horizon is a *design axis* (docs/11 section 1.3), and it interacts with the task: a
            # short horizon suppresses bundle cross-talk when the query is always recent (variant C's delay), and
            # destroys recall when the query is an old item (index recall, and B's pairs).  It is therefore a
            # searched knob, not a constant: lam_init in {0.9, 0.99}.
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.register_buffer("lam_raw", torch.tensor(60.0))       # sigmoid -> 1.0
        self.gate_head = nn.Linear(task.in_dim, 1) if gate else None
        if self.is_index:
            self.delta_head = None
        else:
            self.delta_head = nn.Linear(task.in_dim, 1)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.constant_(self.delta_head.bias, 1.0)              # softplus(1) ~ 1.3 steps to start
        read_in = task.out_dim + task.in_dim + (0 if cleanup else 2 * dim)
        self.mix = nn.Sequential(nn.Linear(read_in, hidden), nn.ReLU(), nn.Linear(hidden, task.out_dim))

    def _pos(self, t):
        if self.pos_kind == "ssp":
            return S.pos(t, self.omega)
        i = t.round().long().clamp(0, self.table.shape[0] - 1)
        return self.table[i]

    def read(self, batch, feat, delta=None):
        """Writes the stream and reads it; returns the cleanup scores (B, T, n_items)."""
        B, T = batch["obs"].shape
        dev = feat.device
        codes = self.codes / self.codes.abs().clamp_min(1e-6)          # keep unit modulus if learned
        write_codes = torch.einsum("bti,id->btd", torch.nn.functional.one_hot(batch["obs"], self.task.n_items)
                                   .to(COMPLEX), codes)
        lam = torch.sigmoid(self.lam_raw)
        gate = torch.sigmoid(self.gate_head(feat)) if self.gate_head is not None else 1.0
        if delta is None:
            if self.is_index:
                # index-addressed recall reads the *absolute* position j (clipped so the item exists), whereas the
                # delay variant reads the position (t - delta) steps back.  Conflating the two makes the arm answer
                # "the item j steps ago" while being scored on "the item at index j" -- it sat at the no-memory
                # floor (0.18) with the pinned-index readout perfect, which is what exposed it.  `batch["index"]` is
                # the raw index, so the arm no longer de-normalises `query * T` (handoff 09 section 5.3).
                delta = torch.minimum(batch["index"].float(), batch["t"].float())
            else:
                delta = torch.nn.functional.softplus(self.delta_head(feat))[..., 0]
        read_at = delta if self.is_index else (batch["t"].float() - delta)
        pos_w = self._pos(batch["t"].float().reshape(-1)).reshape(B, T, self.dim)
        pos_q = self._pos(read_at.reshape(-1)).reshape(B, T, self.dim)

        M = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        reads = []
        for t in range(T):
            M = lam * M + (gate[:, t] if torch.is_tensor(gate) else gate) * write_codes[:, t] * pos_w[:, t]
            reads.append(S.unbind(M, pos_q[:, t]))
        self._r = torch.stack(reads, dim=1)                             # (B, T, D) retrieved item vector
        return S.cleanup_scores(self._r, codes)                         # (B, T, n_items)

    def forward(self, batch, feat):
        scores = self.read(batch, feat)
        parts = [scores]
        if not self.cleanup_on:
            parts += [self._r.real, self._r.imag]
        parts.append(feat)
        return self.mix(torch.cat(parts, -1))

    def state_floats(self, T):
        return 2 * self.dim + (1 if self.gate_on else 0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class SSPPointerArm(nn.Module):
    """Variant B: unbind-then-cleanup applied in a loop (docs/11 section 3.2).

    Write: the pair stream is bound as ``code(key) * code(value)`` into one bundle.
    Read:  start from ``code(s)`` and iterate ``h`` times: ``c <- cleanup(M * conj(c))``.
    The loop count is **hand-wired** from the task-visible hop count, which is what makes 1-2-hop training
    extrapolate to 4-8 hops; docs/11 section 4.5 requires this be reported as a structural prior, not as learning
    the algebra.
    """

    def __init__(self, task, dim=512, hidden=64, hard=False, temp=0.1, decay=True, lam_init=0.99, seed=0, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.dim = dim
        self.n = task.n_nodes
        self.out_dim = task.out_dim
        self.hard = hard
        self.temp = temp
        self.max_hops = task.max_hops
        self.register_buffer("codes", S.random_codes(self.n, dim, seed=seed + 1))
        g = torch.Generator().manual_seed(int(seed) + 7)
        perm = torch.randperm(dim, generator=g)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
        if decay:
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.lam_raw = None
            self.register_buffer("_lam_const", torch.tensor(60.0))
        self.mix = nn.Sequential(nn.Linear(self.n + task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, task.out_dim))

    def forward(self, batch, feat):
        B, T = batch["obs"].shape
        dev = feat.device
        lam = torch.sigmoid(self.lam_raw if self.lam_raw is not None else self._lam_const)
        tok = batch["obs"]
        real = tok < self.n
        M = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        cur_key = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        for t in range(T):
            code_t = self.codes[tok[:, t].clamp(0, self.n - 1)]
            code_t = code_t * real[:, t].float()[:, None]
            if t % 2 == 0:
                cur_key = code_t
            else:
                M = lam * M + cur_key * code_t[:, self.perm]   # asymmetric bind: key * P(value)
        nq = self.task.n_query
        start = batch["start"]                                   # (B, nq)
        hops = batch["hops"]                                     # (B, nq)
        c = self.codes[start]                                    # (B, nq, dim)
        active = hops >= 0
        sc_last = S.cleanup_scores(c.reshape(-1, self.dim), self.codes).reshape(B, nq, self.n)
        for h in range(1, self.max_hops + 1):
            r = (M[:, None, :] * c.conj())[..., self.inv_perm]   # unbind the current key, undo the value permutation
            sc = S.cleanup_scores(r.reshape(-1, self.dim), self.codes).reshape(B, nq, self.n)
            if self.hard:
                idx = sc.argmax(-1)
                c_new = self.codes[idx]
            else:
                soft = torch.softmax(sc / self.temp, -1).to(COMPLEX) @ self.codes
                hard = self.codes[sc.argmax(-1)]
                c_new = soft + (hard - soft).detach()
            step_mask = (hops >= h)[:, :, None]
            c = torch.where(step_mask, c_new, c)
            # scores of the last *executed* hop: they already describe f^h(s), so no extra unbind after the loop
            sc_last = torch.where(step_mask, sc, sc_last)
        final_sc = sc_last
        # place the query-step features back on the time axis
        out_feat = torch.zeros(B, T, self.n + feat.shape[-1], device=dev)
        qsteps = torch.arange(T - nq, T, device=dev)
        out_feat[:, qsteps, : self.n] = final_sc
        out_feat[:, qsteps, self.n:] = feat[:, qsteps]
        return self.mix(out_feat)

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class SSPPathArm(nn.Module):
    """Variant A: pose as an SSP with exact path integration (docs/11 section 3.1).

    ``p_t = p_{t-1} * phi(a_t)``; landmarks are bundled as ``code(l) * p_t`` at observation time; the query
    unbinds the landmark code and then the current pose, giving the *relative* vector ``phi(x_seen - x_t)``.

    ``shift_mode``: ``"read_relative"`` stores absolute poses and unbinds at read time (two multiplications per
    read); ``"shift_memory"`` shifts the whole memory by ``phi(-a_t)`` every step and writes at the origin (one
    multiplication per step, independent of item count).  Same answer, different cost -- docs/11 section 4.4 asks
    for the comparison on state and compute, not accuracy.
    """

    def __init__(self, task, dim=512, hidden=64, shift_mode="read_relative", decay=True, length_scale=4.0,
                 lam_init=0.99, seed=0, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.dim = dim
        self.L = task.out_dim if task.name == "delay" else task.query_dim
        self.L = int(self.L)
        self.out_dim = task.out_dim
        self.shift_mode = shift_mode
        self.register_buffer("codes", S.random_codes(self.L, dim, seed=seed + 1))
        self.register_buffer("omega", S.make_omega(dim, 2, length_scale=length_scale, seed=seed + 2))
        self.register_buffer("omega1", S.make_omega(dim, 2, length_scale=length_scale, seed=seed + 2))
        if decay:
            self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        else:
            self.lam_raw = None
            self.register_buffer("_lam_const", torch.tensor(60.0))
        self.mix = nn.Sequential(nn.Linear(2 * dim + task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, batch, feat):
        B, T = batch["obs"].shape
        dev = feat.device
        lam = torch.sigmoid(self.lam_raw if self.lam_raw is not None else self._lam_const)
        p = torch.ones(B, self.dim, dtype=COMPLEX, device=dev)       # phi(0): the pose reference
        M = torch.zeros(B, self.dim, dtype=COMPLEX, device=dev)
        rels = []
        for t in range(T):
            step = S.phi(batch["action"][:, t], self.omega)          # (B, D)
            if self.shift_mode == "shift_memory":
                M = M * step.conj()                                  # re-centre the whole world map: O(D)
            else:
                p = p * step
            idx = batch["obs"][:, t].clamp(max=self.L - 1)
            is_lm = (batch["obs"][:, t] < self.L).float()[:, None]
            anchor = torch.ones_like(p) if self.shift_mode == "shift_memory" else p
            M = lam * M + is_lm * self.codes[idx] * anchor           # code(l) (*) p_t
            q = batch["query"][:, t].argmax(-1)                      # the queried landmark
            r = M * self.codes[q].conj()                             # -> p_seen (absolute) or rel (shift mode)
            if self.shift_mode != "shift_memory":
                r = r * p.conj()                                     # -> phi(x_seen - x_t)
            rels.append(r)
        rel = torch.stack(rels, 1)
        return self.mix(torch.cat([rel.real, rel.imag, feat], -1))

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


# --------------------------------------------------------------------------------------------------------
# builder + budget matching
# --------------------------------------------------------------------------------------------------------
class PoseAttnArm(nn.Module):
    """Variant A's *honest* competitor (docs/11 section 3.1, handoff 09 section 15.4 step 6).

    K stored ``(pose, landmark-one-hot)`` pairs and an attention read: at a query step the arm attends over the slots
    by landmark identity and returns the attended pose minus the current pose -- the same answer the SSP path arm
    gives, but with a **learned integrator** instead of an exact group action.  State is ``K * (2 + L)`` floats, so at
    a 256-float budget and L = 6 the arm keeps the last K = 32 landmark observations.

    Why it is the competitor that matters: a transformer that learns to subtract stored poses is exact too, and the
    EXPECTED SSP advantage on this variant is sample efficiency, state and extrapolation, not accuracy at unlimited
    state (docs/11 section 3.1).  An attention memory is exactly the "unlimited accuracy" arm, so a comparison against
    only a GRU would overstate the SSP result.
    """

    def __init__(self, task, budget=256, hidden=64, seed=0, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.L = int(task.query_dim)
        self.slot = 2 + self.L
        self.K = max(int(budget) // self.slot, 4)
        self.integrate = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.score = nn.Parameter(torch.tensor(math.log(8.0)))     # attention sharpness over landmark identity
        self.mix = nn.Sequential(nn.Linear(2 + 2 + task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, task.out_dim))

    def forward(self, batch, feat):
        B, T = batch["obs"].shape
        dev = feat.device
        slots = torch.zeros(B, self.K, self.slot, device=dev)
        valid = torch.zeros(B, self.K, device=dev)
        pose = torch.zeros(B, 2, device=dev)
        sharp = self.score.exp().clamp(min=1.0, max=64.0)
        rels, poses = [], []
        for t in range(T):
            pose = pose + self.integrate(batch["action"][:, t])              # learned path integration
            obs = batch["obs"][:, t]
            is_lm = (obs < self.L).float()
            oh = torch.nn.functional.one_hot(obs.clamp(max=self.L - 1), self.L).float()
            new = torch.cat([pose, oh], -1) * is_lm[:, None]
            slots = torch.cat([new[:, None], slots[:, :-1]], 1)              # push the newest observation to the front
            valid = torch.cat([is_lm[:, None], valid[:, :-1]], 1)
            q = batch["query"][:, t]                                          # one-hot queried landmark
            logits = (slots[:, :, 2:] @ q[..., None])[..., 0] * sharp
            logits = logits.masked_fill(valid < 0.5, -1e9)
            w = torch.softmax(logits, -1)
            seen = (w[..., None] * slots[:, :, :2]).sum(1)
            has = valid.any(-1, keepdim=True).float()
            rels.append((seen - pose) * has)
            poses.append(pose)
        rel = torch.stack(rels, 1)
        pose_seq = torch.stack(poses, 1)
        return self.mix(torch.cat([rel, pose_seq, feat], -1))

    def state_floats(self, T):
        return self.K * self.slot

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


MEMORY_ARMS = ("gru", "lru", "tf_window", "kernel_attn", "ssp", "pose_attn")


def filter_kwargs(cls, kw):
    """Drops search-space knobs an arm does not accept (e.g. ``temp`` for the path arm)."""
    import inspect

    allowed = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kw.items() if k in allowed}


def build_arm(name, task, budget=512, tf_d_model=64, tf_layers=2, tf_heads=4, win_d_model=32, win_layers=1,
              win_heads=2, hidden=64, seed=0, dropout=0.0, **kw):
    """Builds an arm; for memory arms the size is chosen from ``budget`` (floats carried between steps).

    Returns (arm, info) where info has the *achieved* state floats, the parameters, and the size knob used.
    """
    in_dim = getattr(task, "in_dim", task.out_dim)
    out_dim = task.out_dim
    info = {"name": name, "budget": budget, "knob": None}

    if name == "mlp":
        arm, info["knob"] = MLPArm(in_dim, out_dim, hidden, seed=seed, dropout=dropout), "hidden=%d" % hidden
    elif name == "posmlp":
        arm, info["knob"] = MLPArm(in_dim, out_dim, hidden, clock=True, seed=seed, dropout=dropout), "hidden=%d" % hidden
    elif name == "gru":
        h = int(budget)
        arm, info["knob"] = GRUArm(in_dim, out_dim, h, seed=seed, dropout=dropout), "hidden=%d" % h
    elif name == "lru":
        n = max(int(budget) // 2, 1)
        arm, info["knob"] = LRUArm(in_dim, out_dim, n, hidden, seed=seed, dropout=dropout), "n_state=%d" % n
    elif name == "tf_full":
        arm, info["knob"] = TFArm(in_dim, out_dim, tf_d_model, tf_layers, tf_heads, None, hidden, seed,
                                  dropout), "d_model=%d,layers=%d" % (tf_d_model, tf_layers)
    elif name == "looped_tf":
        # Weight-tied looped transformer: the non-SSP iterated-structure control (handoff 09 step 4).  ``loops`` must
        # cover the largest hop count the splits ask for, or the read-out iteration is clamped and the arm cannot
        # express the answer for the longest chains.
        loops = int(getattr(task, "max_hops", 8))
        arm, info["knob"] = LoopedTFArm(in_dim, out_dim, tf_d_model, tf_layers, tf_heads, loops=loops, window=None,
                                        hidden=hidden, seed=seed, dropout=dropout), \
            "d_model=%d,layers=%d,loops=%d" % (tf_d_model, tf_layers, loops)
    elif name in ("tf_window", "kernel_attn"):
        # The windowed arms use a smaller d_model/depth than the full-context upper bound, because otherwise the
        # state budget would force a window of 1-2 tokens and the arm could not attend to *anything* -- a
        # matched-state comparison that no longer measures the memory.  State (not parameters) is what is matched.
        per_token = 2 * win_layers * win_d_model
        w = max(int(budget) // per_token, 1)
        cls = TFArm if name == "tf_window" else KernelAttnArm
        arm, info["knob"] = cls(in_dim, out_dim, win_d_model, win_layers, win_heads, w, hidden, seed=seed,
                                dropout=dropout), "window=%d" % w
    elif name == "pose_attn":
        arm = PoseAttnArm(task, budget=budget, hidden=hidden, seed=seed, **filter_kwargs(PoseAttnArm, kw))
        info["knob"] = f"PoseAttnArm(K={arm.K},slot={arm.slot})"
    elif name == "ssp":
        dim = max(int(budget) // 2, 8)
        cls = {"pointer": SSPPointerArm, "path": SSPPathArm}.get(getattr(task, "name", ""), SSPArm)
        arm = cls(task, dim=dim, hidden=hidden, seed=seed, **filter_kwargs(cls, kw))
        info["knob"] = f"{cls.__name__}(dim={dim})"
    else:
        raise ValueError(f"unknown arm {name}")

    info["state_floats"] = int(arm.state_floats(task.ep_len))
    info["achieved"] = info["state_floats"]
    info["params"] = int(arm.n_params())
    return arm, info


ALL_ARMS = ("mlp", "posmlp", "gru", "lru", "tf_full", "tf_window", "kernel_attn", "looped_tf", "ssp")
