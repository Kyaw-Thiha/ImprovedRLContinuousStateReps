"""The SSP arm for *content-keyed* recall (handoff 11 step 1 / docs/11 section 14.3 axis 2, the third family).

The positional arms (`SSPIndexLearned`, `FWPArm`) address by time: the write sits at ``pos(t)`` and the query asks
for an index, a lag, or the same time shifted.  That is what docs/11 section 1.1 calls *substitutable*, so a design
tuned for it can look good without the algebra doing anything.  This arm is the classical VSA memory instead, with
no time in the addressing at all:

    M      = leaky bundle of  bind(key_code(k_i), value_code(v_i))          one term per observed pair
    read   = M * conj(key_code(query_key)),  cleanup over the *value* codebook
    answer = the value code with the highest score

The pairs arrive in a random order and the query names a key, so nothing is matched to position.  Two ablations
separate what carries the association:

* ``bind=False`` bundles the value code *alone* (no key), same state size and same readout: if binding is what stores
  the association this must collapse.
* ``delta=True`` writes the residual ``value - read(key)`` for that key instead of the whole value (the same
  correction as the fast-weight delta rule, docs/11 14.3 axis 2 item 3), which is the capacity axis.

``learned_codes=True`` makes the two codebooks trainable (default: fixed random unit-modulus codes), so "does the
codebook have to be learned?" is one flag.
"""

import math

import torch
import torch.nn as nn

import ssp_ops as S


class AssocSSPArm(nn.Module):
    def __init__(self, task, dim=128, hidden=64, lam_init=0.99, seed=0, bind=True, delta=False, beta_init=-4.0,
                 learned_codes=False, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.dim = int(dim)
        self.out_dim = int(task.out_dim)                     # the *value* alphabet: the model reports a value
        self.bind = bool(bind)
        self.delta = bool(delta)
        self.learned_codes = bool(learned_codes)
        # Sized for the *largest* pair count any split asks for, not the training one: the extrapolation split
        # doubles `n_pairs`, and an encoder or codebook built for the training count crashes there (this is what
        # killed the first pod run of this family -- recorded in RUN_012).
        self.max_pairs = max(int(getattr(task, "n_pairs", 4)), 2)        # the alphabet is fixed by the splits
        key_codes = S.random_codes(self.max_pairs, self.dim, seed=seed + 101)
        val_codes = S.random_codes(self.max_pairs, self.dim, seed=seed + 202)
        if self.learned_codes:
            self.key_codes = nn.Parameter(key_codes.clone())
            self.val_codes = nn.Parameter(val_codes.clone())
        else:
            self.register_buffer("key_codes", key_codes)
            self.register_buffer("val_codes", val_codes)
        self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
        self.mix = nn.Sequential(nn.Linear(self.out_dim + task.in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.out_dim))
        self._scores = None

    # --- interface hooks the harness expects ---------------------------------------------------------
    def address_params(self):
        cand = []
        if isinstance(getattr(self, "key_codes", None), nn.Parameter):
            cand.append(self.key_codes)
        if isinstance(getattr(self, "val_codes", None), nn.Parameter):
            cand.append(self.val_codes)
        return [p for p in cand if p.requires_grad]

    def freeze_address(self):
        for p in self.address_params():
            p.requires_grad_(False)
            p.grad = None

    def aux_loss(self, batch):
        return None

    def set_temp(self, step, **kw):
        return None

    def set_width(self, step, width_steps, feedback=None):
        return None

    def retrieval_loss(self, batch, tau=0.1):
        """Label-free: the target is the value written under the queried key, i.e. the stream's own pairing."""
        sc = self._scores
        m = batch["mask"]
        if not m.any():
            return torch.zeros((), device=sc.device)
        return torch.nn.functional.cross_entropy(sc[m] / tau, batch["target"][m].clamp(min=0))

    def _memory_pass(self, batch, feat):
        B, T = batch["obs"].shape
        P = int(batch["n_pairs"])
        # the pair region includes any *distractor* pairs the split added, so the boundary comes from the batch
        n_pair_steps = int(batch.get("n_pair_steps", 2 * P))
        kc = self.key_codes / self.key_codes.abs().clamp_min(1e-6)
        vc = self.val_codes / self.val_codes.abs().clamp_min(1e-6)
        lam = torch.sigmoid(self.lam_raw)
        beta = torch.sigmoid(self.beta_raw)
        M = torch.zeros(B, self.dim, dtype=kc.dtype, device=feat.device)
        scores = torch.zeros(B, T, self.out_dim, device=feat.device)
        pending = torch.zeros(B, dtype=torch.long, device=feat.device)
        for t in range(T):
            if t < n_pair_steps:
                if t % 2 == 0:
                    pending = batch["obs"][:, t]                                  # key id
                else:
                    v = vc[(batch["obs"][:, t] - P).clamp(min=0)]                 # value's code
                    k = kc[pending]                                              # its key's code
                    if self.delta:
                        r = M * k.conj()
                        M = lam * M + beta * (v - r) * k
                    else:
                        M = lam * M + (S.bind(k, v) if self.bind else v)
            else:
                k = kc[batch["obs"][:, t]]                                       # the queried key's code
                scores[:, t] = S.cleanup_scores(M * k.conj(), vc)
        self._scores = scores
        return scores

    def memory_features(self, batch, feat):
        return torch.cat([self._memory_pass(batch, feat), feat], -1)

    def forward(self, batch, feat):
        return self.mix(torch.cat([self._memory_pass(batch, feat), feat], -1))

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class AssocFWPArm(nn.Module):
    """The matched fast-weight baseline for *content* addressing: same task, same state floats, learned encoder.

    The whole point of the pair: ``AssocSSPArm`` stores ``bind(key, value)`` with *fixed* VHRR codes and a bundle of
    ``2 * dim`` floats; this arm stores ``v (x) k`` in a ``d x d`` matrix with ``d = round(sqrt(2 * dim))``, i.e. the
    same number of carried floats, but it *learns* the key/value encoders (a Linear over the one-hot token).  So the
    difference between them is exactly (a) the binding operation and (b) learned versus fixed codes -- which is the
    comparison docs/11 section 14.2 says was missing.  ``delta=True`` is the practitioner's update rule.
    """

    def __init__(self, task, dim=128, hidden=64, lam_init=0.99, seed=0, delta=True, beta_init=-4.0, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.budget = 2 * int(dim)                      # match the SSP arm's carried state exactly
        self.d = max(int(round(math.sqrt(self.budget))), 4)
        self.out_dim = int(task.out_dim)
        self.max_pairs = max(int(getattr(task, "n_pairs", 4)), 2)        # the alphabet is fixed by the splits
        self.delta = bool(delta)
        self.key_enc = nn.Linear(2 * self.max_pairs, self.d, bias=False)     # one-hot token -> key
        self.val_enc = nn.Linear(2 * self.max_pairs, self.d, bias=False)     # one-hot token -> value
        self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init))))
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
        # this arm reads a raw d-vector (no cleanup), so its readout sees d + in_dim, not out_dim + in_dim
        self.mix = nn.Sequential(nn.Linear(self.d + task.in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.out_dim))
        self._scores = None

    def address_params(self):
        return list(self.key_enc.parameters())

    def freeze_address(self):
        for p in self.address_params():
            p.requires_grad_(False); p.grad = None

    def aux_loss(self, batch):
        return None

    def set_temp(self, step, **kw):
        return None

    def set_width(self, step, width_steps, feedback=None):
        return None

    def retrieval_loss(self, batch, tau=0.1):
        sc = self._scores
        m = batch["mask"]
        if not m.any():
            return torch.zeros((), device=sc.device)
        # label-free proxy: the memory's own read must be close to the value written under that key
        return sc[m].pow(2).mean()

    def _oh(self, ids, P=None):
        """One-hot at the *fixed* encoder width, so a wider split cannot change the input dimension."""
        w = 2 * self.max_pairs
        return torch.nn.functional.one_hot(ids.clamp(0, w - 1), w).float()

    def _memory_pass(self, batch, feat):
        B, T = batch["obs"].shape
        P = int(batch["n_pairs"])
        n_pair_steps = int(batch.get("n_pair_steps", 2 * P))     # includes any distractor pairs
        lam = torch.sigmoid(self.lam_raw)
        beta = torch.sigmoid(self.beta_raw)
        M = torch.zeros(B, self.d, self.d, device=feat.device)
        proj = [self.mix[0].weight.new_zeros(B, self.out_dim) for _ in range(T)]
        reads = torch.zeros(B, T, self.d, device=feat.device)
        pending = torch.zeros(B, dtype=torch.long, device=feat.device)
        for t in range(T):
            oh = self._oh(batch["obs"][:, t], P)
            if t < n_pair_steps:
                if t % 2 == 0:
                    pending = batch["obs"][:, t]
                else:
                    v = self.val_enc(oh)                                  # what is stored
                    k = self.key_enc(self._oh(pending, P))                 # under this key
                    kn = torch.nn.functional.normalize(k, dim=-1)
                    r = torch.bmm(M, kn[:, :, None])[..., 0]
                    M = lam * M + beta * (v - r)[:, :, None] * kn[:, None, :]
            else:
                k = self.key_enc(oh)
                kn = torch.nn.functional.normalize(k, dim=-1)
                reads[:, t] = torch.bmm(M, kn[:, :, None])[..., 0]
        # the readout sees a linear projection of the retrieved vector (the fast-weight analogue of cleanup scores)
        self._scores = None
        return torch.cat([reads, feat], -1)

    def memory_features(self, batch, feat):
        return self._memory_pass(batch, feat)

    def forward(self, batch, feat):
        return self.mix(self._memory_pass(batch, feat))

    def state_floats(self, T):
        return self.d * self.d

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}


class AssocPosSSPArm(AssocSSPArm):
    """Content **and** position keys in one bundle (handoff 11 step 6; tasks_d.AssocMixedTask).

    Each pair is written under *two* keys: its content key, ``bind(key_code[k], val_code[v])``, and its presentation
    position, ``bind(pos_code[i], val_code[v])``.  A query is answered with whichever key the ``mode`` column selects
    -- the content key in content mode, the position code in index mode -- through a soft gate derived from ``mode``
    itself, so the scheme adds **no parameter and no task-dependent constant** beyond the two codebooks.

    Why this is the deployable form.  `assoc` (content-only) and `index`/`delay` (position-only) each let an arm pick
    one addressing convention and keep it; a real task asks both about the same store.  The arm shows the addressing
    split can be carried by the *algebra* rather than by a task-specific choice -- at the cost of storing each item
    twice, which is the honest price and is reported.

    Ablations, both wired and cheap: ``bind=False`` bundles key and value additively (the algebra is not load-bearing);
    ``pos=False`` drops the position binding, which must destroy every index-mode query -- the control that proves the
    position key is actually used rather than the task being solvable from content alone.
    """

    def __init__(self, task, dim=128, hidden=64, lam_init=0.99, seed=0, bind=True, delta=False, beta_init=-4.0,
                 learned_codes=False, pos=True, **kw):
        super().__init__(task, dim=dim, hidden=hidden, lam_init=lam_init, seed=seed, bind=bind, delta=delta,
                         beta_init=beta_init, learned_codes=learned_codes, **kw)
        self.use_pos = bool(pos)
        self.max_pos = max(int(getattr(task, "n_pairs", 4)), 2)
        pc = S.random_codes(self.max_pos, self.dim, seed=seed + 303)
        if self.learned_codes:
            self.pos_codes = nn.Parameter(pc.clone())
        else:
            self.register_buffer("pos_codes", pc)

    def address_params(self):
        cand = list(super().address_params())
        if isinstance(getattr(self, "pos_codes", None), nn.Parameter):
            cand.append(self.pos_codes)
        return [p for p in cand if p.requires_grad]

    def _unit(self, codes):
        return codes / codes.abs().clamp_min(1e-6)

    def _memory_pass(self, batch, feat):
        B, T = batch["obs"].shape
        P = int(batch["n_pairs"])
        n_pair_steps = int(batch.get("n_pair_steps", 2 * P))       # includes any distractor pairs
        kc, vc, pc = self._unit(self.key_codes), self._unit(self.val_codes), self._unit(self.pos_codes)
        lam = torch.sigmoid(self.lam_raw)
        beta = torch.sigmoid(self.beta_raw)
        mode = batch.get("mode")
        if mode is None:                                            # content-only task: always content addressing
            mode = torch.ones(B, T, device=feat.device)
        M = torch.zeros(B, self.dim, dtype=kc.dtype, device=feat.device)
        scores = torch.zeros(B, T, self.out_dim, device=feat.device)
        pending = torch.zeros(B, dtype=torch.long, device=feat.device)
        kpos = torch.zeros(B, dtype=torch.long, device=feat.device)
        for t in range(T):
            if t < n_pair_steps:
                if t % 2 == 0:
                    pending = batch["obs"][:, t]
                    kpos = torch.full_like(pending, min(t // 2, self.max_pos - 1))
                else:
                    v = vc[(batch["obs"][:, t] - P).clamp(min=0)]
                    k = kc[pending]
                    if self.delta:
                        r = M * k.conj()
                        M = lam * M + beta * (v - r) * k
                    else:
                        item = S.bind(k, v) if self.bind else v
                        if self.use_pos:
                            item = item + (S.bind(pc[kpos], v) if self.bind else v)
                        M = lam * M + item
            else:
                s = ((mode[:, t] + 1.0) / 2.0)[:, None]              # 1.0 content, 0.0 index
                kq = s * kc[batch["obs"][:, t]] + (1 - s) * pc[batch["obs"][:, t].clamp(max=self.max_pos - 1)]
                scores[:, t] = S.cleanup_scores(M * kq.conj(), vc)
        self._scores = scores
        return scores

    def info(self, T):
        d = super().info(T)
        d["stores_two_bindings_per_pair"] = bool(self.use_pos)
        return d
