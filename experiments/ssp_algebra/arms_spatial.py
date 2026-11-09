"""Arms for the spatial recall task (`tasks_e`), including the SSP arm whose motion update *is* the group action.

Handoff 11 step 3.  The arm under test is `SpatialSSPArm`: it holds ``K`` landmarks in **one** bundle,

    M = sum_k  role_k (*) phi(x_k)          x_k = the k-th landmark in the frame it was shown in

and each step it moves the whole memory into the new frame with a single element-wise multiplication,

    M <- M (*) phi(-d_t)                    d_t = this step's egocentric motion increment

so a query reads ``M (*) conj(role_q) = phi(x_q - (p_t - p_q))`` -- the landmark in the current frame -- with **no
rewriting of slots and a cost independent of K**.  That is precisely the capability `docs/11` section 14.4 lists as
potentially unique to SSP, and the one earlier testbed failed to exercise (variant A asked for the displacement to
where a landmark was *seen*, which a plain integrator can also supply).

Ablations, both wired:
  * ``shift=False``  -- drop the group action, so the stored item stays in the frame it was written in and the readout
    must undo the motion itself.  This is the load-bearing check for the *algebra*: it should collapse.
  * ``roles="shared"`` -- every landmark written under the **same** role code, so the bundle holds a superposition of
    positions that cannot be told apart by the query; the query one-hot then carries no addressing power.
"""

import math

import torch
import torch.nn as nn

import feats
import ssp_ops as S


class SpatialSSPArm(nn.Module):
    def __init__(self, task, dim=128, hidden=64, seed=0, shift=True, roles="distinct", decay=True, lam_init=0.99,
                 learned_roles=False, **kw):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.K = int(getattr(task, "K", getattr(task, "n_landmarks", 3)))
        self.dim = int(dim)
        self.shift = bool(shift)
        self.roles_distinct = str(roles) != "shared"
        self.learned_roles = bool(learned_roles)
        rc = S.random_codes(self.K, self.dim, seed=seed + 11)
        if not self.roles_distinct:
            rc = rc[:1].repeat(self.K, 1).contiguous()
        if self.learned_roles:
            self.role_codes = nn.Parameter(rc.clone())
        else:
            self.register_buffer("role_codes", rc)
        self.register_buffer("omega", S.make_omega(self.dim, 2, seed=seed + 22))
        self.lam_raw = nn.Parameter(torch.tensor(math.log(lam_init / (1 - lam_init)))) if decay else None
        self.dec = nn.Sequential(nn.Linear(2 * self.dim + task.in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self._scores = None

    # --- harness hooks -------------------------------------------------------------------------------
    def address_params(self):
        return [self.role_codes] if isinstance(getattr(self, "role_codes", None), nn.Parameter) else []

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
        return None

    def memory_features(self, batch, feat):
        return self._memory_pass(batch, feat)

    def forward(self, batch, feat):
        return self._memory_pass(batch, feat)

    def state_floats(self, T):
        return 2 * self.dim

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def info(self, T):
        return {"state_floats": int(self.state_floats(T)), "params": int(self.n_params())}

    # --- the memory -----------------------------------------------------------------------------------
    def _memory_pass(self, batch, feat):
        B, T = batch["obs"].shape
        K = self.K
        rc = self.role_codes / self.role_codes.abs().clamp_min(1e-6)      # unit-modulus roles
        lam = torch.sigmoid(self.lam_raw) if self.lam_raw is not None else 1.0
        M = torch.zeros(B, self.dim, dtype=rc.dtype, device=feat.device)
        out = torch.zeros(B, T, 2, device=feat.device)
        shown = batch["obs"] < K                                         # which step shows which landmark
        for t in range(T):
            if self.shift:
                M = M * S.phi(-batch["action"][:, t], self.omega)         # enter frame t: one group action on the whole memory
            M = lam * M
            if bool(shown[:, t].any()):
                idx = batch["obs"][:, t].clamp(max=K - 1)
                item = rc[idx] * S.phi(batch["lpos"][:, t], self.omega)
                M = M + torch.where(shown[:, t].reshape(-1, 1), item, torch.zeros_like(item))
            r = M * (batch["query"][:, t].to(rc.dtype) @ rc).conj()        # address by the queried landmark's role
            out[:, t] = self.dec(torch.cat([r.real, r.imag, feat[:, t]], dim=-1))
        return out


ARMS = {"ssp_spatial": SpatialSSPArm}
