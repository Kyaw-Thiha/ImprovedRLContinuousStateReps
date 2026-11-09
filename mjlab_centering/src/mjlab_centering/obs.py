"""Observation arms: raw mjlab terms, or SSP-encoded low-dimensional terms.

`install_encoder(cfg, arm, ...)` rewrites the observation groups of an env cfg in place:

  raw      - untouched (mjlab defaults)
  ssp_low  - `command` (3-d) and `projected_gravity` (3-d) are replaced by HexSSP encodings of the same
             information; every other term is untouched. Actor and critic get the same encoder, so the arms differ
             in representation, not in information.
  rff_low  - the same two terms encoded by random Fourier features at the **same output width** as `ssp_low`. It is
             the equal-width control for `ssp_low`: it separates "the input got 22x wider, so the KL-adaptive
             schedule takes smaller steps" from "the hexagonal structured code itself hurts".

Inputs are divided by a per-dimension scale before encoding (the H2 fix from RUN_005): the command by the largest
magnitude in the task's command range, gravity by 1, so every encoded coordinate lies in about [-1, 1] and one
`length_scale` is meaningful for all of them.

Noise: mjlab applies observation noise after the term function, which for an SSP term would corrupt the feature
vector rather than the state. The holder therefore applies the original term's uniform noise to the state itself,
for the actor group only, and the term's own noise is disabled.

(An earlier draft read the raw term back through the observation manager, which after replacement is the SSP term
itself, i.e. infinite recursion, and was never wired into training. The original function and params are now
captured at install time. `ssp_full` was an exact duplicate of `ssp_low` and has been removed.)
"""

from __future__ import annotations

import torch

from mjlab.managers.observation_manager import ObservationTermCfg

from mjlab_centering.ssp_torch import HexSSP

SSP_LOW_TERMS = ("command", "projected_gravity")


class _SSPTerm:
  """Callable observation term: original term -> (actor noise) -> HexSSP feature."""

  def __init__(self, func, params, noise, actor: bool, scale, length_scale: float, n_rotates: int, n_scales: int,
               rff: bool = False, rff_dim: int | None = None, rff_seed: int = 0):
    self.func = func
    self.params = dict(params or {})
    self.noise = noise
    self.actor = actor
    self.enc = HexSSP(len(scale), scale=list(scale), length_scale=length_scale, n_rotates=n_rotates, n_scales=n_scales,
                      rff=rff, rff_dim=rff_dim, seed=rff_seed)

  @property
  def out_dim(self) -> int:
    return self.enc.out_dim

  def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
    if not self.actor or self.noise is None:
      return x
    lo = torch.as_tensor(getattr(self.noise, "n_min", 0.0), dtype=x.dtype, device=x.device)
    hi = torch.as_tensor(getattr(self.noise, "n_max", 0.0), dtype=x.dtype, device=x.device)
    return x + lo + (hi - lo) * torch.rand_like(x)

  def __call__(self, env, **_unused) -> torch.Tensor:
    val = self.func(env, **self.params)
    if val.dim() == 1:
      val = val.unsqueeze(0).repeat(env.num_envs, 1)
    if self.enc.A.device != val.device:
      self.enc.to(val.device)
    return self.enc.encode(self._corrupt(val))


def _command_scale(cfg) -> list[float]:
  r = cfg.commands["twist"].ranges
  return [max(abs(float(r.lin_vel_x[0])), abs(float(r.lin_vel_x[1])), 1e-6),
          max(abs(float(r.lin_vel_y[0])), abs(float(r.lin_vel_y[1])), 1e-6),
          max(abs(float(r.ang_vel_z[0])), abs(float(r.ang_vel_z[1])), 1e-6)]


def install_encoder(cfg, arm: str = "raw", length_scale: float = 0.3, n_rotates: int = 8, n_scales: int = 8,
                    rff_seed: int = 0) -> dict:
  """Rewrite observation terms in place. Returns {term: encoded_dim} for logging."""
  if arm == "raw":
    return {}
  if arm not in ("ssp_low", "rff_low"):
    raise ValueError(f"unknown encoder arm {arm!r} (ssp_full was removed: it duplicated ssp_low)")
  ref_dim = HexSSP(3, n_rotates=n_rotates, n_scales=n_scales).out_dim  # width to match for the rff control
  scales = {"command": _command_scale(cfg), "projected_gravity": [1.0, 1.0, 1.0]}
  dims: dict[str, int] = {}
  for group_name, group in cfg.observations.items():
    is_actor = group_name == "actor"
    for term in SSP_LOW_TERMS:
      if term not in group.terms:
        continue
      old = group.terms[term]
      holder = _SSPTerm(old.func, old.params, old.noise, is_actor, scales[term], length_scale, n_rotates, n_scales,
                        rff=arm == "rff_low", rff_dim=ref_dim if arm == "rff_low" else None, rff_seed=rff_seed)
      new = ObservationTermCfg(func=holder)
      new.noise = None
      group.terms[term] = new
      dims[f"{group_name}/{term}"] = holder.out_dim
  return dims
