"""The SSP observation arm must actually run inside a real mjlab env (it was never wired before)."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from mjlab_centering.env_cfg import build_env_cfg
from mjlab_centering.obs import _SSPTerm


def _obs_dims(encoder: str):
  cfg = build_env_cfg("go1", num_envs=4, continuing=True, encoder=encoder)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
  obs, _ = env.reset()
  out = {g: tuple(v.shape) for g, v in obs.items()}
  finite = all(bool(torch.isfinite(v).all()) for v in obs.values())
  step_obs = env.step(torch.zeros(4, 12, device=env.device))[0]
  finite = finite and all(bool(torch.isfinite(v).all()) for v in step_obs.values())
  env.close()
  return out, finite


def test_ssp_low_runs_in_env_and_grows_the_observation():
  raw, ok_raw = _obs_dims("raw")
  ssp, ok_ssp = _obs_dims("ssp_low")
  assert ok_raw and ok_ssp
  for g in raw:
    assert ssp[g][1] > raw[g][1], (g, raw[g], ssp[g])


def test_holder_does_not_recurse_and_noise_is_actor_only():
  calls = []

  class Env:
    num_envs = 5

  def func(env, **p):
    calls.append(p)
    return torch.zeros(5, 3)

  noise = type("N", (), {"n_min": -0.5, "n_max": 0.5})()
  actor = _SSPTerm(func, {"a": 1}, noise, True, [1.0, 1.0, 1.0], 0.3, 4, 4)
  critic = _SSPTerm(func, {"a": 1}, noise, False, [1.0, 1.0, 1.0], 0.3, 4, 4)
  a1, a2 = actor(Env()), actor(Env())
  c1, c2 = critic(Env()), critic(Env())
  assert calls[0] == {"a": 1}
  assert not torch.allclose(a1, a2)  # actor state is corrupted
  assert torch.allclose(c1, c2)  # critic state is exact


def test_rff_low_control_has_the_same_width_as_ssp_low():
  from mjlab_centering.obs import install_encoder
  from mjlab_centering.env_cfg import build_env_cfg

  a = install_encoder(build_env_cfg("go1", num_envs=2, continuing=True, encoder="raw"), "ssp_low")
  b = install_encoder(build_env_cfg("go1", num_envs=2, continuing=True, encoder="raw"), "rff_low")
  assert a == b and a  # same per-term widths, so width is not confounded with code type
