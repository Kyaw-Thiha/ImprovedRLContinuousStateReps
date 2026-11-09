"""Task and agent configuration for the continuing Go1 (and G1) velocity tasks.

`build_go1` starts from mjlab's shipped flat-terrain config and makes the task continuing:
- the `time_out` termination is removed (no free reset),
- `episode_length_s` is set very large so no time-based reset occurs,
- the terrain curriculum is already disabled on flat; the velocity curriculum is kept (it changes r_bar, which
  is exactly the non-stationarity the study wants to expose).

Nothing in mjlab itself is modified. `action_rate` shocks rewrite `cfg.rewards["action_rate_l2"].weight`,
which the reward manager reads at compute time (verified in tests/test_env_shock.py).
"""

from __future__ import annotations

from dataclasses import asdict

from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

FOREVER_S = 1.0e6


def build_env_cfg(robot: str = "go1", num_envs: int = 4096, continuing: bool = True, encoder: str = "raw",
                  ssp_length_scale: float = 0.3, ssp_n_rotates: int = 8, ssp_n_scales: int = 8,
                  ssp_rff_seed: int = 0, falls_reset: bool = True, time_outs: bool = True, obs_v130: bool = False):
  make = unitree_go1_flat_env_cfg if robot == "go1" else unitree_g1_flat_env_cfg
  cfg = make()
  cfg.scene.num_envs = int(num_envs)
  cfg.seed = 0
  if continuing and not time_outs:
    # Legacy protocol (waves 1-9, decision 4): no time-outs at all. Decision 77 makes this opt-in only.
    cfg.terminations.pop("time_out", None)
    cfg.episode_length_s = FOREVER_S
    cfg.is_finite_horizon = False
  if not falls_reset:
    # Decision 81: dropping `fell_over` also drops the auto-reset it triggers, so the robot lies on the floor until the
    # next time-out (verified: root z 0.41 -> 0.29 over three steps, episode buffer still climbing). Only for
    # "robot stays down" experiments; never the study default.
    cfg.terminations.pop("fell_over", None)
  if obs_v130:
    # Un-do the 1.3.0 -> 1.6.0 observation change (decision 74 follow-up): 1.6.0 feeds the actor a per-env *biased*
    # joint_pos and gives the critic the unbiased one; 1.3.0 (the version the external G1 run used) fed both the
    # unbiased joint_pos. Reproducing the 1.3.0 layout isolates "version" from "our harness".
    for group in ("actor", "critic"):
      cfg.observations[group].terms["joint_pos"].params = {}
  if encoder != "raw":
    from mjlab_centering.obs import install_encoder

    install_encoder(cfg, encoder, length_scale=ssp_length_scale, n_rotates=ssp_n_rotates, n_scales=ssp_n_scales,
                    rff_seed=ssp_rff_seed)
  return cfg


def build_agent_cfg(
  robot: str = "go1",
  max_iterations: int = 500,
  seed: int = 2000,
  center: str = "none",
  eta: float = 0.01,
  beta: float = 0.01,
  value_clip: bool | None = None,
  ret_norm: str = "none",
  reward_scale: str = "none",
  r_bar_init: str = "zero",
  learning_rate: float | None = None,
  num_steps_per_env: int = 24,
  save_interval: int = 10**9,
  log_rew_std: bool = False,
  vn_clamp: float | None = 5.0,
  rs_clip: float = 10.0,
  bias_lr: float | None = None,
  warmup: bool = False,
  warmup_frac: float = 0.95,
  warmup_max_iters: int = 1000,
  actor_lr_scale: float | None = None,
  actor_lr_iters: int = 0,
  level_target: float | None = None,
  lam: float | None = None,
  adv_mode: str = "global",
  log_adv_pos: bool = False,
) -> dict:
  rl = unitree_go1_ppo_runner_cfg() if robot == "go1" else unitree_g1_ppo_runner_cfg()
  rl.seed = int(seed)
  rl.max_iterations = int(max_iterations)
  rl.num_steps_per_env = int(num_steps_per_env)
  rl.save_interval = int(save_interval)
  rl.experiment_name = "mjlab_centering"
  rl.logger = "tensorboard"
  if learning_rate is not None:
    rl.algorithm.learning_rate = float(learning_rate)

  cfg = asdict(rl)
  cfg["multi_gpu"] = None
  # mjlab's train.py strips None-valued optional model configs before the runner sees them (MLPModel would
  # otherwise receive `cnn_cfg=None` etc.); replicate that here.
  for name in ("actor", "critic"):
    part = cfg[name]
    for opt in ("cnn_cfg", "distribution_cfg"):
      if part.get(opt) is None:
        part.pop(opt, None)
    if part.get("rnn_type") is None:
      for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        part.pop(opt, None)
  cfg["algorithm"]["class_name"] = "mjlab_centering.algo:CenteredPPO"
  cfg["algorithm"].update(
    dict(
      center=center,
      eta=eta,
      beta=beta,
      ret_norm=ret_norm,
      reward_scale=reward_scale,
      r_bar_init=r_bar_init,
      log_rew_std=bool(log_rew_std),
      vn_clamp=vn_clamp,
      rs_clip=float(rs_clip),
      bias_lr=bias_lr,
      warmup=bool(warmup),
      warmup_frac=float(warmup_frac),
      warmup_max_iters=int(warmup_max_iters),
      actor_lr_scale=actor_lr_scale,
      actor_lr_iters=int(actor_lr_iters),
      level_target=level_target,
      adv_mode=adv_mode,
      log_adv_pos=bool(log_adv_pos),
    )
  )
  if lam is not None:
    cfg["algorithm"]["lam"] = float(lam)
  if value_clip is not None:
    cfg["algorithm"]["value_clip"] = bool(value_clip)
  return cfg


def apply_shock(env, kind: str, factor: float = 5.0) -> float:
  """Reward-redesign shock, applied mid-training: rescale one reward term's weight.

  The reward manager deep-copies its cfg at construction and reads `term_cfg.weight` at compute time, so the
  shock must be written to the manager's *live* term configs, not only to `env.cfg`. Returns the new weight.
  """
  if kind == "none":
    return float("nan")
  if kind != "action_rate":
    raise ValueError(kind)
  term = "action_rate_l2"
  cfg = getattr(env, "cfg", None)
  if cfg is not None and term in cfg.rewards:
    cfg.rewards[term].weight *= factor
  rm = getattr(env, "reward_manager", None)
  if rm is None:
    return float(cfg.rewards[term].weight)
  idx = rm.active_terms.index(term)
  rm._term_cfgs[idx].weight *= factor  # the copy `compute()` actually reads
  return float(rm._term_cfgs[idx].weight)
