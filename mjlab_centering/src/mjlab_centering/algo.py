"""CenteredPPO: reward centering for PPO, plus the critic-fix comparison arms.

Centering is applied at step-collection time, *before* rsl-rl folds `gamma * V(s)` into the reward on
time-outs (`PPO.process_env_step`), not inside `compute_returns`. rsl-rl 5.5.1 already bootstraps on
time-outs by rewriting the stored reward, so subtracting r_bar in `compute_returns` would also subtract it
from that bootstrap term. rsl-rl 5.5.1 never masks terminal transitions in this task anyway (the wrapper
zeroes dones), but the ordering is the correct one regardless.

Arms (`center`):
  none    - no centering (control)
  value   - TD-based: r_bar += eta * mean(delta), once per iteration
  simple  - running average of the observed reward rate: r_bar += beta * (r_rate - r_bar)

Critic-fix rivals (all uncentered, gamma=0.999, offset cells) test whether centering is doing anything that
is not simply "make the critic easier to fit":
  value_clip=False   - unclipped value loss (rsl-rl's clipped loss moves the critic by at most clip_param/update)
  ret_norm="meanstd" - normalised value regression (batch running mean+std of returns, PopArt-lite): the critic
                       fits (R - mu)/sigma, so the clip bound is relative to sigma instead of absolute reward units
  reward_scale="std" - rewards divided by a running std of the raw reward (scale only, no mean shift)

Round-7 arms (d155), all uncentered unless stated:
  center="simple_raw"   - r_bar += beta * (mean RAW reward - r_bar), the raw reward being task - penalty * fell + offset.
                          `simple` tracks the task reward only, which does not contain the offset, so it could not
                          remove one; it is left unchanged for old runs.
  ret_norm="rlgames"    - rl_games value normalisation: the critic regresses normalised returns, its output is
                          denormalised before GAE and the time-out bootstrap, cumulative mean/std over every return, the
                          +-5 clamp, and the (clipped) value loss in normalised space.
  reward_scale="ret_std" - SB3 VecNormalize: reward / running std of the discounted return, no mean subtraction.
  log_rew_std=True       - log the per-step reward std (task and raw) every iteration; pure logging.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rsl_rl.algorithms import PPO


def next_r_bar(center: str, r_bar: float, *, eta: float, beta: float, delta_mean: float, rate: float) -> float:
  """The centering update rule, as a pure function so it can be tested without an env.

  value      - TD-based (Naik et al. Algorithm 3): the fixed point is mean(delta) = 0, i.e. r_bar = the average reward.
  simple     - running average of the observed TASK reward rate.
  simple_raw - d155: the same running average, driven by the raw reward PPO is handed (task - penalty * fell + offset).
               The caller decides which statistic is in `rate`; eta is unused by both simple arms.
  """
  if center == "value":
    return r_bar + eta * delta_mean
  if center in ("simple", "simple_raw"):
    return r_bar + beta * (rate - r_bar)
  return r_bar


class ValueNorm:
  """rl_games `RunningMeanStd(1)` as used for value normalisation (d155).

  Cumulative mean/variance over every return the critic is regressed on (Chan/Welford parallel update, prior mean 0,
  var 1, count 1 - the historical rl_games `init_count`). `normalize` is (x - mean) / sqrt(var + 1e-5) clamped to
  +-clamp; `denormalize` clamps its input to +-clamp before inverting, exactly as rl_games' forward(denorm=True) does.
  `clamp=None` (or 0) turns both clamps off, which is the diagnostic arm.
  """

  def __init__(self, epsilon: float = 1e-5, init_count: float = 1.0, clamp: float | None = 5.0) -> None:
    self.mean = 0.0
    self.var = 1.0
    self.count = float(init_count)
    self.epsilon = float(epsilon)
    self.clamp = float(clamp) if clamp else None

  @property
  def std(self) -> float:
    return math.sqrt(self.var + self.epsilon)

  def update(self, x: torch.Tensor) -> None:
    x = x.detach().double()
    bc = float(x.numel())
    bm = float(x.mean())
    bv = float(x.var(unbiased=False))
    delta = bm - self.mean
    tot = self.count + bc
    m2 = self.var * self.count + bv * bc + delta * delta * self.count * bc / tot
    self.mean = self.mean + delta * bc / tot
    self.var = m2 / tot
    self.count = tot

  def normalize(self, x: torch.Tensor) -> torch.Tensor:
    y = (x - self.mean) / self.std
    return torch.clamp(y, -self.clamp, self.clamp) if self.clamp else y

  def denormalize(self, y: torch.Tensor) -> torch.Tensor:
    if self.clamp:
      y = torch.clamp(y, -self.clamp, self.clamp)
    return y * self.std + self.mean

  def state(self) -> dict:
    return {"mean": self.mean, "var": self.var, "count": self.count}

  def load(self, d: dict) -> None:
    self.mean, self.var, self.count = float(d["mean"]), float(d["var"]), float(d["count"])


class RetStd:
  """SB3 `VecNormalize(norm_reward=True)` reward scaling (d155): r / sqrt(var(discounted return) + 1e-8).

  Per env `R <- gamma * R + r`; the running variance (prior mean 0, var 1, count 1e-4) is updated from the vector of
  per-env returns once per step; the scaled reward is clipped to +-clip (10); `R` is zeroed where an episode ends. No
  mean is subtracted, so a constant offset survives (scaled by 1/std). All state lives on the device: the step
  involves no host sync.
  """

  def __init__(self, num_envs: int, gamma: float, device, epsilon: float = 1e-8, clip: float = 10.0,
               init_count: float = 1e-4) -> None:
    self.gamma = float(gamma)
    self.epsilon = float(epsilon)
    self.clip = float(clip)
    self.ret = torch.zeros(num_envs, device=device, dtype=torch.float64)
    self.mean = torch.zeros((), device=device, dtype=torch.float64)
    self.var = torch.ones((), device=device, dtype=torch.float64)
    self.count = torch.tensor(float(init_count), device=device, dtype=torch.float64)

  @property
  def std(self) -> torch.Tensor:
    return torch.sqrt(self.var + self.epsilon)

  def scale(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    r = rewards.double()
    self.ret = self.ret * self.gamma + r
    bc = float(self.ret.numel())
    bm = self.ret.mean()
    bv = self.ret.var(unbiased=False)
    delta = bm - self.mean
    tot = self.count + bc
    m2 = self.var * self.count + bv * bc + delta * delta * self.count * bc / tot
    self.mean = self.mean + delta * bc / tot
    self.var = m2 / tot
    self.count = tot
    out = torch.clamp(r / self.std, -self.clip, self.clip)
    self.ret = self.ret * (1.0 - dones.double().reshape(-1))
    return out.to(rewards.dtype)

  def state(self) -> dict:
    return {"ret": self.ret.clone(), "mean": self.mean.clone(), "var": self.var.clone(), "count": self.count.clone()}

  def load(self, d: dict) -> None:
    self.ret, self.mean, self.var, self.count = (d[k].to(self.ret.device) for k in ("ret", "mean", "var", "count"))


def adv_by_mode(raw: torch.Tensor, mode: str) -> torch.Tensor:
  """normalise a (num_steps, num_envs) raw advantage for the advantage mode of REPORT_009 round 9b.

  global      - rsl-rl's own rule: one mean and std over the whole batch. A bias that varies with rollout position
                survives it, which is the leak this round tests.
  pos_norm    - standardise each rollout position separately (the pre-registered causal test): removes any
                position-dependent bias exactly, and also reweights positions to equal variance.
  pos_center  - subtract each position's mean, then divide by ONE global std: removes the position bias without
                reweighting positions. The control for "removed the leak" vs "reweighted the positions".
  """
  if mode == "global":
    return (raw - raw.mean()) / (raw.std() + 1e-8)
  if mode not in ("pos_norm", "pos_center"):
    raise ValueError(mode)
  centred = raw - raw.mean(dim=1, keepdim=True)
  # pos_norm: each position rescaled to unit variance. pos_center: one global scale taken from the CENTRED values, so
  # the result has unit global std AND is exactly invariant to a per-position constant (a bias b*w_t is constant within
  # a position, so it cancels in `centred` before the scale is computed).
  denom = centred.std(dim=1, keepdim=True) if mode == "pos_norm" else centred.std()
  return centred / (denom + 1e-8)


def adv_position_stats(raw: torch.Tensor) -> dict:
  """Per-position mean/std of the raw advantage and the share of its variance explained by the rollout position."""
  m = raw.mean(dim=1)
  sd = raw.std(dim=1)
  resid = raw - m.unsqueeze(1)
  total = raw.var(unbiased=False)
  return {"adv_pos_mean": [float(x) for x in m], "adv_pos_std": [float(x) for x in sd],
          "adv_pos_r2": float(1.0 - resid.var(unbiased=False) / (total + 1e-12))}


def critic_output_bias(critic: nn.Module) -> nn.Parameter:
  """The critic's output-layer bias (the parameter a fast output-bias arm learns; d216 is the earlier init trick)."""
  lin = [m for m in critic.modules() if isinstance(m, nn.Linear)]
  assert lin and lin[-1].bias is not None, "critic has no biased linear output layer"
  return lin[-1].bias


class _LROverrideOptimizer:
  """Wrap the PPO optimizer so some parameter groups keep their own learning-rate rule.

  rsl-rl's adaptive-KL schedule writes ``param_group["lr"] = self.learning_rate`` on *every* group, once per
  mini-batch, and then calls ``optimizer.step()``. Overriding ``lr`` inside ``step()`` is therefore the exact point at
  which a group can escape that schedule: the KL loop has just written the global rate, and the step that would use it
  has not happened yet. The wrapper delegates everything else to the real optimizer, so state_dict/load are untouched.

  ``overrides`` maps a group index to a zero-argument callable returning the group's lr for this step (``None`` means
  "no override for this group on this step").
  """

  def __init__(self, opt, owned: dict[int, callable]) -> None:
    self._opt = opt
    self._owned = owned  # group index -> callable(group) -> float | None

  def step(self, closure=None):
    for i, fn in self._owned.items():
      lr = fn()
      if lr is not None:
        self._opt.param_groups[i]["lr"] = float(lr)
    return self._opt.step(closure)

  def __getattr__(self, name):
    return getattr(self._opt, name)

  def __setattr__(self, name, value):
    if name in ("_opt", "_owned"):
      object.__setattr__(self, name, value)
    else:
      setattr(self._opt, name, value)


def init_critic_output_bias(critic: nn.Module, value: float) -> float:
  """Set the bias of the critic's last linear layer to `value` (Step 0 causal pilot, d114).

  The critic's output layer is linear, so a constant bias is the cheapest way to hand the critic the level of the
  return (c / (1 - gamma) for a constant offset c) before any learning happens. Everything else stays at its default
  initialisation. Works through the `_NormCritic` wrapper because it walks `modules()`.
  """
  lin = [m for m in critic.modules() if isinstance(m, nn.Linear)]
  assert lin, "critic has no linear layer"
  with torch.no_grad():
    lin[-1].bias.fill_(float(value))
  return float(value)


class _NormCritic(nn.Module):
  """Wrapper that lets the critic's output be interpreted in normalised return units."""

  def __init__(self, critic: nn.Module) -> None:
    super().__init__()
    self.critic = critic
    self.is_recurrent = bool(getattr(critic, "is_recurrent", False))

  def forward(self, *args, **kwargs):
    return self.critic(*args, **kwargs)

  def reset(self, *args, **kwargs):
    return self.critic.reset(*args, **kwargs)

  def update_normalization(self, *args, **kwargs):
    return self.critic.update_normalization(*args, **kwargs)

  def __getattr__(self, name):  # delegate anything else (output_std, distribution, ...)
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.critic, name)


class CenteredPPO(PPO):
  """PPO with reward centering and the critic-fix arms. Selected by `class_name` in the algorithm config."""

  # Defaults live on the class as well as in __init__ so that the half-constructed instances the unit tests build with
  # `object.__new__` (to exercise one method in isolation) keep the study's default behaviour: no position mode, no
  # per-position logging. Every real run goes through __init__, which sets both.
  adv_mode = "global"
  log_adv_pos = False

  def __init__(
    self,
    *args,
    center: str = "none",
    eta: float = 0.01,
    beta: float = 0.01,
    value_clip: bool | None = None,
    ret_norm: str = "none",
    reward_scale: str = "none",
    r_bar_init: str = "zero",
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
    adv_mode: str = "global",
    log_adv_pos: bool = False,
    **kwargs,
  ) -> None:
    if value_clip is not None:
      kwargs["use_clipped_value_loss"] = bool(value_clip)
    super().__init__(*args, **kwargs)
    assert center in ("none", "value", "simple", "simple_raw"), center
    assert ret_norm in ("none", "meanstd", "rlgames"), ret_norm
    assert reward_scale in ("none", "std", "ret_std"), reward_scale
    assert r_bar_init in ("zero", "first_batch"), r_bar_init
    self.r_bar_init = r_bar_init
    self._r_bar_inited = False
    self.center = center
    self.eta = float(eta)
    self.beta = float(beta)
    self.ret_norm = ret_norm
    self.reward_scale = reward_scale
    self.log_rew_std = bool(log_rew_std)

    # --- round-9 arms (REPORT_009). All three act on the same quantity: how fast the critic's *level* travels.
    # A1: the output bias gets its own optimiser group at a fixed lr (out of the adaptive-KL schedule), so it moves
    #     ~bias_lr per mini-batch instead of ~learning_rate.
    # A2: the actor is not updated (gradients zeroed, actor lr 0) until the critic's mean value reaches warmup_frac of
    #     the analytic level c/(1-gamma), capped at warmup_max_iters iterations; the KL schedule is pinned meanwhile.
    # A3: the actor group runs at actor_lr_scale x the current adaptive lr for the first actor_lr_iters iterations.
    self.bias_lr = None if bias_lr is None else float(bias_lr)
    self.warmup = bool(warmup)
    self.warmup_frac = float(warmup_frac)
    self.warmup_max_iters = int(warmup_max_iters)
    self.actor_lr_scale = None if actor_lr_scale is None else float(actor_lr_scale)
    self.actor_lr_iters = int(actor_lr_iters)
    self.level_target = None if level_target is None else float(level_target)
    assert adv_mode in ("global", "pos_norm", "pos_center"), adv_mode
    if adv_mode != "global":
      # a per-minibatch restandardisation would partly undo per-position centring; the study's config has it False
      assert not getattr(self, "normalize_advantage_per_mini_batch", False), \
        "adv_mode %s needs normalize_advantage_per_mini_batch=False" % adv_mode
    self.adv_mode = adv_mode
    self.log_adv_pos = bool(log_adv_pos)
    self._iter = 0
    self._warmup_active = False
    self._norm_snapshot: dict | None = None
    self._warmup_done_at: int | None = None
    self._warmup_lr0 = float(self.learning_rate)
    self._bias_idx: int | None = None
    self._actor_idx = 0
    self._setup_param_groups()

    self.r_bar = 0.0
    self._rew_mean = 0.0
    self._rew_var = 1.0
    self._ret_mean = 0.0
    self._ret_var = 1.0

    if ret_norm == "meanstd":
      self._raw_critic = self.critic  # save()/export use the unwrapped module
      self.critic = _NormCritic(self.critic)

    # d155 arms. Value normalisation leaves the critic module alone (so save()/compile() are untouched) and converts its
    # output through a forward hook: denormalised everywhere except inside update(), which regresses normalised targets.
    self.value_norm = ValueNorm(clamp=vn_clamp) if ret_norm == "rlgames" else None
    self._vn_raw = True
    if self.value_norm is not None:
      self._raw_critic.register_forward_hook(self._vn_hook)
    self.ret_std = (RetStd(self.storage.num_envs, float(self.gamma), self.device, clip=rs_clip)
                    if reward_scale == "ret_std" else None)

    # per-iteration accumulators, kept as GPU tensors: converting to float inside the rollout forces a device
    # sync every step, which serialises the whole iteration (measured 4 s/iteration against 1 s with syncs removed).
    self._acc: dict[str, torch.Tensor] = {}
    self._acc_reset()
    self.last_diag: dict[str, float] = {}

  def _vn_hook(self, module, inputs, output):
    return self.value_norm.denormalize(output) if self._vn_raw else None

  # ------------------------------------------------------- round-9 arm plumbing
  def _setup_param_groups(self) -> None:
    """Give the actor and the critic's output bias their own groups when an arm needs its own learning-rate rule.

    Without an arm the optimiser is left exactly as rsl-rl built it (one group), so every earlier cell keeps its
    optimiser behaviour bit-for-bit. With an arm the groups are still stepped by one Adam, so a bias lr means "bias
    moves ~lr per mini-batch", which is the quantity the A1 rule is written in.
    """
    self._actor_hooks = []
    self._last_lr: dict[int, float | None] = {}
    if self.bias_lr is None and self.actor_lr_scale is None and not self.warmup:
      return
    opt = self.optimizer
    base = opt.param_groups[0]
    lr = float(base["lr"])
    bias = critic_output_bias(self.critic)
    critic_rest = [p for p in self.critic.parameters() if p is not bias]
    groups = [{"params": list(self.actor.parameters()), "lr": lr}, {"params": critic_rest, "lr": lr}]
    if self.bias_lr is not None:
      groups.append({"params": [bias], "lr": self.bias_lr})
      self._bias_idx = 2
    kw = {k: v for k, v in opt.defaults.items() if k != "lr"}
    self._opt = type(opt)(groups, **kw)
    # During the warm-up EVERY group is pinned: a frozen actor produces kl = 0, the KL schedule then raises its own
    # rate to the 1e-2 ceiling within one iteration, and any group left unowned (the critic) would train at that rate.
    owned_idx = set(range(len(groups))) if self.warmup else {0}
    if self._bias_idx is not None:
      owned_idx.add(self._bias_idx) if not self.warmup else None
    owned: dict[int, callable] = {i: (lambda i=i: self._lr_for(i)) for i in owned_idx}
    self.optimizer = _LROverrideOptimizer(self._opt, owned)

  def _lr_for(self, i: int) -> float | None:
    """Learning rate this group must use on the step about to happen (None = leave the schedule's value)."""
    lr = self._lr_for_inner(i)
    self._last_lr[i] = lr  # what the last mini-batch step actually used; the diagnostic must report this, not lr0
    return lr

  def _lr_for_inner(self, i: int) -> float | None:
    if self._warmup_active:
      return 0.0 if i == self._actor_idx else self._warmup_lr0
    if self.bias_lr is not None and i == self._bias_idx:
      return self.bias_lr
    if (self.actor_lr_scale is not None and i == self._actor_idx and self._iter < self.actor_lr_iters):
      return self.actor_lr_scale * float(self.learning_rate)
    return None

  def _freeze_actor(self) -> None:
    """Freeze the policy completely: gradients zeroed as they accumulate (so Adam's momentum stays flat), lr 0, and the
    actor's observation normaliser held at its warm-up state (the normaliser is part of what the policy computes, and
    rsl-rl updates it every iteration; leaving it to drift would make a "frozen" policy slowly change anyway)."""
    if self._actor_hooks:
      return
    norm = getattr(self.actor, "obs_normalizer", None)
    if norm is not None and getattr(self.actor, "obs_normalization", False):
      self._norm_snapshot = {k: v.detach().clone() for k, v in norm.state_dict().items()}
    def _zero(p):
      if p.grad is not None:
        p.grad.zero_()
    for p in self.actor.parameters():
      self._actor_hooks.append(p.register_post_accumulate_grad_hook(_zero))

  def _unfreeze_actor(self) -> None:
    for h in self._actor_hooks:
      h.remove()
    self._actor_hooks = []
    self._warmup_active = False
    self._norm_snapshot = None
    # the KL schedule ran against a frozen actor (kl = 0) and tried to raise the rate; start from the pre-warm-up rate
    self.learning_rate = self._warmup_lr0
    for g in self.optimizer.param_groups:
      g["lr"] = self._warmup_lr0

  def _warmup_step(self) -> None:
    """A2: freeze the actor until the critic's mean value is at warmup_frac of c/(1-gamma), capped."""
    if not self.warmup or self._warmup_done_at is not None:
      return
    lvl = float(self.last_diag.get("v_mean", 0.0))
    tgt = float(self.level_target or 0.0)
    reached = tgt > 0 and lvl >= self.warmup_frac * tgt
    if reached or self._iter >= self.warmup_max_iters:
      self._unfreeze_actor()
      self._warmup_done_at = self._iter
      return
    self._warmup_active = True
    self._freeze_actor()

  def _acc_reset(self) -> None:
    z = torch.zeros((), device=self.device)
    zd = torch.zeros((), device=self.device, dtype=torch.float64)
    self._acc = {"task": z.clone(), "raw": z.clone(), "seen": z.clone(), "fell": z.clone(), "n": z.clone(),
                 "since_max": z.clone(), "since_p95": z.clone(),
                 "t1": zd.clone(), "t2": zd.clone(), "w1": zd.clone(), "w2": zd.clone()}

  # ------------------------------------------------------------------ collection
  def process_env_step(self, obs, rewards, dones, extras) -> None:
    cc = extras.get("cc")
    if cc is None:
      super().process_env_step(obs, rewards, dones, extras)
      return
    task = cc["task"]
    fell = cc["fell"]
    n = self.storage.num_envs
    raw_rewards = rewards
    self._acc["raw"] += rewards.sum()  # reward as the env emitted it, before any scaling or centering

    if self.reward_scale == "std":
      # the running statistics are updated once per iteration in compute_returns; here the current scale is used
      rewards = rewards / math.sqrt(max(self._rew_var, 1e-8))
    elif self.reward_scale == "ret_std":
      # SB3 VecNormalize style (d155): divide by the running std of the discounted return, no mean subtraction. Applied
      # before super(), so the time-out bootstrap term gamma * V(s) is added in the same scaled units.
      rewards = self.ret_std.scale(rewards, dones)

    if self.center != "none":
      rewards = rewards - self.r_bar

    acc = self._acc
    acc["task"] += task.sum()
    acc["seen"] += rewards.sum()
    acc["fell"] += fell.sum()
    acc["n"] += n
    sr = cc["since_reset"].float()
    acc["since_max"] = torch.maximum(acc["since_max"], sr.max())
    acc["since_p95"] = torch.maximum(acc["since_p95"], torch.quantile(sr, 0.95))
    if self.reward_scale == "std":
      acc["rew_sum"] = acc.get("rew_sum", task.new_zeros(())) + task.sum()
      acc["rew_sqsum"] = acc.get("rew_sqsum", task.new_zeros(())) + (task * task).sum()
    if self.log_rew_std:
      # per-step spread of the reward, over every env-step of this iteration (d155: S_rew was not logged in d154)
      tk, rw = task.double(), raw_rewards.double()
      acc["t1"] += tk.sum()
      acc["t2"] += (tk * tk).sum()
      acc["w1"] += rw.sum()
      acc["w2"] += (rw * rw).sum()
    super().process_env_step(obs, rewards, dones, extras)

  # ------------------------------------------------------------------ targets
  def _maybe_init_r_bar(self) -> None:
    """r_bar_init="first_batch": start r_bar at the first iteration's mean raw reward instead of 0.

    The first iteration was collected with r_bar = 0, so the stored rewards are re-centred here, before returns and
    advantages are computed from them. Done once, centered arms only.
    """
    if self.center == "none" or getattr(self, "r_bar_init", "zero") != "first_batch" or getattr(self, "_r_bar_inited", True):
      return
    n = float(self._acc["n"]) or 1.0
    r0 = float(self._acc["raw"]) / n
    self.storage.rewards -= r0
    self.r_bar = r0
    self._r_bar_inited = True

  def compute_returns(self, obs) -> None:
    st = self.storage
    self._maybe_init_r_bar()
    critic_hidden_state = None
    if getattr(self.critic, "is_recurrent", False):
      critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    if critic_hidden_state is not None:
      self.critic.reset(hidden_state=critic_hidden_state)
    super().compute_returns(obs)

    with torch.no_grad():
      nxt = torch.cat([st.values[1:], last_values.unsqueeze(0)], dim=0)
      delta = st.rewards + (1.0 - st.dones.float()) * self.gamma * nxt - st.values
      delta_mean = float(delta.mean())

      ret, val = st.returns, st.values
      err = ret - val
      err_mean = float(err.mean())
      mse = float((err * err).mean())
      var_ret = float(ret.var(unbiased=False))
      ev = 1.0 - float((err * err).mean()) / (var_ret + 1e-8)

      acc = self._acc
      n = float(acc["n"]) or 1.0
      rate_task = float(acc["task"]) / n
      rate_raw = float(acc["raw"]) / n  # what PPO was handed: task - penalty * fell + offset, before scaling/centering
      if self.reward_scale == "std" and "rew_sum" in acc:
        bm = float(acc["rew_sum"]) / n
        bv = max(float(acc["rew_sqsum"]) / n - bm * bm, 1e-8)
        self._rew_mean = 0.9 * self._rew_mean + 0.1 * bm
        self._rew_var = 0.9 * self._rew_var + 0.1 * bv
      self.r_bar = next_r_bar(
        self.center, self.r_bar, eta=self.eta, beta=self.beta, delta_mean=delta_mean,
        rate=rate_raw if self.center == "simple_raw" else rate_task,
      )

      self.last_diag = {
        "r_task_rate": rate_task,
        "r_seen_rate": float(acc["seen"]) / n,
        "fall_rate_per_1k": 1000.0 * float(acc["fell"]) / n,
        "resets": float(acc["fell"]),
        "steps_since_reset_max": float(acc["since_max"]),
        "steps_since_reset_p95": float(acc["since_p95"]),
        "r_bar": self.r_bar,
        "r_bar_gap": self.r_bar - rate_task,
        "delta_mean": delta_mean,
        "v_mean": float(val.mean()),
        "v_std": float(val.std(unbiased=False)),
        "ret_mean": float(ret.mean()),
        "ev": ev,
        "err_mean": err_mean,
        "err_mse": mse,
        "const_frac": (err_mean * err_mean) / (mse + 1e-8),
        "rew_std": math.sqrt(max(self._rew_var, 1e-12)),
        # self-describing runs: a resumed or re-mixed cell can be read without its config.json (round 9b writes both)
        "lam": float(getattr(self, "lam", float("nan"))),
      }

      if self.log_adv_pos:
        # raw advantage = returns - values (ld in `err` above); logged only for the diagnostic arms
        self.last_diag.update(adv_position_stats(st.returns - st.values))
      if self.adv_mode != "global":
        # d1101: replace rsl-rl's single global standardisation with a per-position one. The critic's targets are
        # untouched -- only what the actor's loss sees changes -- so this isolates the leak from the value level.
        self.last_diag["adv_mode"] = self.adv_mode
        st.advantages = adv_by_mode(st.returns - st.values, self.adv_mode)

      if self.ret_norm == "meanstd":
        bm, bv = float(ret.mean()), float(ret.var(unbiased=False))
        self._ret_mean = 0.9 * self._ret_mean + 0.1 * bm
        self._ret_var = 0.9 * self._ret_var + 0.1 * max(bv, 1e-8)
        sd = math.sqrt(max(self._ret_var, 1e-12))
        st.returns = (st.returns - self._ret_mean) / sd
        st.values = (st.values - self._ret_mean) / sd
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
          st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
        self.last_diag["ret_norm_mean"] = self._ret_mean
        self.last_diag["ret_norm_std"] = sd
        self.last_diag["v_mean_raw"] = self.last_diag["v_mean"] * sd + self._ret_mean
      elif self.value_norm is not None:
        # rl_games value normalisation (d155). Everything above ran on raw rewards and DENORMALISED values (the critic
        # forward hook), so delta, advantages and every diagnostic are in raw units, like the uncentered arm's. Only
        # the regression targets change: the running stats absorb this batch's raw returns, then the returns and the
        # stored old values are expressed in the updated normalised units, so the clipped value loss clips in
        # normalised space exactly as rl_games does.
        vn = self.value_norm
        vn.update(ret)
        st.returns = vn.normalize(st.returns)
        st.values = vn.normalize(st.values)
        self.last_diag["vn_mean"] = vn.mean
        self.last_diag["vn_std"] = vn.std
        self.last_diag["vn_count"] = vn.count
        if vn.clamp:
          self.last_diag["vn_target_clamped"] = float((st.returns.abs() >= vn.clamp).float().mean())

      if self.ret_std is not None:
        self.last_diag["rs_ret_mean"] = float(self.ret_std.mean)
        self.last_diag["rs_ret_std"] = float(self.ret_std.std)
      if self.log_rew_std:
        for name, (s1, s2) in (("task_rew", ("t1", "t2")), ("raw_rew", ("w1", "w2"))):
          m = float(acc[s1]) / n
          self.last_diag[f"{name}_mean"] = m
          self.last_diag[f"{name}_std"] = math.sqrt(max(float(acc[s2]) / n - m * m, 0.0))

    self._acc_reset()

  # ------------------------------------------------------------------ branching state
  _EXTRA_KEYS = ("r_bar", "_rew_mean", "_rew_var", "_ret_mean", "_ret_var", "_r_bar_inited")

  def _state_extra(self) -> dict:
    out = {}
    if self.value_norm is not None:
      out["vn"] = self.value_norm.state()
    if self.ret_std is not None:
      out["rs"] = self.ret_std.state()
    return out

  def export_state(self) -> dict:
    """Everything needed to continue (or branch) a run: weights, optimizer, adaptive lr, centering and norm state."""
    d = self.save()
    d["cc_state"] = {k: getattr(self, k) for k in self._EXTRA_KEYS}
    d["cc_state"]["learning_rate"] = float(self.learning_rate)
    d["cc_state"].update({"_iter": self._iter, "_warmup_done_at": self._warmup_done_at,
                          "_warmup_active": self._warmup_active})
    d["cc_state"].update(self._state_extra())
    return d

  def import_state(self, d: dict) -> None:
    self.load(d, None, True)  # weights (incl. observation normaliser), optimizer, learning rate
    for k, v in d["cc_state"].items():
      if k == "vn" and self.value_norm is not None:
        self.value_norm.load(v)
      elif k == "rs" and self.ret_std is not None:
        self.ret_std.load(v)
      elif k == "_warmup_active":
        self._warmup_active = bool(v)
        if v:
          self._freeze_actor()
      elif k != "learning_rate":
        setattr(self, k, v)
    self.learning_rate = float(d["cc_state"]["learning_rate"])
    for g in self.optimizer.param_groups:
      g["lr"] = self.learning_rate

  # ------------------------------------------------------------------ update
  def update(self) -> dict[str, float]:
    self._warmup_step()
    if self.value_norm is not None:
      # the critic is regressed on normalised targets; every other forward pass (act, compute_returns, eval) sees raw
      self._vn_raw = False
    try:
      loss = super().update()
    finally:
      self._vn_raw = True
    if self._warmup_active:
      # the KL schedule saw a frozen actor (kl = 0) and multiplied the rate up; the warm-up owns the rate until it ends
      if self._norm_snapshot is not None:
        self.actor.obs_normalizer.load_state_dict(self._norm_snapshot)
      self.learning_rate = self._warmup_lr0
      for g in self.optimizer.param_groups:
        g["lr"] = self._warmup_lr0
    # arm diagnostics: the rates actually used, and where the level stands relative to the analytic target
    self.last_diag["bias_lr"] = float(self.bias_lr) if self.bias_lr is not None else float("nan")
    if self.bias_lr is not None:
      self.last_diag["critic_bias"] = float(critic_output_bias(self.critic).detach())
    opt = getattr(self, "optimizer", None)
    eff = getattr(self, "_last_lr", {}).get(self._actor_idx)
    if eff is None and opt is not None:
      eff = float(opt.param_groups[self._actor_idx]["lr"])
    self.last_diag["actor_lr_eff"] = float(eff) if eff is not None else float("nan")
    self.last_diag["warmup_active"] = 1.0 if self._warmup_active else 0.0
    self.last_diag["warmup_done_at"] = float(self._warmup_done_at if self._warmup_done_at is not None else -1)
    tgt = float(self.level_target or 0.0)
    self.last_diag["lvl_frac"] = float(self.last_diag.get("v_mean", float("nan")) / tgt) if tgt else float("nan")
    self._iter += 1
    self.last_diag["value_loss"] = float(loss.get("value", float("nan")))
    self.last_diag["surrogate_loss"] = float(loss.get("surrogate", float("nan")))
    self.last_diag["entropy"] = float(loss.get("entropy", float("nan")))
    # plasticity diagnostics (cheap, freed norms only)
    with torch.no_grad():
      cn = [p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None]
      an = [p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None]
      self.last_diag["critic_grad_norm"] = float(sum(cn) / max(len(cn), 1))
      self.last_diag["actor_grad_norm"] = float(sum(an) / max(len(an), 1))
    return loss
