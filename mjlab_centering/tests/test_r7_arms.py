"""Round-7 arms (d155): value normalisation, return-std reward scaling, and simple centering on the raw reward.

The two required value-normalisation properties are checked directly: shifting every reward by a constant must leave
the normalised targets unchanged once the stats have adapted, and denormalised critic outputs must match the targets.
"""

from __future__ import annotations

import math
import types

import pytest
import torch

from mjlab_centering.algo import CenteredPPO, RetStd, ValueNorm, next_r_bar

# ---------------------------------------------------------------------------------------------------- value normalisation


def _stream(n_batches: int, batch: int, shift: float = 0.0, seed: int = 0) -> list[torch.Tensor]:
  """Return-like batches: mean 1000, sd 50 (roughly the Go1 return scale at gamma .999, offset +1)."""
  g = torch.Generator().manual_seed(seed)
  base = (1000.0 + 50.0 * torch.randn(n_batches, batch, generator=g, dtype=torch.float64))
  return [base[i] + shift for i in range(n_batches)]


def test_denormalised_outputs_match_the_targets():
  vn = ValueNorm()
  x = torch.tensor([900.0, 1000.0, 1100.0, 1010.0])
  vn.update(x)
  assert torch.allclose(vn.denormalize(vn.normalize(x)), x, atol=1e-6)
  # the clamp is rl_games': +-5 in normalised units, applied in both directions
  assert float(vn.normalize(torch.tensor([1e6]))) == pytest.approx(5.0)
  assert float(vn.denormalize(torch.tensor([1e6]))) == pytest.approx(vn.mean + 5.0 * vn.std, rel=1e-6)
  assert float(vn.denormalize(torch.tensor([0.0]))) == pytest.approx(vn.mean)


def test_a_constant_shift_leaves_the_normalised_targets_unchanged():
  """The required property, with the size of the residual stated rather than assumed.

  rl_games' prior (mean 0, var 1, count 1) is not shift-invariant: after N samples the running mean carries a residual
  of K / (N+1) and the variance a term of the same order. So the normalised targets converge to the unshifted ones as the
  stats adapt: the residual has to fall with N, and be negligible at a realistic run length (~200 batches of 1024, the
  order of a 2000-iteration run at 2048 envs).
  """
  def probe_gap(n_batches: int) -> tuple[float, float]:
    a, b = ValueNorm(), ValueNorm()
    for i, (xa, xb) in enumerate(zip(_stream(n_batches, 1024), _stream(n_batches, 1024, shift=1000.0))):
      a.update(xa)
      b.update(xb)
    g = torch.Generator().manual_seed(5)
    x = 1000.0 + 50.0 * torch.randn(512, generator=g, dtype=torch.float64)
    return float((a.normalize(x) - b.normalize(x + 1000.0)).abs().max()), b.mean - a.mean

  gap200, dmean200 = probe_gap(200)
  gap2000, dmean2000 = probe_gap(2000)
  n = 200 * 1024
  assert dmean200 == pytest.approx(1000.0 * n / (n + 1), rel=1e-3)  # the prior's pull, analytically
  assert gap200 < 1e-2 and gap2000 < 2e-3, (gap200, gap2000)       # targets unchanged to 1% then 0.2% of a std
  assert gap2000 < gap200                                         # and the residual decays with N
  # denormalising the normalised target returns the target itself: what "denormalise before GAE" needs
  a, b = ValueNorm(), ValueNorm()
  for xa, xb in zip(_stream(50, 1024), _stream(50, 1024, shift=1000.0)):
    a.update(xa)
    b.update(xb)
  assert torch.allclose(b.denormalize(b.normalize(xb)), xb, atol=1e-6)


def test_a_mid_stream_shift_decays_as_the_post_shift_data_dominate():
  """The half of the property that cannot hold instantly. A constant shift present from the start is absorbed (above).
  One that arrives mid-run is not: the cumulative mean is displaced by K * N_pre / N_total, and the cumulative variance
  is inflated by the mixture term p * (1 - p) * K^2, which peaks when the pre- and post-shift masses are comparable and
  decays as the pre-shift mass vanishes. The level displacement must fall monotonically; the variance inflation must
  peak in the transition and fall afterwards. Both are why VN should handle an offset present from iteration 0 but
  cannot absorb a mid-run level shock the way reward centering does."""
  def stats_after(n_batches: int):
    vn = ValueNorm()
    for x in _stream(300, 2048):
      vn.update(x)
    shifted = _stream(n_batches, 2048, shift=1000.0, seed=3)
    for x in shifted:
      vn.update(x)
    oracle = ValueNorm()
    for x in _stream(300 + n_batches, 2048, shift=1000.0, seed=3):
      oracle.update(x)
    return abs(vn.mean - oracle.mean), vn.std / oracle.std

  gaps = [stats_after(k) for k in (10, 100, 1000, 3000)]
  means = [g[0] for g in gaps]
  scales = [g[1] for g in gaps]
  assert means[0] > 500.0                                          # un-absorbed right after the shift
  assert all(b < a for a, b in zip(means, means[1:])), means       # ... decaying as 1/N
  assert means[-1] < 0.12 * 1000.0, means                          # within 12% of the shift after 3000 batches
  assert scales[1] > scales[0] > 1.0                               # variance inflation grows into the transition ...
  assert scales[-1] < scales[1]                                    # ... and falls as the old mass vanishes


class _Crit(torch.nn.Module):
  """Critic stub: returns a fixed output of shape (N, 1)."""

  is_recurrent = False

  def __init__(self, value: float, n: int = 2) -> None:
    super().__init__()
    self.v = torch.full((n, 1), float(value))

  def forward(self, obs):
    return self.v.clone()


def _stub_alg(*, center="none", ret_norm="none", reward_scale="none", gamma=0.999, n=2, last_value=1000.0):
  alg = object.__new__(CenteredPPO)
  alg.center = center
  alg.r_bar_init = "first_batch" if center in ("value", "simple_raw") else "zero"
  alg._r_bar_inited = True
  alg.r_bar = 0.0
  alg._rew_mean = 0.0
  alg._rew_var = 1.0
  alg._ret_mean = 0.0
  alg._ret_var = 1.0
  alg.eta = 0.03
  alg.beta = 0.03
  alg.ret_norm = ret_norm
  alg.reward_scale = reward_scale
  alg.log_rew_std = False
  alg.value_norm = ValueNorm() if ret_norm == "rlgames" else None
  alg._vn_raw = True
  alg.ret_std = None
  alg.gamma = gamma
  alg.normalize_advantage_per_mini_batch = False
  alg.device = torch.device("cpu")
  alg.critic = _Crit(last_value, n)
  alg.storage = types.SimpleNamespace(num_envs=n)
  alg.last_diag = {}
  alg._acc_reset()
  return alg


def _fake_parent_returns(self, obs):
  """rsl-rl's compute_returns with the GAE chain collapsed: returns = the stored rewards, in whatever units they are."""
  st = self.storage
  st.returns = st.rewards.clone()
  st.advantages = st.returns - st.values
  if not self.normalize_advantage_per_mini_batch:
    st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)


def _storage(alg, rewards, values, dones=None):
  alg.storage.rewards = rewards.clone()
  alg.storage.values = values.clone()
  alg.storage.dones = torch.zeros_like(rewards) if dones is None else dones.clone()
  alg.storage.returns = torch.zeros_like(rewards)
  alg.storage.advantages = torch.zeros_like(rewards)


def _acc(alg, raw_sum, task_sum, n):
  alg._acc_reset()
  alg._acc["raw"] += raw_sum
  alg._acc["task"] += task_sum
  alg._acc["n"] += n


def test_value_norm_keeps_gae_in_raw_units_and_normalises_only_the_targets(monkeypatch):
  import rsl_rl.algorithms.ppo as ppo_mod

  monkeypatch.setattr(ppo_mod.PPO, "compute_returns", _fake_parent_returns)
  alg = _stub_alg(ret_norm="rlgames")
  g = torch.Generator().manual_seed(1)
  rew = 1.1 + 0.1 * torch.randn(4, 2, generator=g).unsqueeze(-1)
  val = 1100.0 + 5.0 * torch.randn(4, 2, generator=g).unsqueeze(-1)
  _storage(alg, rew, val)
  _acc(alg, rew.sum(), rew.sum(), rew.numel())
  raw_ret, raw_val = rew.clone(), val.clone()
  raw_adv = (raw_ret - raw_val) - (raw_ret - raw_val).mean()
  raw_adv = raw_adv / (raw_adv.std() + 1e-8)
  alg.compute_returns(None)

  vn = ValueNorm()
  vn.update(raw_ret)
  assert torch.allclose(alg.storage.returns, vn.normalize(raw_ret), atol=1e-6)
  assert torch.allclose(alg.storage.values, vn.normalize(raw_val), atol=1e-6)
  assert torch.allclose(alg.storage.advantages, raw_adv, atol=1e-6)  # the advantage is the raw one, untouched
  # every diagnostic stays in raw units, so the arm is readable beside the uncentered and value cells
  assert alg.last_diag["v_mean"] == pytest.approx(float(raw_val.mean()), rel=1e-6)
  assert alg.last_diag["ret_mean"] == pytest.approx(float(raw_ret.mean()), rel=1e-6)
  assert alg.last_diag["vn_mean"] == pytest.approx(float(vn.mean), rel=1e-9)


def test_the_difference_from_the_old_meanstd_rival_is_the_value_target_not_the_advantage(monkeypatch):
  """What differed from ret_norm="meanstd" (d155), stated precisely: rsl-rl batch-standardises the advantage, which is
  affine invariant, so both arms hand the actor the same policy gradient. The difference is the value regression target
  and the clip space: meanstd rescales returns and stored values by an EMA of the current batch (so the clip bound
  tracks one batch), VN normalises with the cumulative stats and denormalises the critic before GAE."""
  import rsl_rl.algorithms.ppo as ppo_mod

  monkeypatch.setattr(ppo_mod.PPO, "compute_returns", _fake_parent_returns)
  g = torch.Generator().manual_seed(2)
  rew = 1.1 + 0.1 * torch.randn(4, 2, generator=g).unsqueeze(-1)
  val = 1100.0 + 5.0 * torch.randn(4, 2, generator=g).unsqueeze(-1)
  got = {}
  for arm in ("rlgames", "meanstd"):
    alg = _stub_alg(ret_norm=arm)
    _storage(alg, rew, val)
    _acc(alg, rew.sum(), rew.sum(), rew.numel())
    alg.compute_returns(None)
    got[arm] = (alg.storage.returns.clone(), alg.storage.values.clone(), alg.storage.advantages.clone())
  raw_adv = (rew - val) - (rew - val).mean()
  raw_adv = raw_adv / (raw_adv.std() + 1e-8)
  assert torch.allclose(got["rlgames"][2], raw_adv, atol=1e-6)   # the advantage is the raw one
  assert torch.allclose(got["meanstd"][2], raw_adv, atol=1e-3)   # ... and so is the rival's, after standardising
  # the value target differs: cumulative stats for VN, an EMA seeded at mean 0 / var 1 for the rival
  vn = ValueNorm()
  vn.update(rew)
  assert torch.allclose(got["rlgames"][0], vn.normalize(rew), atol=1e-6)
  assert not torch.allclose(got["meanstd"][0], got["rlgames"][0], atol=1e-3)
  assert not torch.allclose(got["meanstd"][1], got["rlgames"][1], atol=1e-3)


def test_the_hook_denormalises_everywhere_but_inside_update():
  class Alg:
    pass

  class M(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.out = torch.tensor([[1.0], [0.0]])

    def forward(self, x):
      return self.out

  a = Alg()
  a.value_norm = ValueNorm()
  a.value_norm.mean, a.value_norm.var, a.value_norm.count = 1000.0, 400.0, 50.0
  a._vn_raw = True
  m = M()
  m.register_forward_hook(CenteredPPO._vn_hook.__get__(a, Alg))
  assert torch.allclose(m(None), torch.tensor([[1020.0], [1000.0]]))  # denormalised: raw units for GAE
  a._vn_raw = False
  assert torch.allclose(m(None), torch.tensor([[1.0], [0.0]]))       # normalised targets for the regression


def test_update_flips_the_critic_into_normalised_units(monkeypatch):
  import rsl_rl.algorithms.ppo as ppo_mod

  seen = {}

  def fake_update(self):
    seen["raw"] = self._vn_raw
    return {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}

  monkeypatch.setattr(ppo_mod.PPO, "update", fake_update)
  alg = object.__new__(CenteredPPO)
  alg.value_norm, alg._vn_raw = ValueNorm(), True
  alg.critic = _Crit(0.0)
  alg.actor = torch.nn.Linear(1, 1)
  alg.last_diag = {}
  # round-9 arm state (no arm selected here: the object is built without __init__ on purpose)
  alg.warmup, alg._warmup_active, alg._warmup_done_at, alg._iter = False, False, None, 0
  alg.bias_lr, alg.actor_lr_scale, alg.level_target = None, None, None
  alg._actor_idx = 0
  CenteredPPO.update(alg)
  assert seen["raw"] is False and alg._vn_raw is True


# ----------------------------------------------------------------------------------------------- return-std scaling


def _ref_ret_std(rewards, dones, gamma, epsilon=1e-8, clip=10.0, init_count=1e-4):
  ret, mean, var, count = 0.0, 0.0, 1.0, init_count
  out = []
  for r, d in zip(rewards, dones):
    ret = gamma * ret + r
    bc, bm = 1.0, ret
    delta = bm - mean
    tot = count + bc
    m2 = var * count + delta * delta * count * bc / tot
    mean += delta * bc / tot
    var = m2 / tot
    count = tot
    out.append(max(-clip, min(clip, r / math.sqrt(var + epsilon))))
    ret *= 1.0 - d
  return out


def test_return_std_scaling_matches_the_sb3_normaliser():
  torch.manual_seed(0)
  rs = RetStd(1, 0.99, torch.device("cpu"))
  rewards = torch.randn(60, dtype=torch.float64)
  dones = torch.zeros(60)
  dones[[17, 41]] = 1.0
  got = torch.cat([rs.scale(rewards[i:i + 1], dones[i:i + 1]) for i in range(60)]).double()
  ref = _ref_ret_std(rewards.tolist(), dones.tolist(), 0.99)
  assert torch.allclose(got, torch.tensor(ref, dtype=torch.float64), atol=1e-9)


def test_return_std_scaling_keeps_a_constant_offset():
  """No mean subtraction: a +1 offset survives, divided by the return std only - which is why this arm should not fix
  the offset failure."""
  rs = RetStd(1, 0.999, torch.device("cpu"))
  for _ in range(200):
    rs.scale(torch.tensor([1.1]), torch.zeros(1))
  out = float(rs.scale(torch.tensor([1.1]), torch.zeros(1)))
  assert out == pytest.approx(1.1 / float(rs.std), rel=1e-6)
  assert out > 0.0  # the level is still there: a mean-removing arm would put it at 0
  assert float(rs.mean) > 0.0


def test_return_std_is_applied_before_the_parent_sees_the_reward(monkeypatch):
  import rsl_rl.algorithms.ppo as ppo_mod

  captured = {}
  monkeypatch.setattr(ppo_mod.PPO, "process_env_step",
                      lambda self, obs, rewards, dones, extras: captured.setdefault("r", rewards.clone()))
  alg = object.__new__(CenteredPPO)
  alg.center = "none"
  alg.reward_scale = "ret_std"
  alg._rew_var = 1.0
  alg.log_rew_std = False
  alg.device = torch.device("cpu")
  alg._acc_reset()
  alg.storage = types.SimpleNamespace(num_envs=2)
  alg.ret_std = RetStd(2, 0.999, torch.device("cpu"))
  alg.ret_std.var = torch.tensor(4.0)
  seen = torch.tensor([1.0, 2.0])
  cc = {"task": torch.tensor([0.0, 1.0]), "fell": torch.zeros(2), "since_reset": torch.zeros(2, dtype=torch.long)}
  CenteredPPO.process_env_step(alg, None, seen, torch.zeros(2), {"cc": cc})
  # SB3's prior (var 1, count 1e-4) is washed out by the batch, so the scale is read from the updated stats
  assert torch.allclose(captured["r"], seen / float(alg.ret_std.std), rtol=1e-6)


# ------------------------------------------------------------------------------------------------- simple centering


def test_simple_centering_is_driven_by_the_raw_reward_not_the_task_reward(monkeypatch):
  """SC's r_bar must converge to the reward PPO is handed (task + offset), not to the task term alone."""
  import rsl_rl.algorithms.ppo as ppo_mod

  monkeypatch.setattr(ppo_mod.PPO, "compute_returns", _fake_parent_returns)
  bars = {}
  for center in ("simple", "simple_raw"):
    alg = _stub_alg(center=center)
    alg.r_bar = 1.2  # the first batch had task 0.2 + offset 1.0
    for _ in range(300):  # then the policy only collects the offset: raw 1.0, task 0.0
      _storage(alg, torch.ones(2, 2, 1), torch.zeros(2, 2, 1))
      _acc(alg, torch.tensor(4.0), torch.tensor(0.0), 4.0)
      alg.compute_returns(None)
    bars[center] = alg.r_bar
  assert bars["simple_raw"] == pytest.approx(1.0, abs=1e-3)  # the offset is included, so r_bar removes it
  assert bars["simple"] == pytest.approx(0.0, abs=1e-3)      # the task term carries no offset


def test_simple_centering_updates_with_beta_and_ignores_eta():
  assert next_r_bar("simple_raw", 1.0, eta=0.03, beta=0.03, delta_mean=99.0, rate=2.0) == pytest.approx(1.03)
  assert next_r_bar("simple_raw", 1.0, eta=0.0, beta=0.0, delta_mean=99.0, rate=2.0) == 1.0


# ------------------------------------------------------------------------------------------------- reward-std logging


def test_reward_std_logging_reports_the_per_step_spread(monkeypatch):
  import rsl_rl.algorithms.ppo as ppo_mod

  monkeypatch.setattr(ppo_mod.PPO, "process_env_step", lambda self, *a, **k: None)
  monkeypatch.setattr(ppo_mod.PPO, "compute_returns", _fake_parent_returns)
  alg = _stub_alg()
  alg.log_rew_std = True
  rew = torch.tensor([0.0, 2.0])  # per-step reward with mean 1 and std 1
  cc = {"task": rew.clone(), "fell": torch.zeros(2), "since_reset": torch.zeros(2, dtype=torch.long)}
  CenteredPPO.process_env_step(alg, None, rew, torch.zeros(2), {"cc": cc})
  _storage(alg, rew.reshape(2, 1, 1).expand(2, 2, 1).clone(), torch.zeros(2, 2, 1))
  alg.compute_returns(None)
  assert alg.last_diag["task_rew_mean"] == pytest.approx(1.0)
  assert alg.last_diag["task_rew_std"] == pytest.approx(1.0)
  assert alg.last_diag["raw_rew_std"] == pytest.approx(1.0)
  assert alg.last_diag["raw_rew_mean"] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------- checkpoints


def test_value_norm_state_survives_export_and_import():
  alg = object.__new__(CenteredPPO)
  alg.value_norm = ValueNorm()
  alg.value_norm.update(torch.tensor([10.0, 12.0, 14.0]))
  alg.ret_std = None
  d = {"cc_state": {"learning_rate": 1e-3, **alg._state_extra()}}
  alg2 = object.__new__(CenteredPPO)
  alg2.value_norm = ValueNorm()
  alg2.ret_std = None
  alg2.load = lambda *a, **k: True
  alg2.optimizer = types.SimpleNamespace(param_groups=[{}])
  CenteredPPO.import_state(alg2, d)
  assert (alg2.value_norm.mean, alg2.value_norm.var, alg2.value_norm.count) == pytest.approx(
    (alg.value_norm.mean, alg.value_norm.var, alg.value_norm.count))
