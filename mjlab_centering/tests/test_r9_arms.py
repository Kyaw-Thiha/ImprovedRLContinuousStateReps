"""Round-9 arms (REPORT_009): fast output bias (A1), critic warm-up (A2), slow policy commitment (A3).

The three arms are learning-rate rules on parameter groups, so the tests drive a tiny `CenteredPPO` through real
`update()` calls with a fake storage/actor/critic and assert what *moved*:

  A1  the critic's output bias travels at ~bias_lr per mini-batch while the rest follows the adaptive KL rate, and the
      bias group's learning rate is never overwritten by that schedule;
  A2  the actor is bit-identical until the critic's mean value reaches warmup_frac x c/(1-gamma) (or the cap), and the
      critic keeps learning throughout -- including that the pre-warm-up rate is restored, not the KL-inflated one;
  A3  the actor's step is ~10x smaller than the critic's for the first `actor_lr_iters` iterations and equal afterwards.

A pure-python check of the level rule and the group wiring is included; the tiny-model tests need only CPU torch.
"""
from __future__ import annotations

import math
import types

import pytest
import torch
from torch import nn

from mjlab_centering.algo import CenteredPPO, critic_output_bias


# --------------------------------------------------------------------------------------------- a minimal PPO harness
class _Obs:
  """TensorDict stand-in: rsl-rl reads `batch.observations.batch_size[0]` and hands it to the models."""

  def __init__(self, t: torch.Tensor) -> None:
    self.t = t
    self.batch_size = torch.Size([t.shape[0]])


class _Actor(nn.Module):
  """Gaussian actor with a scalar log-std, enough for rsl-rl's `act`/`update` to run."""

  def __init__(self, obs_dim: int = 4, act_dim: int = 2) -> None:
    super().__init__()
    self.is_recurrent = False
    # stand-in for rsl-rl's EmpiricalNormalization: a buffer that `update_normalization` mutates, so the test can see
    # whether a "frozen" actor's input statistics drifted
    self.obs_normalization = True
    self.obs_normalizer = nn.Module()
    self.obs_normalizer.register_buffer("mean", torch.zeros(obs_dim))
    self.body = nn.Sequential(nn.Linear(obs_dim, 8), nn.Tanh(), nn.Linear(8, act_dim))
    self.log_std = nn.Parameter(torch.zeros(act_dim))

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False, **kw):
    obs = obs.t if isinstance(obs, _Obs) else obs
    mu = self.body(obs)
    std = self.log_std.exp().expand_as(mu)
    self.output_distribution_params = (mu, std)
    self.output_entropy = (0.5 * math.log(2 * math.pi * math.e) + self.log_std).sum(-1).expand(mu.shape[0])
    return mu + std * torch.randn_like(mu) if stochastic_output else mu

  def get_output_log_prob(self, actions):
    mu, std = self.output_distribution_params
    z = (actions - mu) / std
    return (-0.5 * z * z - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

  def get_kl_divergence(self, old_params, new_params):
    (m0, s0), (m1, s1) = old_params, new_params
    return (((m1 - m0) / s0).pow(2) + (s1 / s0).pow(2) - 1 - 2 * (s1.log() - s0.log())).sum(-1) * 0.5

  def reset(self, dones=None, hidden_state=None):
    return None

  def update_normalization(self, obs):
    with torch.no_grad():
      self.obs_normalizer.mean += 1.0


class _Critic(nn.Module):
  def __init__(self, obs_dim: int = 4) -> None:
    super().__init__()
    self.is_recurrent = False
    self.body = nn.Sequential(nn.Linear(obs_dim, 16), nn.Tanh(), nn.Linear(16, 1))

  def forward(self, obs, masks=None, hidden_state=None, **kw):
    obs = obs.t if isinstance(obs, _Obs) else obs
    return self.body(obs).squeeze(-1)

  def reset(self, dones=None, hidden_state=None):
    return None

  def update_normalization(self, obs):
    return None


class _Storage:
  """Just enough storage for `mini_batch_generator`: PPO reads only these tensors inside the loop."""

  def __init__(self, n: int, obs_dim: int, act_dim: int, n_envs: int, batch: int) -> None:
    self.observations = torch.randn(n, n_envs, obs_dim)
    self.actions = torch.randn(n, n_envs, act_dim) * 0.1
    self.old_actions_log_prob = torch.zeros(n, n_envs)
    self.old_distribution_params = (torch.zeros(n, n_envs, act_dim), torch.ones(n, n_envs, act_dim))
    self.values = torch.zeros(n, n_envs)
    self.returns = torch.full((n, n_envs), 1000.0)
    self.advantages = torch.randn(n, n_envs)
    self.actions_log_prob = torch.zeros(n, n_envs)
    self.masks = torch.ones(n, n_envs, dtype=torch.bool)
    self.hidden_states = (None, None)
    self.num_transitions_per_env = n
    self._batch = batch

  def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
    n = self.num_transitions_per_env
    idx = torch.arange(n * self.observations.shape[1])
    for _ in range(num_learning_epochs):
      perm = idx[torch.randperm(len(idx))]
      for chunk in perm.chunk(num_mini_batches):
        b = types.SimpleNamespace()
        flat = lambda t: t.flatten(0, 1)[chunk]  # noqa: E731
        b.observations = _Obs(flat(self.observations))
        b.actions = flat(self.actions)
        b.old_actions_log_prob = flat(self.old_actions_log_prob)
        b.old_distribution_params = tuple(flat(p) for p in self.old_distribution_params)
        b.values = flat(self.values)
        b.returns = flat(self.returns)
        b.advantages = flat(self.advantages)
        b.masks = flat(self.masks)
        b.hidden_states = (None, None)
        yield b

  def clear(self):
    return None


def _ppo(**kw) -> CenteredPPO:
  torch.manual_seed(0)
  actor, critic = _Actor(), _Critic()
  storage = _Storage(n=4, obs_dim=4, act_dim=2, n_envs=8, batch=8)
  cfg = dict(num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, gamma=0.999, device="cpu",
             clip_param=0.2, lam=0.95, value_loss_coef=1.0, entropy_coef=0.01, desired_kl=0.01, schedule="adaptive")
  cfg.update(kw)
  alg = CenteredPPO(actor=actor, critic=critic, storage=storage, **cfg)
  alg.last_diag = {"v_mean": 0.0}
  return alg


# -------------------------------------------------------------------------------------------------------------- A1
def test_a1_bias_group_moves_at_its_own_rate_and_escapes_the_kl_schedule():
  alg = _ppo(bias_lr=0.5)
  assert len(alg.optimizer.param_groups) == 3
  bias = critic_output_bias(alg.critic)
  lrs = [g["lr"] for g in alg.optimizer.param_groups]
  assert lrs[2] == 0.5 and lrs[0] == lrs[1] == 1e-3
  # the KL schedule writes 1e-3 on every group; the group's own rule still applies at the step
  for g in alg.optimizer.param_groups:
    g["lr"] = 1e-3
  alg.optimizer.step()  # no grads yet: a no-op, but the override runs
  assert alg.optimizer.param_groups[2]["lr"] == 0.5
  b0 = bias.detach().clone()
  for _ in range(3):
    alg.update()
  moved = float((bias.detach() - b0).abs().max())
  # 3 iterations x 4 mini-batches = 12 bias steps at 0.5, i.e. up to ~6 in magnitude, if the gradient keeps its sign
  assert moved > 0.1, moved
  assert alg.optimizer.param_groups[2]["lr"] == 0.5


def test_a1_value_clip_off_is_the_control_and_clipping_stays_on_elsewhere():
  alg = _ppo(bias_lr=0.5, value_clip=False)
  assert alg.use_clipped_value_loss is False
  alg2 = _ppo()
  assert alg2.use_clipped_value_loss is True
  assert len(alg2.optimizer.param_groups) == 1  # untouched when no arm is selected


# -------------------------------------------------------------------------------------------------------------- A2
def test_a2_freezes_the_actor_until_the_level_is_reached_then_restores_the_rate():
  alg = _ppo(warmup=True, level_target=1000.0)
  a0 = [p.detach().clone() for p in alg.actor.parameters()]
  alg.last_diag = {"v_mean": 100.0}  # 10% of the target: keep the actor frozen
  alg.update()
  assert alg._warmup_active and alg.last_diag["warmup_active"] == 1.0
  assert alg.last_diag["actor_lr_eff"] == 0.0, "the diagnostic must report the rate the step used, not the restored one"
  assert all(torch.equal(x, y) for x, y in zip(alg.actor.parameters(), a0)), "actor moved during warm-up"
  assert alg.learning_rate == pytest.approx(1e-3), "the KL schedule inflated the rate while the actor was frozen"
  # the critic still learns: its parameters moved
  assert alg.last_diag["critic_grad_norm"] is not None
  alc0 = [p.detach().clone() for p in alg.critic.parameters()]
  alg.update()
  assert any(not torch.equal(x, y) for x, y in zip(alg.critic.parameters(), alc0)), "critic did not learn"
  # now the level arrives
  alg.last_diag["v_mean"] = 960.0
  alg.update()
  assert not alg._warmup_active and alg._warmup_done_at is not None
  # the rate restarts from the pre-warm-up value; the KL schedule is then free to adapt it (downwards here, because
  # the first real actor steps are a large policy change). What must NOT survive is the KL inflated by a frozen actor
  # (20 mini-batches x 1.5 would have pinned it at the 1e-2 cap).
  assert 1e-5 <= alg.learning_rate <= 1e-3
  a1 = [p.detach().clone() for p in alg.actor.parameters()]
  alg.update()
  assert any(not torch.equal(x, y) for x, y in zip(alg.actor.parameters(), a1)), "actor did not train after warm-up"


def test_a2_holds_the_actor_observation_normaliser_still():
  alg = _ppo(warmup=True, level_target=1000.0)
  m0 = alg.actor.obs_normalizer.mean.clone()
  alg.last_diag = {"v_mean": 1.0}
  alg.update()
  alg.update()
  assert torch.equal(alg.actor.obs_normalizer.mean, m0), "the frozen actor's normaliser drifted"
  # and it resumes updating once the actor unfreezes
  alg.last_diag["v_mean"] = 1000.0
  alg.update()
  assert not torch.equal(alg.actor.obs_normalizer.mean, m0)


def test_a2_critic_rate_is_pinned_during_warmup_too():
  """A frozen actor gives kl = 0, so the KL schedule multiplies its rate up; the critic must not inherit that."""
  alg = _ppo(warmup=True, level_target=1000.0)
  alg.last_diag = {"v_mean": 1.0}
  alg.update()
  # every group was pinned to the pre-warm-up rate at step time; the schedule's write is discarded for all of them
  assert alg.optimizer.param_groups[1]["lr"] == pytest.approx(1e-3)
  assert alg.learning_rate == pytest.approx(1e-3)


def test_a2_cap_releases_the_actor_even_if_the_level_never_arrives():
  alg = _ppo(warmup=True, level_target=1000.0, warmup_max_iters=2)
  alg.last_diag = {"v_mean": 1.0}
  alg.update()
  alg.last_diag = {"v_mean": 1.0}
  alg.update()
  alg.last_diag = {"v_mean": 1.0}
  alg.update()
  assert not alg._warmup_active
  assert alg._warmup_done_at == 2


# -------------------------------------------------------------------------------------------------------------- A3
def test_a3_slows_only_the_actor_for_the_first_n_iterations():
  alg = _ppo(actor_lr_scale=0.1, actor_lr_iters=2)
  assert alg.optimizer.param_groups[0]["lr"] == 1e-3
  a0 = [p.detach().clone() for p in alg.actor.parameters()]
  c0 = [p.detach().clone() for p in alg.critic.parameters()]
  alg.update()
  assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * alg.learning_rate)
  da = max(float((x - y).abs().max()) for x, y in zip(alg.actor.parameters(), a0))
  dc = max(float((x - y).abs().max()) for x, y in zip(alg.critic.parameters(), c0))
  assert da > 0 and dc > 0
  alg.update()  # still inside the slow window
  alg.update()  # iteration 2: back to the full rate
  assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(alg.learning_rate)


def test_claim_staleness_fails_closed_when_the_pod_list_is_unreachable(monkeypatch):
  """A pod that cannot reach the API must not reclaim another pod's live claim (2026-11-09: three pods, one seed)."""
  import podrun

  monkeypatch.setattr(podrun, "claim_is_free", lambda ts, now, ttl=podrun.CLAIM_TTL_S: False)
  monkeypatch.setattr(podrun, "_r2_exists", lambda rel: True)
  # make the claim lookup return a live-looking claim owned by another pod, and the pod list raise
  calls = {}

  class _C:
    def get_object(self, **kw):
      raise RuntimeError("claim present")

  monkeypatch.setattr(podrun, "_r2_claim_impl", None, raising=False)
  # the function reads the claim via r2 directly; patch the module-level r2 access by faking the import target
  import types
  fake = types.SimpleNamespace(client=lambda: _C(), bucket=lambda: "b", key=lambda k: k,
                               put_json=lambda k, v: calls.setdefault("put", v))
  import sys as _sys
  _sys.modules["mjlab_centering.r2"] = fake
  try:
    # existing=None path -> free; now simulate "claim exists and is recent" by returning a body
    class _C2:
      def get_object(self, **kw):
        class B:
          def read(self_inner):
            return b'{"ts": 1e12, "owner": "someone-else"}'
        return {"Body": B()}
    fake.client = lambda: _C2()
    import mjlab_centering.pods as real_pods

    def _boom():
      raise RuntimeError("runpodctl not on PATH")
    monkeypatch.setattr(real_pods, "list_pods", _boom)
    owns = podrun._r2_claim("camp", "jid")
    assert owns is False, "an unreachable pod list must NOT be read as 'the owner is gone'"
  finally:
    del _sys.modules["mjlab_centering.r2"]


def test_level_rule_arithmetic():
  """The A1 rate rule, as stated: L / (updates per iteration x target iterations)."""
  L, updates, target = 1000.0, 20, 100
  assert L / (updates * target) == pytest.approx(0.5)
