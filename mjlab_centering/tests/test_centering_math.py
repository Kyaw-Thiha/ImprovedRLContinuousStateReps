"""Centering math: the r_bar rules, and the ordering of centering relative to the time-out bootstrap."""

from __future__ import annotations

import types

import torch

from mjlab_centering.algo import CenteredPPO, next_r_bar


def test_value_rule_converges_to_average_reward():
  # delta_mean = r - r_bar is exactly the TD mean of a constant-reward task with a state-independent V
  r, r_bar = 0.7, 0.0
  for _ in range(4000):
    delta = r - r_bar
    r_bar = next_r_bar("value", r_bar, eta=0.01, beta=0.0, delta_mean=delta, rate=r)
  assert abs(r_bar - r) < 1e-2, r_bar


def test_value_rule_sign_moves_toward_the_reward():
  up = next_r_bar("value", 1.0, eta=0.1, beta=0.0, delta_mean=2.0, rate=0.0)
  down = next_r_bar("value", 1.0, eta=0.1, beta=0.0, delta_mean=-2.0, rate=0.0)
  assert up > 1.0 > down


def test_simple_rule_is_a_running_average():
  r_bar = 0.0
  for _ in range(5000):
    r_bar = next_r_bar("simple", r_bar, eta=0.0, beta=0.05, delta_mean=0.0, rate=0.25)
  assert abs(r_bar - 0.25) < 1e-6


def test_none_rule_is_a_no_op():
  assert next_r_bar("none", 3.0, eta=1.0, beta=1.0, delta_mean=5.0, rate=9.0) == 3.0


def test_centering_is_subtracted_before_the_parent_sees_the_reward(monkeypatch):
  """Centering must happen at collection time: rsl-rl adds gamma*V(s) to the stored reward on time-outs, so
  subtracting r_bar inside compute_returns would also subtract it from that bootstrap term."""
  import rsl_rl.algorithms.ppo as ppo_mod

  captured = {}

  def fake_process_env_step(self, obs, rewards, dones, extras):
    captured["r"] = rewards.clone()

  monkeypatch.setattr(ppo_mod.PPO, "process_env_step", fake_process_env_step)

  alg = object.__new__(CenteredPPO)
  alg.center = "value"
  alg.r_bar = 0.5
  alg.reward_scale = "none"
  alg.log_rew_std = False
  alg._rew_var = 1.0
  alg.device = torch.device("cpu")
  alg._acc_reset()  # accumulators are GPU tensors (no per-step host sync); same contract as the real constructor
  alg.storage = types.SimpleNamespace(num_envs=2)

  seen = torch.tensor([1.0, 2.0])
  cc = {
    "task": torch.tensor([1.0, 2.0]),
    "fell": torch.zeros(2),
    "since_reset": torch.zeros(2, dtype=torch.long),
  }
  CenteredPPO.process_env_step(alg, None, seen, None, {"cc": cc})
  assert torch.allclose(captured["r"], torch.tensor([0.5, 1.5]))
  assert abs(float(alg._acc["task"]) - 3.0) < 1e-6
  assert abs(float(alg._acc["seen"]) - 2.0) < 1e-6


def test_centering_none_passes_the_reward_through(monkeypatch):
  import rsl_rl.algorithms.ppo as ppo_mod

  captured = {}
  monkeypatch.setattr(
    ppo_mod.PPO,
    "process_env_step",
    lambda self, obs, rewards, dones, extras: captured.setdefault("r", rewards.clone()),
  )
  alg = object.__new__(CenteredPPO)
  alg.center = "none"
  alg.r_bar = 99.0
  alg.reward_scale = "none"
  alg.log_rew_std = False
  alg._rew_var = 1.0
  alg.device = torch.device("cpu")
  alg._acc_reset()  # accumulators are GPU tensors (no per-step host sync); same contract as the real constructor
  alg.storage = types.SimpleNamespace(num_envs=2)
  cc = {"task": torch.ones(2), "fell": torch.zeros(2), "since_reset": torch.zeros(2, dtype=torch.long)}
  CenteredPPO.process_env_step(alg, None, torch.tensor([3.0, 4.0]), None, {"cc": cc})
  assert torch.allclose(captured["r"], torch.tensor([3.0, 4.0]))


def _alg(center, init, raw_sum, n_steps_envs, rewards):
  alg = object.__new__(CenteredPPO)
  alg.center = center
  alg.r_bar_init = init
  alg._r_bar_inited = False
  alg.r_bar = 0.0
  alg.device = torch.device("cpu")
  alg._acc_reset()
  alg._acc["raw"] += raw_sum
  alg._acc["n"] += n_steps_envs
  alg.storage = types.SimpleNamespace(rewards=rewards.clone())
  return alg


def test_first_batch_init_sets_r_bar_to_batch_mean_and_recentres_stored_rewards():
  rew = torch.full((4, 3, 1), 1.1)  # 4 steps x 3 envs, collected with r_bar = 0
  alg = _alg("value", "first_batch", raw_sum=1.1 * 12, n_steps_envs=12, rewards=rew)
  alg._maybe_init_r_bar()
  assert abs(alg.r_bar - 1.1) < 1e-5
  assert torch.allclose(alg.storage.rewards, torch.zeros_like(rew), atol=1e-5)
  alg._maybe_init_r_bar()  # once only
  assert torch.allclose(alg.storage.rewards, torch.zeros_like(rew), atol=1e-5)


def test_zero_init_and_uncentered_arms_are_untouched():
  rew = torch.full((4, 3, 1), 1.1)
  for center, init in (("value", "zero"), ("none", "first_batch")):
    alg = _alg(center, init, raw_sum=13.2, n_steps_envs=12, rewards=rew)
    alg._maybe_init_r_bar()
    assert alg.r_bar == 0.0 and torch.allclose(alg.storage.rewards, rew)
