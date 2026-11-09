"""Environment-side checks: the continuing wrapper, the shock hook, and the SSP parity port.

These build real mjlab environments (mujoco-warp), so they need a GPU or CPU fallback and take ~1 min on a cold
kernel cache. Keep them small: 4 envs, a few dozen steps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mjlab_centering.env_cfg import apply_shock, build_env_cfg
from mjlab_centering.ssp_torch import HexSSP
from mjlab_centering.wrapper import ContinuingVecEnv


@pytest.fixture(scope="module")
def go1():
  from mjlab.envs import ManagerBasedRlEnv

  # The legacy protocol, kept explicit: this fixture exists to test "dones are zero even when the env terminates".
  cfg = build_env_cfg(robot="go1", num_envs=4, continuing=True, time_outs=False, falls_reset=True)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
  vec = ContinuingVecEnv(env, penalty=10.0, offset=1.0)
  yield env, vec
  env.close()


def test_continuing_config_keeps_time_outs_and_the_fall_reset():
  """Decisions 77 + 81: time-outs stay (bootstrapped) AND `fell_over` stays, because that termination is what makes
  the env reset a fallen robot in place. Removing it left the robot lying on the floor (decision 81)."""
  cfg = build_env_cfg(robot="go1", num_envs=2, continuing=True)
  assert "time_out" in cfg.terminations
  assert "fell_over" in cfg.terminations
  assert cfg.episode_length_s < 1e5


def test_legacy_protocol_is_opt_in():
  """Waves 1-9 ran with no time-outs at all; that is now opt-in so no new cell inherits it silently."""
  cfg = build_env_cfg(robot="go1", num_envs=2, continuing=True, time_outs=False, falls_reset=True)
  assert "time_out" not in cfg.terminations
  assert cfg.episode_length_s > 1e5
  assert "fell_over" in cfg.terminations


def test_dones_are_always_zero_even_when_robots_fall(go1):
  env, vec = go1
  vec.unwrapped.reset()
  torch.manual_seed(0)
  falls = 0.0
  for _ in range(1500):  # ~6k env-steps: falls are rare (about 4 per 1k steps early in training)
    acts = 1.5 * torch.randn(vec.num_envs, vec.num_actions, device=vec.device)
    _, rew, dones, extras = vec.step(acts)
    cc = extras["cc"]
    assert torch.all(dones == 0)
    assert torch.allclose(rew, cc["task"] - 10.0 * cc["fell"] + 1.0)
    falls += float(cc["fell"].sum())
    if falls > 0:
      break
  # a quadruped driven with saturating actions falls quickly; the wrapper must charge penalties without terminals
  assert falls > 0, "expected at least one fall within 200 steps"


def test_terms_and_since_reset_are_exported(go1):
  env, vec = go1
  _, _, _, extras = vec.step(torch.zeros(vec.num_envs, vec.num_actions, device=vec.device))
  cc = extras["cc"]
  assert len(cc["term_names"]) == len(cc["terms"])
  assert "track_linear_velocity" in cc["term_names"]
  assert cc["since_reset"].shape == (vec.num_envs,)


def test_action_rate_shock_is_read_at_compute_time(go1):
  env, vec = go1
  rm = env.reward_manager
  idx = rm.active_terms.index("action_rate_l2")
  # alternate the action so action_rate_l2 is non-zero, and step once to seed the previous-action buffer
  a1 = torch.full((vec.num_envs, vec.num_actions), 0.3, device=vec.device)
  a2 = -a1
  vec.step(a1)
  before = float(vec.step(a2)[3]["cc"]["terms"][idx])
  apply_shock(env, "action_rate", 5.0)
  after = float(vec.step(a1)[3]["cc"]["terms"][idx])
  assert after < before * 2.0, (before, after)  # larger penalty; ratio is action-dependent, sign must hold
  assert after < 0.0
  apply_shock(env, "action_rate", 0.2)  # restore (x5 then /5)


def test_ssp_forward_matches_sspspace():
  sspspace = pytest.importorskip("sspspace")
  from sspspace import HexagonalSSPSpace

  domain_dim, n_rotates, n_scales, ls = 3, 5, 5, 0.3
  enc = HexagonalSSPSpace(
    domain_dim=domain_dim, n_rotates=n_rotates, n_scales=n_scales, kernel="hypergeom", length_scale=ls
  )
  x = np.random.default_rng(0).normal(size=(7, domain_dim)).astype(np.float32)
  ref = np.asarray(enc.encode(x).v).reshape(x.shape[0], -1)

  mine = HexSSP(domain_dim, scale=1.0, length_scale=ls, n_rotates=n_rotates, n_scales=n_scales)
  out = mine.forward(torch.from_numpy(x)).numpy()
  assert out.shape == ref.shape
  assert np.allclose(out, ref, atol=1e-4), np.abs(out - ref).max()


def test_ssp_per_dimension_scaling_changes_the_kernel_width():
  """The H2 fix: dividing a dimension by a larger scale widens the kernel along that dimension only."""
  enc = HexSSP(2, scale=[1.0, 1.0], length_scale=0.5, n_rotates=4, n_scales=4)
  a = enc.encode(torch.tensor([[0.0, 0.0]]))
  b = enc.encode(torch.tensor([[1.0, 0.0]]))
  sim_unscaled = float((a * b).sum())
  enc.scale = torch.tensor([10.0, 1.0])
  a = enc.encode(torch.tensor([[0.0, 0.0]]))
  b = enc.encode(torch.tensor([[1.0, 0.0]]))
  sim_scaled = float((a * b).sum())
  assert sim_scaled > sim_unscaled + 1e-3


def test_staying_down_is_opt_in():
  """Decision 81: "no fall reset" is a diagnostic, not a default."""
  cfg = build_env_cfg(robot="go1", num_envs=2, continuing=False, falls_reset=False)
  assert "time_out" in cfg.terminations
  assert "fell_over" not in cfg.terminations


def test_obs_v130_drops_the_biased_actor_joint_pos():
  """Decision 74 follow-up: mjlab 1.3.0 fed an unbiased joint_pos to both actor and critic; 1.6.0 biases the actor."""
  modern = build_env_cfg(robot="g1", num_envs=2, continuing=False)
  assert modern.observations["actor"].terms["joint_pos"].params == {"biased": True}
  assert modern.observations["critic"].terms["joint_pos"].params == {}
  old = build_env_cfg(robot="g1", num_envs=2, continuing=False, obs_v130=True)
  assert old.observations["actor"].terms["joint_pos"].params == {}
  assert old.observations["critic"].terms["joint_pos"].params == {}


def test_default_protocol_passes_time_outs_through_and_still_counts_falls():
  """Decision 77: with time-outs kept, an episode end is a real dones>0 on the time-out, PPO gets `time_outs` to
  bootstrap on, and falls are still measured (they are no longer terminal)."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = build_env_cfg(robot="go1", num_envs=4, continuing=True)  # new default: time-outs kept, falls non-terminal
  cfg.episode_length_s = 1.0  # 50 steps at 50 Hz: reachable inside a test
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
  vec = ContinuingVecEnv(env, penalty=0.0, offset=0.0, time_outs=True, fall_check_deg=70.0)
  try:
    vec.unwrapped.reset()
    torch.manual_seed(0)
    saw_done = saw_bootstrap = False
    for _ in range(80):
      acts = 1.5 * torch.randn(vec.num_envs, vec.num_actions, device=vec.device)
      _, rew, dones, extras = vec.step(acts)
      assert "time_outs" in extras, "PPO must be able to bootstrap on the time-out"
      saw_bootstrap = saw_bootstrap or bool(extras["time_outs"].any())
      saw_done = saw_done or bool((dones > 0).any())
      assert torch.allclose(rew, extras["cc"]["task"] + 0.0, atol=1e-5)  # no penalty, no offset
    assert saw_done, "a 1 s episode must end inside 80 steps"
    assert saw_bootstrap, "the episode end must be reported as a time-out, not as a fall"
  finally:
    env.close()


def test_a_fall_resets_the_robot_without_becoming_a_done():
  """Decision 81: `fell_over` must stay in the env, because it is what resets a fallen robot. Verified by driving G1
  with saturating actions: at the fall step the env resets (episode buffer 0, root back up), dones stays 0 and the
  time-out flag is false, so PPO sees a continuing task with predefined resets."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = build_env_cfg(robot="g1", num_envs=2, continuing=True)
  assert "fell_over" in cfg.terminations, "removing it also removes the reset (that was the decision-81 bug)"
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
  vec = ContinuingVecEnv(env, penalty=0.0, offset=0.0, time_outs=True)
  try:
    vec.unwrapped.reset()
    robot = env.scene["robot"]
    for _ in range(60):
      acts = 6.0 * torch.ones(vec.num_envs, vec.num_actions, device=vec.device)
      _, _, dones, extras = vec.step(acts)
      fell = extras["cc"]["fell"] > 0
      if fell.any():
        j = int(fell.nonzero()[0])
        assert int(env.episode_length_buf[j]) == 0, "a fall must reset the robot in place"
        assert float(robot.data.root_link_pos_w[j, 2]) > 0.5, "the reset must put the robot back up"
        assert int(dones[j]) == 0, "PPO must not see the fall as an episode end"
        assert not bool(extras["time_outs"][j])
        return
  finally:
    env.close()
  raise AssertionError("no fall within 60 saturating steps")


def test_penalty_is_charged_once_on_the_fall_step_only():
    """Decision 81 follow-up: with the reset restored, the fall flag must be a one-step event, so the penalty is
    charged once per fall and never again on later steps. Also pins the reward identity task - P*fell + offset."""
    from mjlab.envs import ManagerBasedRlEnv

    P, OFF = 10.0, 1.0
    cfg = build_env_cfg(robot="g1", num_envs=2, continuing=True)  # corrected default
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda" if torch.cuda.is_available() else "cpu")
    vec = ContinuingVecEnv(env, penalty=P, offset=OFF, time_outs=True)
    try:
        vec.unwrapped.reset()
        robot = env.scene["robot"]
        falls = penalty_total = 0.0
        saw_fall = False
        for _ in range(80):
            acts = 6.0 * torch.ones(vec.num_envs, vec.num_actions, device=vec.device)
            _, rew, dones, extras = vec.step(acts)
            cc = extras["cc"]
            assert torch.allclose(rew, cc["task"] - P * cc["fell"] + OFF, atol=1e-4)
            fell = cc["fell"] > 0
            if fell.any():
                j = int(fell.nonzero()[0])
                saw_fall = True
                assert int(env.episode_length_buf[j]) == 0, "the fall must reset the episode counter"
                assert float(robot.data.root_link_pos_w[j, 2]) > 0.5, "the reset must put the robot back up"
                assert int(dones[j]) == 0, "PPO must not see the fall as a done"
                assert abs(float(rew[j] - cc["task"][j]) - (OFF - P)) < 1e-3, "the fall step carries exactly one penalty"
                _, rew2, _, extras2 = vec.step(acts)
                assert float(extras2["cc"]["fell"][j]) == 0.0, "the fall flag must not stay set after the reset"
                assert abs(float(rew2[j] - extras2["cc"]["task"][j]) - OFF) < 1e-3
                break
            falls += float(cc["fell"].sum())
            penalty_total += float((cc["task"] - rew + OFF).sum())
        assert saw_fall, "no fall within 80 saturating steps"
        assert penalty_total == P * falls
    finally:
        env.close()
