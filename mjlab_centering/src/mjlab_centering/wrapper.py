"""Continuing-task wrapper: penalty-and-reset with no terminal flags.

mjlab's robot tasks are episodic: a fall terminates the episode and a 20 s time-out truncates it. Reward
centering needs a continuing criterion, so this wrapper converts the task the way Wan, Korenkevych & Zhu
(ICLR 2025) convert MuJoCo tasks to "continuing with predefined resets":

- `dones` is always 0, so PPO never sees a terminal transition. Its GAE chain is never cut on a fall.
- A fall is charged `penalty` and the env resets that robot in place (mjlab auto-reset). The reset is a jump
  back into the task's normal state distribution, which is what keeps the average reward well defined.
- The task's time-out termination is removed by `env_cfg.build_go1`, so there is no free reset.
- An optional constant `offset` per step is added (the reward-offset manipulation), with an optional mid-run
  shift (the reward-redesign shock, see `shock_at`).

The wrapper also exports, through `extras["cc"]`, the quantities the algorithm cannot see by itself:
the pre-penalty task reward, the fall/reset counts and the steps since the last reset.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class ContinuingVecEnv(RslRlVecEnvWrapper):
  """Penalty-and-reset continuing wrapper around a mjlab env."""

  def __init__(
    self,
    env,
    penalty: float = 10.0,
    offset: float = 0.0,
    shock_at: int | None = None,
    shock_delta: float = 0.0,
    p_reset: float = 0.0,
    seed: int | None = None,
    time_outs: bool = False,
    fall_check_deg: float | None = None,
  ) -> None:
    super().__init__(env)
    self.penalty = float(penalty)
    self.offset = float(offset)
    self.shock_at = shock_at
    self.shock_delta = float(shock_delta)
    self.p_reset = float(p_reset)
    # Decision 77: the default protocol keeps mjlab's time-outs (episodes still end on time) and makes a fall
    # non-terminal at the env level. Then the time-out is a real `truncated`, so dones pass through and PPO
    # bootstraps on it; the fall is measured separately so the fall-rate diagnostic survives.
    self.time_outs = bool(time_outs)
    self.fall_check_deg = fall_check_deg
    self.iter = 0
    n = self.num_envs
    self.since_reset = torch.zeros(n, dtype=torch.long, device=self.device)
    self._term_names = list(self.unwrapped.reward_manager.active_terms)
    if seed is not None:
      self.unwrapped.reset(seed=seed)

  @property
  def offset_now(self) -> float:
    """Constant offset in force at the current iteration (offset + any shock that has landed)."""
    if self.shock_at is not None and self.iter >= self.shock_at:
      return self.offset + self.shock_delta
    return self.offset

  def step(self, actions: torch.Tensor):
    env = self.env
    # Optional free random reset (a trap escape, Wan et al. use ~1e-3 per step). Default off.
    if self.p_reset > 0.0:
      mask = torch.rand(self.num_envs, device=self.device) < self.p_reset
      if bool(mask.any()):
        env.reset(env_ids=mask.nonzero(as_tuple=False).squeeze(-1))
        self.since_reset = torch.where(mask, torch.zeros_like(self.since_reset), self.since_reset)

    if self.clip_actions is not None:
      actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = env.step(actions)
    if not isinstance(rew, torch.Tensor):
      rew = torch.as_tensor(rew, dtype=torch.float, device=self.device)

    if self.fall_check_deg is None:
      fell = terminated.to(dtype=torch.float)
      ends = terminated
    else:
      from mjlab.envs.mdp.terminations import bad_orientation

      fell = bad_orientation(self.unwrapped, math.radians(self.fall_check_deg)).to(dtype=torch.float)
      ends = fell > 0
    if self.time_outs:
      ends = ends | truncated
    self.since_reset = torch.where(ends, torch.zeros_like(self.since_reset), self.since_reset + 1)

    task = rew  # the env's own reward: sum of weighted terms, dt-scaled, no penalty and no offset
    seen = task - self.penalty * fell + self.offset_now

    rm = self.unwrapped.reward_manager
    terms = rm.term_rewards.mean(dim=0)  # (num_terms,) applied per-term rewards, already scaled

    extras = dict(extras or {})
    if self.time_outs:
      # Truncation is a real episode boundary: pass it through and let PPO bootstrap on it.
      extras["time_outs"] = truncated
    else:
      # Legacy protocol: dones are zero, and the parent's bootstrap is unused.
      extras.pop("time_outs", None)
    extras["cc"] = {
      "task": task,
      "seen": seen,
      "fell": fell,
      "terms": terms,
      "since_reset": self.since_reset,
      "term_names": self._term_names,
    }
    if self.time_outs:
      dones = truncated.to(dtype=torch.long)
    else:
      dones = torch.zeros_like(terminated, dtype=torch.long)
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      seen,
      dones,
      extras,
    )

  def reset(self):
    obs_dict, extras = self.env.reset()
    self.since_reset.zero_()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras
