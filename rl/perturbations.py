"""Test-time perturbations of CartPole-v1 for the robustness suite.

Why: the saved policies are evaluated on the SAME 21 conditions with the SAME
per-episode seeds (common random numbers), so differences between policies
reflect the policy, not the draw of initial states or noise.

Semantic choices:
  * Physics conditions scale attributes on ``env.unwrapped`` and recompute
    ``total_mass`` and ``polemass_length`` (cached in the env, so scaling only
    ``length``/``masspole``/``masscart`` would silently do nothing for them).
    ``length`` is the HALF pole length; scaling it by f scales the pole by f.
  * obs_noise_f adds N(0, (f * noise_sigma)^2) to the observation the AGENT
    sees (including the reset observation); the true state, termination and
    reward are untouched.
  * action_delay_d: the action applied at step t is the one decided at step
    t-d. The first d applied actions repeat the decision made on the reset
    observation (no-op fill is not available in CartPole).
  * random_action_p: each step, with probability p, the applied action is
    replaced by a uniformly random one (which may coincide with the decision,
    so the effective mismatch rate is p/2). Applied after the delay.
  * wide_init_b: the initial state is uniform in [-b, b]^4.
"""
from __future__ import annotations

import zlib

import gymnasium as gym
import numpy as np

FAMILIES: dict[str, list[str]] = {
    "baseline": ["baseline"],
    "pole_length": ["pole_length_x0.5", "pole_length_x1.5", "pole_length_x2", "pole_length_x4"],
    "pole_mass": ["pole_mass_x0.5", "pole_mass_x2"],
    "cart_mass": ["cart_mass_x0.5", "cart_mass_x2"],
    "gravity": ["gravity_x0.8", "gravity_x1.2"],
    "force": ["force_x0.7", "force_x1.3"],
    # obs_noise is scaled by the pooled per-dimension state std (see scripts/make_probe_states.py).
    # 0.01/0.05/0.1 were the pre-registered levels, but the pilot (discrete/none/linear and
    # tile_coding/simple/deep, 20 seeds each) showed retained 0.99-1.03 -- indistinguishable from
    # baseline. At 0.1 the pole-angle noise is 0.0027 rad against a 0.209 rad termination limit
    # (~1%), so the levels were too mild to probe the family. 0.25/0.5/1.0 were added after that
    # pilot (decision log); at 1.0 the noise equals the pooled state scale in every dimension.
    "obs_noise": ["obs_noise_0.01", "obs_noise_0.05", "obs_noise_0.1",
                  "obs_noise_0.25", "obs_noise_0.5", "obs_noise_1.0"],
    "action_delay": ["action_delay_1", "action_delay_2"],
    "random_action": ["random_action_0.05"],
    "wide_init": ["wide_init_0.1", "wide_init_0.2"],
}
ALL_CONDITIONS: list[str] = [c for names in FAMILIES.values() for c in names]

# physics family -> (kind label, attribute that is scaled)
_PHYSICS = {
    "pole_length": "length",
    "pole_mass": "masspole",
    "cart_mass": "masscart",
    "gravity": "gravity",
    "force": "force_mag",
}
# non-physics families: name prefix -> family
_PREFIX = {
    "obs_noise_": "obs_noise",
    "action_delay_": "action_delay",
    "random_action_": "random_action",
    "wide_init_": "wide_init",
}


def family_of(condition: str) -> str:
    for fam, names in FAMILIES.items():
        if condition in names:
            return fam
    raise ValueError(f"unknown condition: {condition!r}")


def condition_spec(name: str) -> dict:
    """Parse a condition name into {family, kind, value}.

    kind is "baseline", "scale" (value = factor, physics attribute named by
    family), "noise", "delay", "random_action" or "wide_init".
    Names are parsed (not only looked up) so the value is exactly the name's.
    """
    if name == "baseline":
        return {"family": "baseline", "kind": "baseline", "value": None}
    try:
        for fam in _PHYSICS:
            pre = fam + "_x"
            if name.startswith(pre):
                return {"family": fam, "kind": "scale", "value": float(name[len(pre):])}
        for pre, fam in _PREFIX.items():
            if name.startswith(pre):
                raw = name[len(pre):]
                if fam == "action_delay":
                    return {"family": fam, "kind": "delay", "value": int(raw)}
                kind = {"obs_noise": "noise", "random_action": "random_action",
                        "wide_init": "wide_init"}[fam]
                return {"family": fam, "kind": kind, "value": float(raw)}
    except ValueError:
        pass
    raise ValueError(f"unknown condition: {name!r}")


def episode_seed(condition: str, i: int) -> int:
    """Deterministic across processes (crc32, never builtin hash())."""
    return zlib.crc32(condition.encode()) % 2**30 + i


def make_env(condition: str):
    """CartPole-v1 with the condition's physics applied (others: plain env)."""
    spec = condition_spec(condition)
    env = gym.make("CartPole-v1")
    if spec["kind"] == "scale":
        u = env.unwrapped
        attr = _PHYSICS[spec["family"]]
        setattr(u, attr, getattr(u, attr) * spec["value"])
        u.total_mass = u.masspole + u.masscart
        u.polemass_length = u.masspole * u.length
    return env


def run_episode(policy, condition: str, episode_seed: int, noise_sigma=None,
                max_steps: int = 500) -> float:
    """One greedy episode under `condition`; returns the summed reward.

    The caller supplies the (already cloned) policy; policy.reset() is called
    first. Seeds: env.reset(seed=episode_seed); random-action rng
    default_rng(episode_seed); obs-noise rng default_rng([episode_seed, 1]).
    """
    spec = condition_spec(condition)
    kind, val = spec["kind"], spec["value"]
    sigma = None
    if kind == "noise":
        if noise_sigma is None:
            raise ValueError("obs_noise conditions require noise_sigma")
        sigma = np.asarray(noise_sigma, dtype=np.float64) * val
    env = make_env(condition)
    act_rng = np.random.default_rng(episode_seed)
    noise_rng = np.random.default_rng([episode_seed, 1])

    def seen(o):
        if sigma is None:
            return o
        return (np.asarray(o, dtype=np.float64) + noise_rng.normal(0.0, sigma)).astype(np.float32)

    policy.reset()
    options = {"low": -val, "high": val} if kind == "wide_init" else None
    obs, _ = env.reset(seed=episode_seed, options=options)
    delay = val if kind == "delay" else 0
    decisions: list[int] = []
    total = 0.0
    for t in range(max_steps):
        decisions.append(int(policy.act(seen(obs))))
        action = decisions[max(t - delay, 0)]
        if kind == "random_action" and act_rng.random() < val:
            action = int(act_rng.integers(2))
        obs, r, term, trunc, _ = env.step(action)
        total += float(r)
        if term or trunc:
            break
    env.close()
    return total


def run_condition(policy, condition: str, n_episodes: int = 20, noise_sigma=None,
                  max_steps: int = 500) -> list[float]:
    """n_episodes with common-random-number seeds; clones the policy per episode."""
    return [
        run_episode(policy.clone(), condition, episode_seed(condition, i),
                    noise_sigma=noise_sigma, max_steps=max_steps)
        for i in range(n_episodes)
    ]
