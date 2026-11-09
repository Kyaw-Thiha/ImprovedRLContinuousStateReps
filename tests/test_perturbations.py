import os
import sys
import zlib

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rl import perturbations as P  # noqa: E402

SIGMA = np.array([0.5, 0.5, 0.05, 0.5])


class PD:
    n_clones = 0

    def __init__(self):
        self.seen = []

    def act(self, obs):
        self.seen.append(np.array(obs))
        return int(obs[2] + 0.5 * obs[3] > 0)

    def clone(self):
        PD.n_clones += 1
        return PD()

    def reset(self):
        pass


class Const:
    def __init__(self, a=0):
        self.a = a

    def act(self, obs):
        return self.a

    def clone(self):
        return Const(self.a)

    def reset(self):
        pass


class Seq:
    """Decides a fixed pseudo-random sequence regardless of obs."""

    def __init__(self):
        self.t = 0
        self.seq = list(np.random.default_rng(5).integers(0, 2, 500))

    def act(self, obs):
        a = int(self.seq[self.t])
        self.t += 1
        return a

    def clone(self):
        return Seq()

    def reset(self):
        self.t = 0


class Rec(gym.Wrapper):
    def __init__(self, env, log):
        super().__init__(env)
        self.log = log

    def step(self, a):
        self.log.append(int(a))
        return self.env.step(a)


def spy(monkeypatch):
    log = []
    orig = P.make_env
    monkeypatch.setattr(P, "make_env", lambda c: Rec(orig(c), log))
    return log


def reference(seed, policy):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    tot = 0.0
    for _ in range(500):
        obs, r, te, tr, _ = env.step(policy.act(obs))
        tot += r
        if te or tr:
            break
    return tot


def test_all_conditions_run():
    # 24 = 21 pre-registered + obs_noise 0.25/0.5/1.0 added after the robustness pilot showed
    # 0.01-0.1 was indistinguishable from baseline (decision log). Derived, not hardcoded, so a
    # future deliberate change only has to update FAMILIES.
    assert len(P.ALL_CONDITIONS) == sum(len(v) for v in P.FAMILIES.values()) == 24
    assert P.ALL_CONDITIONS[0] == "baseline"
    for c in P.ALL_CONDITIONS:
        out = P.run_condition(PD(), c, n_episodes=2, noise_sigma=SIGMA)
        assert len(out) == 2 and all(isinstance(x, float) for x in out)
        assert c in P.FAMILIES[P.family_of(c)]


def test_baseline_matches_reference():
    for i in range(3):
        s = P.episode_seed("baseline", i)
        assert P.run_episode(PD(), "baseline", s) == reference(s, PD())


def test_physics_attrs():
    u = P.make_env("pole_length_x2").unwrapped
    assert u.length == 1.0 and u.polemass_length == pytest.approx(0.1)
    assert u.total_mass == pytest.approx(1.1)
    u = P.make_env("cart_mass_x2").unwrapped
    assert u.masscart == 2.0 and u.total_mass == pytest.approx(2.1)
    assert u.polemass_length == pytest.approx(0.05)
    u = P.make_env("pole_mass_x2").unwrapped
    assert u.total_mass == pytest.approx(1.2) and u.polemass_length == pytest.approx(0.1)
    assert P.make_env("gravity_x1.2").unwrapped.gravity == pytest.approx(11.76)
    assert P.make_env("force_x0.7").unwrapped.force_mag == pytest.approx(7.0)


def test_episode_seed():
    assert P.episode_seed("baseline", 3) == zlib.crc32(b"baseline") % 2**30 + 3
    assert P.episode_seed("gravity_x0.8", 0) == zlib.crc32(b"gravity_x0.8") % 2**30
    assert P.episode_seed("baseline", 0) == P.episode_seed("baseline", 0)


def test_obs_noise(monkeypatch):
    with pytest.raises(ValueError):
        P.run_episode(PD(), "obs_noise_0.05", 1, noise_sigma=None)
    a, b = PD(), PD()
    ra = P.run_episode(a, "obs_noise_0.1", 7, noise_sigma=SIGMA)
    rb = P.run_episode(b, "obs_noise_0.1", 7, noise_sigma=SIGMA)
    assert ra == rb and np.array_equal(a.seen[0], b.seen[0])
    clean = PD()
    P.run_episode(clean, "baseline", 7)
    assert not np.allclose(a.seen[0], clean.seen[0])
    # true dynamics unaffected: a constant policy ignores noise -> same return
    assert P.run_episode(Const(1), "obs_noise_0.1", 7, SIGMA) == P.run_episode(Const(1), "baseline", 7)


def test_action_delay(monkeypatch):
    for d in (1, 2):
        log = spy(monkeypatch)
        pol = Seq()
        P.run_episode(pol, f"action_delay_{d}", 11)
        n = len(log)
        assert n > d + 2
        dec = pol.seq[:n]
        expected = [dec[max(t - d, 0)] for t in range(n)]
        assert log == expected
        monkeypatch.undo()


def test_random_action(monkeypatch):
    log = spy(monkeypatch)
    rets = [P.run_episode(Const(0), "random_action_0.05", s) for s in range(300)]
    assert len(log) == sum(int(r) for r in rets) or len(log) > 0
    frac1 = np.mean(log)  # decision is always 0; a 1 appears w.p. p/2 = 0.025
    assert 0.01 < frac1 < 0.045
    monkeypatch.undo()
    assert P.run_episode(PD(), "random_action_0.05", 3) == P.run_episode(PD(), "random_action_0.05", 3)


def test_wide_init():
    env = P.make_env("wide_init_0.2")
    mx = 0.0
    for i in range(50):
        obs, _ = env.reset(seed=i, options={"low": -0.2, "high": 0.2})
        assert np.all(np.abs(obs) <= 0.2)
        mx = max(mx, np.abs(obs).max())
    assert mx > 0.05
    # run_episode must really pass the widened reset options
    class ResetSpy(gym.Wrapper):
        def __init__(self, env, log):
            super().__init__(env)
            self.log = log

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            self.log.append((kw.get("options"), np.array(obs)))
            return obs, info

    def collect(cond, n):
        log = []
        orig = P.make_env
        mp = pytest.MonkeyPatch()
        mp.setattr(P, "make_env", lambda c: ResetSpy(orig(c), log))
        try:
            for i in range(n):
                P.run_episode(Const(0), cond, i, max_steps=1)
        finally:
            mp.undo()
        return log

    wide = collect("wide_init_0.2", 200)
    assert all(o == {"low": -0.2, "high": 0.2} for o, _ in wide)
    ext = max(np.abs(x).max() for _, x in wide)
    assert 0.05 < ext <= 0.2
    base = collect("baseline", 200)
    assert all(o is None for o, _ in base)
    assert max(np.abs(x).max() for _, x in base) <= 0.05


def test_run_condition_clones():
    PD.n_clones = 0
    out = P.run_condition(PD(), "baseline", n_episodes=5)
    assert PD.n_clones == 5 and len(out) == 5


def test_unknown():
    for bad in ("nope", "pole_length_xabc", "action_delay_x"):
        with pytest.raises(ValueError):
            P.condition_spec(bad)
    with pytest.raises(ValueError):
        P.family_of("nope")
    with pytest.raises(ValueError):
        P.run_episode(PD(), "nope", 1)
