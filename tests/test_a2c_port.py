"""Tests for the NumPy A2C port (rl/networks/acLinear.py, experiments/trial_cartpole_a2c.py).

The strongest checks drive the REAL legacy rule code (rl/rules/td0_nengo.py, td0centering_nengo.py) with a
stubbed `nengo` module (the rules only use it for the Process base class) and require the port to
match it. The stub lives only inside these tests; nothing in the port imports Nengo.
"""

import importlib
import os
import subprocess
import sys
import types

import gymnasium as gym
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments.trial_cartpole_a2c import (  # noqa: E402
    A2C_BASE_PARAMS,
    build_agent,
    build_rep,
    select_action,
    train_a2c,
)
from rl.networks.acLinear import LinearActorCritic  # noqa: E402
from rl.representations.ssp_legacy import LegacySSPRep  # noqa: E402

N_ACTIONS = 2


# ---------------------------------------------------------------------------------------------
# legacy oracle (real legacy rule under a nengo stub)
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def legacy_rules():
    stub = types.ModuleType("nengo")
    stub.processes = types.SimpleNamespace(Process=type("Process", (), {"__init__": lambda self, *a, **k: None}))
    saved = {k: sys.modules.get(k) for k in ("nengo", "rl.rules.td0_nengo", "rl.rules.td0centering_nengo")}
    sys.modules["nengo"] = stub
    for k in ("rl.rules.td0_nengo", "rl.rules.td0centering_nengo"):
        sys.modules.pop(k, None)
    try:
        yield (
            importlib.import_module("rl.rules.td0_nengo").TD0,
            importlib.import_module("rl.rules.td0centering_nengo").TD0Center,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class LegacyAgent:
    """What `ActorCritic.step` does with state_neurons=None: pack [phi, onehot(a), rho, r, reset] and run the rule."""

    def __init__(self, rules, dim, mode, lr, beta, eta, act_dis=0.9, state_dis=0.99):
        TD0, TD0Center = rules
        kw = dict(n_actions=N_ACTIONS, lr=lr, act_dis=act_dis, state_dis=state_dis, env_dt=0.001, learnTrials=1000)
        if mode == "none":
            self.rule = TD0(**kw)
        else:
            self.rule = TD0Center(
                reward_center_mode=mode, reward_center_beta=beta, reward_center_eta=eta, reward_center_init=0.0, **kw
            )
        shape_in = (dim + N_ACTIONS + 3,)
        self.state = self.rule.make_state(shape_in, (N_ACTIONS + 1,), 0.001)
        self._step = self.rule.make_step(shape_in, (N_ACTIONS + 1,), 0.001, None, self.state)

    def step(self, phi, action, reward, reset=False, rho=1.0):
        a = np.zeros(N_ACTIONS)
        a[action] = 1
        out = self._step(0.0, np.concatenate([phi, a, [rho, reward, float(reset)]]))
        return out[1:].copy()

    @property
    def W(self):
        return self.state["w"]

    @property
    def avg(self):
        return self.state.get("avg_reward", 0.0)


MODES = [
    ("none", {}),
    ("simple", dict(beta=0.05)),
    ("value", dict(eta=0.5)),
]


def _rand_phi(rng, dim, binary):
    if binary:
        idx = np.unique(rng.integers(0, dim, size=6))
        v = np.zeros(dim)
        v[idx] = 1.0
        return v, idx
    v = rng.normal(size=dim)
    return v, v


@pytest.mark.parametrize("mode,kw", MODES)
@pytest.mark.parametrize("binary", [False, True])
def test_rule_matches_legacy_on_random_trace(legacy_rules, mode, kw, binary):
    dim, lr = 40, 0.6
    beta, eta = kw.get("beta", 0.001), kw.get("eta", 1.0)
    old = LegacyAgent(legacy_rules, dim, mode, lr, beta, eta)
    new = LinearActorCritic(dim, N_ACTIONS, lr=lr, centering=mode, beta=beta, eta=eta, sparse=binary)
    rng = np.random.default_rng(3)
    for _ in range(6):  # episodes
        dense, port = _rand_phi(rng, dim, binary)
        p_old = old.step(dense, 0, 0.0, reset=True)
        p_new = new.reset(port)
        np.testing.assert_allclose(p_new, p_old, atol=1e-12)
        for _ in range(int(rng.integers(5, 25))):
            dense, port = _rand_phi(rng, dim, binary)
            a, r = int(rng.integers(0, 2)), float(rng.choice([1.0, 1.0, 0.5, -1.0]))
            p_old = old.step(dense, a, r)
            p_new = new.step(port, a, r)
            np.testing.assert_allclose(p_new, p_old, atol=1e-12)
        # terminal pseudo-step (same state, same action, reward 0)
        old.step(dense, a, 0.0)
        new.step(port, a, 0.0)
        np.testing.assert_allclose(new.W, old.W, atol=1e-11)
        assert new.avg_reward == pytest.approx(old.avg, abs=1e-12)


def test_dense_rule_is_bit_exact_with_legacy(legacy_rules):
    dim = 30
    old = LegacyAgent(legacy_rules, dim, "value", 0.7, 0.001, 0.3)
    new = LinearActorCritic(dim, N_ACTIONS, lr=0.7, centering="value", eta=0.3, sparse=False)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=dim)
    old.step(phi, 0, 0.0, reset=True)
    new.reset(phi)
    for _ in range(200):
        phi = rng.normal(size=dim)
        a = int(rng.integers(0, 2))
        old.step(phi, a, 1.0)
        new.step(phi, a, 1.0)
    assert np.array_equal(new.W, old.W)
    assert new.avg_reward == old.avg


# ---------------------------------------------------------------------------------------------
# loop-level differential test: legacy ACTrial loop (copied below) driving the legacy rule, versus train_a2c
# ---------------------------------------------------------------------------------------------
def _legacy_behavior(action_logits, eps):  # copy of behavior_policy(mode="epsilon_greedy")
    n = len(action_logits)
    probs = np.full(n, eps / n, dtype=float)
    probs[np.argmax(action_logits)] += 1.0 - eps
    return probs


def _oracle_run(rules, params, seed, trials):
    """Literal copy of the ACTrial.evaluate core (force_rho_one, softmax target irrelevant), dense features."""
    env = gym.make(params["env"])
    obs_dim = len(env.observation_space.high)
    rep, _, _ = build_rep(params, env)
    mode = params.get("reward_center_mode", "none")
    agent = LegacyAgent(
        rules, rep.size_out, mode, params["lr"], params["reward_center_beta"], params["reward_center_eta"]
    )
    rs = np.random.RandomState(seed)
    eps, E = params["eps"], []
    for trial in range(trials):
        reset_obj = env.reset(seed=seed)[0] if trial == 0 else env.reset()[0]
        update_state = rep.get_state(reset_obj[:obs_dim], env)
        action_logits = agent.step(rep.map(update_state), 0, 0, reset=True)
        r_sum = 0.0
        for _ in range(params["steps"]):
            probs = _legacy_behavior(action_logits, eps)
            action = rs.choice(2, p=probs)
            obs, reward, term, trunc, _ = env.step(action)
            current = rep.get_state(obs[:obs_dim], env)
            action_logits = agent.step(rep.map(current), action, reward, rho=1.0)
            r_sum += reward
            if term or trunc:
                for _ in range(params["n_done"]):
                    agent.step(rep.map(current), action, 0.0, rho=1.0)
                break
        E.append(r_sum)
        if params["dynamic_epsilon"] and trial >= 20:
            if np.mean(E[trial - 10 : trial]) > np.mean(E[trial - 20 : trial - 10]) + np.std(E[trial - 20 : trial - 10]):
                eps = float(np.clip(eps - 0.001, 0.0, 1.0))
    return E, agent, eps


REPS = {
    "ssp": dict(rep_="PlaceSSP", length_scale=0.6, n_rotates=2),
    "tile": dict(rep_="TileCoding", num_tilings=8, tiles_per_dim=[6, 6, 6, 6], iht_size=2048, tile_state_indices=[0, 1, 2, 3]),
    "discrete": dict(rep_="Discrete", n_bins=6),
}


@pytest.mark.parametrize("mode,kw", MODES)
@pytest.mark.parametrize("rep", ["ssp", "tile", "discrete"])
def test_loop_matches_legacy_loop(legacy_rules, rep, mode, kw):
    params = dict(A2C_BASE_PARAMS)
    params.update(REPS[rep])
    params.update(lr=0.6, eps=0.2, reward_center_mode=mode, reward_center_beta=kw.get("beta", 0.001),
                  reward_center_eta=kw.get("eta", 1.0), trials=32)
    E_old, agent, eps_old = _oracle_run(legacy_rules, params, seed=11, trials=32)
    out = train_a2c(params, seed=11, rng=np.random.RandomState(11))
    assert out["returns"] == E_old
    assert out["final_eps"] == pytest.approx(eps_old)
    np.testing.assert_allclose(out["ac"].W, agent.W, atol=1e-8)
    assert out["ac"].avg_reward == pytest.approx(agent.avg, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# smaller unit tests
# ---------------------------------------------------------------------------------------------
def test_select_action_matches_numpy_choice():
    rs_a, rs_b = np.random.RandomState(5), np.random.RandomState(5)
    rng = np.random.default_rng(1)
    for _ in range(500):
        prefs = rng.normal(size=2)
        eps = float(rng.uniform(0, 1))
        probs = _legacy_behavior(prefs, eps)
        assert select_action(prefs, eps, rs_a.random()) == rs_b.choice(2, p=probs)


def test_hex_ssp_matches_golden_vendored_encoder():
    g = np.load(os.path.join(REPO, "tests", "data", "legacy_ssp_golden.npz"))
    X = g["X"]
    for key in g.files:
        if key == "X" or key.startswith("rot"):
            continue
        ls, nr = key.split("_")
        rep = LegacySSPRep(4, length_scale=float(ls[2:]), n_rotates=int(nr[2:]))
        got = np.stack([rep.map(x) for x in X])
        assert got.shape == g[key].shape
        np.testing.assert_allclose(got, g[key], atol=1e-12)


def test_legacy_rotations_match_scipy_173():
    """Golden generated with the real vendored code under Python 3.9 / scipy 1.7.3 / numpy 1.22.4."""
    from rl.representations.ssp_legacy import legacy_special_ortho_rvs

    g = np.load(os.path.join(REPO, "tests", "data", "legacy_ssp_golden.npz"))
    for key in g.files:
        if key.startswith("rot"):
            n = int(key[3:])
            np.testing.assert_allclose(legacy_special_ortho_rvs(4, n, seed=1), g[key], atol=1e-14)


def test_ssp_dim_formula():
    assert LegacySSPRep(4, 1.0, 6).size_out == 2 * 6 * 8 * 5 + 1


def test_stale_bootstrap_and_preupdate_preferences():
    """Hand-checkable: the returned preferences use the weights from BEFORE the update."""
    ac = LinearActorCritic(3, 2, lr=0.5, act_dis=1.0, state_dis=1.0, sparse=False)
    phi = np.array([1.0, 0.0, 0.0])
    ac.reset(phi)
    prefs = ac.step(phi, 0, 1.0)  # td = 1 + 1*0 - 0 = 1; scale = 1; W[0,0] += .5, W[1,0] += 1
    assert np.array_equal(prefs, [0.0, 0.0])  # pre-update
    assert ac.W[0, 0] == 0.5 and ac.W[1, 0] == 1.0 and ac.W[2, 0] == 0.0
    prefs = ac.step(phi, 1, 0.0)  # now V = .5 (computed pre-update), prev_value stale = 0
    assert np.array_equal(prefs, [1.0, 0.0])


def test_value_centering_uses_old_avg_then_updates():
    ac = LinearActorCritic(2, 2, lr=1.0, centering="value", eta=0.5, sparse=False)
    phi = np.array([1.0, 0.0])
    ac.reset(phi)
    ac.step(phi, 0, 2.0)  # td = (2-0) + 0 - 0 = 2 -> avg = 0 + .5*1*2 = 1
    assert ac.avg_reward == 1.0


def test_sparse_matches_dense_in_a_real_run():
    params = dict(A2C_BASE_PARAMS)
    params.update(REPS["tile"], lr=0.6, eps=0.2, reward_center_mode="none", trials=15)
    env = gym.make("CartPole-v1")
    rep, encode, sparse = build_rep(params, env)
    assert sparse
    sparse_run = train_a2c(params, seed=2, rng=np.random.RandomState(2))
    # dense path through the same agent class
    env2 = gym.make("CartPole-v1")
    rep2, _, _ = build_rep(params, env2)
    ac = build_agent(params, rep2, sparse=False)
    rs = np.random.RandomState(2)
    returns = []
    for t in range(15):
        obs = env2.reset(seed=2)[0] if t == 0 else env2.reset()[0]
        prefs = ac.reset(rep2.map(rep2.get_state(obs, env2)))
        total = 0.0
        for _ in range(500):
            a = select_action(prefs, 0.2, rs.random())
            obs, r, te, tr, _ = env2.step(a)
            phi = rep2.map(rep2.get_state(obs, env2))
            prefs = ac.step(phi, a, r)
            total += r
            if te or tr:
                ac.step(phi, a, 0.0)
                break
        returns.append(total)
    assert returns == sparse_run["returns"]
    np.testing.assert_allclose(ac.W, sparse_run["ac"].W, atol=1e-9)


def test_a2c_chain_imports_no_nengo():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "import rl.networks.acLinear, rl.representations.ssp_legacy, experiments.trial_cartpole_a2c\n"
        "bad = [m for m in sys.modules if m == 'nengo' or m.startswith('nengo.') or m.startswith('nengo_spa')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
