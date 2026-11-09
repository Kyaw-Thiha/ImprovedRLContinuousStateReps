"""
Tests for rl/policy_io.py: exact save/reload of trained DQN policies.

What must hold (see the module docstring for the why):
  * a tiny real DQNTrial run saves a policy whose reload replays the in-memory verify episodes with
    max_abs_diff == 0.0, for every rep x architecture combination;
  * a FRESH process reproduces the Q-values of the in-process load (no hidden global state in the file);
  * the IHT overflow regime (hash collisions) round-trips;
  * Policy.encode/q_values reproduce the trial loop's get_state -> normalize -> encode -> get_q_values
    exactly (the trial hot loop itself does not call policy_io);
  * clone() isolates the mutable IHT; the file loads with weights_only=True and holds only plain types.
"""

import json
import os
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from experiments.trial_cartpole_dqn import DQNTrial  # noqa: E402
from rl.networks.dqnBasic import DQN  # noqa: E402
from rl.policy_io import Policy, load_policy, save_policy, verify_saved_policy  # noqa: E402
from rl.representations.onehot import OneHotRepCP  # noqa: E402
from rl.representations.ssp import SSPRep  # noqa: E402
from rl.representations.tilecoding import TileCodingRep  # noqa: E402

REP_PARAMS = {
    "discrete": dict(rep_="Discrete", n_bins=4),
    "ssp": dict(rep_="HexSSP", n_rotates=2, n_scales=2, length_scale=1.0),
    "tile": dict(rep_="TileCoding", num_tilings=8, iht_size=256),
    # 16 slots for 8 tilings: the table fills within the first few states, so collisions are certain.
    "tile_overflow": dict(rep_="TileCoding", num_tilings=8, iht_size=16),
}
COMBOS = [("discrete", "linear"), ("discrete", "deep"), ("ssp", "linear"), ("ssp", "deep"),
          ("tile", "linear"), ("tile", "deep"), ("tile_overflow", "linear")]
STEPS = 60


def _run_tiny(tmp_dir, rep, arch, seed=3000):
    path = os.path.join(tmp_dir, f"{rep}_{arch}.pt")
    meta = DQNTrial().run(
        trials=12, steps=STEPS, eval_freq=4, n_eval_episodes=2, final_eval_episodes=3,
        verify_episodes=3, seed_env=True, init_seed=-1, save_policy_path=path,
        seed=seed, architecture=arch, learning_starts=50, batch_size=16, buffer_size=1000,
        data_dir=tmp_dir, verbose=False, **REP_PARAMS[rep],
    )
    return path, meta


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("policies"))
    cache = {}

    def get(rep, arch):
        if (rep, arch) not in cache:
            cache[(rep, arch)] = _run_tiny(tmp_dir, rep, arch)
        return cache[(rep, arch)]

    return get


@pytest.mark.parametrize("rep,arch", COMBOS)
def test_saved_policy_replays_exactly(trained, rep, arch):
    path, meta = trained(rep, arch)
    assert meta["policy_saved"]["path"] == path and os.path.exists(path)
    assert meta["policy_saved"]["bytes"] == os.path.getsize(path)
    assert len(meta["verify_returns"]) == 3
    res = verify_saved_policy(path, meta["verify_env_seed"], 3, meta["verify_returns"], max_steps=meta["verify_max_steps"])
    assert res["ok"] is True
    assert res["max_abs_diff"] == 0.0
    assert res["returns"] == meta["verify_returns"]

    pol = load_policy(path)
    assert isinstance(pol, Policy)
    assert pol.hidden_dims == (None if arch == "linear" else [64, 64])
    assert pol.sparse == (rep != "ssp")
    assert pol.rep_kind == {"discrete": "discrete", "ssp": "ssp"}.get(rep, "tile")
    assert pol.n_params == meta["n_params"]
    assert pol.dim == meta["dimensionality"]
    assert pol.meta["seed"] == 3000 and pol.meta["format_version"] == 1
    assert {"torch", "gymnasium", "numpy", "created_utc"} <= set(pol.meta["provenance"])
    if rep == "tile_overflow":
        assert pol.rep.iht.overfullCount > 0
        assert len(pol.rep.iht.dictionary) == 16


def _fixed_states(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([-2.0, -2.0, -0.2, -2.0], [2.0, 2.0, 0.2, 2.0], size=(n, 4)).astype(np.float32)


_SUBPROC = r"""
import sys, json
sys.path.insert(0, {root!r})
import numpy as np
from rl.policy_io import load_policy
pol = load_policy({path!r})
states = np.load({states!r})
q = np.stack([pol.q_values(s) for s in states])
np.save({out!r}, q)
"""


@pytest.mark.parametrize("rep,arch", COMBOS)
def test_fresh_process_reload_matches(trained, tmp_path, rep, arch):
    path, _ = trained(rep, arch)
    states = _fixed_states()
    states_path = str(tmp_path / "states.npy")
    out_path = str(tmp_path / "q_sub.npy")
    np.save(states_path, states)
    code = _SUBPROC.format(root=ROOT, path=path, states=states_path, out=out_path)
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, capture_output=True, text=True)
    q_sub = np.load(out_path)
    pol = load_policy(path)
    q_here = np.stack([pol.q_values(s) for s in states])
    assert q_here.dtype == np.float32 and q_here.shape == (len(states), 2)
    np.testing.assert_allclose(q_sub, q_here, atol=1e-6, rtol=0)


def test_weights_only_load_and_plain_contents(trained):
    path, _ = trained("tile", "deep")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["format_version"] == 1

    def check(v):
        if isinstance(v, dict):
            assert all(isinstance(k, str) for k in v)
            for x in v.values():
                check(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                check(x)
        else:
            assert v is None or isinstance(v, (bool, int, float, str, torch.Tensor)), type(v)

    check(blob)
    json.dumps(blob["params"])  # params were made jsonable
    assert blob["rep"]["kind"] == "tile" and blob["rep"]["iht_keys"].dtype == torch.int64


class _Evil:
    def __reduce__(self):
        return (os.getcwd, ())


def test_loader_refuses_pickled_objects(tmp_path):
    p = str(tmp_path / "evil.pt")
    torch.save({"format_version": 1, "x": _Evil()}, p)
    with pytest.raises(Exception):
        load_policy(p)


def test_tampered_ssp_phase_matrix_is_rejected(trained, tmp_path):
    path, _ = trained("ssp", "linear")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    blob["rep"]["phase_matrix"][0, 0] += 1.0
    bad = str(tmp_path / "bad.pt")
    torch.save(blob, bad)
    with pytest.raises(ValueError, match="checksum"):
        load_policy(bad)


def test_atomic_write_leaves_no_tmp_files(trained):
    path, _ = trained("discrete", "linear")
    assert not [f for f in os.listdir(os.path.dirname(path)) if ".tmp" in f]


# ---------------------------------------------------------------------------------------------
# Equivalence with the trial loop's encode path (hand-built DQN, no training involved)
# ---------------------------------------------------------------------------------------------
STATE_SCALE = np.array([4.8, 10.0, 0.418, 10.0])


def _hand_build(kind, arch, normalize=False, iht_size=1024):
    """Mirror of the rep construction in DQNTrial.evaluate (same bounds tweaks)."""
    env = gym.make("CartPole-v1")
    low = env.observation_space.low.copy()
    low[1] = low[3] = -10
    high = env.observation_space.high.copy()
    high[1] = high[3] = 10
    params = {"rep_": kind}
    if kind == "ssp":
        rep = SSPRep(domain_dim=4, length_scale=1.0, n_rotates=2, n_scales=2)
        params.update(n_rotates=2, n_scales=2, length_scale=1.0)
    elif kind == "tile":
        tl = np.array([-1.0] * 4) if normalize else low
        th = np.array([1.0] * 4) if normalize else high
        rep = TileCodingRep(env, num_tilings=8, tiles_per_dim=None, iht_size=iht_size,
                            bounds_low=tl, bounds_high=th, state_indices=None)
    else:
        rep = OneHotRepCP((4, 4, 4, 4), bounds_low=low, bounds_high=high)
    hidden = None if arch == "linear" else [64, 64]
    dqn = DQN(rep=rep, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99, buffer_size=100, batch_size=16,
              target_update_freq=10, hidden_dims=hidden, init_seed=5)
    encode = rep.map_indices if hasattr(rep, "map_indices") else rep.map
    return env, rep, dqn, encode, params


def _trial_q(env, rep, dqn, encode, obs, normalize):
    state = rep.get_state(obs[:4], env)
    if normalize:
        state = state / STATE_SCALE
    return dqn.get_q_values(encode(state)).numpy()


@pytest.mark.parametrize("kind", ["discrete", "ssp", "tile"])
@pytest.mark.parametrize("arch", ["linear", "deep"])
@pytest.mark.parametrize("normalize", [False, True])
def test_policy_encode_matches_trial_path(tmp_path, kind, arch, normalize):
    if normalize and kind == "discrete":
        pytest.skip("normalize_state is not used with the one-hot representation")
    env, rep, dqn, encode, params = _hand_build(kind, arch, normalize)
    states = _fixed_states(40, seed=1)
    # Warm the IHT (tile) so the saved file carries a non-trivial dictionary.
    for s in states[:10]:
        _trial_q(env, rep, dqn, encode, s, normalize)
    path = str(tmp_path / "hand.pt")
    save_policy(path, dqn, rep, params, seed=7, normalize_state=normalize,
                state_scale=STATE_SCALE if normalize else None)
    pol = load_policy(path)
    assert pol.meta["avg_reward"] == dqn.avg_reward
    if kind == "ssp":
        assert pol.meta["ssp_rebuilt_equal"] is True
    # Same state sequence on both sides => identical IHT allocation order for tile coding.
    for s in states:
        q_ref = _trial_q(env, rep, dqn, encode, s, normalize)
        q_pol = pol.q_values(s)
        assert q_pol.dtype == np.float32
        assert np.array_equal(q_ref, q_pol)
        assert pol.act(s) == int(torch.as_tensor(q_ref).argmax())
    if kind == "tile":
        assert pol.rep.iht.dictionary == rep.iht.dictionary


def test_q_values_batch_matches_single(tmp_path):
    env, rep, dqn, encode, params = _hand_build("tile", "deep")
    path = str(tmp_path / "b.pt")
    save_policy(path, dqn, rep, params, seed=1)
    states = _fixed_states(10, seed=2)
    pol_a, pol_b = load_policy(path), load_policy(path)
    single = np.stack([pol_a.q_values(s) for s in states])
    batch = pol_b.q_values_batch(states)
    assert batch.shape == (10, 2)
    np.testing.assert_allclose(batch, single, atol=1e-6)


def test_discrete_out_of_range_observation_is_clipped_not_crashing(tmp_path):
    env, rep, dqn, encode, params = _hand_build("discrete", "linear")
    path = str(tmp_path / "d.pt")
    save_policy(path, dqn, rep, params, seed=1)
    pol = load_policy(path)
    q = pol.q_values(np.array([50.0, 50.0, 5.0, -50.0], dtype=np.float32))
    assert q.shape == (2,) and np.all(np.isfinite(q))


# ---------------------------------------------------------------------------------------------
# clone() isolation
# ---------------------------------------------------------------------------------------------
def test_clone_isolates_iht_and_shares_weights(tmp_path):
    env, rep, dqn, encode, params = _hand_build("tile", "linear", iht_size=4096)
    for s in _fixed_states(5, seed=3):
        _trial_q(env, rep, dqn, encode, s, False)
    path = str(tmp_path / "c.pt")
    save_policy(path, dqn, rep, params, seed=1)
    orig = load_policy(path)
    n0 = len(orig.rep.iht.dictionary)
    snapshot = dict(orig.rep.iht.dictionary)

    clone = orig.clone()
    assert clone.net is orig.net
    assert clone.rep.iht is not orig.rep.iht
    rng = np.random.default_rng(9)
    wide = rng.uniform([-4.0, -9.0, -0.4, -9.0], [4.0, 9.0, 0.4, 9.0], size=(300, 4)).astype(np.float32)
    for s in wide:
        clone.act(s)
    assert len(clone.rep.iht.dictionary) > n0
    assert len(orig.rep.iht.dictionary) == n0
    assert orig.rep.iht.dictionary == snapshot
    # a second clone starts from the ORIGINAL table, not from the first clone's
    assert len(orig.clone().rep.iht.dictionary) == n0


def test_clone_of_dense_policies_shares_rep(tmp_path):
    env, rep, dqn, encode, params = _hand_build("ssp", "deep")
    path = str(tmp_path / "s.pt")
    save_policy(path, dqn, rep, params, seed=1)
    pol = load_policy(path)
    c = pol.clone()
    s = _fixed_states(1)[0]
    assert c.rep is pol.rep and c.act(s) == pol.act(s)
    c.reset()


# ---------------------------------------------------------------------------------------------
# Regression: duplicate module identities (tests/test_sspspace_parity.py purges sys.modules['rl*'] and
# re-imports, so a trial can build reps from one copy of rl.representations while policy_io holds another).
# ---------------------------------------------------------------------------------------------
def _load_duplicate(modname):
    import importlib.util
    path = os.path.join(ROOT, "rl", "representations", f"{modname}.py")
    spec = importlib.util.spec_from_file_location(f"_dup_{modname}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod


@pytest.mark.parametrize("kind", ["discrete", "ssp", "tile"])
def test_save_policy_accepts_reps_from_a_duplicate_module_copy(tmp_path, kind):
    env = gym.make("CartPole-v1")
    low = env.observation_space.low.copy()
    low[1] = low[3] = -10
    high = env.observation_space.high.copy()
    high[1] = high[3] = 10
    params = {}
    if kind == "discrete":
        dup = _load_duplicate("onehot")
        assert dup.OneHotRepCP is not OneHotRepCP  # really a different class object
        rep = dup.OneHotRepCP((4, 4, 4, 4), bounds_low=low, bounds_high=high)
    elif kind == "ssp":
        dup = _load_duplicate("ssp")
        assert dup.SSPRep is not SSPRep
        rep = dup.SSPRep(domain_dim=4, length_scale=1.0, n_rotates=2, n_scales=2)
        params = dict(n_rotates=2, n_scales=2, length_scale=1.0)
    else:
        dup = _load_duplicate("tilecoding")
        assert dup.TileCodingRep is not TileCodingRep
        rep = dup.TileCodingRep(env, num_tilings=8, iht_size=1024, bounds_low=low, bounds_high=high)
    hidden = None
    dqn = DQN(rep=rep, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99, buffer_size=100, batch_size=16,
              target_update_freq=10, hidden_dims=hidden, init_seed=5)
    encode = rep.map_indices if hasattr(rep, "map_indices") else rep.map
    states = _fixed_states(12, seed=4)
    q_ref = []
    for s in states:
        q_ref.append(dqn.get_q_values(encode(rep.get_state(s[:4], env))).numpy())
    path = str(tmp_path / f"dup_{kind}.pt")
    # tile: save AFTER the encodes above so the dictionary is non-empty and must round-trip
    save_policy(path, dqn, rep, params, seed=1)
    pol = load_policy(path)
    assert pol.rep_kind == {"discrete": "discrete", "ssp": "ssp", "tile": "tile"}[kind]
    for s, q in zip(states, q_ref):
        assert np.array_equal(pol.q_values(s), q)
