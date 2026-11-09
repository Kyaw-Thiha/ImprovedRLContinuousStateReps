"""
Tests for the eval_v3 hooks in experiments/trial_cartpole_dqn.py (seed_env, probe_path,
final_eval_returns, verify_episodes, save_policy_path). All hooks default off; with defaults the
run must be unchanged apart from the extra optional metadata keys.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from experiments.trial_cartpole_dqn import DQNTrial  # noqa: E402

REP_PARAMS = {
    "discrete": dict(rep_="Discrete", n_bins=4),
    "ssp": dict(rep_="HexSSP", n_rotates=2, n_scales=2, length_scale=1.0),
    "tile": dict(rep_="TileCoding", num_tilings=8, iht_size=256),
}

# Keys returned by DQNTrial before the eval_v3 hooks existed.
BASE_KEYS = {
    "dimensionality", "terminal_reward_learning", "terminal_reward", "terminal_eval_reward",
    "episodes_to_learn", "trial_ID", "build_time", "total_time", "avg_trial_time", "eval_history",
    "eval_time", "train_time", "env_steps", "n_updates", "n_params", "n_active", "final_eval_reward",
}
NEW_KEYS = {
    "final_eval_returns", "probe_history", "avg_reward_final", "verify_returns", "verify_env_seed",
    "verify_max_steps", "policy_saved",
}


def _run(tmp_path, rep="discrete", arch="linear", **kw):
    args = dict(
        trials=12, steps=60, eval_freq=4, n_eval_episodes=2, seed=3001, init_seed=-1,
        architecture=arch, learning_starts=50, batch_size=16, buffer_size=1000,
        data_dir=str(tmp_path), verbose=False,
    )
    args.update(REP_PARAMS[rep])
    args.update(kw)
    return DQNTrial().run(**args)


def _probe_file(tmp_path, n=16):
    rng = np.random.default_rng(0)
    states = rng.uniform([-2, -2, -0.2, -2], [2, 2, 0.2, 2], size=(n, 4)).astype(np.float32)
    p = str(tmp_path / "probe.npz")
    np.savez(p, states=states, noise_sigma=np.ones(4), source=np.array("test"))
    return p


def test_defaults_off_returns_base_keys_plus_only_new_optional_ones(tmp_path):
    meta = _run(tmp_path, trials=3, steps=30, eval_freq=0)
    assert set(meta) == BASE_KEYS | NEW_KEYS
    assert meta["final_eval_returns"] is None
    assert meta["probe_history"] == []
    assert meta["verify_returns"] is None and meta["verify_env_seed"] is None
    assert meta["policy_saved"] is None
    assert meta["eval_history"] == []
    assert meta["avg_reward_final"] == 0.0


def test_probe_history_has_one_entry_per_checkpoint(tmp_path):
    meta = _run(tmp_path, probe_path=_probe_file(tmp_path))
    hist = meta["probe_history"]
    assert len(hist) == len(meta["eval_history"]) == 3
    for k, h in enumerate(hist, start=1):
        assert set(h) == {"checkpoint", "episode", "mean_max_q", "mean_action_gap", "avg_reward"}
        assert h["checkpoint"] == k and h["episode"] == 4 * k
        assert np.isfinite(h["mean_max_q"]) and h["mean_action_gap"] >= 0.0
        assert isinstance(h["avg_reward"], float)


@pytest.mark.parametrize("rep", ["discrete", "ssp", "tile"])
def test_probe_does_not_change_training(tmp_path, rep):
    """The probe is a diagnostic only: it must not shift the IHT (tile) or any RNG stream."""
    common = dict(rep=rep, seed_env=True, final_eval_episodes=2)
    a = _run(tmp_path, **common)
    b = _run(tmp_path, probe_path=_probe_file(tmp_path), **common)
    assert a["eval_history"] == b["eval_history"]
    assert a["final_eval_returns"] == b["final_eval_returns"]
    assert a["env_steps"] == b["env_steps"] and a["n_updates"] == b["n_updates"]


def test_final_eval_returns_length_and_mean(tmp_path):
    meta = _run(tmp_path, final_eval_episodes=5, seed_env=True)
    rets = meta["final_eval_returns"]
    assert len(rets) == 5 and all(isinstance(r, float) for r in rets)
    assert meta["final_eval_reward"] == pytest.approx(float(np.mean(rets)))


@pytest.mark.parametrize("rep", ["discrete", "ssp", "tile"])
def test_seed_env_makes_runs_reproducible(tmp_path, rep):
    a = _run(tmp_path, rep=rep, seed_env=True, final_eval_episodes=3)
    b = _run(tmp_path, rep=rep, seed_env=True, final_eval_episodes=3)
    assert a["eval_history"] == b["eval_history"]
    assert a["final_eval_returns"] == b["final_eval_returns"]
    assert a["env_steps"] == b["env_steps"]
    assert a["n_updates"] == b["n_updates"]


def test_save_without_verify_and_verify_fields(tmp_path):
    path = str(tmp_path / "p.pt")
    meta = _run(tmp_path, save_policy_path=path, seed_env=True)
    assert meta["policy_saved"]["path"] == path and os.path.exists(path)
    assert len(meta["policy_saved"]["sha256"]) == 64
    assert meta["verify_returns"] is None and meta["verify_env_seed"] is None

    path2 = str(tmp_path / "p2.pt")
    meta2 = _run(tmp_path, save_policy_path=path2, seed_env=True, verify_episodes=2)
    assert len(meta2["verify_returns"]) == 2
    assert meta2["verify_env_seed"] == 3001 + 2_000_000

    # verify_episodes without a save path does nothing
    meta3 = _run(tmp_path, verify_episodes=2)
    assert meta3["verify_returns"] is None and meta3["policy_saved"] is None
