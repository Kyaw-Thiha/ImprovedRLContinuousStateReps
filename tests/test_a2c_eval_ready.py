"""Eval-readiness of the A2C port: checkpoints/probes/final eval/save hooks in train_a2c, the saved-policy
round trip, and that the saved policy works unchanged with the DQN eval code (robustness, deployment)."""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from experiments.trial_cartpole_a2c import A2C_BASE_PARAMS, train_a2c  # noqa: E402

REPS = {
    "ssp": dict(rep_="PlaceSSP", length_scale=0.6, n_rotates=2),
    "tile": dict(rep_="TileCoding", num_tilings=8, tiles_per_dim=[6, 6, 6, 6], iht_size=16384,
                 tile_state_indices=[0, 1, 2, 3]),
    "discrete": dict(rep_="Discrete", n_bins=6),
}


def _params(rep, mode="none", trials=30):
    p = dict(A2C_BASE_PARAMS)
    p.update(REPS[rep])
    p.update(lr=0.6, eps=0.2, reward_center_mode=mode, reward_center_beta=0.01, reward_center_eta=0.5, trials=trials)
    return p


def _probe(n=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, (n, 4)) * np.array([2.0, 2.0, 0.2, 2.0])).astype(np.float32)


@pytest.mark.parametrize("rep", ["ssp", "tile", "discrete"])
def test_eval_hooks_do_not_change_training(rep):
    base = train_a2c(_params(rep), seed=5, rng=np.random.RandomState(5))
    hooked = train_a2c(_params(rep), seed=5, rng=np.random.RandomState(5), eval_every=10, n_eval_episodes=2,
                       final_eval_episodes=3, probe_states=_probe())
    assert hooked["returns"] == base["returns"]
    if rep != "tile":
        np.testing.assert_array_equal(hooked["ac"].W, base["ac"].W)
    else:
        # Checkpoint evals allocate tile-coding IHT entries (as in the DQN trial), which only permutes the
        # column order of W; the multiset of weights is the same.
        np.testing.assert_allclose(np.sort(hooked["ac"].W, axis=None), np.sort(base["ac"].W, axis=None), atol=1e-9)
    assert len(hooked["eval_history"]) == 3 and len(hooked["probe_history"]) == 3
    assert len(hooked["final_eval_returns"]) == 3
    ph = hooked["probe_history"][-1]
    assert all(np.isfinite(ph[k]) for k in ("mean_value", "mean_max_q", "mean_action_gap", "avg_reward"))
    assert ph["episode"] == 30 and ph["checkpoint"] == 3


@pytest.mark.parametrize("mode", ["none", "simple", "value"])
@pytest.mark.parametrize("rep", ["ssp", "tile", "discrete"])
def test_save_load_roundtrip_is_exact(tmp_path, rep, mode):
    from rl.policy_io import load_policy, verify_saved_policy

    path = str(tmp_path / "p.pt")
    out = train_a2c(_params(rep, mode), seed=7, rng=np.random.RandomState(7), eval_every=15, n_eval_episodes=2,
                    final_eval_episodes=2, save_policy_path=path, verify_episodes=3)
    assert os.path.exists(path) and out["policy_saved"]["bytes"] > 0
    ver = verify_saved_policy(path, out["verify_env_seed"], len(out["verify_returns"]), out["verify_returns"],
                              max_steps=out["verify_max_steps"])
    assert ver["ok"] and ver["max_abs_diff"] == 0.0

    pol = load_policy(path)
    assert type(pol).__name__ == "LinearACPolicy" and pol.meta["algo"] == "a2c_linear"
    assert pol.meta["agent"]["centering"] == mode
    # greedy action of the reloaded policy == greedy action of the trained agent on a batch of states
    rng = np.random.default_rng(1)
    for _ in range(40):
        obs = (rng.uniform(-1, 1, 4) * np.array([2.0, 2.0, 0.2, 2.0])).astype(np.float32)
        phi = out["rep"].map(out["rep"].get_state(obs, None)) if rep == "ssp" else pol.encode(obs)
        assert pol.act(obs) == int(np.argmax(out["ac"].preferences(phi if rep == "ssp" else
                                                                   (np.unique(out["rep"].map_indices(obs)) if rep == "tile"
                                                                    else out["rep"].map_indices(out["rep"].get_state(obs, None))))))


def test_dqn_policies_still_load(tmp_path):
    """The A2C dispatch must not touch DQN files (no `algo` key)."""
    import torch
    from rl import policy_io

    src = open(os.path.join(REPO, "rl", "policy_io.py")).read()
    assert 'blob.get("algo") == "a2c_linear"' in src and "FORMAT_VERSION = 1" in src
    assert hasattr(policy_io, "Policy") and hasattr(torch, "load")


def _saved(tmp_path, rep, mode="none"):
    path = str(tmp_path / f"{rep}.pt")
    train_a2c(_params(rep, mode, trials=25), seed=3, rng=np.random.RandomState(3), save_policy_path=path,
              verify_episodes=2)
    from rl.policy_io import load_policy
    return load_policy(path)


@pytest.mark.parametrize("rep", ["ssp", "tile", "discrete"])
def test_policy_works_with_robustness_code(tmp_path, rep):
    from rl.perturbations import ALL_CONDITIONS, run_condition

    pol = _saved(tmp_path, rep)
    for cond in ("baseline", ALL_CONDITIONS[1], "obs_noise_0.1", "action_delay_1", "random_action_0.05"):
        if cond not in ALL_CONDITIONS:
            continue
        rets = run_condition(pol, cond, n_episodes=2, noise_sigma=np.array([0.43, 0.31, 0.03, 0.21]))
        assert len(rets) == 2 and all(r > 0 for r in rets)
    # the saved policy must not be mutated by rollouts (clone semantics)
    n0 = len(pol.rep.iht.dictionary) if rep == "tile" else None
    run_condition(pol, "wide_init_0.2", n_episodes=2, noise_sigma=None)
    if rep == "tile":
        assert len(pol.rep.iht.dictionary) == n0


@pytest.mark.parametrize("rep", ["ssp", "tile", "discrete"])
def test_policy_works_with_deployment_code(tmp_path, rep):
    import run_deployment as rd

    pol = _saved(tmp_path, rep)
    rpt = rd.size_report(pol)
    A, D = pol.n_actions, pol.dim
    assert rpt["n_params"] == A * D and rpt["network"]["n_params"] == A * D
    assert rpt["network"]["fp32"] == 4 * A * D and rpt["network"]["fp16"] == 2 * A * D
    assert rpt["network"]["int8"] == A * D + 4 * A  # one fp32 scale per output row
    ops = rd.operation_counts(pol)
    expect_macs = (rd.n_active_of(pol) * A) if pol.sparse else (D * A)
    assert ops["network_macs"] == expect_macs

    ref = pol.clone().q_values_batch(_probe(8))
    variants = ["fp32", "fp16_weights", "int8_weights"] + (["fp16_encoder"] if rep == "ssp" else [])
    for v in variants:
        base = rd.make_variant(pol, v)
        rets = rd.greedy_returns(rd._episode_factory(base, v), 123, 2)
        assert len(rets) == 2
        if v != "fp32":
            diff = np.abs(base.clone().q_values_batch(_probe(8)) - ref).max()
            assert diff > 0.0, f"{v} did not change the function (override ignored)"
    # the original must be untouched by the variants
    np.testing.assert_array_equal(pol.clone().q_values_batch(_probe(8)), ref)
    t = rd.time_calls(pol, _probe(8), n_calls=20, warmup=2)
    assert t["act"]["median_us"] > 0


def test_layout_and_cell_helpers():
    import eval_v3_layout as layout

    assert layout.study_name("ssp/none/a2c") == "a2c_ssp_none_port"
    assert layout.cell_from_study("a2c_tile_coding_value_port") == "tile_coding/value/a2c"
    assert "ssp/none/deep" in layout.ALL_CELLS and not any(c.endswith("/a2c") for c in layout.ALL_CELLS)
    assert len(layout.A2C_CELLS) == 9


@pytest.mark.parametrize("cell", ["ssp/value/a2c", "tile_coding/none/a2c"])
def test_tiny_policy_pass_end_to_end(tmp_path, cell):
    """Offline tiny pass through the SAME entrypoint the pod launcher uses (run_policy_pass.py delegates),
    then the robustness runner and analysis on the saved policies."""
    import json
    import subprocess

    import eval_v3_layout as layout

    root = str(tmp_path / "ev3")
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=REPO, env=env)  # noqa: E731
    r = run("scripts/run_policy_pass.py", "--cell", cell, "--tiny", "--no-r2", "--local-root", root,
            "--workers", "2", "--n-seeds", "2")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    study = layout.study_name(cell)
    summ = json.load(open(layout.local_path(layout.train_summary_key(study), root)))
    man = json.load(open(layout.local_path(layout.manifest_key(study), root)))
    assert summ["n_seeds"] == 2 and len(man["policy_files"]) == 2
    assert all(p["verify_max_abs_diff"] == 0.0 for p in man["policy_files"])
    runs = [json.loads(line) for line in open(layout.local_path(layout.train_runs_key(study), root))]
    assert all(x["policy_verified"] and len(x["eval_history"]) == 6 and x["probe_history"] for x in runs)
    for seed in (3000, 3001):
        assert os.path.exists(layout.local_path(layout.policy_key(study, seed), root))
    r = run("scripts/run_robustness.py", "--cell", cell, "--no-r2", "--local-root", root, "--n-seeds", "2",
            "--episodes", "2", "--workers", "2", "--conditions", "baseline,gravity_x1.2")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert len(open(layout.local_path(layout.robustness_key(study), root)).readlines()) == 4
