"""Pure-logic tests for scripts/run_policy_pass.py (no training, no network)."""
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)

import eval_v3_layout as layout  # noqa: E402
import parallel_jobs as pj  # noqa: E402
import run_policy_pass as rpp  # noqa: E402


def test_winner_yaml_payload_and_bare(tmp_path):
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump({"params": {"lr": 0.1, "n_bins": 7}, "winner_search_trial": 42}))
    w = rpp.read_winner_yaml(str(p))
    assert w["params"] == {"lr": 0.1, "n_bins": 7} and w["winner_trial"] == 42
    q = tmp_path / "b.yaml"
    q.write_text(yaml.safe_dump({"lr": 0.2}))
    w = rpp.read_winner_yaml(str(q))
    assert w["params"] == {"lr": 0.2} and w["winner_trial"] is None


def test_params_discrete_and_defaults():
    p = rpp.build_trial_params("discrete/none/linear", {"n_bins": 7, "eps": 0.1, "lr": 0.05})
    assert p["rep_"] == "Discrete" and p["architecture"] == "linear" and p["reward_center_mode"] == "none"
    assert p["trials"] == 1000 and p["steps"] == 500 and p["buffer_size"] == 5000
    assert p["target_update_freq"] == 10 and p["n_bins"] == 7


def test_params_tile_and_deep():
    p = rpp.build_trial_params("tile_coding/value/deep", {"tiles_per_dim": 12, "num_tilings": 16})
    assert p["rep_"] == "TileCoding" and p["tiles_per_dim"] == [12] * 4
    assert p["tile_state_indices"] == [0, 1, 2, 3] and p["architecture"] == "deep"
    assert p["reward_center_mode"] == "value"
    # converting an already-converted v2 yaml is idempotent
    again = rpp.build_trial_params("tile_coding/value/deep", {k: v for k, v in p.items()
                                                              if k in ("tiles_per_dim", "num_tilings",
                                                                       "rep_", "tile_state_indices")})
    assert again["tiles_per_dim"] == [12] * 4 and again["rep_"] == "TileCoding"


def test_variant_overrides_and_tiny():
    base = rpp.build_trial_params("ssp/none/linear", {"lr": 0.1, "buffer_size": 9000})
    v = rpp.build_trial_params("ssp/none/linear", {"lr": 0.1, "buffer_size": 9000},
                               buffer_size=50000, target_update_freq=100)
    assert base["buffer_size"] == 9000 and v["buffer_size"] == 50000 and v["target_update_freq"] == 100
    assert {k: x for k, x in v.items() if k not in ("buffer_size", "target_update_freq")} == \
           {k: x for k, x in base.items() if k not in ("buffer_size", "target_update_freq")}
    t = rpp.build_trial_params("ssp/none/linear", {"lr": 0.1}, tiny=True)
    assert t["trials"] == 24 and t["steps"] == 80 and t["learnTrials"] == 24


def test_variant_folder_naming():
    assert rpp.variant_of(None, None) is None
    assert rpp.variant_of(50000, None) == "buf50000"
    assert rpp.variant_of(None, 100) == "tu100"
    assert rpp.variant_of(50000, 100) == "buf50000_tu100"
    study = layout.study_name("ssp/none/deep")
    key = layout.train_runs_key(study, rpp.variant_of(50000, None))
    assert key == f"{layout.ROOT}/dqn_settings/{study}/buf50000/train_runs.jsonl"
    assert layout.train_runs_key(study) == f"{layout.ROOT}/{study}/train_runs.jsonl"


def test_jobs_keys_and_seeds():
    params = {"lr": 0.1}
    jobs = rpp.build_jobs("ssp/none/deep", params, 13, 3, layout.POLICY_SEED_OFFSET, "/tmp/r", save=True,
                          upload=True, eval_freq=33, n_eval_episodes=10, final_eval_episodes=100,
                          verify_episodes=20, probe_path="/tmp/p.npz", data_dir="/tmp/d")
    assert [pj.job_key(j) for j in jobs] == [("policy", 13, 0), ("policy", 13, 1), ("policy", 13, 2)]
    study = layout.study_name("ssp/none/deep")
    assert jobs[1]["policy_key"] == layout.policy_key(study, 3001)
    assert jobs[1]["policy_path"] == layout.local_path(layout.policy_key(study, 3001), "/tmp/r")
    assert jobs[0]["seed_offset"] == 3000 and jobs[0]["save"] and jobs[0]["upload"]
    nj = rpp.build_jobs("ssp/none/deep", params, 13, 1, 3000, "/tmp/r", save=False, upload=True, eval_freq=1,
                        n_eval_episodes=1, final_eval_episodes=1, verify_episodes=0, probe_path="",
                        data_dir="/tmp/d", variant="buf200")
    assert nj[0]["save"] is False and nj[0]["upload"] is False


def _rec(i, score, fin, q, gap):
    return {"seed_idx": i, "seed": 3000 + i, "score": score, "last1": score + 1, "final_eval_reward": fin,
            "terminal_reward": 10.0 + i, "train_time": 1.0, "eval_time": 0.5, "total_time": 2.0,
            "dimensionality": 101, "n_params": 202, "env_steps": 100, "n_updates": 10,
            "probe_history": [{"checkpoint": 1, "mean_max_q": 0.0, "mean_action_gap": 0.0},
                              {"checkpoint": 2, "mean_max_q": q, "mean_action_gap": gap}],
            "policy_key": f"k{i}", "policy_sha256": "ab" * 32, "policy_bytes": 123 + i,
            "policy_verified": True, "verify_max_abs_diff": 0.0}


def test_summarize_runs():
    runs = [_rec(0, 500.0, 500.0, 40.0, 1.0), _rec(1, 300.0, 200.0, 20.0, 3.0), _rec(2, 100.0, 100.0, 30.0, 2.0)]
    s = rpp.summarize_runs(runs)
    assert s["n_seeds"] == 3 and s["score_mean"] == pytest.approx(300.0)
    assert s["score_std"] == pytest.approx((2 * 200.0 ** 2 / 3) ** 0.5)      # np.std (ddof=0)
    assert s["score_min"] == 100.0 and s["score_max"] == 500.0
    assert s["share_solved_lastk"] == pytest.approx(1 / 3)
    assert s["share_solved_final_policy"] == pytest.approx(1 / 3)
    assert s["per_seed_scores"] == [500.0, 300.0, 100.0]
    assert s["final_policy_eval_mean"] == pytest.approx(800.0 / 3)
    assert s["probe"]["per_seed_final_mean_max_q"] == [40.0, 20.0, 30.0]
    assert s["probe"]["mean_action_gap"]["mean"] == pytest.approx(2.0)
    assert s["cost"]["n_params"] == 202 and s["cost"]["total_time_mean_s"] == 2.0
    files = rpp.policy_files(runs)
    assert [f["bytes"] for f in files] == [123, 124, 125] and files[0]["seed"] == 3000


def test_summarize_without_optional_fields():
    s = rpp.summarize_runs([{"score": 10.0, "last1": 11.0, "terminal_reward": 5.0}])
    assert s["final_policy_eval_mean"] is None and s["probe"]["mean_max_q"] is None
