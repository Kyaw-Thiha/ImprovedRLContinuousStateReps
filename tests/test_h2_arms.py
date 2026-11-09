"""H2 (d131): the CartPole length-scale arms. The three arms must differ only in what the test is about, and a quick
smoke run must show that `normalize_state` really changes the encoder's input (not just a stored flag)."""

import os
import sys

import numpy as np
import optuna
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

import run_hparam_search_dqn as s  # noqa: E402
import launch_cell_pod as lc  # noqa: E402


def _sample(rep, seed=0):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    out = {}

    def obj(trial):
        out["p"] = s.sample_representation_params(trial, rep)
        out["t"] = dict(trial.params)
        return 0.0

    study.optimize(obj, n_trials=1)
    return out["p"], out["t"]


def test_h2_ssp_arms_differ_only_in_normalisation_and_share_the_wide_length_scale_range():
    pn, _ = _sample("ssp_h2")
    pr, _ = _sample("ssp_h2raw")
    assert pn["normalize_state"] is True and pr["normalize_state"] is False
    assert {k: v for k, v in pn.items() if k != "normalize_state"} == {k: v for k, v in pr.items() if k != "normalize_state"}
    lengths = [_sample("ssp_h2", i)[0]["length_scale"] for i in range(200)]
    assert min(lengths) < 0.05 and max(lengths) > 0.5 and all(0.01 <= x <= 1.0 for x in lengths)


def test_the_original_ssp_space_and_the_tile_space_are_untouched():
    old = [_sample("ssp", i)[0]["length_scale"] for i in range(100)]
    assert min(old) >= 0.1
    p, _ = _sample("tile_h2")
    q, _ = _sample("tile_coding")
    assert p == q and p["rep_"] == "TileCoding"


def test_trial_params_round_trip_keeps_the_normalisation_flag_for_reeval_and_final_eval():
    for rep, flag in (("ssp_h2", True), ("ssp_h2raw", False)):
        _, t = _sample(rep)
        out = s.trial_params_from_search_params(rep, "none", t, "linear")
        assert out["rep_"] == "PlaceSSP" and out["normalize_state"] is flag
    _, t = _sample("tile_h2")
    out = s.trial_params_from_search_params("tile_h2", "none", t, "linear")
    assert out["rep_"] == "TileCoding" and out["tile_state_indices"] == [0, 1, 2, 3]


def test_h2_arms_are_never_warm_started_and_the_launcher_knows_them():
    study = optuna.create_study()
    for rep in s.H2_ALL:
        assert s._enqueue_warmstart(study, rep, "none") is False
    assert len(study.trials) == 0
    for rep in s.H2_ALL:
        assert lc.parse_cell(f"{rep}/none/linear")[0] == rep
    assert lc.study_name("ssp_h2/none/linear") == "dqn_ssp_h2_none_v4_hparam_search"
    assert "ssp_h2" not in " ".join(lc.ALL_CELLS)  # the 18 original cells are unchanged
    a = lc.cell_args("tile_h2/none/linear", 16, 100, "--final-seed-offset 3000")
    assert "--final-seed-offset 3000" in a and "--representation tile_h2" in a


@pytest.mark.skipif(os.environ.get("SKIP_SLOW"), reason="runs a few CartPole episodes")
def test_normalize_state_changes_what_the_encoder_sees():
    from experiments.trial_cartpole_dqn import DQNTrial
    p = dict(s.BASE_PARAMS)
    p.update(s.DQN_DEFAULTS)
    p.update(trials=3, steps=40, architecture="linear", reward_center_mode="none", rep_="PlaceSSP",
             length_scale=0.2, n_rotates=2, n_scales=5, eps=0.1, lr=0.01, eval_freq=0)
    seen = {}
    import rl.representations.ssp as sspmod
    orig = sspmod.SSPRep.map

    def spy(self, state):
        seen.setdefault("max", []).append(float(np.max(np.abs(state))))
        return orig(self, state)

    sspmod.SSPRep.map = spy
    try:
        import experiments.trial_cartpole_dqn as t
        t.SSPRep.map = spy
        DQNTrial().run(seed=1, data_dir="/tmp/h2_unit", normalize_state=False, **p)
        raw_max = max(seen["max"])
        seen["max"] = []
        DQNTrial().run(seed=1, data_dir="/tmp/h2_unit", normalize_state=True, **p)
        norm_max = max(seen["max"])
    finally:
        sspmod.SSPRep.map = orig
    assert norm_max < raw_max and norm_max <= 1.0 + 1e-6
