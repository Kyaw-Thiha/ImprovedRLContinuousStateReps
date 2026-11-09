"""Tests for speed-aware config selection (handoff 09 section 15.4 step 1).

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python test_select.py

Why: ``pick_winners`` used to choose by ``last5_id`` on ONE search seed per config.  Under ties near ceiling that choice
is arbitrary with respect to speed (RUN_009: the same arm needed 225 or 50 updates to reach 0.90 depending on which
tied config won).  These tests pin the new behaviour and that the old one is reproduced when asked for.
"""
import argparse
import sys

import run_sweep as R


def _row(cfg_idx, seed, last5, conv, aulc, arm="ssp", variant="index", phase="search"):
    return {"variant": variant, "arm": arm, "arm_real": arm, "phase": phase, "cfg_idx": cfg_idx, "seed": seed,
            "cfg": {"lr": 0.001 * (cfg_idx + 1)},
            "summary": {"last5_id": last5, "conv_id": conv, "aulc_id": aulc}}


def test_tie_goes_to_the_fast_config():
    # three configs, all tied at ceiling on converged accuracy; config 2 is the fast one (largest AULC)
    rows = [_row(0, 0, 1.0, 1.0, 0.70), _row(1, 1, 1.0, 1.0, 0.60), _row(2, 2, 1.0, 1.0, 0.95)]
    w = R.pick_winners(rows, select="aulc")[("index", "ssp")]
    assert w["cfg_idx"] == 2, w["cfg_idx"]


def test_last5_reproduces_the_old_pick():
    rows = [_row(0, 0, 0.90, 0.99, 0.95), _row(1, 1, 0.99, 0.99, 0.60), _row(2, 2, 0.99, 0.99, 0.95)]
    w = R.pick_winners(rows, select="last5")[("index", "ssp")]
    assert w["cfg_idx"] == 1, "stable sort: the first of the tied maxima, exactly as before"
    w_default = R.pick_winners(rows)[("index", "ssp")]
    assert w_default["cfg_idx"] == w["cfg_idx"], "default must stay the old behaviour"


def test_aulc_averages_over_search_seeds():
    # config 0 got one lucky seed and one bad one; config 1 is steadily good: the mean decides
    rows = [_row(0, 0, 1.0, 1.0, 0.99), _row(0, 100, 1.0, 1.0, 0.40),
            _row(1, 1, 1.0, 1.0, 0.80), _row(1, 101, 1.0, 1.0, 0.82)]
    w = R.pick_winners(rows, select="aulc")[("index", "ssp")]
    assert w["cfg_idx"] == 1, w["cfg_idx"]


def test_aulc_ties_are_broken_by_converged_accuracy():
    rows = [_row(0, 0, 0.9, 0.90, 0.80), _row(1, 1, 0.9, 0.99, 0.80)]
    assert R.pick_winners(rows, select="aulc")[("index", "ssp")]["cfg_idx"] == 1


def test_arms_and_final_rows_are_not_mixed():
    rows = [_row(0, 0, 1.0, 1.0, 0.5, arm="gru"), _row(1, 1, 1.0, 1.0, 0.9, arm="gru"),
            _row(0, 0, 1.0, 1.0, 0.9, arm="ssp"), _row(1, 1, 1.0, 1.0, 0.5, arm="ssp"),
            _row(2, 1000, 1.0, 1.0, 9.9, arm="ssp", phase="final")]
    win = R.pick_winners(rows, select="aulc")
    assert win[("index", "gru")]["cfg_idx"] == 1 and win[("index", "ssp")]["cfg_idx"] == 0


def test_incomplete_config_groups_are_not_compared():
    # config 1 has only one of the two search seeds (an interrupted run): it must not win on a lone lucky seed
    rows = [_row(0, 0, 1.0, 1.0, 0.70), _row(0, 100, 1.0, 1.0, 0.72), _row(1, 1, 1.0, 1.0, 0.99)]
    assert R.pick_winners(rows, select="aulc")[("index", "ssp")]["cfg_idx"] == 0


def test_min_conv_is_a_floor_not_a_ranking():
    """docs/11 step 2: select by AULC *subject to* converged accuracy >= threshold.

    A config with the best AULC but a converged score under the floor must lose to the fastest config that clears it.
    """
    rows = [_row(0, 0, 1.0, 0.70, 0.99), _row(1, 1, 1.0, 0.96, 0.80)]
    assert R.pick_winners(rows, select="aulc")[("index", "ssp")]["cfg_idx"] == 0, "without a floor AULC wins"
    w = R.pick_winners(rows, select="aulc", min_conv=0.95)[("index", "ssp")]
    assert w["cfg_idx"] == 1, w["cfg_idx"]


def test_min_conv_falls_back_when_nothing_clears_it():
    rows = [_row(0, 0, 1.0, 0.80, 0.99), _row(1, 1, 1.0, 0.90, 0.70)]
    w = R.pick_winners(rows, select="aulc", min_conv=0.95)[("index", "ssp")]
    assert w["cfg_idx"] == 1, "with no qualifying config the best converged one is the safest fallback"


def test_min_conv_does_not_apply_to_the_last5_rule():
    rows = [_row(0, 0, 0.99, 0.10, 0.10), _row(1, 1, 0.50, 0.99, 0.99)]
    assert R.pick_winners(rows, select="last5", min_conv=0.95)[("index", "ssp")]["cfg_idx"] == 0


def _args(search_seeds):
    return argparse.Namespace(variants=["index"], arms=["gru"], budget=256, steps=10, batch=8, eval_every=5,
                              task_kw={"ep_len": 16, "n_items": 8, "n_landmarks": 6}, tag="t", arm_cfg={},
                              label=None, eval_steps=None, pool=0, robust=False, search_seeds=search_seeds)


def test_search_seeds_offsets_and_same_config_draw():
    one = R.build_search_jobs(_args(1), n_configs=3)
    three = R.build_search_jobs(_args(3), n_configs=3)
    assert len(one) == 3 and len(three) == 9
    assert [j["seed"] for j in one] == [0, 1, 2], "K=1 must keep the old seeds so existing rows are reused"
    by_cfg = {}
    for j in three:
        by_cfg.setdefault(j["cfg_idx"], []).append(j)
    for ci, js in by_cfg.items():
        assert sorted(j["seed"] for j in js) == [ci, ci + 100, ci + 200]
        assert all(j["cfg"] == js[0]["cfg"] for j in js), "all seeds of a config share the one drawn config"
        assert js[0]["cfg"] == one[ci]["cfg"], "adding seeds must not change the drawn configs"
    keys = {(j["cfg_idx"], j["seed"]) for j in three}
    assert len(keys) == 9, "resume keys must stay distinct"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
