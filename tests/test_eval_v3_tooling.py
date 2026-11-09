"""Tests for the eval_v3 tooling: pod watcher, launcher, and plot helpers.

The important one is `test_watcher_names_match_launcher`: the watcher maps a running pod back to its
(job, cell) by regenerating the name, so if the launcher's naming ever changes the watcher would go
blind and the run would silently lose its cost backstop. That binding is asserted directly here.
"""
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def watch():
    return _load("watch_eval_v3", "scripts/watch_eval_v3.py")


@pytest.fixture(scope="module")
def plot():
    return _load("plot_eval_v3", "scripts/plot_eval_v3.py")


@pytest.fixture(scope="module")
def launcher():
    return _load("launch_cell_pod", "scripts/launch_cell_pod.py")


def test_watcher_names_match_launcher(watch, launcher):
    """The launcher's pod name for every (job, cell) must be exactly what the watcher looks for."""
    import eval_v3_layout as layout
    fake_env = {k: "x" for k in launcher.lp.R2_KEYS}
    expected = watch.expected_names()
    cells = list(layout.ALL_CELLS) + [c for c in layout.A2C_CELLS if c not in layout.ALL_CELLS]
    assert len(cells) == 27, "18 DQN + 9 A2C cells"
    for job in watch.JOBS:
        for cell in cells:
            body = launcher.build_job_body(fake_env, job, cell, 4, "code_key", "")
            assert body["name"] == expected[(job, cell)], (job, cell, body["name"])
    # the tile-coding cells are the ones a naive "/"->"-" round-trip would break
    assert expected[("policy_pass", "tile_coding/value/deep")] == "eval-v3-policy-pass-tile-coding-value-deep"


def test_launcher_does_not_duplicate_workers(launcher):
    """A duplicate --workers would silently override the pod's vCPU-matched worker count."""
    assert launcher.job_args("policy_pass", "ssp/none/deep", 4, "") == "--cell ssp/none/deep --workers 4"
    assert (launcher.job_args("policy_pass", "ssp/none/deep", 2, "--n-seeds 20")
            == "--cell ssp/none/deep --workers 2 --n-seeds 20")
    explicit = launcher.job_args("policy_pass", "ssp/none/deep", 2, "--workers 4")
    assert explicit.count("--workers") == 1 and "--workers 4" in explicit
    # jobs whose runner takes no --cell (the latency sweep covers every cell itself)
    assert launcher.job_args("latency", "ssp/none/linear", 4, "--latency-calls 2000") == "--workers 4 --latency-calls 2000"
    assert "--cell" in launcher.job_args("dqn_settings", "tile_coding/none/deep", 4, "--n-seeds 10")


def test_watcher_age_parsing(watch):
    assert watch.age_min("") != watch.age_min("")  # nan != nan
    mins = watch.age_min("Rented by User: Tue Oct 06 2026 18:36:31 GMT+0000 (Coordinated Universal Time)")
    assert 0 < mins < 60 * 24 * 365
    assert watch.age_min("garbage") != watch.age_min("garbage")


def test_only_eval_v3_names_are_ours(watch):
    expected = set(watch.expected_names().values())
    assert all(n.startswith("eval-v3-") for n in expected)
    assert "cerb-confirm" not in expected
    assert not any(n.startswith("cerb") for n in expected)


def test_plot_run003_table_covers_all_dqn_cells(plot):
    import eval_v3_layout as layout
    assert set(plot.RUN003) == set(layout.ALL_CELLS)
    for cell, (mean, std, filt) in plot.RUN003.items():
        assert 0 <= mean <= 500 and 0 <= std <= 200 and 0 <= filt <= 500


def test_plot_condition_levels(plot):
    assert plot.cond_level("baseline") == 1.0
    assert plot.cond_level("pole_length_x0.5") == 0.5
    assert plot.cond_level("obs_noise_0.01") == 0.01
    assert plot.cond_level("action_delay_2") == 2.0
    assert plot.cond_level("gravity_x1.2") == 1.2


def test_plot_stability_and_band_helpers(plot):
    import numpy as np
    runs = [{"eval_history": [0.0] * 5 + [500.0] * 10},      # solves and stays
            {"eval_history": [0.0] * 12 + [500.0, 10.0, 20.0]},  # solves then collapses
            {"eval_history": [10.0] * 15}]                    # never solves
    st = plot.stability_stats(runs, solved=475.0)
    assert st["n"] == 3 and st["ever"] == 2 and st["stay"] == 1
    assert st["first_reach"] == [5, 12]
    mean, lo, hi = plot.band(np.array([[1.0, 2.0], [3.0, 4.0]]), "p5_95")
    assert np.allclose(mean, [2.0, 3.0]) and np.all(lo <= mean) and np.all(mean <= hi)
    lo, hi = plot.bootstrap_ci([])  # must return a pair rather than raise
    assert np.isnan(lo) and np.isnan(hi)
