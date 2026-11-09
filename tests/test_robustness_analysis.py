"""Robustness analysis statistics and runner job construction, on synthetic records (no policies needed)."""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, ".."))

import analyze_robustness as ar  # noqa: E402
import eval_v3_layout as layout  # noqa: E402
import run_robustness as rr  # noqa: E402

FAMS = {"baseline": ["baseline"], "gravity": ["gravity_x0.8", "gravity_x1.2"]}


def make_records(seed_idx, means, n_ep=3):
    return [{"kind": "rob", "seed_idx": seed_idx, "seed": 3000 + seed_idx, "condition": c, "n_episodes": n_ep,
             "returns": [float(m)] * n_ep, "mean": float(m)} for c, m in means.items()]


def fake_cell(n_seeds, base, grav, n_bad=0, bad_base=300.0, jitter=5.0, rng_seed=0):
    """grav is the mean gravity score; the first n_bad seeds get a failed baseline (< 475)."""
    rng = np.random.default_rng(rng_seed)
    recs = []
    for s in range(n_seeds):
        b = bad_base if s < n_bad else base
        g = grav + rng.uniform(-jitter, jitter)
        recs += make_records(s, {"baseline": b, "gravity_x0.8": g + 10, "gravity_x1.2": g - 10})
    return recs


def test_family_score_and_retained_formulas():
    recs = (make_records(0, {"baseline": 500, "gravity_x0.8": 300, "gravity_x1.2": 100})
            + make_records(1, {"baseline": 250, "gravity_x0.8": 200, "gravity_x1.2": 100}))
    fs = ar.family_scores(recs, FAMS)
    assert fs[0] == {"baseline": 500.0, "gravity": 200.0}
    assert fs[1] == {"baseline": 250.0, "gravity": 150.0}
    num, den = ar._arrays(fs, "gravity", retained=True)
    ci = ar.bootstrap_ci_mean(num, 200, den=den)
    assert ci["mean"] == pytest.approx(175.0 / 375.0)  # ratio of means, not mean of ratios
    cell = ar.summarize_cell(recs, FAMS, n_boot=200)
    assert cell["families"]["gravity"]["retained"]["mean"] == pytest.approx(175.0 / 375.0)
    assert cell["families"]["gravity"]["score"]["mean"] == pytest.approx(175.0)
    assert cell["worst_condition"]["condition"] == "gravity_x1.2"
    assert cell["baseline_solved_share"] == pytest.approx(0.5)


def test_incomplete_family_dropped_and_errors_ignored():
    recs = make_records(0, {"baseline": 500, "gravity_x0.8": 300})  # gravity_x1.2 missing
    recs.append({"kind": "rob", "seed_idx": 0, "condition": "gravity_x1.2", "error": "boom"})
    fs = ar.family_scores(recs, FAMS)
    assert fs[0] == {"baseline": 500.0}


def test_bootstrap_ci_contains_true_mean():
    x = 100.0 + np.linspace(-10, 10, 40)  # sample mean is exactly 100
    ci = ar.bootstrap_ci_mean(x, 1000, seed=1)
    assert ci["mean"] == pytest.approx(100.0)
    assert ci["lo"] < 100.0 < ci["hi"]
    assert ar.bootstrap_ci_mean(x, 1000, seed=1) == ci  # reproducible


def test_bootstrap_diff_sign_and_null():
    a = 400.0 + np.linspace(-10, 10, 20)
    b = 300.0 + np.linspace(-10, 10, 20)
    r = ar.bootstrap_diff(a, b, 1000, seed=2)
    assert r["diff"] == pytest.approx(100.0)
    assert r["lo"] > 0 and r["p"] < 0.01
    r2 = ar.bootstrap_diff(b, a, 1000, seed=2)
    assert r2["diff"] == pytest.approx(-100.0) and r2["hi"] < 0
    same = ar.bootstrap_diff(a, a, 1000, seed=3)
    assert same["lo"] < 0 < same["hi"] and same["p"] > 0.05


def test_holm_hand_computed():
    adj, rej = ar.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # sorted 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max 0.02, 0.03, 0.06, 0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej == [True, False, False, True]
    assert ar.holm([]) == ([], [])
    assert ar.holm([0.9, 0.99])[0] == pytest.approx([1.0, 1.0])


def test_sensitivity_filter_drops_failed_baselines():
    recs = fake_cell(10, base=500.0, grav=400.0, n_bad=3)
    fs = ar.family_scores(recs, FAMS)
    kept = ar.filter_solved(fs)
    assert sorted(kept) == list(range(3, 10))
    assert all(row["baseline"] >= 475 for row in kept.values())


def test_analyze_writes_files_and_labels(tmp_path):
    data = {
        "tile_coding/none/linear": fake_cell(12, 500.0, 420.0, n_bad=2, rng_seed=1),
        "ssp/none/linear": fake_cell(12, 500.0, 300.0, n_bad=2, rng_seed=2),
        "tile_coding/simple/linear": fake_cell(12, 500.0, 410.0, rng_seed=3),
        "ssp/simple/linear": fake_cell(12, 500.0, 310.0, rng_seed=4),
    }
    out = tmp_path / "out"
    summary = ar.analyze(data, str(out), n_boot=300, families=FAMS)
    assert (out / "robustness_tables.md").exists()
    loaded = json.loads((out / "robustness_summary.json").read_text())
    assert set(loaded["cells"]) == set(data)
    rows = loaded["comparisons"]
    prim = [r for r in rows if r["label"] == "primary"]
    assert {r["cell_a"] for r in prim} == {"tile_coding/none/linear"}
    # 2 score rows (baseline, gravity) + 1 retained row (gravity)
    assert len(prim) == 3 and all(r["p_holm"] is not None for r in prim)
    gs = [r for r in prim if r["family"] == "gravity" and r["metric"] == "score"][0]
    assert gs["diff"] > 0 and gs["reject"] is True
    expl = [r for r in rows if r["label"] == "exploratory"]
    assert expl and all(r["cell_a"].startswith("tile_coding/simple") and r["p_holm"] is None for r in expl)
    sens = [r for r in loaded["comparisons_sensitivity"] if r["label"] == "primary" and r["metric"] == "score"
            and r["family"] == "baseline"][0]
    assert sens["n_a"] == 10 and sens["n_b"] == 10  # 2 failed baselines dropped from each cell
    md = (out / "robustness_tables.md").read_text()
    assert "tile_coding/none/linear" in md and "SIGNIFICANT" in md
    assert summary["cells"]["ssp/none/linear"]["n_seeds"] == 12


# ---------------------------------------------------------------------------------------------
# Runner job construction
# ---------------------------------------------------------------------------------------------
def test_build_jobs_keys_and_order(tmp_path):
    study = layout.study_name("ssp/none/deep")
    conds = ["baseline", "gravity_x1.2", "obs_noise_0.05"]
    jobs = rr.build_jobs(study, str(tmp_path), n_seeds=2, seed_offset=3000, conditions=conds, episodes=7,
                         noise_sigma=np.array([0.1, 0.2, 0.3, 0.4]))
    keys = [rr.rob_key(j) for j in jobs]
    assert keys == [("rob", i, c) for i in range(2) for c in conds]  # seed-major
    assert jobs[0]["seed"] == 3000 and jobs[-1]["seed"] == 3001
    assert jobs[0]["n_episodes"] == 7 and jobs[0]["noise_sigma"] == [0.1, 0.2, 0.3, 0.4]
    expected = layout.local_path(layout.policy_key(study, 3001), str(tmp_path))
    assert jobs[-1]["policy_path"] == expected
    json.dumps(jobs)  # picklable / serialisable payloads only


def test_missing_policy_detection(tmp_path):
    study = layout.study_name("tile_coding/value/linear")
    paths = rr.policy_paths(study, str(tmp_path), 3, 3000)
    os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
    open(paths[0], "wb").close()
    with pytest.raises(rr.MissingPolicyError, match="2 of 3 saved policies"):
        rr.ensure_policies(study, str(tmp_path), 3, 3000, no_r2=True)
    for p in paths[1:]:
        open(p, "wb").close()
    assert rr.ensure_policies(study, str(tmp_path), 3, 3000, no_r2=True) == paths


def test_worker_never_raises_on_missing_policy(tmp_path):
    job = {"kind": "rob", "study": "s", "seed_idx": 0, "seed": 3000, "condition": "baseline", "n_episodes": 1,
           "policy_path": str(tmp_path / "nope.pt"), "noise_sigma": None}
    rec = rr.robustness_worker(job)
    assert "error" in rec and rr.rob_key(rec) == ("rob", 0, "baseline")
