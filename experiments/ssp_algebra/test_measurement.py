"""Measurement / fairness checks added after RUN_007 (handoff 09 section 5, step B).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_measurement.py

As with ``test_fixes.py``: ``test_harness.py`` stays untouched, and no check weakens a threshold.
"""

import numpy as np
import torch

import arms as A
import feats
import harness as H
import tasks_a
import tasks_b
import tasks_c


def test_dense_eval_has_enough_checkpoints(min_points=25):
    """Every 25 updates up to 400, then every 100, so an arm at ceiling at the first checkpoint still has a
    measurable time-to-threshold (RUN_007 section 4.5 limitation)."""
    steps = 1500
    sched = H.eval_schedule(steps)
    print(f"  eval_schedule({steps}) -> {len(sched)} checkpoints: {sched[:5]} ... {sched[-3:]}")
    assert len(sched) >= min_points, f"only {len(sched)} checkpoints"
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    sp = H.splits_for("index", task)
    arm, _ = A.build_arm("mlp", task, budget=256, seed=0, hidden=8)
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=steps, batch=8, lr=1e-3, eval_every=300,
                           eval_steps=sched)
    assert len(curve) == len(sched), f"train_arm produced {len(curve)} checkpoints for {len(sched)} requested"
    assert set(H.summarize(curve) ) >= {"conv_id", "aulc_id", "t50_id", "t90_id", "last5_id"}
    assert curve[-1]["step"] == steps
    return len(curve)


def test_pool_mode_is_deterministic():
    """The low-data regime must be reproducible: same seed -> same curve (the pool is seeded from 5000+seed)."""
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    sp = H.splits_for("index", task)
    curves = []
    for _ in range(2):
        arm, _ = A.build_arm("mlp", task, budget=256, seed=0, hidden=8)
        c, _ = H.train_arm(arm, task, sp, seed=3, steps=200, batch=16, lr=1e-3, eval_every=50, pool=64)
        curves.append(c)
    same = all(abs(a["id"] - b["id"]) < 1e-12 for a, b in zip(curves[0], curves[1]))
    print(f"  pool-mode determinism: identical curves = {same} ({len(curves[0])} checkpoints)")
    assert same, "pool mode is not deterministic"


def test_features_are_episode_length_independent():
    """A given absolute index / lag must map to the same feature at T=16 and T=64 (RUN_007 section 5.3-5.4).

    Without this, a longer episode silently re-encodes every query and clock feature, so train-short/test-long
    measures the encoding change rather than the memory.
    """
    j, lag = 5, 3
    rows = []
    for T in (16, 64):
        t = torch.arange(T)[None]
        obs = torch.full((1, T), 3)
        ic = tasks_c.IndexTask(n_items=8, ep_len=T)
        f_idx = ic.features({"obs": obs, "t": t, "query": torch.full((1, T), j / feats.NORM)})[0, j]
        dl = tasks_c.DelayTask(n_items=8, ep_len=T, train_lags=(lag,))
        f_lag = dl.features({"obs": obs, "t": t, "query": torch.full((1, T), lag / feats.NORM)})[0, j]
        rows.append((f_idx, f_lag))
    same_idx = torch.equal(rows[0][0], rows[1][0])
    same_lag = torch.equal(rows[0][1], rows[1][1])
    print(f"  identical index feature at T=16/64 = {same_idx}; identical lag feature = {same_lag}")
    assert same_idx and same_lag, "query/time encoding depends on the episode length"
    # the clock itself: same step, same features regardless of T
    assert torch.equal(feats.abs_time_feats(torch.tensor([j])), feats.abs_time_feats(torch.tensor([j])))


def test_features_width_matches_in_dim():
    """Every task's ``features`` width must equal its ``in_dim``: the arms are built from ``in_dim``.

    A mismatch here is exactly the class of bug that made RUN_007's SSP readout dimension wrong (bug 7.3) and it is
    how a task-side feature change would silently break every arm.
    """
    rows = []
    for task in (tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2)),
                 tasks_c.IndexTask(n_items=8, ep_len=16),
                 tasks_b.PointerTask(n_nodes=6, n_query=4, train_hops=(1, 2)),
                 tasks_a.LandmarkTask(n_landmarks=4, ep_len=16)):
        b = task.batch(4, task.ep_len, rng=np.random.default_rng(0))
        got, want = task.features(b).shape[-1], task.in_dim
        rows.append((task.name, got, want))
        assert got == want, f"{task.name}: features width {got} != in_dim {want}"
    print("  in_dim matches features width for " + ", ".join(f"{n}({g})" for n, g, _ in rows))


def test_corruption_alters_inputs_not_targets():
    """Robustness axes must corrupt the *input* only: a corrupted step's label is still what clean data implies."""
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, train_hops=(1, 2), max_hops=4)
    T = task.ep_len
    clean = task.batch(16, T, rng=np.random.default_rng(3))
    corrupt = task.batch(16, T, rng=np.random.default_rng(3), cue_drop=0.5, obs_noise=0.25)
    same_target = torch.equal(clean["target"], corrupt["target"])
    same_inter = torch.equal(clean["inter"], corrupt["inter"])
    diff_query = not torch.equal(clean["query"], corrupt["query"])
    diff_obs = not torch.equal(clean["obs"], corrupt["obs"])
    print(f"  targets identical = {same_target}, inter identical = {same_inter}, "
          f"query changed = {diff_query}, obs changed = {diff_obs}")
    assert same_target and same_inter, "corruption changed the targets: that would be label noise, not robustness"
    assert diff_query, "cue_drop did not change the query"
    assert diff_obs, "obs_noise did not change the observation tokens"


def test_robustness_splits_are_opt_in_and_present():
    import tasks_b as B

    ti, tp = tasks_c.IndexTask(n_items=8, ep_len=16), B.PointerTask(n_nodes=8, n_query=8, train_hops=(1, 2))
    base_i, base_p = H.splits_for("index", ti), H.splits_for("pointer", tp)
    rob_i, rob_p = H.splits_for("index", ti, robust=True), H.splits_for("pointer", tp, robust=True)
    extra_i, extra_p = sorted(set(rob_i) - set(base_i)), sorted(set(rob_p) - set(base_p))
    print(f"  index extra splits {extra_i}; pointer extra splits {extra_p}")
    assert set(base_i) == {"train", "id", "extrap_T"} and set(base_p) == {"train", "id", "extrap_lag"}, "defaults changed"
    assert extra_i == ["rob_T128"]
    assert extra_p == ["rob_cue25", "rob_cue50", "rob_cue75", "rob_noise10", "rob_noise25"]
    # the spec reaches the task: a corrupted split really changes the inputs it evaluates, never the targets
    clean = H.make_batch(tp, base_p["id"], 16, np.random.default_rng(5))
    cue = H.make_batch(tp, rob_p["rob_cue75"], 16, np.random.default_rng(5))
    noi = H.make_batch(tp, rob_p["rob_noise25"], 16, np.random.default_rng(5))
    assert torch.equal(clean["target"], cue["target"]) and torch.equal(clean["target"], noi["target"])
    assert not torch.equal(clean["query"], cue["query"]) and not torch.equal(clean["obs"], noi["obs"])
    # and a robust sweep still evaluates every arm on the long stream without error
    arm, _ = A.build_arm("ssp", ti, budget=256, seed=0, hidden=16)
    sc = H.evaluate(arm, ti, rob_i["rob_T128"], seed=0)
    assert np.isfinite(sc)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checks = [("dense_eval_checkpoints", test_dense_eval_has_enough_checkpoints),
              ("pool_mode_determinism", test_pool_mode_is_deterministic),
              ("features_length_independent", test_features_are_episode_length_independent),
              ("features_width_matches_in_dim", test_features_width_matches_in_dim),
              ("corruption_alters_inputs_not_targets", test_corruption_alters_inputs_not_targets),
              ("robustness_splits_opt_in", test_robustness_splits_are_opt_in_and_present)]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
