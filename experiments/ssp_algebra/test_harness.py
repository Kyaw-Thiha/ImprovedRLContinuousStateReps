"""Harness gate (docs/11 section 4.1-4.3).  Every check here must pass before a sweep is worth running.

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_harness.py

1. ``lru_is_a_delay_line``      -- the time-SSP substitute can actually represent the pure delay this variant tests.
2. ``ssp_arm_learns_delay``     -- the SSP arm's learned delta + cleanup reaches high accuracy end to end.
3. ``oracle_delta_readout``     -- with delta pinned to the true lag the cleanup argmax *is* the answer (arm-level
                                  version of ``selftest.py``; isolates the algebra from the delta learning).
4. ``state_budget_matching``    -- every memory arm's carried state is within 5% of the budget in floats.
5. ``determinism``              -- same seed, same curve.
6. ``all_arms_smoke``           -- every arm runs, shapes and losses are finite.
"""

import math

import numpy as np
import torch

import arms as A
import harness as H
import tasks_c


def _train(arm, task, sp, seed, steps, batch, lr, eval_every=100, device="cpu"):
    return H.train_arm(arm, task, sp, seed, steps, batch, lr, eval_every, device)


def test_lru_is_a_delay_line():
    """Freeze the recurrence, train only the readout, and require the delay to be recovered."""
    task = tasks_c.DelayTask(n_items=8, ep_len=32, train_lags=(4,))
    sp = H.splits_for("delay", task)
    arm, info = A.build_arm("lru", task, budget=1024, seed=0, hidden=64)
    for name, p in arm.named_parameters():
        if not name.startswith("head"):
            p.requires_grad_(False)
    curve, _ = _train(arm, task, sp, seed=0, steps=600, batch=64, lr=3e-3)
    acc = H.summarize(curve)["last5_id"]
    print(f"  lru frozen-recurrence delay acc = {acc:.3f} (head-only training)")
    assert acc > 0.9, f"an LRU cannot represent this delay: {acc:.3f}"
    return acc


def test_ssp_arm_learns_delay():
    task = tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2, 3, 4))
    sp = H.splits_for("delay", task)
    arm, info = A.build_arm("ssp", task, budget=512, seed=0, hidden=64, length_scale=1.0, temp=0.1)
    curve, _ = _train(arm, task, sp, seed=0, steps=1500, batch=64, lr=1e-2)
    s = H.summarize(curve)
    # diagnostic: did the delta head learn to reproduce the requested lag?
    with torch.no_grad():
        b = task.batch(64, task.ep_len, rng=np.random.default_rng(7), lags=(1, 2, 3, 4))
        delta = torch.nn.functional.softplus(arm.delta_head(task.features(b)))[..., 0]
        per_row = (delta * b["mask"]).sum(1) / b["mask"].sum(1).clamp_min(1)   # learned offset per episode
        true = b["lag"].float()
        corr = float(torch.corrcoef(torch.stack([per_row, true]))[0, 1])
    lam = float(torch.sigmoid(arm.lam_raw))
    print(f"  ssp arm: id={s['last5_id']:.3f} extrap_lag={s['last5_extrap_lag']:.3f} {info['knob']} "
          f"lambda={lam:.3f} corr(mean delta, lag)={corr:+.3f}")
    assert s["last5_id"] > 0.9, f"the SSP arm does not learn the delay: {s}"
    return s


def test_oracle_delta_readout():
    """Pin delta = lag via the delta head's bias and check that the cleanup argmax *is* the answer."""
    task = tasks_c.DelayTask(n_items=8, ep_len=32, train_lags=(3,))
    arm, info = A.build_arm("ssp", task, budget=1024, seed=0, hidden=64, temp=0.05)
    with torch.no_grad():
        arm.delta_head.weight.zero_()
        arm.delta_head.bias.fill_(math.log(math.e ** 3 - 1))   # softplus(bias) = 3 = the true lag
    rng = np.random.default_rng(0)
    b = task.batch(32, task.ep_len, rng=rng, lags=(3,))
    with torch.no_grad():
        scores = arm.read(b, task.features(b))
        pred = scores.argmax(-1)
        acc = ((pred == b["target"]) & b["mask"]).sum().item() / max(b["mask"].sum().item(), 1)
    print(f"  ssp pinned-delta cleanup acc = {acc:.3f}")
    assert acc > 0.9, f"pinned-delta cleanup is not the answer: {acc:.3f}"
    return acc


def test_state_budget_matching(tol=0.05):
    task = tasks_c.DelayTask(n_items=8, ep_len=32)
    rows = []
    for budget in (256, 512, 1024):
        for name in A.MEMORY_ARMS:
            arm, info = A.build_arm(name, task, budget=budget, seed=0)
            achieved = arm.state_floats(task.ep_len)
            rel = abs(achieved - budget) / budget
            rows.append((budget, name, achieved, rel))
            assert rel <= tol, f"{name} at budget {budget}: {achieved} floats ({rel:.1%} off)"
    # the unlimited-state upper bound is reported, not matched
    for name in ("mlp", "posmlp"):
        arm, _ = A.build_arm(name, task, budget=1024, seed=0)
        assert arm.state_floats(task.ep_len) == 0
    print("  " + "; ".join(f"b{b}:{n}={a}" for b, n, a, _ in rows[::5]))
    return rows


def test_determinism():
    task = tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(2,))
    sp = H.splits_for("delay", task)
    curves = []
    for _ in range(2):
        arm, _ = A.build_arm("gru", task, budget=64, seed=0, hidden=32)
        c, _ = _train(arm, task, sp, seed=5, steps=60, batch=16, lr=1e-3, eval_every=20)
        curves.append(c)
    same = all(abs(a["id"] - b["id"]) < 1e-12 for a, b in zip(curves[0], curves[1]))
    print(f"  determinism: identical curves = {same}")
    assert same, "same seed produced different curves"


def test_all_arms_smoke():
    task = tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(2, 4))
    sp = H.splits_for("delay", task)
    for name in A.ALL_ARMS + ():
        arm, info = A.build_arm(name, task, budget=256, seed=0, hidden=16)
        c, _ = _train(arm, task, sp, seed=0, steps=20, batch=8, lr=1e-3, eval_every=20)
        assert np.isfinite(c[-1]["id"]), f"{name} produced a non-finite metric"
    # the other variants use the same interface; every arm must at least construct, run and stay finite
    import tasks_a, tasks_b

    other = {"index": tasks_c.IndexTask(n_items=8, ep_len=16),
             "pointer": tasks_b.PointerTask(n_nodes=4, n_query=3, train_hops=(1, 2)),
             "path": tasks_a.LandmarkTask(n_landmarks=4, ep_len=16)}
    for variant, tk in other.items():
        sp2 = H.splits_for(variant, tk)
        for name in A.ALL_ARMS:
            arm, _ = A.build_arm(name, tk, budget=256, seed=0, hidden=16)
            c, _ = _train(arm, tk, sp2, seed=0, steps=15, batch=8, lr=1e-3, eval_every=15)
            assert np.isfinite(c[-1]["id"]), f"{name} on {variant} produced a non-finite metric"
    print(f"  all {len(A.ALL_ARMS)} arms train on delay/index/pointer/path")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checks = [("lru_is_a_delay_line", test_lru_is_a_delay_line),
              ("oracle_delta_readout", test_oracle_delta_readout),
              ("ssp_arm_learns_delay", test_ssp_arm_learns_delay),
              ("state_budget_matching", test_state_budget_matching),
              ("determinism", test_determinism),
              ("all_arms_smoke", test_all_arms_smoke)]
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
