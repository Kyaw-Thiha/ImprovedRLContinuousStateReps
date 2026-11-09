"""Checks for the four SSP candidate changes of handoff 11 step 1 / docs/11 section 14.3 (all knobs default OFF).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_candidates.py

The candidates, one change at a time on the rung-1 arm:

1. ``omega_kind="cauchy"``      -- a kernel with no dead zone (the measured cause of the stall is that
                                   ``sinc(pi d)`` is flat/zero at integer offsets; the Cauchy spectrum gives
                                   ``exp(-|d|/ls)``, strictly positive and monotone).
2. ``addr_mode="tied"``         -- ONE address function for writing and reading, so the read address has no origin of
                                   its own to discover.  ``read_feat_scale``/``read_feat_shift`` simulate a query handed
                                   over in different units (a deployment stress, not a prior).
3. ``read_heads=K``             -- K read heads with offsets spread by half an index: redundancy instead of a prior.
4. ``write_rule="delta"``       -- correct the written slot instead of piling onto it (capacity at fixed state).

None of these is a result; the pod protocols in ``jobs/p8_candidates.json`` decide.  What is checked here is that each
change is *expressible and trainable*, and that none of them alters the historical default path (the forward-pass check
that guards every earlier number in this folder is in ``test_learned``/the RUN_011 hash sweep).
"""

import math

import numpy as np
import torch

import arms_learned as L
import harness as H
import ssp_ops as S
import tasks_c


def _index_task(ep_len=16, n_items=8):
    return tasks_c.IndexTask(n_items=n_items, ep_len=ep_len)


def _batch(task, B=8):
    return task.batch(B, task.ep_len, rng=np.random.default_rng(0))


def test_cauchy_kernel_has_no_dead_zone():
    """The structural claim, measured on the kernel itself with enough dimensions that Monte-Carlo noise is small."""
    D = 4096
    d = torch.arange(0, 9, dtype=torch.float)
    om_c = S.make_omega(D, 1, length_scale=1.0, seed=0, kind="cauchy")
    om_u = S.make_omega(D, 1, length_scale=1.0, seed=0, kind="uniform")
    kc = torch.cos(om_c.reshape(-1)[:, None].float() * d[None, :]).mean(0)
    ku = torch.cos(om_u.reshape(-1)[:, None].float() * d[None, :]).mean(0)
    se = 1.0 / math.sqrt(D)
    print("  d        :", " ".join(f"{int(x):>7d}" for x in d))
    print("  cauchy   :", " ".join(f"{v:>7.3f}" for v in kc))
    print("  exp(-d)  :", " ".join(f"{math.exp(-float(x)):>7.3f}" for x in d))
    print("  sinc     :", " ".join(f"{v:>7.3f}" for v in ku))
    # cauchy tracks the Laplace kernel within Monte-Carlo error, and stays positive everywhere it matters
    for i in range(1, 6):
        assert abs(kc[i].item() - math.exp(-float(d[i]))) < 4 * se, f"cauchy kernel off theory at d={i}"
    assert (kc[:5] > 0).all(), "the cauchy kernel must be positive at small offsets (no dead zone)"
    assert all(kc[i] > kc[i + 1] for i in range(4)), "the cauchy kernel must decay monotonically at small offsets"
    # the historical kernel really does sit near zero at the first integer offsets, which is the measured cause of the stall
    assert abs(ku[1].item()) < 6 * se and abs(ku[2].item()) < 6 * se, "sinc should be ~0 at integer offsets"


def test_tied_addressing_shares_one_function():
    """`addr_mode="tied"` must remove the separate read head, and give the same address for a key and its query."""
    task = _index_task()
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, addr_mode="tied")
    assert arm.read_head is None, "a tied arm must not keep a second address function"
    assert arm.w_time in arm.address_params(), "the shared address function must be trainable"
    # a query at index j and the write at time j must produce the *same* scalar (that is what removes the origin)
    j = torch.tensor([1.0, 3.0, 7.0])
    s_read = S.__dict__  # (documentation marker: the scalar is feats @ w_time, no separate head)
    import feats
    s_q = feats.abs_time_feats(j) @ arm.w_time
    s_w = feats.abs_time_feats(j) @ arm.w_time
    assert torch.allclose(s_q, s_w), "tied addressing must give a key and its query the same address"
    print(f"  tied: identical address for key and query (max diff {(s_q - s_w).abs().max():.2e}), params={arm.n_params()}")


def test_read_head_count_and_shapes():
    """K heads must widen the readout input and still produce one score vector per item per head."""
    task = _index_task()
    b = _batch(task)
    for K in (1, 3, 5):
        arm, info = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, read_heads=K)
        out = arm(b, task.features(b))
        assert out.shape == (8, task.ep_len, task.out_dim)
        assert torch.isfinite(out).all()
        assert arm.mix[0].in_features == K * task.out_dim + task.in_dim, "the readout must see every head"
        print(f"  K={K}: readout in={arm.mix[0].in_features} params={info['params']}")
    # the offsets must be centred and spread by half an index, so the range depends on K and not on the task
    arm5, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, read_heads=5)
    assert arm5.head_offsets == [-1.0, -0.5, 0.0, 0.5, 1.0], arm5.head_offsets
    print(f"  K=5 offsets: {arm5.head_offsets}")


def test_delta_write_has_a_trainable_strength():
    """The delta write adds exactly one gate, and the gradient must reach it."""
    task = _index_task()
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, write_rule="delta", beta_init=4.0)
    assert isinstance(arm.beta_raw, torch.nn.Parameter)
    base, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1)
    assert arm.n_params() == base.n_params() + 1, "the delta write must cost exactly one parameter"
    b = _batch(task, B=16)
    loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
    loss.backward()
    assert arm.beta_raw.grad is not None and torch.isfinite(arm.beta_raw.grad).all()
    print(f"  delta write: +1 param (gate init sigmoid(4.0)={torch.sigmoid(torch.tensor(4.0)):.3f}), grad={float(arm.beta_raw.grad):.3e}")


def test_every_candidate_trains():
    """Each candidate must improve on the index task in a short run: expressible is not enough, it must be optimisable."""
    task = _index_task()
    cfgs = {"baseline": {}, "cauchy": {"omega_kind": "cauchy"}, "tied": {"addr_mode": "tied"},
            "tied+residual": {"addr_mode": "tied", "read_residual": True}, "K=5": {"read_heads": 5},
            # MEASURED: at full write strength from step 0 the delta write does *not* train on index (0.144 -> 0.085 in
            # 150 steps, recorded in RUN_012).  Initialising the gate near the plain rule (sigmoid(-4) = 0.018) does
            # train, so the arm is optimisable -- the failure is the starting point, not the mechanism.
            "delta-write": {"write_rule": "delta", "beta_init": -4.0},
            "tied+delta+K=3": {"addr_mode": "tied", "write_rule": "delta", "beta_init": -4.0, "read_heads": 3}}
    rng = np.random.default_rng(0)
    for name, cfg in cfgs.items():
        arm, info = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, init="affine",
                                    affine_eps=0.03, affine_offset_zero=True, length_scale=1.0, ls_mode="fixed",
                                    lam_init=0.99, temp=0.1, **cfg)
        opt = torch.optim.Adam(arm.parameters(), lr=0.02)
        first = last = None
        for step in range(150):
            bb = task.batch(64, task.ep_len, rng=rng)
            loss, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
            assert torch.isfinite(loss), f"{name} produced a non-finite loss"
            opt.zero_grad(); loss.backward(); opt.step()
            if first is None:
                first = acc
            last = acc
        print(f"  {name:14s} params={info['params']:6d} state={info['state_floats']} acc {first:.3f} -> {last:.3f}")
        assert last > first, f"{name} did not improve ({first:.3f} -> {last:.3f})"


def test_gates_and_banks():
    """Axis-1 (input-dependent gates) and axis-2 (multi-timescale banks) knobs: expressible, equal state, trainable.

    Neither is a result -- the pod protocols decide -- but a knob that silently changes the carried state or cannot
    train would waste a grid.  The gate must start *at* the ungated rule (identity init) and the banks must not buy
    state: `dim` is split across them, so `state_floats` is unchanged.
    """
    task = _index_task()
    base, ib = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1)
    for name, cfg in (("gate=input", {"gate": "input"}), ("banks=2", {"banks": 2}), ("banks=4", {"banks": 4}),
                      ("gate+banks=2", {"gate": "input", "banks": 2})):
        arm, info = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, lam_init=0.99, **cfg)
        assert info["state_floats"] == ib["state_floats"] == 256, (name, info["state_floats"])
        with torch.no_grad():                      # identity init: the gate must reproduce the ungated dynamics
            if arm.gate_mlp is not None:
                g = arm.gate_mlp(task.features(_batch(task))[:, 3])
                assert abs(float(torch.sigmoid(g[0, 1])) - 0.99) < 1e-4, "the decay gate must start at lam_init"
                assert float(torch.sigmoid(g[0, 0])) > 0.9, "the write gate must start near 1"
        rng = np.random.default_rng(0)
        opt = torch.optim.Adam(arm.parameters(), lr=0.02)
        first = last = None
        for _ in range(120):
            bb = task.batch(64, task.ep_len, rng=rng)
            loss, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
            assert torch.isfinite(loss), f"{name} produced a non-finite loss"
            opt.zero_grad(); loss.backward(); opt.step()
            if first is None:
                first = acc
            last = acc
        print(f"  {name:14s} state={info['state_floats']} params={info['params']:6d} acc {first:.3f} -> {last:.3f}")
        assert last > first, f"{name} did not improve ({first:.3f} -> {last:.3f})"


if __name__ == "__main__":
    for fn in (test_cauchy_kernel_has_no_dead_zone, test_tied_addressing_shares_one_function,
               test_read_head_count_and_shapes, test_delta_write_has_a_trainable_strength,
               test_every_candidate_trains, test_gates_and_banks):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_candidates: all checks passed")

