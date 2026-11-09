"""Checks for the equal-aid knobs on the fast-weight baseline: `gate="input"` and `banks=K` (RUN_012 step 1).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_fwp_gates.py

Why: the SSP candidates that won RUN_012 (input-dependent gates, multi-timescale banks) were compared against
`fwp_delta`, which had neither. That is not a matched comparison, so the baseline gets the same two knobs here. Both
default OFF, and `banks=1` must reproduce the plain arm *bit-exactly* (the default path is guarded separately by
`test_default_path_hash.py`; the first check below re-asserts it locally so a failure is attributable).

What is checked:
  * `banks=1, gate="none"` == the plain arm, bit-for-bit;
  * the gate's identity init: with its write bias raised, a gated arm reproduces the ungated memory to 1e-4, so the
    gate starts *at* the ungated rule rather than beside it;
  * the default gate init is conservative but not off (sigmoid(4) in (0.95, 1)) and its decay equals `lam_init`;
  * `banks=K` splits the carried state (`state_floats == K * d^2`, within 10% of the budget) rather than growing it;
  * both knobs train on index (120 steps) and the gate's gradient reaches its output layer;
  * the two documented errors raise.
"""

import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_c


def _task(ep_len=16, n_items=8):
    return tasks_c.IndexTask(n_items=n_items, ep_len=ep_len)


def _arm(task, **cfg):
    return L.build_learned("fwp_delta", task, budget=256, seed=0, hidden=16, **cfg)


def test_banks_one_matches_the_plain_arm():
    task = _task()
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    feat = task.features(b)
    plain, _ = L.build_learned("fwp_delta", task, budget=256, seed=0, hidden=16)
    one, info_one = _arm(task, banks=1)
    with torch.no_grad():
        assert torch.equal(plain(b, feat), one(b, feat)), "banks=1 changed the default path"
    assert info_one["state_floats"] == plain.state_floats(task.ep_len)
    print(f"  banks=1 bit-identical to the plain arm; state {info_one['state_floats']}")


def test_gate_identity_init():
    task = _task()
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    feat = task.features(b)
    plain, _ = _arm(task)
    gated, _ = _arm(task, gate="input")
    assert gated.gate_mlp is not None
    with torch.no_grad():                      # raze the gate to exactly the ungated rule
        g = gated.gate_mlp(feat[:, 3])
        assert abs(float(torch.sigmoid(g[0, 1])) - 0.99) < 1e-4, "the decay gate must start at lam_init"
        assert 0.95 < float(torch.sigmoid(g[0, 0])) < 1.0, "the write gate must start near, but below, 1"
        gated.gate_mlp[2].bias[0] = 20.0       # sigmoid(20) == 1 to float32, so this is the identity case
        d_plain = plain._memory_pass(b, feat)
        d_gated = gated._memory_pass(b, feat)
    diff = float((d_plain - d_gated).abs().max())
    print(f"  gated (write bias raised to 20) vs ungated: max |diff| = {diff:.2e}")
    assert diff < 1e-4, "an identity-initialised gate must reproduce the ungated memory"


def test_banks_state_is_split_not_grown():
    task = _task()
    for K in (2, 4):
        arm, info = _arm(task, banks=K)
        expect = K * arm.d * arm.d
        assert info["state_floats"] == expect, (K, info["state_floats"], expect)
        assert abs(expect - 256) <= 0.10 * 256, f"banks={K} state {expect} is far from the 256-float budget"
        print(f"  banks={K}: d={arm.d} state={expect} (budget 256, {100 * abs(expect - 256) / 256:.1f}% off)")


def test_both_knobs_train_on_index():
    task = _task()
    for name, cfg in (("gate=input", {"gate": "input"}), ("banks=2", {"banks": 2}),
                      ("gate+banks=2", {"gate": "input", "banks": 2})):
        arm, info = _arm(task, **cfg)
        opt = torch.optim.Adam(arm.parameters(), lr=2e-2)
        rng = np.random.default_rng(0)
        first = last = None
        for _ in range(120):
            bb = task.batch(64, task.ep_len, rng=rng)
            loss, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
            assert torch.isfinite(loss), f"{name} produced a non-finite loss"
            opt.zero_grad(); loss.backward()
            if arm.gate_mlp is not None and first is None:
                assert arm.gate_mlp[2].weight.grad is not None
                assert torch.isfinite(arm.gate_mlp[2].weight.grad).all()
            opt.step()
            if first is None:
                first = acc
            last = acc
        print(f"  {name:14s} state={info['state_floats']:4d} params={info['params']:6d} acc {first:.3f} -> {last:.3f}")
        assert last > first, f"{name} did not improve ({first:.3f} -> {last:.3f})"
    # the gate's own gradient was checked above; state a separate check for clarity
    arm, _ = _arm(task, gate="input")
    bb = task.batch(16, task.ep_len, rng=np.random.default_rng(1))
    loss, _ = H.loss_and_acc(arm, task, bb, task.features(bb))
    loss.backward()
    assert arm.gate_mlp[2].weight.grad is not None and float(arm.gate_mlp[2].weight.grad.abs().sum()) > 0
    print(f"  gate gradient reaches the output layer ({float(arm.gate_mlp[2].weight.grad.abs().sum()):.3e})")


def test_documented_errors():
    task = _task()
    import tasks_b
    try:
        _arm(task, gate="bogus"); raise AssertionError("a bad gate name must raise")
    except ValueError:
        print("  gate='bogus' -> ValueError")
    ptr = tasks_b.PointerTask(n_nodes=4, n_query=2, train_hops=(1,), max_hops=2)
    for cfg in ({"gate": "input"}, {"banks": 2}):
        try:
            L.build_learned("fwp", ptr, budget=256, seed=0, hidden=16, **cfg)
            raise AssertionError(f"pointer + {cfg} must raise")
        except NotImplementedError:
            print(f"  pointer + {list(cfg)[0]} -> NotImplementedError")


if __name__ == "__main__":
    for fn in (test_banks_one_matches_the_plain_arm, test_gate_identity_init, test_banks_state_is_split_not_grown,
               test_both_knobs_train_on_index, test_documented_errors):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_fwp_gates: all checks passed")
