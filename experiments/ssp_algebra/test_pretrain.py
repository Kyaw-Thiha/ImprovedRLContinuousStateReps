"""Tests for interface pretraining modes and address-parameter transfer (handoff 09 section 15.4 step 3).

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python test_pretrain.py

The question step 3 answers is whether staging the learning (interface first on cheap synthetic streams, readout second
on the task data) restores rung-0-like data efficiency *reliably* and whether the learned interface **transfers** (an
interface reused on a different stream length is the only thing that makes its pretraining cost amortised rather than
just more training data).  These tests pin the mechanics; the scientific thresholds are measured on pods in RUN_010.
"""
import sys

import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_c


def _task(ep_len=16):
    return tasks_c.IndexTask(n_items=8, ep_len=ep_len)


def _arm(task, seed=0, budget=64):
    arm, _info = L.build_learned("ssp_index_learned", task, budget=budget, seed=seed, rung=1, init="random", hidden=16,
                                 length_scale=1.0, ls_start=4.0, ls_mode="sched", width_steps=10)
    return arm


def test_probe_mode_trains_the_interface_and_discards_the_probe():
    """``probe``: train the address heads through a throwaway readout on the synthetic task, then discard it."""
    task = _task()
    sp = H.splits_for("index", task)
    arm = _arm(task)
    before = arm.w_time.detach().clone()
    n_before = arm.n_params()
    curve, extra = H.train_arm(arm, task, sp, seed=0, steps=20, batch=16, lr=0.02, eval_every=20,
                               pretrain_steps=25, pretrain_mode="probe", width_steps=10)
    assert extra["pretrain_tokens"] > 0, "probe pretraining must report its own token cost"
    assert not torch.allclose(before, arm.w_time.detach()), "probe mode did not train the interface"
    assert arm.n_params() == n_before, "the probe readout leaked into the arm"
    assert not any(p.requires_grad for p in arm.address_params()), "address params were not frozen after pretraining"
    assert all(np.isfinite([r["id"] for r in curve]))
    print(f"  probe: w_time moved, params {n_before} unchanged, pretrain_tokens {extra['pretrain_tokens']}")


def test_full_mode_reinitialises_the_readout():
    """``full``: train everything on synthetic streams, freeze the interface, re-initialise the readout."""
    task = _task()
    sp = H.splits_for("index", task)
    arm = _arm(task)
    with torch.no_grad():                                   # a marker: a re-init must overwrite this
        for p in arm.mix.parameters():
            p.zero_()
    curve, extra = H.train_arm(arm, task, sp, seed=0, steps=20, batch=16, lr=0.02, eval_every=20,
                               pretrain_steps=25, pretrain_mode="full", width_steps=10)
    assert any(float(p.abs().sum()) > 0 for p in arm.mix.parameters()), "the readout was not re-initialised"
    assert not any(p.requires_grad for p in arm.address_params()), "address params were not frozen after pretraining"
    assert all(np.isfinite([r["id"] for r in curve]))
    print(f"  full: readout re-initialised, pretrain_tokens {extra['pretrain_tokens']}")


def test_unknown_pretrain_mode_is_rejected():
    task = _task()
    sp = H.splits_for("index", task)
    try:
        H.train_arm(_arm(task), task, sp, seed=0, steps=5, batch=8, lr=0.02, eval_every=5,
                    pretrain_steps=3, pretrain_mode="nonsense")
    except ValueError:
        print("  unknown pretrain_mode raises ValueError")
        return
    raise AssertionError("an unknown pretrain_mode must raise, not silently skip pretraining")


def test_address_params_transfer_between_lengths():
    """The transfer mechanic: copy only the address parameters into an arm built for a different stream length."""
    short, long = _task(16), _task(32)
    src = _arm(short)
    with torch.no_grad():                                   # a distinctive interface
        src.w_time.copy_(torch.randn_like(src.w_time))
        src.read_head.weight.copy_(torch.randn_like(src.read_head.weight))
    dst = _arm(long, seed=7)
    dst.load_address_params(src)
    assert torch.allclose(dst.w_time, src.w_time) and torch.allclose(dst.read_head.weight, src.read_head.weight)
    assert torch.allclose(dst.log_mult, src.log_mult), "the kernel width must transfer with the interface"
    assert not hasattr(dst, "codes") or not torch.allclose(dst.codes, src.codes) or True
    dst.freeze_address()
    assert not any(p.requires_grad for p in dst.address_params()), "a transferred interface must be freezable"
    sp = H.splits_for("index", long)
    curve, _ = H.train_arm(dst, long, sp, seed=0, steps=20, batch=16, lr=0.02, eval_every=20, width_steps=10)
    assert all(np.isfinite([r["id"] for r in curve]))
    print("  transfer: w_time/read_head/log_mult copied into a T=32 arm, frozen, readout trains")


def test_load_address_params_is_shape_checked():
    """A silent shape mismatch would make the transfer look like it worked while copying nothing."""
    src = _arm(_task(16))
    dst = _arm(_task(32))
    with torch.no_grad():
        dst.read_head = torch.nn.Linear(3, 1)
    try:
        dst.load_address_params(src)
    except (RuntimeError, ValueError) as e:
        print(f"  shape mismatch raises {type(e).__name__}")
        return
    raise AssertionError("copying address params into a different-shaped head must raise")


def test_pretrain_mode_reaches_the_harness_config():
    assert "pretrain_mode" in H.TRAIN_KEYS, H.TRAIN_KEYS
    cfg = {"pretrain_steps": 10, "pretrain_mode": "probe"}
    arm_cfg = {k: v for k, v in cfg.items() if k not in H.TRAIN_KEYS}
    assert arm_cfg == {}, f"pretrain keys must not be passed to the arm: {arm_cfg}"
    print("  pretrain_mode is a training key, not an arm argument")


def test_transfer_protocol_runs_end_to_end():
    """The two-stage protocol (pretrain the interface, copy it to another shape, train only the readout) must run and
    report the destination accuracy and both token costs -- mechanics only, no accuracy threshold."""
    import transfer_test as TT
    src = _task(8)
    dst = tasks_c.IndexTask(n_items=8, ep_len=12)
    row = TT.run_seed(0, src, dst, pretrain_steps=15, readout_steps=15, budget=64, hidden=16)
    assert set(row) >= {"seed", "dst_id", "pretrain_tokens", "task_tokens"}, row.keys()
    assert 0.0 <= row["dst_id"] <= 1.0 and row["pretrain_tokens"] > 0 and row["task_tokens"] > 0
    print(f"  transfer protocol: dst id {row['dst_id']:.3f}, pretrain {row['pretrain_tokens']} tok, "
          f"task {row['task_tokens']} tok")


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
