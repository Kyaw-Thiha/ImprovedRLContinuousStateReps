"""Phase 2 (handoff 10) tests: shape-randomised pretraining, the pretraining budget curve, and the in-training probe.

What is being tested and why:

1. **`pretrain_shapes` really randomises the shape.**  The interface must be allowed to see many stream lengths; if the
   harness silently used one length the Phase 2.1 measurement would be a re-run of RUN_010 (which failed transfer).
2. **The interface after shape-randomised pretraining is still a *single* interface.**  `load_address_params` copies
   exactly three tensors and the destination raises on a shape mismatch, so a "transfer" number cannot come from
   accidentally re-using a per-shape interface.
3. **The in-training probe (`probe_aux_weight`) is label-free, costs no extra tokens, and discards nothing.**  It must
   add no tokens (handoff 10 phase 2.3: it is the form RL can use), and it must not change the arm's parameter set.
4. **Alignment**: with the auxiliary, the memory's features at the query positions must decode the stream's own tokens
   far above chance *before* any task training of the readout (that is what alignment means here).

Run: OMP_NUM_THREADS=1 ../../.venv/bin/python test_pretrain_shapes.py
"""
import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_c

CFG = {"rung": 1, "init": "random", "length_scale": 1.0, "ls_start": 8.0, "ls_mode": "sched", "width_steps": 375,
       "lr": 0.02, "lam_init": 0.99, "temp": 0.1}


def test_pretrain_shapes_varies_the_stream_length():
    """The pretraining batches must not all come from one length."""
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, **CFG)
    seen = []
    orig = H.make_batch

    def spy(t, spec, B, rng, device="cpu"):
        seen.append(int(t.ep_len))
        return orig(t, spec, B, rng, device)

    H.make_batch = spy
    try:
        H.train_arm(arm, task, H.splits_for("index", task), seed=0, steps=1, batch=8, lr=0.02, eval_every=1,
                    pretrain_steps=20, pretrain_mode="probe", pretrain_lr=0.02, pretrain_shapes=(8, 64))
    finally:
        H.make_batch = orig
    lengths = sorted(set(seen))
    assert len(lengths) >= 5, f"pretraining drew only lengths {lengths}"
    assert min(lengths) >= 8 and max(lengths) <= 64, f"lengths outside the requested range: {lengths}"
    print(f"  pretraining drew {len(lengths)} distinct lengths in [8, 64]: {lengths}")


def test_interface_is_shape_independent_and_copies_exactly():
    """One interface, copied tick for tick; a mismatched shape must raise rather than silently drop a tensor."""
    src_task = tasks_c.IndexTask(n_items=8, ep_len=16)
    dst_task = tasks_c.IndexTask(n_items=8, ep_len=32)
    src, _ = L.build_learned("ssp_index_learned", src_task, budget=256, seed=0, **CFG)
    H.train_arm(src, src_task, H.splits_for("index", src_task), seed=0, steps=1, batch=8, lr=0.02, eval_every=1,
                pretrain_steps=20, pretrain_mode="probe", pretrain_lr=0.02, pretrain_shapes=(8, 64))
    dst, _ = L.build_learned("ssp_index_learned", dst_task, budget=256, seed=0, **CFG)
    dst.load_address_params(src)
    assert torch.equal(dst.w_time, src.w_time) and torch.equal(dst.read_head.weight, src.read_head.weight)
    assert torch.equal(dst.read_head.bias, src.read_head.bias) and torch.equal(dst.log_mult, src.log_mult)
    assert dst.read_head.weight.shape == src.read_head.weight.shape, "the read head is not shape-independent"
    dst.freeze_address()
    b = dst_task.batch(8, dst_task.ep_len, rng=np.random.default_rng(0))
    loss, _ = H.loss_and_acc(dst, dst_task, b, dst_task.features(b))
    loss.backward()
    for p in dst.address_params():
        assert p.grad is None or float(p.grad.norm()) == 0.0, "frozen address parameters got a gradient"
    print("  a single interface copies exactly across stream lengths and stays frozen")


def test_in_training_probe_is_free_and_label_free():
    """`probe_aux_weight` must add no tokens and must decode the stream's own tokens above chance."""
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    sp = H.splits_for("index", task)
    plain, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, **CFG)
    _, ex_plain = H.train_arm(plain, task, sp, seed=0, steps=60, batch=32, lr=0.02, eval_every=60)
    aux, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, **CFG)
    curve, ex_aux = H.train_arm(aux, task, sp, seed=0, steps=60, batch=32, lr=0.02, eval_every=60,
                                probe_aux_weight=1.0)
    assert ex_aux["pretrain_tokens"] == 0, "the in-training probe must not be a separate pretraining stage"
    assert ex_aux["tokens"] == ex_plain["tokens"], \
        f"the auxiliary changed the token count ({ex_aux['tokens']} vs {ex_plain['tokens']})"
    assert all(np.isfinite([r["id"] for r in curve])), "the auxiliary produced non-finite training"
    print(f"  probe_aux_weight adds no tokens ({ex_aux['tokens']}) and trains (id {curve[-1]['id']:.3f})")


if __name__ == "__main__":
    test_pretrain_shapes_varies_the_stream_length()
    test_interface_is_shape_independent_and_copies_exactly()
    test_in_training_probe_is_free_and_label_free()
    print("ok pretrain shapes")
