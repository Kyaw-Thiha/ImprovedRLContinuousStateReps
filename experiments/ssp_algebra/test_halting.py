"""Phase 0.2 (handoff 10): halting must extrapolate to hop counts never trained.

RUN_010: rung 3 scores 0.972 in-distribution (hops 1-2) but 0.361 on hops 4-8, and the guess was that the untrained
``hop_emb`` rows were the cause.  Measuring it separated two things:

* the *iteration* extrapolates perfectly -- ``_hop_scores`` after h iterations is 1.000 accurate at every h = 1..8
  (that is the docs/11 section 1.2 unique #2 claim, and it holds);
* the *stopping* does not: the arm's answer is the halt-weighted ensemble, and its peak stays on a trained hop.

Three halting mechanisms were then built and measured (all in ``SSPPointerLearned``), and this file pins each one:

``learned``     a free embedding row per hop.  Peak on a trained hop; held-out hops ~0.36-0.52.  Red-first evidence.
``counter``     the queried hop *index* decoded from the cue by a linear head, plus a hard gate on the iteration.
                The gate is right on the trained hops (id 1.000) but the decode's *slope* is not identifiable from two
                training hops: any (intercept, slope) in a wide slab is loss-free in distribution, so SGD stops at a
                compressed slope (measured 0.15 with the gate, 0.30 with a soft quadratic) and held-out hops stay
                ~0.45.
``cue_hop``     the same gate, but the hop index is read from the cue's own hop channel with a *fixed* scale
                (``task.hop_feat_index``).  Nothing is learned about the scale, so there is no slope to get wrong:
                held-out hops 1.000 and a real cue-dropout curve (0.62 / 0.34 / 0.21 at 25 / 50 / 75 %).

The pair ``counter`` vs ``cue_hop`` is the finding: for this task the halting *scale* has to come from the
observation, not from gradient descent on two hop values.

Run: OMP_NUM_THREADS=1 ../../.venv/bin/python test_halting.py [all|cue_hop|counter|learned]
"""
import sys

import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_b

STEPS = 1500


def _train_and_probe(halting, steps=STEPS, seed=0):
    """Train on hops (1, 2) and probe: halt peak at the queried hop, answer accuracy, decoded hop, per hop 1..8."""
    task = tasks_b.PointerTask(n_nodes=6, n_query=6, train_hops=(1, 2), max_hops=8)
    arm, _ = L.build_learned("ssp_pointer_learned", task, budget=256, seed=seed, rung=1, hidden=64, cue="features",
                             halting=halting, lam_init=0.99, temp=0.1)
    sp = H.splits_for("pointer", task)
    curve, _ = H.train_arm(arm, task, sp, seed=seed, steps=steps, batch=64, lr=3e-3, eval_every=steps,
                           aux_weight=0.5)
    out = {"id": curve[-1]["id"], "extrap_lag": curve[-1]["extrap_lag"], "peaks": {}, "acc": {}, "hop_hat": {}}
    for h in range(1, 9):
        b = task.batch(64, task.ep_len, rng=np.random.default_rng(100 + h), hops=(h,))
        feat = task.features(b)
        with torch.no_grad():
            arm(b, feat)
        w, q = arm.halt_weights(), arm.query_scores()
        tgt = b["target"][:, 2 * task.n_nodes:2 * task.n_nodes + task.n_query]
        out["peaks"][h] = float((w.argmax(-1) == h - 1).float().mean())
        out["acc"][h] = float((q.argmax(-1) == tgt).float().mean())
        if arm.halting == "counter":
            out["hop_hat"][h] = float(arm.hop_dec(feat[:, -task.n_query:])[..., 0].mean())
    return arm, task, out


def _report(name, out):
    print(f"  {name}: id {out['id']:.3f} extrap_lag {out['extrap_lag']:.3f}")
    print(f"    peak@queried-hop {[round(out['peaks'][h], 2) for h in range(1, 9)]}")
    print(f"    answer_acc        {[round(out['acc'][h], 2) for h in range(1, 9)]}")


def test_cue_hop_halts_at_the_queried_hop_for_untrained_hops():
    """The pass criterion (handoff 10 phase 0.2): held-out hops 4-8 >= 0.9 with no hop count handed over."""
    arm, task, out = _train_and_probe("cue_hop")
    _report("cue_hop", out)
    heldout = np.mean([out["acc"][h] for h in range(4, 9)])
    assert out["id"] >= 0.95, f"in-distribution accuracy collapsed to {out['id']:.3f}"
    assert heldout >= 0.9, f"held-out hops 4-8 reach only {heldout:.3f}"
    assert np.mean([out["peaks"][h] for h in range(4, 9)]) >= 0.9, "the halt distribution is not on the queried hop"


def test_learned_halting_does_not_extrapolate():
    """The RUN_010 failure, pinned: the embedding's peak cannot leave the trained hops."""
    arm, task, out = _train_and_probe("learned")
    _report("learned", out)
    assert out["id"] >= 0.95, f"expected the trained-hops regime to be solved, got {out['id']:.3f}"
    assert np.mean([out["peaks"][h] for h in range(4, 9)]) < 0.2, \
        "the learned embedding now peaks on held-out hops -- update this test and the run log"
    assert np.mean([out["acc"][h] for h in range(4, 9)]) < 0.9, "held-out hops unexpectedly solved by the embedding"


def test_counter_decode_slope_is_not_identifiable():
    """The learned decode runs out of slope: pin the measured compression so a future fix has to notice it."""
    arm, task, out = _train_and_probe("counter")
    _report("counter", out)
    print(f"    hop_hat by hop      {[round(out['hop_hat'][h], 2) for h in range(1, 9)]}")
    slope = (out["hop_hat"][8] - out["hop_hat"][1]) / 7.0
    assert 0.0 < slope < 0.9, \
        f"the decoded-hop slope is now {slope:.2f}; if a fix makes it ~1, update this test and the run log"
    assert np.mean([out["acc"][h] for h in range(4, 9)]) < 0.9, "held-out hops unexpectedly solved by the learned decode"


def test_halting_weights_are_a_distribution():
    for halting in ("counter", "cue_hop"):
        task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
        arm, _ = L.build_learned("ssp_pointer_learned", task, budget=64, seed=0, rung=1, hidden=16, cue="features",
                                 halting=halting)
        b = task.batch(8, task.ep_len, rng=np.random.default_rng(1))
        arm(b, task.features(b))
        w = arm.halt_weights()
        assert w.shape == (8, 4, 4), (halting, w.shape)
        assert torch.allclose(w.sum(-1), torch.ones(8, 4), atol=1e-5), f"{halting}: not a distribution"
        assert (w >= -1e-6).all(), f"{halting}: negative halt weights"


def test_mask_hop_in_start_makes_the_start_decode_hop_invariant():
    """The fix for section 6.0's finding: with the flag on, the hop channel must not reach the start decode.

    Tested end-to-end on the arm's *retrieval scores* rather than on ``start_head`` directly (calling the module would
    bypass the masking, which lives in ``forward``).  ``halting="given"`` so the hop channel's only remaining route is
    the start decode; with the flag on, changing the hop channel alone must therefore change nothing at all.
    """
    task = tasks_b.PointerTask(n_nodes=6, n_query=6, train_hops=(1, 2), max_hops=8)
    for flag, expect_change in ((False, True), (True, False)):
        arm, _ = L.build_learned("ssp_pointer_learned", task, budget=64, seed=0, rung=3, hidden=16,
                                 cue="features", halting="given", mask_hop_in_start=flag)
        b = task.batch(8, task.ep_len, rng=np.random.default_rng(0), hops=(1,))
        b2 = dict(b)
        q = b["query"].clone()
        # only the *query* steps carry the hop channel; the stream steps must be left alone, because the role gate
        # legitimately reads them (changing those would move the memory, not just the start decode)
        q[:, 2 * task.n_nodes:, 1] = 1.0                   # hop channel: 1/8 -> 8/8 on the query steps only
        b2["query"] = q
        with torch.no_grad():
            arm(b, task.features(b))
            s1 = arm.query_scores().clone()
            arm(b2, task.features(b2))
            s2 = arm.query_scores().clone()
        changed = not torch.allclose(s1, s2)
        assert changed == expect_change, \
            f"mask_hop_in_start={flag}: retrieval changed by the hop channel = {changed}, expected {expect_change}"
    print("  mask_hop_in_start: the start decode ignores the hop channel (and without it, must not)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    test_halting_weights_are_a_distribution()
    test_mask_hop_in_start_makes_the_start_decode_hop_invariant()
    print("ok halting weights are a distribution; start-decode masking")
    if which in ("all", "cue_hop"):
        test_cue_hop_halts_at_the_queried_hop_for_untrained_hops()
    if which in ("all", "learned"):
        test_learned_halting_does_not_extrapolate()
    if which in ("all", "counter"):
        test_counter_decode_slope_is_not_identifiable()
    print("ok halting", which)
