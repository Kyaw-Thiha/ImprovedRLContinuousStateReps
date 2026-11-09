"""Tests for the pointer-family controls (handoff 09 section 15.4 step 4).

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python test_pointer.py

Two things must hold before any pointer claim is restated:

  * **equal aid** -- the baseline arms are given a chain of supervised prefixes rather than the SSP arm's per-hop
    auxiliary loss, and the *metric* must still be the answer (otherwise an arm gets credit for the easy first hops);
  * **a non-SSP iterated control** -- a weight-tied looped transformer that applies a learned update once per hop. If
    it matches the SSP arm, the finding is "iteration plus supervision", not the algebra.
"""
import sys

import numpy as np
import torch

import arms as A
import harness as H
import tasks_b


def _chain_task(n_nodes=6, n_query=4, max_hops=4):
    return tasks_b.PointerTask(n_nodes=n_nodes, n_query=n_query, train_hops=(1, 2), max_hops=max_hops, chain=True)


def test_chain_expansion_labels_every_prefix():
    """Sub-step k of query q must be labelled f^k(s), and only the last sub-step counts as the answer."""
    n, nq, Hh = 6, 4, 4
    task = tasks_b.PointerTask(n_nodes=n, n_query=nq, train_hops=(1, 2), max_hops=Hh, chain=True)
    assert task.ep_len == 2 * n + nq * Hh and task.query_dim == 4
    b = task.batch(5, task.ep_len, rng=np.random.default_rng(0), hops=(1, 2, 3, 4))
    perms, start, hops = b["perms"].numpy(), b["start"].numpy(), b["hops"].numpy()
    power = {0: np.arange(n)[None, :].repeat(perms.shape[0], 0)}
    for k in range(1, Hh + 1):
        power[k] = np.take_along_axis(perms, power[k - 1], axis=1)
    tg, m, ans = b["target"].numpy(), b["mask"].numpy(), b["ans_mask"].numpy()
    checked = 0
    for i in range(5):
        for q in range(nq):
            base = 2 * n + q * Hh
            for k in range(1, Hh + 1):
                col = base + k - 1
                if k <= hops[i, q]:
                    assert m[i, col], f"sub-step (q={q}, k={k}) is not supervised"
                    assert tg[i, col] == power[k][i, start[i, q]], f"label at (q={q}, k={k}) is not f^{k}(s)"
                    checked += 1
                if k == hops[i, q]:
                    assert ans[i, col], f"the answer mask misses the last sub-step (q={q}, k={k})"
                else:
                    assert not ans[i, col], f"the answer mask includes a prefix step (q={q}, k={k})"
    assert ans.sum() == 5 * nq, ans.sum()
    assert m.sum() > ans.sum(), "the loss mask must cover the prefixes, not only the answers"
    assert tg[b["query"].numpy()[:, :, 3] == 0].min() == -100, "padding must carry no supervision"
    print(f"  chain: {checked} labelled sub-steps, ans_mask {int(ans.sum())} == queries {5 * nq}")


def test_standard_task_is_unaffected_and_has_an_answer_mask():
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    b = task.batch(3, task.ep_len, rng=np.random.default_rng(1))
    assert task.query_dim == 3 and task.ep_len == 2 * 6 + 4
    assert torch.equal(b["ans_mask"], b["mask"]), "without a chain the answer mask is the loss mask"
    print("  standard pointer batch unchanged (ans_mask == mask)")


def test_accuracy_is_measured_on_the_answer_not_the_prefixes():
    """An arm that solves every *prefix* but answers only half the queries must score 0.5, not the prefix average.

    With the chain protocol the loss sees every sub-step; if accuracy were taken over the same mask, this arm would
    score about ``1 - 0.5/avg_hops`` (~0.87) and the equal-aid comparison would be wrong.
    """

    class PrefixOnly(torch.nn.Module):
        def __init__(self, batch, n_nodes, max_hops):
            super().__init__()
            self.t = torch.nn.Parameter(torch.zeros(1))
            self.out_dim = n_nodes
            self.b = batch
            self.n, self.H = n_nodes, max_hops

        def __call__(self, batch, feat):
            n = self.out_dim
            out = torch.full((*batch["obs"].shape, n), -5.0)
            for i in range(batch["obs"].shape[0]):
                for q in range(self.b["hops"].shape[1]):
                    h = int(self.b["hops"][i, q])
                    for k in range(h):
                        col = 2 * self.n + q * self.H + k
                        if k < h - 1:                                    # every prefix: correct
                            out[i, col, self.b["inter"][i, q, k]] = 5.0
                        else:                                            # the answer: right for even queries only
                            tgt = int(self.b["inter"][i, q, h - 1])
                            out[i, col, tgt if q % 2 == 0 else (tgt + 1) % n] = 5.0
            return out

    task = _chain_task()
    b = task.batch(4, task.ep_len, rng=np.random.default_rng(2), hops=(1, 2, 3, 4))
    arm = PrefixOnly(b, 6, 4)
    _, acc = H.loss_and_acc(arm, task, b, task.features(b))
    prefix_avg = 1.0 - 0.5 / float(b["hops"][b["hops"] >= 1].float().mean())
    assert abs(acc - 0.5) < 1e-6, f"answer accuracy is {acc:.3f}, expected 0.5"
    assert abs(acc - prefix_avg) > 0.1, "the metric does not discriminate answer accuracy from prefix accuracy"
    print(f"  prefix-solver scores {acc:.3f} on the answer (prefix average would be {prefix_avg:.3f})")


def test_looped_tf_is_registered_and_trainable_on_pointer():
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    arm, info = A.build_arm("looped_tf", task, budget=256, seed=0, tf_d_model=16, tf_layers=1)
    assert "loops=4" in info["knob"], info["knob"]
    assert arm.state_floats(task.ep_len) > 0
    sp = H.splits_for("pointer", task)
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=40, batch=16, lr=1e-2, eval_every=20)
    assert all(np.isfinite([r["id"] for r in curve]))
    print(f"  looped_tf trains on pointer: id {H.summarize(curve)['last5_id']:.3f}, knob {info['knob']}")


def test_looped_tf_reads_out_the_requested_iteration():
    """The output at a query must be the iteration that the query's hop count asks for."""
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    arm, _ = A.build_arm("looped_tf", task, budget=256, seed=0, tf_d_model=16, tf_layers=1)
    b = task.batch(3, task.ep_len, rng=np.random.default_rng(3), hops=(1, 2, 3, 4))
    out = arm(b, task.features(b))
    hop = arm._hop_logits
    assert hop.shape == (3, 4, task.ep_len, 6), hop.shape
    n, nq = 6, 4
    for i in range(3):
        for q in range(nq):
            t = 2 * n + q
            k = int(b["hops"][i, q]) - 1
            assert torch.allclose(out[i, t], hop[i, k, t]), f"query (b={i}, q={q}) did not read iteration {k + 1}"
    print("  query output == the iteration matching its hop count")


def test_looped_tf_aux_loss_gives_per_hop_supervision():
    """The per-hop labels must reach the shared (weight-tied) step block, and be inert without `inter`."""
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    arm, _ = A.build_arm("looped_tf", task, budget=256, seed=0, tf_d_model=16, tf_layers=1)
    b = task.batch(4, task.ep_len, rng=np.random.default_rng(4), hops=(1, 2, 3, 4))
    arm(b, task.features(b))
    loss = arm.aux_loss(b)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    g = float(sum(p.grad.abs().sum() for p in arm.step.parameters() if p.grad is not None))
    assert g > 0, "the per-hop labels do not reach the weight-tied update"
    b2 = {k: v for k, v in b.items() if k != "inter"}
    assert arm.aux_loss(b2) is None, "aux loss must be inert when the task has no per-hop labels"
    print(f"  per-hop CE {float(loss):.3f}, |grad step block|={g:.3e}")


def test_looped_tf_is_deterministic():
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    b = task.batch(2, task.ep_len, rng=np.random.default_rng(5))
    feats = task.features(b)
    outs = []
    for _ in range(2):
        arm, _ = A.build_arm("looped_tf", task, budget=256, seed=11, tf_d_model=16, tf_layers=1)
        arm.eval()
        with torch.no_grad():
            outs.append(arm(b, feats))
    assert torch.equal(outs[0], outs[1]), "looped_tf is not seed-deterministic"
    print("  looped_tf is deterministic across constructions")


def test_chain_is_wired_through_the_sweep_runner():
    """The equal-aid protocol must be reachable from the CLI, not only from a hand-built task."""
    import argparse
    import run_sweep as R
    args = argparse.Namespace(variants=["pointer"], arms=["gru"], budget=256, steps=10, batch=8, eval_every=5,
                              task_kw={"ep_len": 30, "n_items": 8, "n_landmarks": 6, "chain": True}, tag="t",
                              arm_cfg={}, label=None, eval_steps=None, pool=0, robust=False, search_seeds=1)
    jobs = R.build_search_jobs(args, n_configs=1)
    task = R.make_task(jobs[0]["variant"], **jobs[0]["task_kw"])
    assert task.chain and task.query_dim == 4, (task.chain, task.query_dim)
    print("  --chain reaches the task through run_sweep's job kwargs")


def test_splits_for_uses_the_chain_task_length():
    task = _chain_task()
    sp = H.splits_for("pointer", task)
    assert sp["train"]["T"] == task.ep_len, (sp["train"]["T"], task.ep_len)
    arm, _ = A.build_arm("gru", task, budget=64, seed=0, hidden=16)
    acc = H.evaluate(arm, task, sp["id"], seed=0, B=8)
    assert 0.0 <= acc <= 1.0, acc
    print(f"  chain task length {task.ep_len} works with the split machinery (untrained id acc {acc:.3f})")


def test_cue_fix_makes_the_learned_pointer_read_the_query_features():
    """Rung 3 (handoff 09 step 4): the *retrieval* must come from the corruptible query features.

    Today the query code comes from ``batch["start"]``, so zeroing the cue features cannot change what the memory
    retrieves (RUN_008 section 7 item 2) -- even though the corrupted features still reach the readout.  With
    ``cue="features"`` the retrieved scores themselves must change; the old behaviour stays available and stays
    immune at the retrieval level.
    """
    import arms_learned as L
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    b_nocue = dict(b)
    b_nocue["query"] = torch.zeros_like(b["query"])
    feats, feats_nocue = task.features(b), task.features(b_nocue)
    old, _ = L.build_learned("ssp_pointer_learned", task, budget=64, seed=0, rung=1, hidden=16)
    new, _ = L.build_learned("ssp_pointer_learned", task, budget=64, seed=0, rung=1, hidden=16, cue="features")
    for arm in (old, new):
        arm.eval()
    with torch.no_grad():
        old(b, feats)
        old_nocue = old.query_scores()
        old(b_nocue, feats_nocue)
        assert torch.allclose(old_nocue, old.query_scores()), "the batch-cue arm's retrieval must be immune by design"
        new(b, feats)
        new_cue = new.query_scores()
        new(b_nocue, feats_nocue)
        assert not torch.allclose(new_cue, new.query_scores()), \
            "with cue='features' the retrieval ignores the query features, so cue corruption still cannot reach it"
    print("  cue='features' changes the retrieval; the batch-cue arm's retrieval does not")


def test_learned_halting_produces_a_distribution_and_trains():
    """Rung 3 also removes the given hop count: the arm unrolls max_hops and weights the hops by a learned halt."""
    import arms_learned as L
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    arm, _ = L.build_learned("ssp_pointer_learned", task, budget=64, seed=0, rung=1, hidden=16, cue="features",
                             halting="learned")
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(1))
    arm(b, task.features(b))
    w = arm.halt_weights()
    assert w is not None and w.shape == (8, 4, 4), w.shape
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5), "halting weights are not a distribution"
    sp = H.splits_for("pointer", task)
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=30, batch=16, lr=1e-2, eval_every=15)
    assert all(np.isfinite([r["id"] for r in curve]))
    print(f"  learned halting weights sum to 1; trains (id {H.summarize(curve)['last5_id']:.3f})")


def test_rung3_is_reachable_from_the_sweep_config():
    """``rung=3`` must select the hint-free behaviour through the ordinary --arm-cfg path."""
    import arms_learned as L
    import harness as Hh
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    cfg = {"rung": 3, "init": "random", "hidden": 16}
    arm, _info = Hh.build_any("ssp_pointer_learned", task, budget=64, seed=0, cfg=cfg)
    assert arm.rung == 3 and arm.cue == "features" and arm.halting == "learned", (arm.rung, arm.cue, arm.halting)
    print("  rung=3 switches on the feature cue and the learned halting through build_any")


def test_iterated_fwp_reads_once_per_hop():
    """The non-SSP control must actually iterate: 1 hop and 3 hops cannot give the same retrieval."""
    import arms_learned as L
    task = tasks_b.PointerTask(n_nodes=6, n_query=4, max_hops=4)
    arm, info = L.build_learned("fwp", task, budget=256, seed=0, hidden=16)
    assert "FWPArm" in info["knob"]
    b1 = task.batch(6, task.ep_len, rng=np.random.default_rng(0), hops=(1,))
    arm.eval()
    with torch.no_grad():
        arm(b1, task.features(b1))
        one = arm.query_scores().clone()
    b3 = task.batch(6, task.ep_len, rng=np.random.default_rng(0), hops=(3,))
    with torch.no_grad():
        arm(b3, task.features(b3))
        three = arm.query_scores()
    assert one.shape == (6, 4, 6), one.shape
    assert not torch.allclose(one, three), "the iterated fwp read does not depend on the hop count"
    sp = H.splits_for("pointer", task)
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=30, batch=16, lr=1e-2, eval_every=15)
    assert all(np.isfinite([r["id"] for r in curve]))
    print(f"  iterated fwp: 1-hop and 3-hop retrievals differ; trains (id {H.summarize(curve)['last5_id']:.3f})")


def test_index_and_delay_fwp_paths_are_unchanged():
    """The pointer support must not disturb the index/delay control the earlier runs used."""
    import arms_learned as L
    import tasks_c
    task = tasks_c.IndexTask(n_items=8, ep_len=12)
    arm, info = L.build_learned("fwp", task, budget=256, seed=0, hidden=16)
    assert arm.state_floats(task.ep_len) == 256, arm.state_floats(task.ep_len)
    b = task.batch(4, task.ep_len, rng=np.random.default_rng(0))
    out = arm(b, task.features(b))
    assert out.shape == (4, task.ep_len, task.out_dim), out.shape
    print(f"  index fwp path unchanged ({info['knob']}, state {arm.state_floats(task.ep_len)})")


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
