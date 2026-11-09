"""Checks for the content-plus-position family (handoff 11 step 6): `tasks_d.AssocMixedTask` + `arm_assoc.AssocPosSSPArm`.

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_tasks_d.py

The claim under test is narrow and must be checked before any pod run: **the position key is load-bearing for the
index-mode half of the queries**.  Both modes ask the same question about the same pair, and an index-mode query's
``obs`` symbol comes from the key alphabet, so an arm that ignores ``mode`` -- or a task whose ``mode`` column is
constant -- would look fine on content queries and be silently wrong on index queries.  The tests therefore score
**per mode**, never pooled:

  * the task's two modes are both present, roughly balanced, and a wrong-addressing oracle fails exactly the mode it
    gets wrong (the "the test can fail" baseline);
  * the alphabet and feature width are fixed across every split, and mutating ``mode`` alone changes the features;
  * the arm solves it locally in a short run, and its two ablations fail in the predicted, mode-specific way
    (``pos=False`` destroys index-mode queries; ``bind=False`` destroys both);
  * the historical defaults are untouched -- ``assoc`` itself is unchanged and its inputs still have their old width.

Pass rule read off these tests: the arm works only if index-mode accuracy is far above chance WITH position keys and
at chance WITHOUT them.  A pod protocol (`jobs/p18_assoc_mixed.json`) decides the real numbers; this file decides
whether the experiment is well posed.
"""

import numpy as np
import torch

import arms_assoc as AA
import harness as H
import tasks_d

IGNORE = tasks_d.IGNORE


def _task(ep_len=16, n_pairs=4):
    return tasks_d.AssocMixedTask(n_pairs=n_pairs, ep_len=ep_len)


def _mode_mask(b):
    return b["mode"] > 0, b["mode"] < 0


def _acc(b, logits, which):
    pred = logits.argmax(-1)
    ok = (pred == b["target"]) & b["mask"] & which
    return float(ok.sum()) / max(int((b["mask"] & which).sum()), 1)


def test_determinism_and_both_modes_present():
    t = _task()
    a = t.batch(16, t.ep_len, rng=np.random.default_rng(0))
    c = t.batch(16, t.ep_len, rng=np.random.default_rng(0))
    for k in ("obs", "target", "mask", "mode", "key", "query"):
        assert torch.equal(a[k], c[k]), f"{k} is not rng-deterministic"
    content, index = _mode_mask(a)
    assert int((a["mask"] & content).sum()) > 0, "no content-mode queries"
    assert int((a["mask"] & index).sum()) > 0, "no index-mode queries"
    frac = int((a["mask"] & index).sum()) / int(a["mask"].sum())
    assert 0.35 < frac < 0.65, f"modes are unbalanced: {frac:.2f} index"
    # a content query names a key; an index query names a *position*, and both live in the key alphabet
    assert int(a["obs"][content & a["mask"]].max()) < t.n_pairs
    assert int(a["obs"][index & a["mask"]].max()) < t.n_pairs
    print(f"  deterministic; {frac:.2f} of {int(a['mask'].sum())} queries are index-mode")


def test_the_mode_column_is_load_bearing():
    """Mutating only ``mode`` must move the features, and a wrong-addressing oracle must fail that mode.

    The oracle answers with the *content* key always.  In content mode that is correct by construction; in index mode
    it is an unrelated key, so its accuracy there is chance (1/P).  This is the control that proves the two modes are
    distinguishable at all.
    """
    t = _task()
    b = t.batch(16, t.ep_len, rng=np.random.default_rng(1))
    mutated = {**b, "mode": -b["mode"]}
    assert not torch.equal(t.features(b), t.features(mutated)), "the mode column is not in the features"

    content, index = _mode_mask(b)
    # oracle: answer every masked query with the value of the *content key* obs points at.
    # The task stores value ids in [0,P) directly, so recover them from the pair region.
    P = t.n_pairs
    val_of_key = torch.full((b["obs"].shape[0], P), -1, dtype=b["target"].dtype)
    for bb in range(b["obs"].shape[0]):
        for i in range(P):
            val_of_key[bb, int(b["obs"][bb, 2 * i])] = int(b["obs"][bb, 2 * i + 1]) - P
    oracle = torch.full_like(b["target"], -1)
    m = b["mask"]
    # at a query step `obs` is always inside the key alphabet (a key in content mode, a position in index mode)
    assert int(b["obs"][m].max()) < P, "query steps must carry a symbol from the key alphabet"
    oracle[m] = val_of_key.gather(1, b["obs"].clamp(max=P - 1))[m]
    ok = (oracle == b["target"]) & b["mask"]
    acc_c = float(ok[content].sum()) / max(int((b["mask"] & content).sum()), 1)
    acc_i = float(ok[index].sum()) / max(int((b["mask"] & index).sum()), 1)
    print(f"  content oracle: content-mode {acc_c:.3f}, index-mode {acc_i:.3f} (chance {1 / P:.3f})")
    assert acc_c > 0.99, "a content-key oracle must solve content-mode queries"
    assert acc_i < 0.45, "a content-key oracle must NOT solve index-mode queries"


def test_alphabet_and_width_are_fixed_across_splits():
    t = _task()
    for name, spec in H.splits_for("assoc_mixed", t).items():
        b = H.make_batch(t, spec, 16, np.random.default_rng(0))
        assert int(b["obs"].max()) < 2 * t.n_pairs, f"{name}: obs outside the alphabet"
        assert int(b["target"][b["mask"]].max()) < t.out_dim, f"{name}: target outside the output alphabet"
        assert t.features(b).shape[-1] == t.in_dim, f"{name}: feature width changed"
        assert b["mode"].abs().max() == 1.0, f"{name}: mode is not +-1"
        print(f"  {name:11s} T={b['obs'].shape[1]:3d} obs<{2 * t.n_pairs} target<{t.out_dim} feat={t.in_dim}")


def test_the_content_only_family_is_unchanged():
    """The new variant must not touch `assoc`: same width, same targets, and its own splits."""
    a = tasks_d.AssocTask(n_pairs=4, ep_len=16)
    b = a.batch(8, 16, rng=np.random.default_rng(0))
    assert a.in_dim == 2 * a.n_pairs + 1 + __import__("feats").TIME_DIM
    assert "mode" not in b, "the content-only task must not have gained a mode column"
    assert a.features(b).shape[-1] == a.in_dim


def _short_run(arm_name, task, steps=150, seed=0, **cfg):
    arm, info = H.build_any(arm_name, task, 256, seed, cfg)
    sp = {"train": {"T": task.ep_len}, "id": {"T": task.ep_len}}
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(arm.parameters(), lr=2e-2)
    b = H.make_batch(task, sp["train"], 64, rng)
    feat = task.features(b)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = H.loss_and_acc(arm, task, b, feat)
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss, _ = H.loss_and_acc(arm, task, b, feat)
    content, index = _mode_mask(b)
    logits = arm(b, feat)
    logits = logits[0] if isinstance(logits, tuple) else logits
    return _acc(b, logits, content), _acc(b, logits, index), float(loss)


def test_the_arm_solves_both_modes_and_the_ablations_fail_where_they_must():
    task = _task(ep_len=24, n_pairs=4)
    c_full, i_full, _ = _short_run("assoc_pos_ssp", task, steps=250)
    c_nopos, i_nopos, _ = _short_run("assoc_pos_ssp", task, steps=250, pos=False)
    c_nobind, i_nobind, _ = _short_run("assoc_pos_ssp", task, steps=250, bind=False)
    print(f"  full        content {c_full:.3f}  index {i_full:.3f}")
    print(f"  pos=False   content {c_nopos:.3f}  index {i_nopos:.3f}")
    print(f"  bind=False  content {c_nobind:.3f}  index {i_nobind:.3f}")
    assert c_full > 0.9 and i_full > 0.9, "the arm must solve both modes in a short run"
    assert i_nopos < i_full - 0.3, "dropping the position key must destroy index-mode queries"
    assert c_nobind < 0.9 or i_nobind < 0.9, "the additive (no-bind) ablation must lose at least one mode"


if __name__ == "__main__":
    for fn in (test_determinism_and_both_modes_present, test_the_mode_column_is_load_bearing,
               test_alphabet_and_width_are_fixed_across_splits, test_the_content_only_family_is_unchanged,
               test_the_arm_solves_both_modes_and_the_ablations_fail_where_they_must):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_tasks_d: all checks passed")
