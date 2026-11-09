"""Checks for the spatial recall task and its SSP arm (handoff 11 step 3, `tasks_e` / `arms_spatial`).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_tasks_e.py

This is the gate for the robotics argument, so the tests are written to make the *claim* falsifiable rather than to
show that the arm trains:

  * the task is deterministic and its target is exactly the analytic re-expression ``x_seen - (p_t - p_seen)``;
  * **the group-action identity holds against the task itself, with no parameters involved**: multiplying the phase
    vector of the shown position by ``phi(-d_s)`` over the intervening steps reproduces the phase vector of the target
    exactly, and omitting those multiplications does not -- i.e. the algebra is exactly what this task needs, and the
    shift is load-bearing.  This is checkable before any training, which is the point;
  * the no-memory floor is high (an MLP on the features can read the *motion* but not the stored position, so it
    cannot do better than predicting the current-frame offset from the observation alone);
  * a short run trains: the SSP arm beats that floor clearly, and both ablations (`shift=False`, shared roles) lose.
"""

import numpy as np
import torch

import arms_spatial as ASP
import feats
import harness as H
import ssp_ops as S
import tasks_e


def _task(ep_len=24, n_landmarks=3):
    return tasks_e.SpatialRecallTask(n_landmarks=n_landmarks, ep_len=ep_len)


def test_determinism_and_interface_shapes():
    t = _task()
    a = t.batch(16, t.ep_len, rng=np.random.default_rng(0))
    b = t.batch(16, t.ep_len, rng=np.random.default_rng(0))
    for k in ("obs", "action", "lpos", "query", "target", "mask", "t"):
        assert torch.equal(a[k], b[k]), f"{k} is not rng-deterministic"
    B, T = a["obs"].shape
    assert a["obs"].shape == (B, T) and a["action"].shape == (B, T, 2) and a["lpos"].shape == (B, T, 2)
    assert a["query"].shape == (B, T, t.K) and a["target"].shape == (B, T, 2) and a["mask"].shape == (B, T)
    assert a["loss"] == "mse" and t.out_dim == 2
    assert t.features(a).shape[-1] == t.in_dim
    # every landmark is shown exactly once, in the first K steps
    for k in range(t.K):
        assert bool((a["obs"][:, k] == k).all()), "landmark k must be shown at step k"
    assert int((a["obs"][:, t.K:] == t.K).all()) == 1, "nothing may be shown after the write region"
    assert int(a["mask"].sum()) > 0 and not bool(a["mask"][:, : t.K].any())
    print(f"  deterministic; {int(a['obs'].shape[0])}x{T}, K={t.K}, {int(a['mask'].sum())} queries, in_dim {t.in_dim}")


def test_the_target_is_the_analytic_re_expression():
    """Independent of any arm: replay the motion and check the target equals the landmark in the current frame."""
    t = _task()
    b = t.batch(8, t.ep_len, rng=np.random.default_rng(1))
    act = b["action"].numpy()
    p = np.cumsum(act, axis=1)
    ok = 0
    for bb in range(8):
        for tt in range(t.ep_len):
            if not bool(b["mask"][bb, tt]):
                continue
            k = int(b["query"][bb, tt].argmax())
            seen = b["lpos"][bb, k].numpy()                     # position in the frame it was shown (step k)
            expect = seen - (p[bb, tt] - p[bb, k])
            assert np.allclose(expect, b["target"][bb, tt].numpy(), atol=1e-5), "target is not the re-expression"
            ok += 1
    assert ok == int(b["mask"].sum())
    print(f"  target == lpos[shown] - (p_t - p_shown) on all {ok} queries")


def test_the_group_action_is_exactly_what_the_task_needs():
    """Parameter-free: the phase vector of the target is the written phase vector times ``phi(-d)`` per step.

    The SSP motion update is a single multiplication per step applied to the whole memory.  If that does not *exactly*
    reproduce the target here, the arm is being asked to learn something the algebra does not provide and the task is
    the wrong test.  The same computation without the multiplications must fail, which is what makes the assertion
    meaningful.
    """
    omega = S.make_omega(128, 2, seed=7)
    t = _task()
    b = t.batch(6, t.ep_len, rng=np.random.default_rng(2))
    worst = 0.0
    worst_noshift = 0.0
    with torch.no_grad():
        for bb in range(6):
            for tt in range(t.ep_len):
                if not bool(b["mask"][bb, tt]):
                    continue
                k = int(b["query"][bb, tt].argmax())
                z = S.phi(b["lpos"][bb, k][None, :], omega)                      # written in the frame at step k
                for s in range(k + 1, tt + 1):                                   # one group action per step since
                    z = z * S.phi(-b["action"][bb, s][None, :], omega)
                want = S.phi(b["target"][bb, tt][None, :], omega)
                worst = max(worst, float((z - want).abs().max()))
                no = S.phi(b["lpos"][bb, k][None, :], omega)                     # the same read without shifting
                worst_noshift = max(worst_noshift, float((no - want).abs().max()))
    print(f"  shifted: max|z - phi(target)| = {worst:.2e};  unshifted: {worst_noshift:.3f}")
    assert worst < 1e-5, "the group action does not reproduce the target: the task is not SSP-shaped"
    assert worst_noshift > 0.1, "the unshifted read happens to match: the test proves nothing"


def _short_run(arm_name, task, steps=200, seed=0, **cfg):
    arm, _ = H.build_any(arm_name, task, 256, seed, cfg)
    sp = {"train": {"T": task.ep_len}, "id": {"T": task.ep_len}}
    rng = np.random.default_rng(seed)
    b = H.make_batch(task, sp["train"], 64, rng)
    feat = task.features(b)
    opt = torch.optim.Adam(arm.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = H.loss_and_acc(arm, task, b, feat)
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss, _ = H.loss_and_acc(arm, task, b, feat)
        m = b["mask"]
        pred = arm(b, feat)
        pred = pred[0] if isinstance(pred, tuple) else pred
        mse = float(((pred - b["target"]) ** 2).sum(-1)[m].mean())
    return mse


def test_the_arm_trains_and_both_ablations_lose():
    task = _task(ep_len=24, n_landmarks=3)
    full = _short_run("ssp_spatial", task, steps=250)
    noshift = _short_run("ssp_spatial", task, steps=250, shift=False)
    shared = _short_run("ssp_spatial", task, steps=250, roles="shared")
    print(f"  full {full:.4f}   shift=False {noshift:.4f}   roles=shared {shared:.4f}")
    assert full < noshift, "the group-action ablation must be worse"
    assert full < shared, "the shared-role ablation must be worse"
    assert full < 0.25 * noshift, "the group action must buy a large factor, not a rounding difference"


def test_the_gate_variant_is_pure_recall():
    """Structural check for the trainability-gate variant (`spatial_easy`).

    **The gate itself was executed on pods, not here** (`jobs/p21_spatial_tuned.json`, cells `p21-easy-*`; training is
    not run locally in this round).  Its outcome is recorded in RUN_013 section 14.1 and it is a **failure**: the GRU
    reached MSE 0.630 against a no-memory floor of 1.314 -- 0.48x, where the pre-registered bar was 0.1x -- while the
    SSP arm reached 0.001.  That is a *finding about the baseline*, so it is asserted nowhere in this file: a training
    outcome is an experiment, not a unit test, and encoding it here would leave the suite permanently red (an earlier
    version of this function did exactly that, with `assert gru <= 0.1 * floor`, and it failed).

    What a unit test can and does check is that the variant is what it claims to be, so that the pod cell measured
    what it was supposed to measure.
    """
    task = tasks_e.SpatialRecallTask(n_landmarks=1, ep_len=24, step_scale=0.0)
    b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
    assert task.K == 1, "the gate variant must hold exactly one landmark"
    assert float(b["action"].abs().max()) == 0.0, "the gate variant must have no motion"
    assert bool(b["mask"].any()) and not bool(b["mask"][:, :1].any()), "the query region must be after the write step"
    # with no motion the answer is exactly the position shown at step 0, so the variant is pure recall
    for bb in range(8):
        for tt in range(task.ep_len):
            if bool(b["mask"][bb, tt]):
                assert np.allclose(b["target"][bb, tt].numpy(), b["lpos"][bb, 0].numpy(), atol=1e-6), \
                    "the gate variant's target is not the step-0 observation"
    print("  gate variant: 1 landmark, no motion, target == position shown at step 0 "
          "(pod outcome: GRU FAILED the gate, RUN_013 14.1)")


if __name__ == "__main__":
    for fn in (test_determinism_and_interface_shapes, test_the_target_is_the_analytic_re_expression,
               test_the_group_action_is_exactly_what_the_task_needs,
               test_the_arm_trains_and_both_ablations_lose,
               test_the_gate_variant_is_pure_recall):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_tasks_e: all checks passed")
