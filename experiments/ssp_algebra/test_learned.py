"""Checks for the learned-structure arms and the FWP control (handoff 09 step E).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_learned.py

The important one is the last: ``rung=1, init="identity"`` must reproduce rung 0 (write at ``t``, read at ``j``) and
therefore reach high accuracy after readout-only training.  That proves rung 1 can *express* rung 0; it is a
capability check, not a thesis result (docs/11 section 3.4).
"""

import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_b
import tasks_c


def _index_task(ep_len=16):
    return tasks_c.IndexTask(n_items=8, ep_len=ep_len)


def _batch(task, B=8):
    return task.batch(B, task.ep_len, rng=np.random.default_rng(0))


def test_determinism():
    task = _index_task()
    outs = []
    for _ in range(2):
        arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1)
        b = _batch(task)
        with torch.no_grad():
            outs.append(arm(b, task.features(b)))
    same = torch.equal(outs[0], outs[1])
    print(f"  determinism (identical logits) = {same}")
    assert same, "construction is not seed-determined"


def test_smoke_on_index_and_delay():
    for task in (_index_task(), tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2, 3, 4))):
        for name in ("ssp_index_learned", "fwp"):          # the variant-C arms; pointer has its own tests
            arm, info = L.build_learned(name, task, budget=256, seed=0, hidden=16)
            b = _batch(task)
            logits = arm(b, task.features(b))
            assert logits.shape == (8, task.ep_len, task.out_dim), (name, task.name, logits.shape)
            assert torch.isfinite(logits).all(), f"{name} on {task.name} produced non-finite logits"
            print(f"  {name:20s} on {task.name:6s} state={info['state_floats']:5d} params={info['params']:6d}")


def test_rung2_gradients_on_omega_and_codes():
    """Rung 2 must actually train ``Omega`` and the codes, and must not produce NaNs."""
    task = _index_task()
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=2, hidden=16)
    assert isinstance(arm.omega, torch.nn.Parameter) and isinstance(arm.codes, torch.nn.Parameter)
    opt = torch.optim.Adam(arm.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    for step in range(50):
        b = task.batch(16, task.ep_len, rng=rng)
        loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(arm.parameters(), 1.0)
        opt.step()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
    gw = float(arm.omega.grad.abs().sum()) if arm.omega.grad is not None else 0.0
    gc = float(arm.codes.grad.abs().sum()) if arm.codes.grad is not None else 0.0
    print(f"  rung 2 after 50 steps: |grad omega|={gw:.3e} |grad codes|={gc:.3e}")
    assert gw > 0 and gc > 0, "rung 2 did not train Omega / the codes"
    assert torch.isfinite(arm.omega).all() and torch.isfinite(arm.codes).all()


def test_identity_init_reproduces_rung0(min_acc=0.95):
    """Rung 1 with an identity-initialised address heads *is* rung 0, so a readout-only fit must reach >= 0.95."""
    task = _index_task()
    sp = H.splits_for("index", task)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, init="identity", hidden=64)
    for name, p in arm.named_parameters():
        if not name.startswith("mix"):
            p.requires_grad_(False)
    curve, _ = H.train_arm(arm, task, sp, seed=0, steps=400, batch=64, lr=1e-2, eval_every=200)
    acc = curve[-1]["id"]
    print(f"  rung 1 (identity init), readout-only: id={acc:.3f} after 400 updates")
    assert acc >= min_acc, f"identity init does not reproduce rung 0: {acc:.3f}"


class _Pass(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[..., : self.n]


def _pointer_task():
    return tasks_b.PointerTask(n_nodes=8, n_query=8, train_hops=(1, 2), max_hops=8)


def test_pointer_identity_init_is_rung0(min_acc=0.99):
    """Oracle-initialised learned pointer arm == the hand-wired rung 0: exact answers at every hop 1..8 with a
    pass-through readout and no forgetting.  This proves the soft role gate / soft bind / learned query code *can*
    express rung 0 (a capability check, never a thesis result)."""
    task = _pointer_task()
    arm, _ = L.build_learned("ssp_pointer_learned", task, budget=512, seed=0, rung=1, init="identity",
                             decay=False, hidden=8)
    arm.mix = _Pass(task.n_nodes)
    rng = np.random.default_rng(0)
    worst, rows = 1.0, []
    for h in range(1, 9):
        b = task.batch(64, task.ep_len, rng=rng, hops=(h,))
        with torch.no_grad():
            pred = arm(b, task.features(b)).argmax(-1)
        m = b["mask"]
        acc = ((pred == b["target"]) & m).sum().item() / m.sum().item()
        rows.append(f"h{h}={acc:.2f}")
        worst = min(worst, acc)
    print("  learned pointer, identity init, oracle readout: " + " ".join(rows))
    assert worst >= min_acc, f"identity init does not reproduce rung 0: worst hop {worst:.3f}"


def test_pointer_determinism_and_aux():
    task = _pointer_task()
    outs, auxs = [], []
    for _ in range(2):
        arm, _ = L.build_learned("ssp_pointer_learned", task, budget=256, seed=0, rung=1, hidden=16)
        b = task.batch(8, task.ep_len, rng=np.random.default_rng(0))
        with torch.no_grad():
            outs.append(arm(b, task.features(b)))
        auxs.append(float(arm.aux_loss(b)))
    assert torch.equal(outs[0], outs[1]), "construction is not seed-determined"
    print(f"  pointer determinism ok; per-hop aux loss = {auxs[0]:.3f} (finite: {np.isfinite(auxs[0])})")
    assert np.isfinite(auxs[0]) and abs(auxs[0] - auxs[1]) < 1e-9


def test_pointer_gradients_reach_role_gate_and_query_map():
    """The structure heads must actually receive gradient through the soft bind and the hop loop."""
    task = _pointer_task()
    arm, _ = L.build_learned("ssp_pointer_learned", task, budget=256, seed=0, rung=1, hidden=16)
    b = task.batch(16, task.ep_len, rng=np.random.default_rng(1))
    loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
    (loss + arm.aux_loss(b)).backward()
    gr = float(arm.role.weight.grad.abs().sum())
    gq = float(arm.qmap.weight.grad.abs().sum())
    print(f"  |grad role gate|={gr:.3e} |grad query map|={gq:.3e}")
    assert gr > 0 and gq > 0, "a learned structure head receives no gradient"
    assert all(torch.isfinite(p.grad).all() for p in arm.parameters() if p.grad is not None)


def test_scrambled_init_differs_from_prior():
    task = _index_task()
    a, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, init="random")
    b, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, init="scrambled")
    same_read = torch.equal(a.read_head.weight, b.read_head.weight)
    differs = not torch.equal(a.w_time, b.w_time)
    print(f"  scrambled: write phase differs from SSP prior = {differs}; read head same stream = {same_read}")
    assert differs and a.w_time[0].item() == 32.0, "scrambled init did not randomise the write phase"


def test_width_schedule_and_learned_width():
    import math

    task = _index_task()
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, length_scale=1.0, ls_start=4.0,
                             ls_mode="sched")
    got = []
    for step in (0, 375, 750, 1500):
        arm.set_width(step, 750)
        got.append(round(float(arm._mult()), 3))
    print(f"  sched mult at steps 0/375/750/1500 = {got} (want 4 -> 2.5 -> 1 -> 1)")
    assert got == [4.0, 2.5, 1.0, 1.0], got
    # the fixed and unset modes are exactly rung 1 as before
    a, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1)
    b, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, ls_start=1.0, ls_mode="fixed")
    bt = _batch(task)
    with torch.no_grad():
        same = torch.equal(a(bt, task.features(bt)), b(bt, task.features(bt)))
    print(f"  ls_start=None vs ls_start=1 logits identical = {same}")
    assert same, "width control changed the default behaviour"
    # learned mode: a trained scalar that actually receives gradient
    lw, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, length_scale=1.0, ls_start=4.0,
                            ls_mode="learned", hidden=16)
    assert isinstance(lw.log_mult, torch.nn.Parameter)
    loss, _ = H.loss_and_acc(lw, task, bt, task.features(bt))
    loss.backward()
    g = float(lw.log_mult.grad.abs())
    print(f"  learned width: mult0={float(lw._mult()):.2f}, |grad log_mult|={g:.3e}")
    assert g > 0 and math.isfinite(g)


def test_diagnostic_hooks_address_params_freeze_and_retrieval_loss():
    task = _index_task()
    a, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, freeze_write=True, hidden=16)
    names = {id(p) for p in a.address_params()}
    assert id(a.w_time) not in names and not a.w_time.requires_grad, "freeze_write did not freeze the write phase"
    assert id(a.read_head.weight) in names and id(a.read_head.bias) in names
    b = _batch(task)
    a(b, task.features(b))
    loss = a.retrieval_loss(b)
    loss.backward()
    g = float(a.read_head.weight.grad.abs().sum())
    print(f"  retrieval loss {float(loss):.3f}, |grad read head|={g:.3e}, write phase frozen")
    assert torch.isfinite(loss) and g > 0
    a.freeze_address()
    assert not any(p.requires_grad for p in (a.read_head.weight, a.read_head.bias))


def test_adaptive_width_narrows_on_plateau_and_respects_the_ceiling():
    task = _index_task()
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, length_scale=1.0, ls_start=8.0,
                             ls_mode="adaptive", hidden=16)
    for step in range(101):                       # a flat loss: the width must shrink faster than the linear schedule
        arm.set_width(step, 1000, feedback=1.0)
    stalled, linear = float(arm._mult()), 8.0 + (1.0 - 8.0) * 0.1
    arm2, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, length_scale=1.0, ls_start=8.0,
                              ls_mode="adaptive", hidden=16)
    for step in range(101):                       # a loss that keeps improving: it must follow (not beat) the linear ceiling
        arm2.set_width(step, 1000, feedback=1.0 * 0.9 ** step)
    improving = float(arm2._mult())
    print(f"  adaptive mult at step 100: plateau {stalled:.2f}, improving {improving:.2f}, linear ceiling {linear:.2f}")
    assert stalled < linear - 0.5, "a plateau did not narrow the kernel"
    assert abs(improving - linear) < 1e-6, "an improving loss should leave the width at the linear schedule"
    arm.set_width(1000, 1000, feedback=1.0)
    assert abs(float(arm._mult()) - 1.0) < 1e-6, "the schedule must finish at the final width"


def test_param_group_lr_scale_and_pretraining_freezes_the_address():
    task = _index_task()
    sp = H.splits_for("index", task)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, ls_start=4.0, ls_mode="sched",
                             hidden=16)
    opt = H.make_optimizer(arm, 1e-2, 0.1)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    print(f"  optimizer group lrs = {lrs}")
    assert len(lrs) == 2 and abs(lrs[0] / lrs[1] - 0.1) < 1e-9
    base = H.make_optimizer(arm, 1e-2, 1.0)
    assert len(base.param_groups) == 1, "default must be a single group (existing runs unchanged)"
    curve, extra = H.train_arm(arm, task, sp, seed=0, steps=10, batch=16, lr=1e-2, eval_every=10, pretrain_steps=15)
    frozen = not any(p.requires_grad for p in arm.address_params()) and len(arm.address_params()) == 0
    print(f"  after pretraining: address frozen = {frozen}, mult = {float(arm._mult()):.2f}, "
          f"pretrain_tokens = {extra['pretrain_tokens']}, task tokens = {extra['tokens']}")
    assert frozen and abs(float(arm._mult()) - 1.0) < 1e-6
    assert extra["pretrain_tokens"] > 0 and extra["tokens"] > 0


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    checks = [("determinism", test_determinism),
              ("smoke", test_smoke_on_index_and_delay),
              ("rung2_gradients", test_rung2_gradients_on_omega_and_codes),
              ("identity_init_reproduces_rung0", test_identity_init_reproduces_rung0),
              ("pointer_identity_init_is_rung0", test_pointer_identity_init_is_rung0),
              ("pointer_determinism_and_aux", test_pointer_determinism_and_aux),
              ("pointer_gradients", test_pointer_gradients_reach_role_gate_and_query_map),
              ("scrambled_init", test_scrambled_init_differs_from_prior),
              ("width_schedule_and_learned", test_width_schedule_and_learned_width),
              ("diagnostic_hooks", test_diagnostic_hooks_address_params_freeze_and_retrieval_loss),
              ("adaptive_width", test_adaptive_width_narrows_on_plateau_and_respects_the_ceiling),
              ("param_groups_and_pretraining", test_param_group_lr_scale_and_pretraining_freezes_the_address)]
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
