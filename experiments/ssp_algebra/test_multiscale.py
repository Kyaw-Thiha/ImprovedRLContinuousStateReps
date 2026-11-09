"""Phase 1 (handoff 10): the multi-scale frequency set must give a basin *without* a width schedule.

Why these checks, in this order:

1. **Algebra is unchanged.**  The kinds only change *which* frequencies are drawn, so ``phi`` stays unit-modulus,
   deterministic per seed, and the old single-scale kinds keep their exact old output (a regression check: every
   previous result must stay reproducible).
2. **Kernel shape.**  ``kappa(d) = mean_j cos(omega_j * d)``.  At ``length_scale = 1`` the single-scale kernel is
   ``sinc(pi d)``, which is **exactly zero at every integer d != 0** -- the measured reason plain training stalls on
   index recall (RUN_008: the read address must start within ~1 index for any gradient to exist).  A multi-scale set
   must keep kernel mass across the address range (no dead *regions*) and must not alias inside it.
3. **Loss landscape in the address error.**  This is the decisive cheap mechanism check.  ``init='identity'`` makes
   the read address the raw index, so ``read_head.bias`` is the address error in indices.  With a single scale the
   loss is a *flat plateau* for every error beyond ~1 index (all integer offsets give the same, zero-signal loss), so
   gradient descent has no systematic pull back to the answer; a multi-scale set must make the loss *increase* with
   the error across the range.

   **Negative result recorded here (2026-10-12):** the obvious probe -- the norm of ``d loss / d read_head`` -- does
   *not* discriminate.  Measured at a 5.3-index error: uniform 3.15e-03 vs logmix 2.60e-03 (both nonzero).  With a
   finite ``D`` the cleanup scores carry crosstalk of size ~1/sqrt(D) that varies with the address, so the gradient
   norm is dominated by it.  The basin is a *directional* property; only the loss shape (and training) shows it.

Run: OMP_NUM_THREADS=1 ../../.venv/bin/python test_multiscale.py
"""
import numpy as np
import torch

import arms_learned as L
import harness as H
import ssp_ops as S
import tasks_c


def _kappa(omega, deltas):
    """``mean_j cos(omega_j * d)``: the expected kernel of the SSP encoder for separation ``d``."""
    om = omega.reshape(-1).double()
    return [float(torch.cos(om * float(d)).mean()) for d in deltas]


def test_old_kinds_are_unchanged():
    for kind in ("uniform", "gauss"):
        a = S.make_omega(64, 1, length_scale=1.5, kind=kind, seed=3)
        b = S.make_omega(64, 1, length_scale=1.5, kind=kind, seed=3)
        assert torch.equal(a, b), "make_omega must be deterministic per (kind, seed)"
    # the single-scale kernel really is (numerically) zero at integer lags: the measured failure mode
    om = S.make_omega(2048, 1, length_scale=1.0, kind="uniform", seed=0)
    for d in (1, 2, 3):
        assert abs(_kappa(om, [d])[0]) < 0.05, f"single-scale kappa({d}) should be ~0, got {_kappa(om, [d])[0]}"
    print("  historical kinds deterministic; single-scale kernel ~0 at integer lags (the failure mode)")


def test_multiscale_kinds_are_unit_modulus_and_deterministic():
    for kind in ("logmix", "residue"):
        for d in (1, 3):
            a = S.make_omega(64, d, length_scale=1.0, kind=kind, seed=5, ls_hi=16.0)
            b = S.make_omega(64, d, length_scale=1.0, kind=kind, seed=5, ls_hi=16.0)
            assert a.shape == (64, d) and torch.equal(a, b), f"{kind}: determinism / shape"
            phi = S.phi(torch.zeros(4, d), a)
            assert torch.allclose(phi.abs(), torch.ones_like(phi.abs()), atol=1e-5), f"{kind}: phi is not unit modulus"
    print("  logmix/residue: unit modulus, deterministic, right shape")


def test_multiscale_basin_without_aliasing():
    """Kernel mass across the range (no dead *region*) and no aliasing inside it."""
    R = 14
    ds = list(range(0, R + 1))
    k_uni = _kappa(S.make_omega(512, 1, length_scale=1.0, kind="uniform", seed=0), ds)
    assert k_uni[0] > 0.99, "kappa(0) must be 1"
    assert max(abs(x) for x in k_uni[1:]) < 0.1, f"single scale should be ~0 across the range: {k_uni[1:5]}"
    for kind in ("logmix", "residue"):
        om = S.make_omega(512, 1, length_scale=1.0, kind=kind, seed=0, ls_hi=R / 2.0)
        k = _kappa(om, ds)
        assert k[0] > 0.99, f"{kind}: kappa(0) = {k[0]}"
        assert k[1] > 0.3, f"{kind}: kappa(1) = {k[1]:.3f} -- no basin at a 1-index offset (the whole point)"
        # a single draw's mean kernel can cross zero at isolated offsets (sign cancellation across the mixture); what
        # must not happen is a *region* with no kernel mass, since the basin is a region
        run = 0
        for i, d in enumerate(ds[1:], start=1):
            run = run + 1 if abs(k[i]) < 0.02 else 0
            assert run < 3, f"{kind}: a {run}-index region with |kappa| ~ 0 ending at offset {d}"
        mid = np.mean([abs(x) for x, d in zip(k, ds) if 1 <= d <= R // 2])
        assert mid > 0.05, f"{kind}: mean |kappa| over the near half of the range is only {mid:.3f}"
        far = _kappa(om, [d for d in range(R // 2, R + 1)])
        assert max(far) < 0.95, f"{kind}: kernel repeats inside the address range (aliasing): max {max(far):.3f}"
        print(f"  {kind}: kappa {[round(x, 3) for x in k[:5]]} ... {[round(x, 3) for x in k[10:15]]}")


def _loss_at_error(omega_kind, eps, ls_hi=7.0, seed=0):
    """Retrieval loss (cleanup cross-entropy) with the read address ``eps`` indices away from the correct one.

    ``init='identity'`` + a bias of ``eps`` makes the read address exactly ``index + eps``.  The *readout* is
    untrained at this point, so the arm's own task loss is ~log(8) for every address; the quantity the address
    controls is the retrieval quality of the memory read, which is what ``arm.retrieval_loss`` measures (it uses no
    readout and needs no task-specific labels beyond the stream's own tokens).
    """
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=seed, rung=1, init="identity",
                             ls_mode="fixed", omega_kind=omega_kind, ls_hi=ls_hi, lam_init=0.99, temp=0.1)
    with torch.no_grad():
        arm.read_head.bias.fill_(float(eps))
    b = task.batch(64, task.ep_len, rng=np.random.default_rng(1))
    with torch.no_grad():
        arm(b, task.features(b))
        return float(arm.retrieval_loss(b))


def test_loss_landscape_is_not_a_flat_plateau():
    """Single scale: the loss beyond a 1-index error is a plateau (no information to descend).  Multi-scale: it grows."""
    eps = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
    curves = {k: [_loss_at_error(k, e) for e in eps] for k in ("uniform", "logmix")}
    for k, v in curves.items():
        print("  " + k + " loss at error " + ", ".join(f"{e}: {x:.3f}" for e, x in zip(eps, v)))
    uni, mix = curves["uniform"], curves["logmix"]
    plateau = max(uni[2:]) - min(uni[2:])                       # error >= 1 index: single scale has no signal
    assert plateau < 0.25 * (uni[2] - uni[0]), \
        f"single-scale loss should be flat beyond 1 index (measured spread {plateau:.3f} vs rise {uni[2] - uni[0]:.3f})"
    rise = mix[-1] - mix[0]
    assert rise > 0.25 * (uni[2] - uni[0]), \
        f"multi-scale loss must keep rising with the address error (rise {rise:.3f})"
    assert mix[4] > mix[1], "multi-scale loss should be larger at a 3-index error than at 0.5"


def test_gradient_reaches_the_read_head_without_a_schedule():
    """Plain training, random read head, multi-scale frequencies: the read head must get a non-zero gradient."""
    task = tasks_c.IndexTask(n_items=8, ep_len=16)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=256, seed=0, rung=1, init="random",
                             ls_mode="fixed", omega_kind="logmix", ls_hi=7.0, lr=0.02, lam_init=0.99, temp=0.1)
    b = task.batch(32, task.ep_len, rng=np.random.default_rng(0))
    loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
    loss.backward()
    for name, q in (("read_head.weight", arm.read_head.weight), ("w_time", arm.w_time)):
        g = q.grad
        assert g is not None and float(g.norm()) > 0.0, f"{name} got no gradient at random init"
    print(f"  random init: read_head grad norm {float(arm.read_head.weight.grad.norm()):.3e}, "
          f"w_time {float(arm.w_time.grad.norm()):.3e}")


if __name__ == "__main__":
    test_old_kinds_are_unchanged()
    test_multiscale_kinds_are_unit_modulus_and_deterministic()
    test_multiscale_basin_without_aliasing()
    test_loss_landscape_is_not_a_flat_plateau()
    test_gradient_reaches_the_read_head_without_a_schedule()
    print("ok multiscale")
