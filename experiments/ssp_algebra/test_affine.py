"""Affine read-head prior: the invariants RUN_011's phase 0.3 and section 4.4 depend on.

1. **`eps = 0` is exactly `init="identity"`** (bit-for-bit).  Everything measured about the *identity* floor (RUN_009's
   "50 updates to 0.90") must remain the same arm, so this is checked against the other init rather than against a
   constant.
2. **Determinism**: the same seed gives the same prior.
3. **The `affine_abs` form is scale-free**: the *absolute* address error must be the same at every stream length R,
   which is what makes it a rule rather than a per-variant constant.  The `affine_eps` form is deliberately the
   opposite (error proportional to R) and is checked to differ.
4. **The prior is a prior, not the answer**: at a large `affine_abs` the initial address must actually be wrong by a
   measurable amount, or the "learned from a bad prior" claim would be vacuous.
5. **Gradient reaches the read head** from a perturbed prior (otherwise nothing could be learned from it).

Run: OMP_NUM_THREADS=1 ../../.venv/bin/python test_affine.py
"""
import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_c


def _arm(T=16, budget=256, seed=0, **kw):
    task = tasks_c.IndexTask(n_items=8, ep_len=T)
    arm, _ = L.build_learned("ssp_index_learned", task, budget=budget, seed=seed, rung=1, **kw)
    return arm, task


def _addr_rms(arm, task, seed=3, B=64):
    b = task.batch(B, task.ep_len, rng=np.random.default_rng(seed))
    with torch.no_grad():
        arm(b, task.features(b))
        _, s_q, addr = arm._read_phase(b)
    err = (s_q - addr)[b["mask"]]
    return float(err.pow(2).mean().sqrt())


def test_eps_zero_is_exactly_identity():
    a, _ = _arm(init="affine", affine_eps=0.0)
    b, _ = _arm(init="identity")
    assert torch.equal(a.read_head.weight, b.read_head.weight), "eps=0 no longer reproduces the identity read head"
    assert torch.equal(a.read_head.bias, b.read_head.bias), "eps=0 no longer reproduces the identity read head"
    print("  eps = 0 is bit-for-bit init='identity'")


def test_prior_is_deterministic():
    a, _ = _arm(init="affine", affine_abs=0.5)
    b, _ = _arm(init="affine", affine_abs=0.5)
    assert torch.equal(a.read_head.weight, b.read_head.weight) and torch.equal(a.read_head.bias, b.read_head.bias)
    print("  the prior is deterministic per seed")


def test_absolute_form_is_scale_free_and_the_relative_form_is_not():
    abs_err, eps_err = {}, {}
    for T, budget in ((16, 256), (32, 256), (64, 512)):
        arm, task = _arm(T=T, budget=budget, init="affine", affine_abs=0.5)
        abs_err[T] = _addr_rms(arm, task)
        arm2, task2 = _arm(T=T, budget=budget, init="affine", affine_eps=0.4)
        eps_err[T] = _addr_rms(arm2, task2)
    spread = max(abs_err.values()) - min(abs_err.values())
    assert spread < 0.35, f"affine_abs error is not scale-free: {abs_err}"
    assert eps_err[64] > 2 * eps_err[16], f"affine_eps error should grow with R: {eps_err}"
    print(f"  affine_abs RMS error {[round(abs_err[t], 2) for t in (16, 32, 64)]} (spread {spread:.2f}); "
          f"affine_eps {[round(eps_err[t], 2) for t in (16, 32, 64)]}")


def test_the_prior_is_actually_wrong():
    arm, task = _arm(init="affine", affine_abs=4.0)
    rms = _addr_rms(arm, task)
    assert rms > 2.0, f"affine_abs=4.0 should start several indices away, measured {rms:.2f}"
    print(f"  affine_abs=4.0 starts {rms:.1f} indices away (a prior, not the answer)")


def test_gradient_reaches_the_read_head_from_a_perturbed_prior():
    arm, task = _arm(init="affine", affine_abs=0.5)
    b = task.batch(32, task.ep_len, rng=np.random.default_rng(0))
    loss, _ = H.loss_and_acc(arm, task, b, task.features(b))
    loss.backward()
    assert arm.read_head.weight.grad is not None and float(arm.read_head.weight.grad.norm()) > 0.0
    print(f"  read-head grad norm from the prior: {float(arm.read_head.weight.grad.norm()):.3e}")


def test_deterministic_channels_give_exactly_the_requested_error():
    """``affine_slope_err``/``affine_offset`` must set the address error *exactly*, since the basin map reads cells off
    a grid and a draw would blur them."""
    arm, task = _arm(init="affine", affine_slope_err=0.0, affine_offset=3.0)
    b = task.batch(64, task.ep_len, rng=np.random.default_rng(3))
    with torch.no_grad():
        arm(b, task.features(b))
        _, s_q, addr = arm._read_phase(b)
    err = (s_q - addr)[b["mask"]]
    # slope error 0 and offset 3 -> the address is exactly 3 indices out at every index
    assert torch.allclose(err, torch.full_like(err, 3.0), atol=1e-5), \
        f"offset channel is not exact: min {float(err.min()):.4f} max {float(err.max()):.4f}"
    print(f"  slope_err=0, offset=3 -> address error exactly 3.0 (RMS {float(err.pow(2).mean().sqrt()):.4f})")


def test_slope_error_grows_with_the_index():
    """A slope error is a *relative* error: it must grow with the address, unlike the offset channel."""
    T = 32
    arm, task = _arm(T=T, budget=256, init="affine", affine_slope_err=0.1, affine_offset=0.0)
    b = task.batch(64, task.ep_len, rng=np.random.default_rng(3))
    with torch.no_grad():
        arm(b, task.features(b))
        _, s_q, addr = arm._read_phase(b)
    m = b["mask"]
    err = (s_q - addr)[m]
    idx = addr[m]
    ratio = err / idx.clamp_min(1e-6)
    assert torch.allclose(err, 0.1 * idx, atol=1e-4), \
        f"slope channel is not exact: err - 0.1*addr ranges {float((err - 0.1 * idx).min()):.2e}..{float((err - 0.1 * idx).max()):.2e}"
    nz = ratio[idx > 0]
    assert nz.numel() and torch.allclose(nz, torch.full_like(nz, 0.1), atol=1e-4), "err/addr must be the slope error"
    lo = err[idx <= idx.median()].abs().mean()
    hi = err[idx > idx.median()].abs().mean()
    assert hi > 1.6 * lo, f"slope error must grow with the index: small-index {lo:.2f} vs large-index {hi:.2f}"
    print(f"  slope_err=0.1: err/addr is exactly 0.1; mean |err| {lo:.2f} below the median index, {hi:.2f} above it")


def test_zero_channels_equal_identity_and_are_default_off():
    a, _ = _arm(init="affine", affine_slope_err=0.0, affine_offset=0.0)
    b, _ = _arm(init="identity")
    assert torch.equal(a.read_head.weight, b.read_head.weight) and torch.equal(a.read_head.bias, b.read_head.bias), \
        "affine with both channels zero must be bit-for-bit the identity read head"
    # and the channels must be inert unless asked for: the plain affine_eps form is unchanged
    c, _ = _arm(init="affine", affine_eps=0.4)
    assert c.affine_slope_err == 0.0 and c.affine_offset == 0.0
    print("  zero channels == init='identity'; the random forms are untouched")


def test_offset_zero_keeps_the_slope_channel_only():
    """``affine_offset_zero`` must keep the drawn slope error and force the offset to exactly 0 (RUN_011 section 3.2)."""
    arm, task = _arm(T=32, budget=256, init="affine", affine_eps=0.4, affine_offset_zero=True)
    assert float(arm.read_head.bias.detach()) == 0.0, f"offset must be exactly 0, got {float(arm.read_head.bias.detach())}"
    slope = float(arm.read_head.weight[0, 0]) / 32.0          # the ramp weight is 32 (feats.RAMP_SCALE)
    assert abs(slope - 1.0) > 0.0, "the slope channel must still carry the requested error"
    # and the plain eps form (offset not zeroed) must still have a nonzero offset in general
    arm2, _ = _arm(T=32, budget=256, init="affine", affine_eps=0.4)
    assert arm2.affine_offset_zero is False
    print(f"  offset_zero: bias exactly 0, slope {slope:.3f} (error {slope - 1:+.3f})")


if __name__ == "__main__":
    test_eps_zero_is_exactly_identity()
    test_prior_is_deterministic()
    test_absolute_form_is_scale_free_and_the_relative_form_is_not()
    test_the_prior_is_actually_wrong()
    test_gradient_reaches_the_read_head_from_a_perturbed_prior()
    test_deterministic_channels_give_exactly_the_requested_error()
    test_slope_error_grows_with_the_index()
    test_zero_channels_equal_identity_and_are_default_off()
    test_offset_zero_keeps_the_slope_channel_only()
    print("ok affine")
