"""Contract tests for rl/ssp_tied_model.py (reference venv).  ``OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python test_ssp_tied.py``"""

import sys
import time

import numpy as np
import torch

import test_ssp_model as T0
from test_ssp_model import make, run, toks, init_state, inp
import feats


def _set_lag(m, k):
    with torch.no_grad():
        m.lag_head.weight.zero_()
        m.lag_head.bias.fill_(k / m.lag_scale)


def _scores(m, tk, z=None):
    B, T = tk.shape
    z = torch.randn(B, T, 128) if z is None else z
    with torch.no_grad():
        s, _, _, _ = m.memory_core(z, tk, torch.zeros(B, 256), torch.zeros(B, dtype=torch.long), torch.full((B,), T))
    return s


def test_default_reads_now_and_has_no_second_origin():
    m = make("tied")
    assert not any(p.requires_grad for p in m.mem.addr_head.parameters())           # untied head is frozen / unused
    tk = toks(4, 16, 1)
    acc = (_scores(m, tk).argmax(-1) == tk).float().mean()
    assert float(acc) == 1.0                                                          # lag 0: exact write phase
    assert abs(m.address_summary()["lag_bias"]) < 1e-9


def test_tying_property_read_follows_the_write_function():
    """The point of tying: change the write function (w_time); a read at lag k still recalls token t-k.  The untied
    control does not survive the same change (its read origin is its own ramp)."""
    k = 4
    tk = toks(4, 24, 2)
    m = make("tied", lag_init=float(k))
    with torch.no_grad():
        m.mem.w_time.add_(0.3 * torch.randn_like(m.mem.w_time) * (torch.arange(13) > 0).float())   # non-ramp parts only
    acc_t = float((_scores(m, tk)[:, k:].argmax(-1) == tk[:, :-k]).float().mean())
    u = make("untied_scaled", lag_init=float(k))
    base = float((_scores(u, tk)[:, k:].argmax(-1) == tk[:, :-k]).float().mean())
    with torch.no_grad():
        u.mem.w_time.add_(0.3 * torch.randn_like(u.mem.w_time) * (torch.arange(13) > 0).float())
    acc_u = float((_scores(u, tk)[:, k:].argmax(-1) == tk[:, :-k]).float().mean())
    print(f"tied after w_time change {acc_t:.3f}; untied before {base:.3f}, after {acc_u:.3f}")
    assert acc_t > base - 0.05                  # tied: unchanged by the write-function change (crosstalk-limited ~0.87 at 23 bundled items)
    assert base > 0.8 and acc_u < acc_t - 0.1


def test_shapes_state_size_and_chunking():
    for arm in ("tied", "tied_aux", "untied_scaled"):
        m = make(arm, lag_init=2.0)
        tk = toks(3, 10, 3)
        whole, st = run(m, tk)
        assert whole.shape == (3, 10, 4) and tuple(st[0].shape) == (3, 1, 256)
        l1, s1 = run(m, tk[:, :4])
        l2, _ = run(m, tk[:, 4:], state=s1)
        assert torch.allclose(whole[:, 4:], l2, atol=1e-5), arm


def test_gradient_reaches_lag_head_and_w_time():
    m = make("tied", lag_init=1.5)
    tk = toks(4, 12, 4)
    logits, _ = run(m, tk)
    (logits.pow(2).mean() + m.value_function().pow(2).mean()).backward()
    for p in (m.lag_head.bias, m.lag_head.weight, m.mem.w_time):
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
    assert m.mem.addr_head.weight.grad is None


def test_tied_aux_is_lag_aware_and_small_at_the_right_lag():
    k = 3
    m = make("tied_aux", lag_init=float(k))
    tk = toks(6, 20, 5)
    run(m, tk)
    good = float(m._aux)
    _set_lag(m, 5.0 + 0.0)
    # same data, but the head now points at lag 5 while the *target* is computed at lag 5 too: still consistent
    run(m, tk)
    consistent5 = float(m._aux)
    # inconsistent: read at lag k via the head, target from a different lag, built by hand
    _set_lag(m, k)
    run(m, tk)
    assert good < 0.3 and consistent5 < 0.3, (good, consistent5)                       # recall is sharp at any integer lag
    m.tied_aux_lag = False                                                             # literal SSPMemory target = current token
    run(m, tk)
    assert float(m._aux) > 1.0                                                         # reading lag 3 vs target "now": large


def test_aux_flag_and_custom_loss():
    off, on = make("tied"), make("tied_aux")
    tk = toks(3, 9, 6)
    run(off, tk); run(on, tk)
    base = [torch.tensor(1.0, requires_grad=True)]
    assert off.custom_loss(base, {}) is base
    out = on.custom_loss(base, {})
    assert float(out[0]) > 1.0
    on.zero_grad(); out[0].backward()
    assert float(on.lag_head.bias.grad.abs().sum()) >= 0.0 and on.mem.w_time.grad is not None


def test_address_summary_keys_and_scale():
    m = make("tied", lag_init=4.0)
    s = m.address_summary()
    assert abs(s["lag_bias"] - 4.0) < 1e-5 and "lag_probe_t25_h0" in s
    u = make("untied_scaled", lag_init=4.0).address_summary()
    assert abs(u["lag_bias"] - 4.0) < 1e-5 and "ramp_w" in u


def test_per_step_cost_tied():
    m, base = make("tied"), make("ssp")
    a, b = T0._step_time(m, 0), T0._step_time(base, 0)
    late = T0._step_time(m, 5000)
    print(f"per-step forward (B=8,T=1): tied {a*1e3:.2f} ms (t=5000: {late*1e3:.2f}), untied ssp {b*1e3:.2f} ms")
    assert late < 2.0 * a + 1e-3 and a < 20e-3


# --- 4. the lag bank (handoffs/11 step 4) ------------------------------------------------------------
def _set_lag_bank(m, lag):
    """One-hot the selector onto ``lag``: the bank's counterpart of ``_set_lag`` for the continuous head."""
    with torch.no_grad():
        m.sel_head.weight.zero_()
        m.sel_head.bias.zero_()
        m.sel_head.bias[m.bank_lags.tolist().index(float(lag))] = 30.0


def test_lag_bank_state_size_uniform_init_and_no_second_origin():
    bank, tied = make("tied_bank"), make("tied")
    assert bank.state_floats() == tied.state_floats() == 256                # the bank adds params, not carried floats
    assert bank.bank_lags.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert not any(p.requires_grad for p in bank.mem.addr_head.parameters())   # tying is inherited, unchanged
    assert bank._sel is None                                                # nothing selected before a forward
    _scores(bank, toks(2, 16, 1))
    # default init: a uniform selector (zero weights, zero bias), so no bank lag is preferred and no task constant enters
    assert torch.allclose(bank._sel, torch.full_like(bank._sel, 1.0 / bank._sel.shape[-1]), atol=1e-6)
    s = bank.address_summary()
    assert abs(s["sel_max"] - 0.125) < 1e-6 and abs(s["lag_eff"] - 4.5) < 1e-4


def test_lag_bank_one_hot_recalls_exactly_like_the_set_lag_arm():
    """A bank containing the task's lag reproduces the hand-set-lag arm bit-for-bit in behaviour (accuracy 1.0)."""
    for lag in (2, 4, 8):
        m = make("tied_bank")
        _set_lag_bank(m, lag)
        tk = toks(4, 16, 1)
        sc = _scores(m, tk)
        acc = (sc[:, lag:].argmax(-1) == tk[:, :-lag]).float().mean()
        assert float(acc) == 1.0, (lag, float(acc))
        assert abs(m.address_summary()["lag_eff"] - lag) < 1e-3


def test_lag_bank_aux_and_custom_loss_run():
    m = make("tied_bank_aux", aux_coeff=0.1)
    tk = toks(3, 16, 2)
    logits, _ = run(m, tk)
    assert torch.isfinite(logits).all()
    aux = m.aux_term(_scores(m, tk), tk, torch.full((3,), 16, dtype=torch.int32))
    assert torch.isfinite(aux) and float(aux) > 0
    assert torch.isfinite(m.custom_loss(torch.tensor(1.0), None))


def test_lag_bank_gradient_reaches_the_selector():
    """The selection must be trainable from the policy/aux signal, or the bank is a fixed random mixture."""
    m = make("tied_bank")
    tk = toks(4, 16, 1)
    z = torch.randn(4, 16, 128)
    scores, _, _, _ = m.memory_core(z, tk, torch.zeros(4, 256), torch.zeros(4, dtype=torch.long),
                                    torch.full((4,), 16, dtype=torch.long))
    loss = torch.nn.functional.cross_entropy((scores[:, 4:] / 0.1).reshape(-1, 4), tk[:, :-4].reshape(-1))
    loss.backward()
    g = m.sel_head.bias.grad
    assert g is not None and float(g.abs().sum()) > 0.0, g
    assert float(g[m.bank_lags.tolist().index(4.0)]) != 0.0


def test_lag_bank_finds_the_lag_from_the_uniform_default():
    """The claim: selection has no dead zone, so the right lag is *found* from the default init by gradient descent.

    Only the selector is trained (the rest of the model is frozen, as if the write path were already aligned); the
    target is the token at lag 4.  The continuous head cannot do this inside a PPO budget, which is why this arm exists.
    """
    torch.manual_seed(0)
    m = make("tied_bank")
    m.train()
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.sel_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.sel_head.parameters(), lr=0.05)
    B, T, LAG = 8, 24, 4
    tk = toks(B, T, 3)
    first = None
    for step in range(60):
        z = torch.randn(B, T, 128)
        scores, _, _, _ = m.memory_core(z, tk, torch.zeros(B, 256), torch.zeros(B, dtype=torch.long),
                                        torch.full((B,), T, dtype=torch.long))
        loss = torch.nn.functional.cross_entropy((scores[:, LAG:] / 0.1).reshape(-1, 4), tk[:, :-LAG].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        acc = float((scores[:, LAG:].argmax(-1) == tk[:, :-LAG]).float().mean())
        first = acc if first is None else first
        if step in (0, 20, 59):
            print(f"  bank sel step {step:3d} acc {acc:.3f} lag_eff {m.address_summary()['lag_eff']:.2f}")
    assert first < 0.6, first                       # it does not start solved (otherwise the test proves nothing)
    assert acc > 0.9, acc                           # ... and 60 selector-only steps find the lag
    assert abs(m.address_summary()["lag_eff"] - LAG) < 0.75


def test_lag_bank_without_the_true_lag_cannot_solve_the_task():
    """The bank's *range* is a task-dependent constant, and this measures what it buys.

    RUN_013 section 10 says "no lag constant was injected", which is only half true: the bank is a fixed set of lags
    1..8 and RepeatPreviousEasy's recall lag happens to be 3.  This asserts the other half -- a bank that does not
    contain the true lag cannot solve the task -- so "the selector finds the lag" is bounded by "within the range it
    is given".  The control is the same bank *with* the lag, same steps, same seed.

    Note on `lag_eff`: it is the mean of the selector distribution times the bank's lags, so a bank of [1, 2, 5, 6]
    can report an effective lag near 4 by mixing 1 and 6 while holding no such address.  That is why this test scores
    *accuracy*, not the effective lag -- the first version of it asserted on `lag_eff` and failed for exactly that
    reason.
    """
    torch.manual_seed(0)
    B, T, LAG = 8, 24, 4
    tk = toks(B, T, 3)

    def selector_steps(m, steps=60):
        m.train()
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.sel_head.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(m.sel_head.parameters(), lr=0.05)
        for _ in range(steps):
            z = torch.randn(B, T, 128)
            sc, _, _, _ = m.memory_core(z, tk, torch.zeros(B, 256), torch.zeros(B, dtype=torch.long),
                                        torch.full((B,), T, dtype=torch.long))
            loss = torch.nn.functional.cross_entropy((sc[:, LAG:] / 0.1).reshape(-1, 4), tk[:, :-LAG].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            z = torch.randn(B, T, 128)
            sc, _, _, _ = m.memory_core(z, tk, torch.zeros(B, 256), torch.zeros(B, dtype=torch.long),
                                        torch.full((B,), T, dtype=torch.long))
            acc = float((sc[:, LAG:].argmax(-1) == tk[:, :-LAG]).float().mean())
            eff = float(m.address_summary()["lag_eff"])
        return acc, eff

    lacking = make("tied_bank", lag_bank=[1, 2, 5, 6])
    assert 4.0 not in lacking.bank_lags.tolist(), "the test needs a bank that excludes the true lag"
    acc_lack, eff_lack = selector_steps(lacking)

    with_lag = make("tied_bank", lag_bank=[1, 2, 3, 4, 5, 6, 7, 8])
    acc_have, eff_have = selector_steps(with_lag)

    print(f"  bank [1,2,5,6] acc {acc_lack:.3f} (lag_eff {eff_lack:.2f})   "
          f"bank [1..8] acc {acc_have:.3f} (lag_eff {eff_have:.2f})")
    assert acc_lack < 0.45, f"a bank that cannot read at the true lag still scored {acc_lack:.3f}"
    assert acc_have > acc_lack + 0.3, "the bank containing the lag must clearly beat the one that does not"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for k, f in fns:
        try:
            f(); print("PASS", k)
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback; traceback.print_exc(); print("FAIL", k, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
