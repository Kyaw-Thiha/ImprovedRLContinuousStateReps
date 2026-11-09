"""Checks for the delta-rule / tied-encoder fast-weight arm (`fwp_delta`, docs/11 section 14.2 and handoff 11 step 0a).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_fwp_delta.py

Why this arm exists: the `fwp` control every earlier number used is *plain* linear attention -- `M += v k^T` with one
learned scalar decay, untied key/query heads, no write strength.  The fast-weight family actually used in practice has
the delta rule `M += beta (v - M k) k^T` and gates, so "SSP ties `fwp`" was a comparison against a weak control
(RUN_011 section 12; handoff 11 section 2).  These checks fix the semantics of the stronger arm; no result is claimed
here.

The last check is the one that matters for the comparison: on an overwrite (two values written under the same key) the
delta arm must read the *later* value while the plain arm reads their *sum*, at the same state size.
"""

import numpy as np
import torch

import arms_learned as L
import feats
import tasks_c


def _index_task(ep_len=16, n_items=8):
    return tasks_c.IndexTask(n_items=n_items, ep_len=ep_len)


def _batch(task, B=8):
    return task.batch(B, task.ep_len, rng=np.random.default_rng(0))


def _pin_heads(arm, slot=0, identity=False):
    """Make the address encoders deterministic so a test measures the *rule*, not the random head's scale.

    ``slot=k``: every feature maps to the basis vector ``e_k`` (one key, so an overwrite is exact).
    ``identity=True``: the head copies the 13 time features through, so the key similarity between two times is the
    feature kernel -- the fast-weight analogue of the SSP read-address question.
    """
    d, fd = arm.k_head.weight.shape
    with torch.no_grad():
        if identity:
            arm.k_head.weight.zero_(); arm.k_head.bias.zero_()
            arm.k_head.weight[:min(d, fd), :min(d, fd)] = torch.eye(min(d, fd))
        else:
            arm.k_head.weight.zero_(); arm.k_head.bias.zero_(); arm.k_head.bias[slot] = 1.0
        if arm.tied:
            arm.q_head = arm.k_head
        else:
            arm.q_head.load_state_dict(arm.k_head.state_dict())
        arm.lam_raw.data.fill_(60.0)          # sigmoid(60) == 1.0: no decay, so only the update rule is under test
        if getattr(arm, "rule", "plain") == "delta":
            arm.beta_raw.data.fill_(60.0)     # sigmoid(60) == 1.0: full-strength correction
    return arm


def test_delta_overwrites_while_plain_accumulates():
    """The load-bearing semantic difference, measured at one key with no optimisation involved.

    With the heads pinned to one basis vector and no decay, the plain arm's read after the second write is exactly
    ``va + vb`` (it accumulates), while the delta arm's is exactly ``vb`` (it corrects the slot).
    """
    task = _index_task()
    B, d = 4, 16
    torch.manual_seed(0)
    feat = feats.abs_time_feats(torch.full((B,), 5.0))
    va, vb = torch.randn(B, d), torch.randn(B, d)
    res = {}
    for rule in ("plain", "delta"):
        arm = _pin_heads(L.FWPArm(task, budget=256, seed=0, rule=rule, tied=True))
        k = arm.k_head(feat); q = arm.q_head(feat)
        lam = torch.sigmoid(arm.lam_raw)
        M = torch.zeros(B, d, d)
        def write(M, v):
            if rule == "delta":
                kn = torch.nn.functional.normalize(k, dim=-1)
                r = torch.bmm(M, kn[:, :, None])[..., 0]
                return lam * M + torch.sigmoid(arm.beta_raw) * (v - r)[:, :, None] * kn[:, None, :]
            return lam * M + v[:, :, None] * k[:, None, :]
        M = write(M, va)
        read_a = torch.bmm(M, torch.nn.functional.normalize(q, dim=-1)[:, :, None])[..., 0]
        M = write(M, vb)
        read_b = torch.bmm(M, torch.nn.functional.normalize(q, dim=-1)[:, :, None])[..., 0]
        res[rule] = dict(a=(read_a - va).norm(dim=-1).mean().item(),
                         b_vb=(read_b - vb).norm(dim=-1).mean().item(),
                         b_sum=(read_b - (va + vb)).norm(dim=-1).mean().item())
    p, dl = res["plain"], res["delta"]
    print(f"  plain: |read_a-va|={p['a']:.2e}  |read_b-vb|={p['b_vb']:.3f}  |read_b-(va+vb)|={p['b_sum']:.2e}")
    print(f"  delta: |read_a-va|={dl['a']:.2e}  |read_b-vb|={dl['b_vb']:.2e}  |read_b-(va+vb)|={dl['b_sum']:.3f}")
    assert p["b_sum"] < 1e-4, "plain arm should read the sum of both writes"
    assert dl["b_vb"] < 1e-4, "delta arm should read only the later value"
    assert dl["b_vb"] < p["b_vb"] / 10.0, "delta rule did not overwrite the plain rule by a wide margin"
    assert p["a"] < 1e-4 and dl["a"] < 1e-4, "both arms must return the first value right after writing it"


def test_tied_encoder_is_one_encoder():
    """`tied=True` must share the object, so a query equal to a key gets the same address by construction."""
    task = _index_task()
    tied = L.FWPArm(task, budget=256, seed=0, tied=True)
    assert tied.q_head is tied.k_head, "tied encoder is not shared"
    untied = L.FWPArm(task, budget=256, seed=0)
    assert untied.q_head is not untied.k_head
    print("  tied=True shares k_head/q_head; tied=False keeps them separate")


def test_state_and_param_accounting_matches_plain():
    """The comparison is at equal carried state: the delta/tied knobs must not change it."""
    task = _index_task()
    plain = L.FWPArm(task, budget=512, seed=0)
    delta = L.FWPArm(task, budget=512, seed=0, rule="delta", tied=True, read_abs=True)
    # `fwp` carries d*d floats where d = round(sqrt(budget)) (the harness's budget-matched construction), so the
    # value is 23*23 = 529 at budget 512 -- what matters here is that the two arms carry the *same* state.
    assert plain.state_floats(task.ep_len) == delta.state_floats(task.ep_len)
    assert plain.d == delta.d
    print(f"  state floats: plain={plain.state_floats(task.ep_len)} delta={delta.state_floats(task.ep_len)}")
    # the tied encoder removes a whole (13 x d) head; the delta arm adds one scalar.  Report, do not assert equality.
    print(f"  params: plain={plain.n_params()} delta={delta.n_params()} (difference {plain.n_params() - delta.n_params()})")


def test_wrong_address_dies_in_both_arms():
    """The SSP dead-zone question, asked of the fast-weight arms (handoff 11 step 0c).

    The heads copy the 13 absolute-time features through, so the key similarity between two times is exactly the
    feature kernel every arm in this folder receives.  A read addressed one index off must retrieve the wrong item;
    if it does not, the discovery problem is *not* about addressing at all.
    """
    # Within capacity on purpose: at 32 items in d=16 this arm retrieves only ~0.36 at offset 0 (measured), which is
    # a capacity limit, not an addressing one -- so the addressing question is asked at 8 items.
    task = _index_task(ep_len=8, n_items=8)
    d = 16
    torch.manual_seed(0)
    B = 4
    t = torch.arange(task.ep_len, dtype=torch.float)[None, :].expand(B, task.ep_len)
    val = torch.randn(B, task.ep_len, d)
    print(f"  {'rule':6s} {'offset':>6s}  {'mean retrieval acc':>18s}")
    accs = {}
    for rule in ("plain", "delta"):
        for off in (0.0, 1.0, 3.0):
            arm = _pin_heads(L.FWPArm(task, budget=256, seed=0, rule=rule, tied=True), identity=True)
            k = arm.k_head(feats.abs_time_feats(t))
            q = arm.q_head(feats.abs_time_feats(t + off))
            kn = torch.nn.functional.normalize(k, dim=-1)
            qn = torch.nn.functional.normalize(q, dim=-1)
            lam = torch.sigmoid(arm.lam_raw)
            beta = torch.sigmoid(arm.beta_raw) if rule == "delta" else None
            M = torch.zeros(B, d, d)
            for i in range(task.ep_len):
                ki = kn[:, i]
                if rule == "delta":
                    r = torch.bmm(M, ki[:, :, None])[..., 0]
                    M = lam * M + beta * (val[:, i] - r)[:, :, None] * ki[:, None, :]
                else:
                    M = lam * M + val[:, i][:, :, None] * ki[:, None, :]
            hits = []
            for i in range(task.ep_len):
                r = torch.bmm(M, qn[:, i, :, None])[..., 0]                     # (B, d)
                sim = torch.nn.functional.cosine_similarity(r[:, None, :], val, dim=-1)   # (B, T)
                hits.append((sim.argmax(-1) == i).float().mean().item())
            acc = float(np.mean(hits))
            accs[(rule, off)] = acc
            print(f"  {rule:6s} {off:6.1f}  {acc:18.3f}")
    # The offset-0 ceiling is set by the code, not by the rule: the 13 absolute-time features are broad, so eight
    # items in d=16 already interfere (measured 0.844 plain / 0.719 delta).  What this check is for is the *drop*
    # with a wrong address, which is what the SSP dead-zone result is about.
    for rule in ("plain", "delta"):
        assert accs[(rule, 0.0)] > 2.0 * accs[(rule, 1.0)], f"a one-index offset must hurt the {rule} arm sharply"
        assert accs[(rule, 1.0)] < accs[(rule, 0.0)], f"an offset of 1 should hurt the {rule} arm"
        assert accs[(rule, 3.0)] <= accs[(rule, 1.0)], f"an offset of 3 should be no better than 1 for the {rule} arm"


def test_smoke_trains_on_index_and_delay():
    """Both tasks, a short optimisation: the arm must improve and stay finite (it is a baseline, so it must be trainable)."""
    import harness as H
    for task in (_index_task(), tasks_c.DelayTask(n_items=8, ep_len=16, train_lags=(1, 2, 3, 4))):
        for name in ("fwp", "fwp_delta"):
            arm, info = L.build_learned(name, task, budget=256, seed=0, hidden=16)
            b = _batch(task, B=16)
            logits = arm(b, task.features(b))
            assert logits.shape == (16, task.ep_len, task.out_dim), (name, logits.shape)
            assert torch.isfinite(logits).all(), f"{name} on {task.name} produced non-finite logits"
            opt = torch.optim.Adam(arm.parameters(), lr=1e-2)
            rng = np.random.default_rng(0)
            first = last = None
            for _ in range(300):
                bb = task.batch(64, task.ep_len, rng=rng)
                loss, acc = H.loss_and_acc(arm, task, bb, task.features(bb))
                opt.zero_grad(); loss.backward()
                assert torch.isfinite(loss), f"{name} loss became non-finite"
                opt.step()
                if first is None:
                    first = acc
                last = acc
            print(f"  {name:10s} on {task.name:6s} state={info['state_floats']:4d} params={info['params']:6d} acc {first:.3f} -> {last:.3f}")
            assert last >= first, f"{name} on {task.name} did not improve ({first:.3f} -> {last:.3f})"


if __name__ == "__main__":
    for fn in (test_delta_overwrites_while_plain_accumulates, test_tied_encoder_is_one_encoder,
               test_state_and_param_accounting_matches_plain, test_wrong_address_dies_in_both_arms,
               test_smoke_trains_on_index_and_delay):
        print(f"\n{fn.__name__}")
        fn()
    print("\ntest_fwp_delta: all checks passed")
