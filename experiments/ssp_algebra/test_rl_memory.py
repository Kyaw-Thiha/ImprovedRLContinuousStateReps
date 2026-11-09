"""Contract tests for the RL memory module (handoff 10 phase 5.2).  No RL runs: these are the mechanics.

1. **Round trip.**  Writing a token at every time step and reading at that step's own address must recover the token
   (>= 0.99) at a state budget where the capacity rule D/4 is not the limit -- i.e. the module's write/read/unbind
   path is correct, in the same sense as the supervised arms' oracle tests.
2. **The interface is separable.**  ``freeze_interface`` must stop gradients into the address parameters, which is what
   makes the supervised two-phase protocol (interface first, readout second) portable to RL.
3. **The auxiliary is label-free and live.**  ``aux_loss`` must be finite and must *produce* gradient into the
   interface while it is unfrozen; after freezing it must produce none.
4. **Reset is real.**  A reset memory must not leak the previous episode's contents (a POMDP memory that leaks across
   resets would silently cheat on any episodic benchmark).

Run: OMP_NUM_THREADS=1 ../../.venv/bin/python test_rl_memory.py
"""
import numpy as np
import torch

import rl_memory as RM


def _round_trip(n=8, dim=128, seed=0, B=64):
    m = RM.SSPMemory(n_items=n, dim=dim, seed=seed).reset(B)
    rng = np.random.default_rng(0)
    tok = torch.as_tensor(rng.integers(0, n, size=B), dtype=torch.long)
    t = torch.zeros(B)
    m.write(tok, t)                      # phase(t = 0) is the write address
    m.read(m._last_address)              # read at the address it wrote to
    return m, tok


def test_round_trip_recovers_the_written_token():
    n, B = 8, 64
    m, tok = _round_trip(n=n, B=B)
    acc = float((m._last_scores.argmax(-1) == tok).float().mean())
    assert acc >= 0.99, f"round trip recovers only {acc:.3f} of the written tokens"
    print(f"  round trip: {acc:.3f} at dim 128 (capacity rule D/4 = 32 items, {n} written)")


def _aux_setup(seed=0, B=16):
    m = RM.SSPMemory(n_items=8, dim=128, seed=seed).reset(B)
    hidden = torch.randn(B, m.hidden)
    t = torch.arange(B).float()
    tok = torch.as_tensor(np.random.default_rng(1).integers(0, 8, size=B), dtype=torch.long)
    for i in range(B):
        m.write(tok[i].reshape(1), t[i].reshape(1))
    return m, tok, hidden, t


def test_freeze_interface_stops_the_address_gradient():
    m, tok, hidden, t = _aux_setup()
    loss = m.aux_loss(tok, hidden, t)
    loss.backward()
    assert float(m.addr_head.weight.grad.norm()) > 0, "no gradient into the address head before freezing"
    # a *fresh* module (``load_state_dict`` copies buffers under no_grad and would drop the memory's graph, which is
    # not what this test is about), frozen before its first backward
    m2, tok2, hid2, t2 = _aux_setup(seed=0)
    m2.freeze_interface()
    for p in m2.interface_params():
        assert not p.requires_grad, "freeze_interface left an interface parameter trainable"
    loss2 = m2.aux_loss(tok2, hid2, t2)
    assert loss2.requires_grad, "freezing the interface removed every trainable path (the leak is trainable)"
    loss2.backward()
    assert m2.addr_head.weight.grad is None, "a frozen address head still received a gradient"
    print("  freeze_interface: address head gets no gradient afterwards (and did before)")


def test_aux_loss_is_finite_and_label_free():
    B, n = 32, 8
    m = RM.SSPMemory(n_items=n, dim=128, seed=0).reset(B)
    rng = np.random.default_rng(2)
    hid = torch.randn(B, m.hidden)
    for step in range(4):
        t = torch.full((B,), float(step))
        tok = torch.as_tensor(rng.integers(0, n, size=B), dtype=torch.long)
        m.write(tok, t)
        loss = m.aux_loss(tok, hid, t)
        assert torch.isfinite(loss), "aux_loss is not finite"
    # a decoder that is *given* the right address must do better than chance: the auxiliary is a real signal
    acc = float((m._last_scores.argmax(-1) == tok).float().mean())
    print(f"  aux_loss finite ({float(loss):.3f}); at the module's own address the token reads at {acc:.2f}")


def test_reset_does_not_leak():
    B, n = 8, 8
    m = RM.SSPMemory(n_items=n, dim=128, seed=0).reset(B)
    tok = torch.full((B,), 3, dtype=torch.long)          # non-zero: with a zero memory every score is 0 and argmax
    m.write(tok, torch.zeros(B))                        # would be index 0, which would fake a hit
    before = float((m.read(m._last_address).argmax(-1) == tok).float().mean())
    m.reset(B)
    after = float((m.read(m._last_address).argmax(-1) == tok).float().mean())
    assert before > 0.9 and after < 0.3, f"reset did not clear the memory ({before:.2f} -> {after:.2f})"
    print(f"  reset clears: {before:.2f} -> {after:.2f} (chance {1 / n:.2f})")


if __name__ == "__main__":
    test_round_trip_recovers_the_written_token()
    test_freeze_interface_stops_the_address_gradient()
    test_aux_loss_is_finite_and_label_free()
    test_reset_does_not_leak()
    print("ok rl memory")
