"""
Unit tests for DQN update rule.

The production DQN contract (rl/networks/dqnBasic.py):
- DQN.push receives ALREADY-ENCODED phi vectors (not raw states)
- DQN.update() does NOT call rep.map(); it uses stored phi directly
- DQN.update() trains via standard autograd (loss.backward()) + torch.optim.Adam
  over the whole QNetwork — NOT per-action hand-rolled Adam (that design was
  removed when hidden layers were introduced; per-action step/gradient-clip
  semantics don't extend to parameters shared across actions, and the
  underlying paper's update rule doesn't specify an optimizer at all — see
  runs/RUN_001.md for the full before/after rationale).
- Loss is MSE (0.5*delta^2), matching the paper's semi-gradient TD update.
- Gradient is clipped by GLOBAL norm across all parameters (not per-action).

The previous version of this file was half-broken: it pushed raw 4D states
into a buffer sized for 91D HighDimRep, which crashed T2/T3 with a NumPy
broadcast error. The torch port of dqnBasic preserves the contract; the test
must too.

Pass criterion: |Q_final - target| < 0.15 * |target|.
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from rl.networks.dqnBasic import DQN


class _Encoder:
    """Identity-encoding stand-in for testing DQN math with a known size.

    size_out matches the encoded phi dimensionality. map() is identity
    (we already push encoded phi into the buffer; update() must NOT call
    rep.map() per the production contract).
    """

    def __init__(self, size_out: int):
        self.size_out = size_out

    def map(self, x):
        return x  # identity — testing DQN math, not the encoder


def _build_dqn(obs_dim: int, **kwargs):
    enc = _Encoder(obs_dim)
    defaults = dict(
        rep=enc,
        state_dim=obs_dim,
        n_actions=2,
        lr=0.1,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
    )
    defaults.update(kwargs)
    return DQN(**defaults)


def _fill_with_targets(dqn, phi_dim, target_q=1.0):
    """Fill buffer with phi vectors of unit norm and target Q=target_q."""
    n = 200
    for _ in range(n):
        phi = np.ones(phi_dim, dtype=np.float32) / np.sqrt(phi_dim)
        dqn.push(phi, 0, target_q, phi, True)


def _q_at(dqn, phi_dim):
    phi = np.ones(phi_dim, dtype=np.float32) / np.sqrt(phi_dim)
    return float(dqn.get_q_values(phi)[0])


def test1_lowdim_nlms():
    """Low-dim: ||phi||=1 (size_out=4). Adam with lr=0.1 should converge Q->1.0."""
    dqn = _build_dqn(4, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                     max_grad_norm=10.0, target_update_freq=1)
    _fill_with_targets(dqn, 4, target_q=1.0)

    for _ in range(80):
        dqn.update()

    q = _q_at(dqn, 4)
    target = 1.0
    err = abs(q - target)
    print(f"  [PASS] T1 (low-dim, size_out=4): Q(phi,0)={q:.4f}, target={target}, |err|={err:.4f}" if err < 0.15 * target else
          f"  [FAIL] T1 (low-dim, size_out=4): Q(phi,0)={q:.4f}, target={target}, |err|={err:.4f}")
    assert err < 0.15 * target


def test2_highdim_nlms():
    """High-dim: ||phi||=sqrt(91) (size_out=91, all-ones phi). With compensated lr should converge Q->1.0."""
    dqn = _build_dqn(91, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                     max_grad_norm=10.0, target_update_freq=1)
    _fill_with_targets(dqn, 91, target_q=1.0)

    # Scale lr by ||phi||^2 to compensate for the per-sample gradient normalization
    # (gradient = mean(delta * phi); with all-ones phi and ||phi||^2=91, gradient norm grows).
    dqn.lr = 9.1

    # Need ~91 Adam steps to span a 91-dim gradient direction. Use 200 to be safe.
    for _ in range(200):
        dqn.update()

    q = _q_at(dqn, 91)
    target = 1.0
    err = abs(q - target)
    print(f"  [PASS] T2 (high-dim, size_out=91): Q(phi,0)={q:.4f}, target={target}, |err|={err:.4f}" if err < 0.15 * target else
          f"  [FAIL] T2 (high-dim, size_out=91): Q(phi,0)={q:.4f}, target={target}, |err|={err:.4f}")
    assert err < 0.15 * target


def test3_math_consistency():
    """Verify training is stable and converges under a skewed action
    distribution using the current standard-Adam (loss.backward()) update —
    replaces the old per-action-Adam-internals check (that mechanism no
    longer exists by design, see module docstring)."""
    obs_dim = 16
    n_actions = 3
    dqn = _build_dqn(obs_dim, n_actions=n_actions, adam_beta1=0.9, adam_beta2=0.999,
                     adam_eps=1e-8, max_grad_norm=10.0, target_update_freq=1)

    # Push 100 fixed-target transitions with action distribution 50/30/20 across
    # actions 0/1/2, all with the same unit-norm phi so each action's target Q is
    # independently checkable afterward (same style as tests 1/2's convergence check).
    rng = np.random.default_rng(42)
    actions = [0] * 50 + [1] * 30 + [2] * 20
    rng.shuffle(actions)
    phi = np.ones(obs_dim, dtype=np.float32) / np.sqrt(obs_dim)
    for a in actions:
        dqn.push(phi, a, 1.0, phi, True)

    for _ in range(80):
        dqn.update()

    q = dqn.get_q_values(phi)
    assert torch.isfinite(q).all(), f"Non-finite Q-values after training: {q}"
    errs = (q - 1.0).abs()
    print(f"  [PASS] T3 (math consistency, skewed actions): Q={q.tolist()}, |err|={errs.tolist()}"
          if (errs < 0.15).all() else
          f"  [FAIL] T3 (math consistency, skewed actions): Q={q.tolist()}, |err|={errs.tolist()}")
    assert (errs < 0.15).all()


if __name__ == "__main__":
    print("DQN update unit tests")
    print("=" * 60)
    test1_lowdim_nlms()
    test2_highdim_nlms()
    test3_math_consistency()
    print("=" * 60)
    print("All tests passed.")
