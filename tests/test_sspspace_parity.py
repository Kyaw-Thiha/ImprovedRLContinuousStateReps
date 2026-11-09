"""
Parity test for the Stage-0 SSP encoding path.

What this verifies:
  1. The upstream `sspspace` package (pinned at the SHA recorded in
     requirements-stage0.txt) is importable.
  2. Its `HexagonalSSPSpace` factory function returns an `SSPEncoder` whose
     `.encode()` returns a wrapped `SSP` object — NOT a raw ndarray.
  3. `.v` exposes the underlying ndarray and supports `.reshape()`.
  4. The ssp_dim matches the closed-form expectation
     n_rotates * n_scales * (domain_dim + 1) * 2 + 1
     for the same (domain_dim, n_rotates, n_scales) the legacy code used.
  5. Encoding is a bounded real-valued signal (no NaN, no blow-up).
  6. rl/representations/ssp.py::SSPRep integrates cleanly with the upstream
     package — its `map(state)` returns a 1-D ndarray with shape (ssp_dim,)
     and the same point encodes to the same vector across calls.
  7. The DQN import chain (experiments.trial_cartpole_dqn and its direct
     submodule dependencies) does not pull in Nengo at any step.

This is the contract the Stage-0 DQN path's SSP representation depends on.
Run once after changes to rl/representations/ssp.py or the sspspace pin
in requirements-stage0.txt. NOT in the DQN hot path.
"""

import sys
import os
import importlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _drop_cached(*prefixes):
    """Remove modules starting with any of the given prefixes from sys.modules."""
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == p or mod_name.startswith(p + ".") for p in prefixes):
            del sys.modules[mod_name]


def test_upstream_import_and_api():
    from sspspace import HexagonalSSPSpace
    enc = HexagonalSSPSpace(domain_dim=4, n_rotates=5, n_scales=5, kernel="hypergeom", length_scale=1.0)
    assert hasattr(enc, "ssp_dim"), "upstream encoder missing ssp_dim"
    assert hasattr(enc, "phase_matrix"), "upstream encoder missing phase_matrix"
    assert enc.phase_matrix.shape[1] == 4
    assert enc.ssp_dim == enc.phase_matrix.shape[0]


def test_upstream_encode_returns_wrapped_ssp():
    """Per docs/handoffs/01-stage0-repo-rebuild.md §2: upstream encode()
    returns a wrapped SSP object. .v is the underlying ndarray."""
    from sspspace import HexagonalSSPSpace
    from sspspace.ssp import SSP
    enc = HexagonalSSPSpace(domain_dim=4, n_rotates=5, n_scales=5, kernel="hypergeom", length_scale=1.0)
    x = np.array([[0.1, 0.2, -0.3, 0.4]])
    out = enc.encode(x)
    assert isinstance(out, SSP), f"expected wrapped SSP, got {type(out)}"
    phi = out.v
    assert phi.shape == (1, enc.ssp_dim)
    # .v must support reshape; the wrapped SSP object itself does NOT.
    flat = phi.reshape(-1)
    assert flat.shape == (enc.ssp_dim,)


def test_upstream_ssp_dim_matches_closed_form():
    """For domain_dim=4, n_rotates=5, n_scales=5 the closed form is
    n_rotates * n_scales * (domain_dim + 1) * 2 + 1 = 5*5*5*2 + 1 = 251.
    This matches what the vendored class produced before the refactor."""
    from sspspace import HexagonalSSPSpace
    enc = HexagonalSSPSpace(domain_dim=4, n_rotates=5, n_scales=5, kernel="hypergeom", length_scale=1.0)
    expected = 5 * 5 * (4 + 1) * 2 + 1
    assert enc.ssp_dim == expected, f"ssp_dim={enc.ssp_dim}, expected {expected}"


def test_upstream_encoding_is_bounded_real_signal():
    """An SSP encoding is a real-valued bounded signal with no NaN/Inf."""
    from sspspace import HexagonalSSPSpace
    enc = HexagonalSSPSpace(domain_dim=4, n_rotates=5, n_scales=5, kernel="hypergeom", length_scale=1.0)
    rng = np.random.default_rng(42)
    x = rng.uniform(-1.0, 1.0, size=(11, 4))
    phi = enc.encode(x).v
    assert phi.shape == (11, enc.ssp_dim)
    assert np.all(np.isfinite(phi)), "encoding contains NaN or Inf"
    assert np.all(np.abs(phi) < 2.0), f"out of expected bounds: max={np.max(np.abs(phi))}"


def test_sp_rep_integration_with_upstream():
    """rl/representations/ssp.py::SSPRep is the integration point for the
    DQN path. After the refactor it must construct cleanly from upstream
    HexagonalSSPSpace and produce a 1-D phi of the expected size."""
    _drop_cached("rl.representations.ssp", "rl")
    sys.path.insert(0, REPO)
    from rl.representations.ssp import SSPRep
    rep = SSPRep(
        domain_dim=4,
        length_scale=1.0,
        n_rotates=5,
        n_scales=5,
    )
    x = np.array([0.1, 0.2, -0.3, 0.4])
    phi = rep.map(x)
    assert isinstance(phi, np.ndarray), f"map() returned {type(phi)}, expected ndarray"
    assert phi.ndim == 1, f"map() returned {phi.shape}, expected 1-D"
    expected_dim = 5 * 5 * (4 + 1) * 2 + 1
    assert phi.shape == (expected_dim,), f"phi shape {phi.shape}, expected ({expected_dim},)"


def test_sp_rep_deterministic():
    """Same state must encode to the same vector across calls (no internal
    RNG state in the encoder path that would change between calls)."""
    _drop_cached("rl.representations.ssp", "rl")
    sys.path.insert(0, REPO)
    from rl.representations.ssp import SSPRep
    rep = SSPRep(
        domain_dim=4,
        length_scale=1.0,
        n_rotates=5,
        n_scales=5,
    )
    x = np.array([0.5, -0.2, 0.0, 0.8])
    phi_a = rep.map(x)
    phi_b = rep.map(x)
    np.testing.assert_array_equal(phi_a, phi_b, "non-deterministic encoding")


def test_sp_rep_distinguishes_states():
    """Two different states must encode to different vectors — guards
    against a phase-matrix all-zeros bug."""
    _drop_cached("rl.representations.ssp", "rl")
    sys.path.insert(0, REPO)
    from rl.representations.ssp import SSPRep
    rep = SSPRep(
        domain_dim=4,
        length_scale=1.0,
        n_rotates=5,
        n_scales=5,
    )
    a = rep.map(np.array([0.1, 0.2, -0.3, 0.4]))
    b = rep.map(np.array([0.4, -0.1, 0.2, -0.3]))
    diff = np.linalg.norm(a - b)
    assert diff > 0.1, f"different states encoded to nearly identical vectors: ||a-b||={diff}"


def test_dqn_path_imports_no_nengo():
    """The Stage-0 DQN import chain (the actual code path the smoke test
    and Optuna runs will use) must not pull in Nengo at any step.

    Specifically: importing experiments.trial_cartpole_dqn and exercising
    the four representation constructors (NormalRep, SSPRep, TileCodingRep,
    OneHotRepCP) plus the DQN network must not leave any nengo* module
    in sys.modules."""
    _drop_cached("rl", "experiments", "base_trial")
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "experiments"))

    # The DQN trial imports these directly — exercise the chain.
    from rl.networks.dqnBasic import DQN
    from rl.representations.normal import NormalRep
    from rl.representations.ssp import SSPRep
    from rl.representations.tilecoding import TileCodingRep
    from rl.representations.onehot import OneHotRepCP

    # Also import the trial module itself to verify its import statements
    # don't touch anything Nengo-bound.
    import base_trial
    from experiments.trial_cartpole_dqn import DQNTrial

    # Construct the four representations and the network to ensure no lazy
    # import fires during use.
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    normal = NormalRep(env)
    ssp = SSPRep(domain_dim=4, length_scale=1.0, n_rotates=5, n_scales=5)
    # TileCoding needs an iht_size etc.
    import numpy as np
    tile = TileCodingRep(
        env,
        num_tilings=8, tiles_per_dim=(8, 8, 8, 8), iht_size=4096,
        bounds_low=env.observation_space.low, bounds_high=env.observation_space.high,
    )
    discrete = OneHotRepCP(
        (7, 7, 7, 7), bounds_low=env.observation_space.low, bounds_high=env.observation_space.high,
    )
    dqn = DQN(
        rep=ssp, state_dim=4, n_actions=2, lr=1e-3, gamma=0.99,
        buffer_size=100, batch_size=8, target_update_freq=10,
    )
    # Exercise the network end-to-end.
    phi = ssp.map(np.array([0.1, 0.2, -0.3, 0.4]))
    q = dqn.get_q_values(phi)
    assert q.shape == (2,)
    env.close()

    nengo_modules = [
        name for name in sys.modules.keys()
        if (name == "nengo" or name.startswith("nengo.")
            or name == "nengo_spa" or name.startswith("nengo_spa.")
            or name == "nengo_extras" or name.startswith("nengo_extras."))
    ]
    assert not nengo_modules, (
        f"Nengo modules leaked into the Stage-0 DQN import chain: {nengo_modules}"
    )


def main():
    print("=" * 60)
    print("sspspace parity (upstream + rl/representations/ssp.py)")
    print("=" * 60)
    tests = [
        test_upstream_import_and_api,
        test_upstream_encode_returns_wrapped_ssp,
        test_upstream_ssp_dim_matches_closed_form,
        test_upstream_encoding_is_bounded_real_signal,
        test_sp_rep_integration_with_upstream,
        test_sp_rep_deterministic,
        test_sp_rep_distinguishes_states,
        test_dqn_path_imports_no_nengo,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
            failed.append((t.__name__, e))
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
