"""FHRR / SSP primitives and the three frequency designs for the Stage-0 memory tests.

NumPy only, no torch, no environment.  Frequency designs (docs/10 section 11):

    legacy : the A2C-era encoder, n_scales=8, scales linspace(0.5, 2)   (rl.representations.ssp_legacy)
    current: upstream sspspace HexagonalSSPSpace, n_scales=5            (what the DQN SSPRep uses)
    rff    : dense Gaussian spectrum control, omega ~ N(0, I) / length_scale

All three return a real (D, d) matrix of independent complex-FHRR frequencies (the Hermitian-symmetric
duplicate half and the DC row of the real-valued SSP are dropped), so D is the true complex dimension.
"""

import importlib.util
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load_legacy():
    path = REPO / "rl" / "representations" / "ssp_legacy.py"
    spec = importlib.util.spec_from_file_location("ssp_legacy_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def half_spectrum(P: np.ndarray, check: bool = True) -> np.ndarray:
    """Real-valued SSP phase matrix (2n+1, d) -> the n independent frequencies (n, d)."""
    D = P.shape[0]
    assert D % 2 == 1, "expected odd (DC + symmetric pairs) layout"
    n = (D - 1) // 2
    if check:
        # layout after ifftshift/fft: row k pairs with row D-k (negated), row 0 is DC
        err = np.abs(P[1 : n + 1] + P[D - 1 : n : -1]).max()
        assert err < 1e-8, f"phase matrix not Hermitian-symmetric in the assumed layout (err={err})"
    return P[1 : n + 1].copy()


def make_omega(design: str, d: int, D_target: int, length_scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Return Omega (D, d) already divided by length_scale, with D >= D_target the nearest the design allows."""
    if design == "rff":
        rng = np.random.default_rng(seed)
        return rng.normal(size=(D_target, d)) / length_scale
    if design == "legacy":
        mod = _load_legacy()
        n_scales = 8
        per_rot = (d + 1) * n_scales  # complex frequencies per rotation
        n_rot = max(1, int(np.ceil(D_target / per_rot)))
        P = mod.legacy_hex_phase_matrix(d, n_rot, n_scales, 0.5, 2.0)
        return half_spectrum(P) / length_scale
    if design == "current":
        from sspspace import HexagonalSSPSpace

        n_scales = 5
        per_rot = (d + 1) * n_scales
        n_rot = max(1, int(np.ceil(D_target / per_rot)))
        enc = HexagonalSSPSpace(d, n_rotates=n_rot, n_scales=n_scales, length_scale=length_scale)
        # upstream's phase_matrix is length-scale-independent (the scale is applied in encode()), so divide here
        return half_spectrum(np.asarray(enc.phase_matrix, dtype=float)) / length_scale
    raise ValueError(design)


def phi(Omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """exp(i Omega x). x: (..., d) -> (..., D) complex."""
    return np.exp(1j * (np.asarray(x) @ Omega.T))


def random_codes(rng, n: int, D: int) -> np.ndarray:
    """n random unit-modulus complex vectors (item memory)."""
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(n, D)))


def bind(a, b):
    return a * b


def unbind(m, key):
    return m * np.conj(key)


def cleanup(q: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Index of nearest codebook entry by real inner product. q: (..., D)."""
    return np.argmax(np.real(q @ np.conj(codebook).T), axis=-1)
