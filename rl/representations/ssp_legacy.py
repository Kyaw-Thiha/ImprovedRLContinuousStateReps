"""Legacy-compatible hexagonal SSP encoder for the A2C port (NumPy only, no Nengo).

The original A2C results (outputs/hparam_search/best/ssp_*.yaml) were produced with the vendored
`sspspace.HexagonalSSPSpace` as configured by the old `SSPRep`:

    n_scales = 8 (SSPRep default; the old trial never passed it), scale_min = 0.5, scale_max = 2,
    scales = linspace(scale_min, scale_max, n_scales), length_scale and n_rotates from the search.

This differs from both the Stage-0 `SSPRep` (upstream `sspspace`, n_scales=5, scales in (0, 1)) and
from upstream's `HexagonalSSPSpace`, which cannot be configured to reproduce it. So the construction
is re-implemented here, clean-room, in plain NumPy/SciPy:

    1. hexagonal simplex phases, scaled by `scales`, rotated by `n_rotates` SO(d) matrices
       (scipy-1.7.3 special_ortho_group with random_state=1, re-implemented in `legacy_special_ortho_rvs`
       because newer scipy draws different rotations for the same seed);
    2. axis vectors via a Hermitian-symmetric inverse FFT, then the phase matrix is RECOVERED as
       -1j*log(fft(axes)) -- the principal log, so phases are wrapped to (-pi, pi]. That wrap is part
       of the legacy behavior and is kept on purpose;
    3. encode(x) = real(ifft(exp(1j * phase_matrix @ (x / length_scale)))).

`domain_bounds` of the old class never affected encode (plotting/decoding only) and is not needed.
Parity with the vendored implementation is checked in tests/test_a2c_port.py.
"""

import numpy as np


def legacy_special_ortho_rvs(dim: int, size: int, seed: int = 1) -> np.ndarray:
    """`scipy.stats.special_ortho_group.rvs(dim, size, random_state=seed)` as implemented in scipy 1.7.3.

    The legacy A2C results were produced with scipy 1.7.3 (requirements.txt). Newer scipy (>= ~1.10) draws a
    DIFFERENT random rotation for the same seed, which changes the SSP phases and therefore the kernel, so
    the current scipy cannot be used for the legacy encoder. This is the 1.7.3 Householder algorithm on a
    single `numpy.random.RandomState(seed)` stream (stable across numpy versions). Returns (size, dim, dim).
    """
    rs = np.random.RandomState(seed)
    mats = []
    for _ in range(size):
        H = np.eye(dim)
        D = np.empty((dim,))
        for n in range(dim - 1):
            x = rs.normal(size=(dim - n,))
            norm2 = np.dot(x, x)
            x0 = x[0].item()
            D[n] = np.sign(x[0]) if x[0] != 0 else 1
            x[0] += D[n] * np.sqrt(norm2)
            x /= np.sqrt((norm2 - x0**2 + x[0] ** 2) / 2.0)
            H[:, n:] -= np.outer(np.dot(H[:, n:], x), x)  # Householder transformation
        D[-1] = (-1) ** (dim - 1) * D[:-1].prod()
        mats.append((D * H.T).T)
    return np.array(mats)


def _axes_from_phases(K: np.ndarray) -> np.ndarray:
    d = K.shape[0]
    F = np.ones((d * 2 + 1, K.shape[1]), dtype="complex")
    F[0:d, :] = np.exp(1.0j * K)
    F[-d:, :] = np.flip(np.conj(F[0:d, :]), axis=0)
    return np.fft.ifft(np.fft.ifftshift(F, axes=0), axis=0).real


def legacy_hex_phase_matrix(domain_dim: int, n_rotates: int, n_scales: int, scale_min: float, scale_max: float):
    """(ssp_dim, domain_dim) phase matrix, identical to the vendored HexagonalSSPSpace."""
    phases_hex = np.hstack(
        [
            np.sqrt(1 + 1 / domain_dim) * np.identity(domain_dim) - (domain_dim ** (-3 / 2)) * (np.sqrt(domain_dim + 1) + 1),
            (domain_dim ** (-1 / 2)) * np.ones((domain_dim, 1)),
        ]
    ).T
    scales = np.linspace(scale_min, scale_max, n_scales)
    phases_scaled = np.vstack([phases_hex * i for i in scales])
    if n_rotates == 1:
        rotated = phases_scaled
    elif domain_dim == 1:
        scales = np.linspace(scale_min, scale_max, n_scales + n_rotates)
        rotated = np.vstack([phases_hex * i for i in scales])
    elif domain_dim == 2:
        angles = np.linspace(0, 2 * np.pi / 3, n_rotates, endpoint=False)
        R = np.stack(
            [np.stack([np.cos(angles), -np.sin(angles)], axis=1), np.stack([np.sin(angles), np.cos(angles)], axis=1)],
            axis=1,
        )
        rotated = (R @ phases_scaled.T).transpose(0, 2, 1).reshape(-1, domain_dim)
    else:
        R = legacy_special_ortho_rvs(domain_dim, n_rotates, seed=1)
        rotated = (R @ phases_scaled.T).transpose(0, 2, 1).reshape(-1, domain_dim)
    axes = _axes_from_phases(rotated)
    return (-1.0j * np.log(np.fft.fft(axes, axis=0))).real


class _EncoderView:
    """Stand-in for the `sspspace` SSPEncoder surface the eval code reads (`phase_matrix`, `length_scale`,
    `ssp_dim`, `domain_dim`), so policy_io / deployment code that looks at `rep._encoder` works unchanged."""

    def __init__(self, phase_matrix, length_scale, domain_dim):
        self.phase_matrix = phase_matrix
        self.length_scale = length_scale * np.ones((domain_dim, 1))
        self.ssp_dim = int(phase_matrix.shape[0])
        self.domain_dim = int(domain_dim)


class LegacySSPRep:
    """State encoder with the same surface as the other reps: size_out, map(), get_state()."""

    def __init__(
        self,
        domain_dim: int,
        length_scale: float = 1.0,
        n_rotates: int = 5,
        n_scales: int = 8,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        phase_matrix=None,
    ):
        self.domain_dim = int(domain_dim)
        self.length_scale = float(length_scale)
        self.n_rotates = int(n_rotates)
        self.n_scales = int(n_scales)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if phase_matrix is None:
            phase_matrix = legacy_hex_phase_matrix(self.domain_dim, self.n_rotates, self.n_scales, scale_min, scale_max)
        self._encoder = _EncoderView(np.asarray(phase_matrix, dtype=np.float64), self.length_scale, self.domain_dim)
        self.size_out = int(self._encoder.ssp_dim)
        # x @ diag(1/ls) in the vendored code == x * (1/ls) elementwise (kept as a product, not a quotient,
        # so the rounding is the same).
        self._inv_ls = 1.0 / (self.length_scale * np.ones(self.domain_dim))

    @property
    def phase_matrix(self):
        return self._encoder.phase_matrix

    def map(self, state):
        x = np.asarray(state, dtype=float).reshape(-1) * self._inv_ls
        return np.fft.ifft(np.exp(1.0j * (self._encoder.phase_matrix @ x))).real

    def get_state(self, state, env=None):
        return state
