"""Torch port of the HexSSP encoder (GPU-batched), plus a random-Fourier control.

The phase matrix is built once by the upstream `sspspace` package (`HexagonalSSPSpace`, pinned SHA in
requirements-stage0.txt of the parent repo) and registered as a buffer. `forward` reproduces sspspace's
`SSPEncoder.encode` exactly (ifft over the encoded axis, real part, no renormalisation) so that
`tests/test_ssp_parity.py` can assert `np.allclose`; `encode` applies the sqrt(D) rescaling used as an observation
feature so that feature magnitudes are O(1) for a linear readout.

Scaling is per-dimension (`scale` vector), which is the H2 fix from RUN_005: an isotropic length scale cannot
resolve mixed-range state dimensions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _hex_phase_matrix(domain_dim: int, n_rotates: int, n_scales: int, kernel: str) -> np.ndarray:
  from sspspace import HexagonalSSPSpace

  enc = HexagonalSSPSpace(
    domain_dim=domain_dim,
    n_rotates=n_rotates,
    n_scales=n_scales,
    kernel=kernel,
    length_scale=1.0,
  )
  return np.asarray(enc.phase_matrix, dtype=np.float32)


def _gaussian_phase_matrix(domain_dim: int, ssp_dim: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  # conjugate-symmetric random frequencies, same layout as sspspace's `conjugate_symmetry`
  d = (ssp_dim - 1) // 2
  K = rng.normal(size=(d, domain_dim)).astype(np.float32)
  F = np.zeros((2 * d + 1, domain_dim), dtype=np.float32)
  F[1 : d + 1] = K
  F[-1 : -d - 1 : -1] = -K
  return F


class HexSSP(nn.Module):
  """Hexagonal spatial semantic pointer encoder, batched over environments."""

  def __init__(
    self,
    domain_dim: int,
    scale: torch.Tensor | list[float] | float = 1.0,
    length_scale: float = 1.0,
    n_rotates: int = 8,
    n_scales: int = 8,
    kernel: str = "hypergeom",
    rff: bool = False,
    rff_dim: int | None = None,
    seed: int = 0,
  ) -> None:
    super().__init__()
    if rff:
      A = _gaussian_phase_matrix(domain_dim, rff_dim or (2 * n_rotates * n_scales + 1), seed)
    else:
      A = _hex_phase_matrix(domain_dim, n_rotates, n_scales, kernel)
    self.register_buffer("A", torch.from_numpy(A))
    s = torch.as_tensor(scale, dtype=torch.float32)
    if s.dim() == 0:
      s = s.repeat(domain_dim)
    self.register_buffer("scale", s)
    self.length_scale = float(length_scale)
    self.out_dim = int(A.shape[0])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """sspspace-exact encoding (used by the parity test)."""
    z = (x / self.scale) / self.length_scale
    ph = z @ self.A.T  # (B, D)
    return torch.fft.ifft(torch.exp(1j * ph.to(torch.complex64)), dim=-1).real

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Observation feature: unit-RMS encoding."""
    return self.forward(x) * (self.out_dim**0.5)
