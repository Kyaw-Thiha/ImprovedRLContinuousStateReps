"""Feature maps on the continuous state coordinates x in [0, 1]^k (all return a (S, D) matrix for given states)."""

from __future__ import annotations

import numpy as np


def raw(X: np.ndarray, bias: bool = False) -> np.ndarray:
  F = X.copy()
  return np.concatenate([F, np.ones((len(X), 1))], 1) if bias else F


def tile(X: np.ndarray, tilings: int = 8, tiles: int = 6, seed: int = 0) -> np.ndarray:
  """Tile coding with offset grids. One-hot per tiling, so the feature vector can represent a constant
  (every tiling is active exactly once)."""
  rng = np.random.default_rng(seed)
  S, k = X.shape
  D = tilings * tiles**k
  F = np.zeros((S, D))
  for t in range(tilings):
    off = rng.uniform(0, 1.0 / tiles, size=k)
    cell = np.clip(np.floor((X + off) * tiles / (1 + 1.0 / tiles)).astype(int), 0, tiles - 1)
    flat = np.zeros(S, dtype=int)
    for j in range(k):
      flat = flat * tiles + cell[:, j]
    F[np.arange(S), t * tiles**k + flat] = 1.0
  return F


def hexssp(X: np.ndarray, length_scale: float = 0.3, n_rotates: int = 4, n_scales: int = 4) -> np.ndarray:
  """Hexagonal SSP features (sspspace), unit-RMS per coordinate. Real-valued, no bias unit."""
  from sspspace import HexagonalSSPSpace

  enc = HexagonalSSPSpace(domain_dim=X.shape[1], n_rotates=n_rotates, n_scales=n_scales, length_scale=1.0)
  A = np.asarray(enc.phase_matrix)
  ph = (X / length_scale) @ A.T
  F = np.fft.ifft(np.exp(1j * ph), axis=-1).real
  return F * np.sqrt(F.shape[1])


def rff(X: np.ndarray, dim: int, length_scale: float = 0.3, seed: int = 0) -> np.ndarray:
  """Random Fourier features at a given dimension (cos(Wx + b)), no bias unit."""
  rng = np.random.default_rng(seed)
  W = rng.normal(size=(X.shape[1], dim)) / length_scale
  b = rng.uniform(0, 2 * np.pi, size=dim)
  return np.sqrt(2.0) * np.cos(X @ W + b)


def make(name: str, X: np.ndarray, seed: int = 0) -> np.ndarray:
  if name == "raw":
    return raw(X, bias=False)
  if name == "raw_bias":
    return raw(X, bias=True)
  if name == "tile":
    return tile(X, seed=seed)
  if name == "hexssp":
    return hexssp(X)
  if name == "rff":
    D = hexssp(X[:2]).shape[1]
    return rff(X, D, seed=seed)
  raise ValueError(name)
