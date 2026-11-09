"""SSP / FHRR algebra primitives for the algebra-controller study (docs/11).

Everything is a differentiable torch op on complex tensors of last dim ``D`` (complex dimension; the real-valued
repo convention is ``2*D+1``).

    phi(x)   = exp(i * x @ Omega.T)            fractional binding / SSP of a point x in R^d
    pos(t)   = exp(i * Omega * t)              the d = 1 case, used as *time* (a delay line is a rotation)
    bind     = element-wise product            == translation:   phi(x) * phi(y) = phi(x + y)
    unbind   = product with conjugate          == difference:    phi(y) * conj(phi(x)) = phi(y - x)
    shift    = M * phi(delta)                  re-addresses EVERY item stored in M at once, O(D)
    bundle   = sum, optionally with leak       M <- lam * M + write
    cleanup  = nearest neighbour in a codebook (hard argmax, or softmax / straight-through for gradients)

Frequency design: ``Omega_j ~ U(-pi, pi) / length_scale`` for integer-valued positions, which makes the kernel
``E[cos(Omega d)] = sinc(pi d / length_scale)``: exactly zero at integer lags d != 0 when ``length_scale == 1`` and
smooth in between, so a *continuous* read offset (variant C) has a gradient.  ``kind="gauss"`` gives the Gaussian
random-Fourier-feature spectrum used in RUN_005 for comparison.
"""

import math

import torch

CDT = torch.complex64


def make_omega(dim, d=1, length_scale=1.0, kind="uniform", seed=0, dtype=torch.float32, ls_hi=None, n_bands=6,
               spread=0.05, fine_frac=0.0):
    """Frequencies, shape (dim, d).

    ``kind='uniform'``  U(-pi, pi) / length_scale         -- the historical single-scale draw
    ``kind='gauss'``    N(0, 1) / length_scale            -- Gaussian random-Fourier-feature spectrum (RUN_005 control)
    ``kind='logmix'``   U(-pi, pi) / ls_j, ``log ls_j`` uniform on [log length_scale, log ls_hi]
    ``kind='residue'``  a geometric ladder of ``n_bands`` scales between ``length_scale`` and ``ls_hi``, assigned
                        round-robin, each jittered by +-``spread`` against exact commensurability (a ladder with
                        exact power-of-two ratios repeats: the kernel aliases, which is the failure mode the
                        logmix/residue kinds exist to avoid)

    Why a mixture (handoff 10 phase 1): with a single scale the address basin is about one index wide at
    ``length_scale = 1`` (RUN_008), while a *wide* single scale raises the basin but lowers the ceiling (0.83 at
    width 8 vs 0.997 at width 1).  A mixture has wide components (basin from step 0, no schedule needed) and narrow
    ones (resolution), so discoverability does not have to be bought with a curriculum.  ``length_scale`` is always
    the *finest* scale: it is what sets the resolution.
    """
    g = torch.Generator().manual_seed(int(seed))
    ls0 = float(length_scale)
    if kind in ("uniform", "logmix", "residue") and ls_hi is not None and float(ls_hi) < ls0:
        raise ValueError(f"ls_hi={ls_hi} must be >= length_scale={ls0}")
    if kind == "uniform":
        om = (torch.rand(dim, d, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
        return (om / ls0).to(dtype)
    if kind == "gauss":
        om = torch.randn(dim, d, generator=g, dtype=torch.float64)
        return (om / ls0).to(dtype)
    if kind == "cauchy":
        # docs/11 14.3 axis 2 item 2: the *dead-zone-free* kernel.  With ``om`` standard Cauchy the expected
        # similarity at a distance of d addresses is ``exp(-|d| / length_scale)`` -- strictly positive and monotone
        # in |d|, with no zero anywhere, unlike ``sinc(pi d)`` which is exactly zero at every integer offset (the
        # measured cause of the discovery dead zone, RUN_011 section 3.2).  The tail is heavy, so clip the draws:
        # an unclipped Cauchy frequency would put all the resolution in a few dimensions.
        u = torch.rand(dim, d, generator=g, dtype=torch.float64)
        om = torch.tan(math.pi * (u - 0.5))
        om = om.clamp(-50.0, 50.0)
        return (om / ls0).to(dtype)
    if kind == "logmix":
        if ls_hi is None:
            raise ValueError("kind='logmix' needs ls_hi (the coarsest scale)")
        lo, hi = math.log(ls0), math.log(float(ls_hi))
        ls = torch.exp(lo + (hi - lo) * torch.rand(dim, d, generator=g, dtype=torch.float64))
        if fine_frac > 0:
            # ``fine_frac`` of the dimensions are pinned to the finest scale.  The first measurement of a plain
            # log-uniform mixture (Phase 1 wave 1) showed its *ceiling* is what fails, not its basin: the wide
            # components carry crosstalk into every read.  Pinning most dimensions fine keeps the resolution while a
            # minority supplies the basin.
            keep = torch.rand(dim, d, generator=g, dtype=torch.float64) < float(fine_frac)
            ls = torch.where(keep, torch.full_like(ls, ls0), ls)
    elif kind == "residue":
        if ls_hi is None:
            raise ValueError("kind='residue' needs ls_hi (the coarsest scale)")
        k = max(int(n_bands), 2)
        ladder = torch.exp(torch.linspace(math.log(ls0), math.log(float(ls_hi)), k, dtype=torch.float64))
        band = torch.arange(dim * d, dtype=torch.long) % k
        jit = 1.0 + float(spread) * (torch.rand(dim * d, generator=g, dtype=torch.float64) * 2 - 1)
        ls = (ladder[band] * jit).reshape(dim, d)
    else:
        raise ValueError(kind)
    om = (torch.rand(dim, d, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    return (om / ls).to(dtype)


def phi(x, omega):
    """x: (..., d) -> (..., D) complex unit-modulus.  omega: (D, d)."""
    ang = x.to(omega.dtype) @ omega.T
    return torch.polar(torch.ones_like(ang), ang)


def pos(t, omega1):
    """Time as an SSP. t: (...) real/int, omega1: (D,) or (D, 1) -> (..., D)."""
    om = omega1.reshape(-1)
    ang = t.to(om.dtype)[..., None] * om
    return torch.polar(torch.ones_like(ang), ang)


def bind(a, b):
    return a * b


def unbind(a, b):
    """Returns ``a (*) conj(b)``: removes ``b`` from ``a`` (relative position when both are SSPs)."""
    return a * b.conj()


def shift(mem, delta, omega):
    """Re-address everything in ``mem`` by ``delta`` (shape (..., d)); one multiplication regardless of item count."""
    return mem * phi(delta, omega)


def sim(a, b):
    """Normalised real inner product ``Re<a, b> / D`` (1 for identical unit-modulus vectors)."""
    return (a * b.conj()).real.sum(-1) / a.shape[-1]


def leak_write(mem, write, lam):
    """Bundle with forgetting: ``lam * mem + write``.  ``lam`` scalar or broadcastable (..., 1)."""
    return mem * lam + write


def cleanup_scores(query, codebook):
    """Cosine-style scores of ``query`` (..., D) against ``codebook`` (n, D).  Returns (..., n)."""
    s = torch.einsum("...d,nd->...n", query, codebook.conj()).real
    qn = query.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
    cn = codebook.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    return s / (qn * cn)


def cleanup(query, codebook, temp=None):
    """Nearest codebook entry.  ``temp=None``: hard argmax (returns the entry and its index, no gradient);
    ``temp>0``: softmax(scores / temp) mixture of entries with a straight-through hard forward value."""
    sc = cleanup_scores(query, codebook)
    idx = sc.argmax(-1)
    hard = codebook[idx]
    if temp is None:
        return hard, idx
    soft = torch.softmax(sc / temp, dim=-1).to(codebook.dtype) @ codebook
    return soft + (hard - soft).detach(), idx


def random_codes(n, dim, seed=0):
    """Random unit-modulus item codebook (n, D) complex64."""
    g = torch.Generator().manual_seed(int(seed))
    ang = (torch.rand(n, dim, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    return torch.polar(torch.ones_like(ang), ang).to(CDT)


def complex_to_real(z):
    """(..., D) complex -> (..., 2D) real, for feeding a real-valued network."""
    return torch.cat([z.real, z.imag], dim=-1)
