#!/usr/bin/env python
"""CartPole-specific encoder diagnostic: is the SSP deficit a length-scale/resolution artefact?

Encoder-only (no RL, no checkpoints).  Answers doc 08 §M3 / hypothesis H2 directly:

  * what feature size is representable along each CartPole axis at the fitted length_scale;
  * whether the encoder can resolve the CartPole value-function cliff at |theta| > 0.209;
  * whether ANY single isotropic scalar can do that without destroying the omega axis
    (this is the "one scalar, two jobs, four heterogeneous dimensions" claim);
  * whether normalising the state per dimension repairs it, and at what cost to omega.

CartPole ranges (gymnasium): x +-4.8, x_dot +-10, theta +-0.418, omega +-10.  The repo feeds SSP the
raw state (`normalize_state` is False in every cell and `domain_bounds_` is never passed).

Usage:
    .venv/bin/python experiments/vsa_memory_stage0/cartpole_ls.py
"""

import json
import pathlib
import sys

import numpy as np

from vsa import make_omega

AXES = ["x", "x_dot", "theta", "omega"]
RANGE = np.array([4.8, 10.0, 0.418, 10.0])
CLIFF = 0.209  # |theta| beyond which CartPole's value function has a hard edge

# fitted length_scale values from the repo's own winners (handoffs/03, outputs/eval_v3 manifests)
FITTED = {
    "legacy(a2c)": [("ssp/none", 0.24773507149141605), ("ssp/value", 0.34330015266171904), ("NNI/representation", 0.527579)],
    "current(dqn)": [("ssp/value/deep", 0.16309222982594232)],
}


def kappa_axis(O, i, deltas):
    """kappa for a displacement purely along axis i: mean_j cos(Omega_ji * delta)."""
    w = O[:, i]
    return np.mean(np.cos(w[None, :] * deltas[:, None]), axis=1)


def halfwidth(O, i, hi, n=400):
    """Delta at which kappa drops below 0.5, or nan if it never does within `hi`."""
    deltas = np.linspace(0.0, hi, n)
    k = kappa_axis(O, i, deltas)
    below = np.where(k < 0.5)[0]
    return float(deltas[below[0]]) if below.size else float("nan")


def axis_resolution(O, i):
    """smallest/largest wavelength representable along axis i: 2*pi / max,min |Omega_ji|.

    NOTE: the smallest representable wavelength is set by max|w|, but kappa is the *mean* over all w,
    so a design with a few high-frequency components and many low-frequency ones can have a fine
    min-wavelength while the kernel the readout actually sees stays coarse.
    """
    wmax = np.abs(O[:, i]).max()
    wmin = np.abs(O[:, i]).min()
    return float(2 * np.pi / wmax) if wmax > 0 else float("inf"), float(2 * np.pi / wmin) if wmin > 0 else float("inf")


def analyse(O, label):
    hw = [halfwidth(O, i, RANGE[i]) for i in range(4)]
    res = [axis_resolution(O, i) for i in range(4)]
    k_cliff = float(kappa_axis(O, 2, np.array([CLIFF]))[0])
    k_extremes = float(kappa_axis(O, 2, np.array([RANGE[2]]))[0])
    return {
        "label": label,
        "D_complex": int(O.shape[0]),
        "ssp_dim_real": int(2 * O.shape[0] + 1),
        "half_width": {a: hw[i] for i, a in enumerate(AXES)},
        "half_width_over_range": {a: hw[i] / RANGE[i] for i, a in enumerate(AXES)},
        "min_wavelength": {a: res[i][0] for i, a in enumerate(AXES)},
        "max_wavelength": {a: res[i][1] for i, a in enumerate(AXES)},
        "kappa_theta_at_cliff": k_cliff,
        "kappa_theta_across_full_range": k_extremes,
        "theta_resolved": bool(k_extremes < 0.5),
    }


def main():
    out = {}
    for design, fits in FITTED.items():
        rep = "legacy" if design.startswith("legacy") else "current"
        for name, ls in fits:
            for Dc in (100, 500):  # 100 complex -> 201 real; 500 -> 1001 real (the repo's search range)
                O = make_omega(rep, 4, Dc, length_scale=ls)
                r = analyse(O, f"{design}/{name}/ls={ls:.4f}/Dreal={2*O.shape[0]+1}")
                r["length_scale"] = ls
                out[r["label"]] = r

    # single-scalar impossibility: pick ls so theta half-width == CLIFF, then see what omega becomes.
    # kappa's half-width scales linearly with length_scale, so measure it once at ls=1 over a wide sweep.
    O1 = make_omega("current", 4, 100, length_scale=1.0)
    hw_theta_at_1 = halfwidth(O1, 2, 400.0)
    if not np.isfinite(hw_theta_at_1):
        forced = {"note": "theta kappa never reaches 0.5 within delta<=400 even at length_scale=1"}
    else:
        # half-width scales linearly with ls, so ls for a target half-width is target/hw_at_1
        ls_target = CLIFF / hw_theta_at_1
        O2 = make_omega("current", 4, 100, length_scale=ls_target)
        forced = {
            "theta_half_width_at_ls1": float(hw_theta_at_1),
            "ls_forced_for_theta_cliff": float(ls_target),
            "theta_half_width_after": halfwidth(O2, 2, RANGE[2]),
            "omega_half_width_after": halfwidth(O2, 3, RANGE[3]),
            "kappa_omega_at_step_1.0": float(kappa_axis(O2, 3, np.array([1.0]))[0]),
            "kappa_x_at_step_1.0": float(kappa_axis(O2, 0, np.array([1.0]))[0]),
            "note": "omega/x move ~1 unit per CartPole step; a kappa near 1 there means those axes are over-resolved into near-noise",
        }

    # normalised variant: equivalent to per-dimension ls_i = ls * range_i
    norm = {}
    for ls in (0.16309222982594232, 0.24773507149141605, 0.5):
        base = make_omega("legacy", 4, 100, length_scale=1.0)
        O = base / (ls * RANGE)[None, :]
        r = analyse(O, f"normalised/legacy/ls={ls}")
        r["length_scale"] = ls
        norm[r["label"]] = r

    result = {"raw": out, "single_scalar_impossibility": forced, "normalised": norm}
    dest = pathlib.Path(__file__).with_name("cartpole_ls.json")
    dest.write_text(json.dumps(result, indent=2, default=float))

    print("=== raw state, fitted length_scale (the config the repo actually ran) ===")
    for lbl, r in out.items():
        print(f"\n{lbl}")
        print(f"  half-width (kappa=0.5) per axis: " + " ".join(f"{a}={r['half_width'][a]:.3f}" for a in AXES))
        print(f"  as fraction of the axis range:   " + " ".join(f"{a}={r['half_width_over_range'][a]:.3f}" for a in AXES))
        print(f"  min representable wavelength:    " + " ".join(f"{a}={r['min_wavelength'][a]:.3f}" for a in AXES))
        print(f"  kappa(theta) at the cliff (0.209): {r['kappa_theta_at_cliff']:.3f}   across full range: {r['kappa_theta_across_full_range']:.3f}   theta resolved: {r['theta_resolved']}")
    print("\n=== single-scalar impossibility ===")
    print(json.dumps(forced, indent=2))
    print("\n=== normalised state (per-dimension effective length scale) ===")
    for lbl, r in norm.items():
        print(f"{lbl}: " + " ".join(f"{a}={r['half_width'][a]:.3f}" for a in AXES) + f"  theta_resolved={r['theta_resolved']}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
