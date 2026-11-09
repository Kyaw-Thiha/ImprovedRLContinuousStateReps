#!/usr/bin/env python
"""Regenerate the RUN_005 supplementary measurements that the CLI battery does not cover.

Everything here is also quoted in runs/RUN_005_hypotesting.md; it lives in its own script so every number
in that log is reproducible from an artifact rather than from terminal history.

  * t4_endpoints  - T4 query capacity at the repo's ACTUAL dimension endpoints
                    (real 201 = 100 complex, real 1001 = 500 complex), plus depth 1 vs 3
  * t7_naive      - T7 with the unmatched length_scale=1, including the off-diagonal kappa of the
                    441-cell position grid (the "why it fails" diagnostic)
  * t7_matched    - T7 with length_scale matched to the grid step, per design
  * cartpole_fill - kappa along theta at each winner's own length_scale, at the repo's smallest real dim

Usage:
    .venv/bin/python experiments/vsa_memory_stage0/extras.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cartpole_ls import CLIFF, RANGE, kappa_axis  # noqa: E402
from stage0 import M_SWEEP, t4_query_capacity  # noqa: E402
from vsa import make_omega, phi, random_codes  # noqa: E402

DESIGNS = ["legacy", "current", "rff"]
GRID_N = 21
GRID = np.linspace(0.0, 1.0, GRID_N)
PTS = np.array([(a, b) for a in GRID for b in GRID])
SEEDS = 10

# (design, length_scale, tag) for every winner whose fitted scale is quoted in the log
WINNERS = [
    ("current", 0.14765906199753728, "dqn_ssp_none_linear"),
    ("current", 0.16309222982594232, "dqn_ssp_value_deep"),
    ("legacy", 0.24773507149141605, "a2c_ssp_none"),
    ("legacy", 0.34330015266171904, "a2c_ssp_value"),
    ("legacy", 0.527579, "a2c_nni_representation"),
]


def t7_case(design, Dc, ls=None, match=False):
    O1 = make_omega(design, 2, Dc, length_scale=1.0)
    if match:
        deltas = np.linspace(0.0, 40.0, 200)
        # match on the first axis's half-width (the designs are near-isotropic across axes at this d)
        w = O1[:, 0]
        kx = np.mean(np.cos(deltas[:, None] * w[None, :]), axis=1)
        below = np.where(kx < 0.5)[0]
        hw = float(deltas[below[0]]) if below.size else float("nan")
        ls = (1.0 / (GRID_N - 1)) / hw
    O = make_omega(design, 2, Dc, length_scale=ls)
    D = int(O.shape[0])
    PC = phi(O, PTS)
    sim = np.real(PC @ np.conj(PC).T) / D
    off = sim - np.eye(len(PTS))
    offdiag = float(off.sum() / (len(PTS) ** 2 - len(PTS)))
    per_N = {}
    for N in (1, 2, 4, 8):
        acc = []
        for s in range(SEEDS):
            rng = np.random.default_rng(555 * s + N)
            roles = random_codes(rng, N, D)
            idx = rng.choice(len(PTS), size=N, replace=False)
            S = sum(roles[i] * PC[idx[i]] for i in range(N))
            pred = [int(np.argmax(np.real(PC @ np.conj(S * np.conj(roles[i]))))) for i in range(N)]
            acc.append(float(np.mean(np.array(pred) == idx)))
        per_N[N] = round(float(np.mean(acc)), 2)
    return {"Dc": D, "ssp_dim_real": 2 * D + 1, "length_scale": ls, "offdiag_kappa": round(offdiag, 3), "per_N": per_N}


def main():
    out = {}

    out["t4_endpoints"] = t4_query_capacity(
        ["legacy", "current"], [100, 500], M_SWEEP, [1, 3], SEEDS
    )

    out["t7_naive"] = {f"{d}_Dc{Dc}": t7_case(d, Dc, ls=1.0) for d in DESIGNS for Dc in (100, 500)}
    out["t7_matched"] = {f"{d}_Dc{Dc}": t7_case(d, Dc, match=True) for d in DESIGNS for Dc in (100, 500)}

    out["cartpole_fill"] = {}
    for design, ls, tag in WINNERS:
        O = make_omega(design, 4, 100, length_scale=ls)
        out["cartpole_fill"][tag] = {
            "design": design,
            "length_scale": ls,
            "Dc": int(O.shape[0]),
            "ssp_dim_real": 2 * int(O.shape[0]) + 1,
            "kappa_theta_at_cliff": round(float(kappa_axis(O, 2, np.array([CLIFF]))[0]), 3),
            "kappa_theta_across_range": round(float(kappa_axis(O, 2, np.array([RANGE[2]]))[0]), 3),
        }

    dest = pathlib.Path(__file__).with_name("results_extras.json")
    dest.write_text(json.dumps(out, indent=2, default=float))

    print("t4 endpoints (real 201 / real 1001):")
    for k, v in sorted(out["t4_endpoints"].items()):
        print(f"  {k:22s} Dc={v['D']:4d} " + " ".join(f"m{m}={v['per_m'][m]:.2f}" for m in M_SWEEP))
    print("t7 naive (length_scale=1):")
    for k, v in out["t7_naive"].items():
        print(f"  {k:14s} offdiag_kappa={v['offdiag_kappa']:+.3f} per_N={v['per_N']}")
    print("t7 matched:")
    for k, v in out["t7_matched"].items():
        print(f"  {k:14s} ls={v['length_scale']:.4f} per_N={v['per_N']}")
    print("cartpole winners:")
    for k, v in out["cartpole_fill"].items():
        print(f"  {k:26s} ls={v['length_scale']:.4f} k_th(cliff)={v['kappa_theta_at_cliff']:.3f} k_th(range)={v['kappa_theta_across_range']:.3f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
