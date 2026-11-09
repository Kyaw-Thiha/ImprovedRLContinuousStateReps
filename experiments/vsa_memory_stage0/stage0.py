#!/usr/bin/env python
"""Stage-0 encoder-only test battery for the VSA-as-memory direction (docs/10 sections 11-14).

Runs T0 (algebra parity), T1 (kernel shape + long-range floor), T4 (query capacity, the crux),
T6 (degraded-cue recall) and T7 (variable entity count).  Task-dependent tests T2/T3/T5 from
docs/10 are NOT run here: they need a chosen dynamics/task.

No environment, no RL, no GPU.  Usage:

    .venv/bin/python experiments/vsa_memory_stage0/stage0.py            # T0,T1,T4,T6,T7
    .venv/bin/python experiments/vsa_memory_stage0/stage0.py --test T0
    .venv/bin/python experiments/vsa_memory_stage0/stage0.py --quick    # timing calibration
"""

import argparse
import json
import pathlib
import resource
import time

import numpy as np

from vsa import make_omega, phi, random_codes, cleanup

DESIGNS = ["legacy", "current", "rff"]
D_TARGETS = [201, 1001, 1601]
M_SWEEP = [1, 2, 5, 10, 20, 50, 100]
DEPTHS = [1, 2, 3]
CODEBOOK = 512  # fixed distractor count for cleanup in T4/T6


def _rng(*tag):
    return np.random.default_rng(abs(hash(tag)) % (2**32))


def t0_parity(designs, d_list=(1, 2, 4), D_target=1001):
    """phi(x)phi(y) == phi(x+y); phi(y)conj(phi(x)) == phi(y-x); kappa depends only on x-y."""
    out = {}
    for design in designs:
        for d in d_list:
            O = make_omega(design, d, D_target)  # half_spectrum asserts Hermitian symmetry
            rng = _rng("t0", design, d)
            x, y = rng.normal(size=(2, d))
            a = np.abs(phi(O, x) * phi(O, y) - phi(O, x + y)).max()
            b = np.abs(phi(O, y) * np.conj(phi(O, x)) - phi(O, y - x)).max()
            # kappa(x,y) == kappa(x-y): compare against phi(x-y) real part
            k_xy = np.real(np.mean(phi(O, x) * np.conj(phi(O, y))))
            k_d = np.real(np.mean(phi(O, x - y)))
            out[f"{design}_d{d}"] = dict(bind_err=float(a), unbind_err=float(b), kappa_err=float(abs(k_xy - k_d)), D=int(O.shape[0]))
    return out


def kappa_curve(O, deltas, n_dirs=64, rng=None):
    """|| mean over directions of cos(omega . (delta * u)) || for each delta."""
    rng = rng or np.random.default_rng(0)
    d = O.shape[1]
    U = rng.normal(size=(n_dirs, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    out = np.empty(len(deltas))
    for i, delta in enumerate(deltas):
        ph = np.cos(delta * (U @ O.T))  # (n_dirs, D)
        out[i] = np.mean(ph)
    return out


def t1_kernel_floor(designs, d_list=(1, 4), D_target=1001, R=1.0, n_pts=400, sweep=40.0):
    """kappa(delta): working-range similarity, decay half-width, first sub-0.1 radius, and the true floor.

    The sweep must be long enough to see the floor: docs/10 section 10.1 item 3 predicts a ~1/sqrt(D) floor
    at long range, not a decay, and different designs reach it at very different deltas.
    """
    out = {}
    deltas = np.linspace(0.0, sweep * R, n_pts)
    for design in designs:
        for d in d_list:
            O = make_omega(design, d, D_target)
            k = kappa_curve(O, deltas)
            below = np.where(k < 0.5)[0]
            half = float(deltas[below[0]]) if below.size else float("nan")
            b01 = np.where(k < 0.1)[0]
            r01 = float(deltas[b01[0]]) if b01.size else float("nan")
            tail = deltas >= 0.9 * sweep * R
            floor = float(np.mean(np.abs(k[tail])))
            floor_max = float(np.max(np.abs(k[tail])))
            out[f"{design}_d{d}"] = dict(
                D=int(O.shape[0]),
                half_width=half,
                radius_below_0p1=r01,
                floor_mean=floor,
                floor_max=floor_max,
                kappa_at_R=float(np.interp(R, deltas, k)),
                floor_below_0p1=bool(floor < 0.1),
            )
    return out


def _bundled_query(O, d, D, m, depth, rng, codebook_size=CODEBOOK, dropout=0.0):
    """Bundle m chains of `depth` binds (item memory: random unit-modulus keys and values), unbind with the
    query keys, cleanup against a fixed codebook.  Returns retrieval accuracy (fraction of the m items).

    This isolates the algebra's superposition capacity from kernel resolution: values are near-orthogonal
    codes, so the only thing under test is crosstalk (theory: sqrt(D/m), docs/10 sections 10.1/10.4).
    """
    C = max(codebook_size, m)
    keys = random_codes(rng, m * depth, D).reshape(m, depth, D)
    val_codes = random_codes(rng, C, D)
    item_val_idx = rng.choice(C, size=m, replace=False)

    S = np.zeros(D, dtype=complex)
    for i in range(m):
        item = val_codes[item_val_idx[i]].copy()
        for k in range(depth):
            item = item * keys[i, k]
        S += item

    q = np.tile(S, (m, 1))
    for k in range(depth):
        q = q * np.conj(keys[:, k])
    if dropout > 0.0:
        n_zero = int(dropout * D)
        if n_zero:
            for i in range(m):
                q[i, rng.choice(D, size=n_zero, replace=False)] = 0.0
    pred = cleanup(q, val_codes)
    return float(np.mean(pred == item_val_idx))


def t4_query_capacity(designs, D_targets, m_sweep, depths, seeds, d=4):
    """The crux: retrieval accuracy vs bundle size m, chain depth, and D (docs/10 section 10.4/11)."""
    out = {}
    for design in designs:
        for Dt in D_targets:
            O = make_omega(design, d, Dt)
            D = int(O.shape[0])
            for depth in depths:
                accs = []
                for seed in range(seeds):
                    rng = np.random.default_rng(1000 * seed + depth)
                    for m in m_sweep:
                        rng2 = np.random.default_rng(7919 * seed + 13 * m + depth)
                        accs.append((m, _bundled_query(O, d, D, m, depth, rng2)))
                per_m = {m: float(np.mean([a for mm, a in accs if mm == m])) for m in m_sweep}
                theory = {m: float(np.sqrt(D / m) / (1 + np.sqrt(D / m))) for m in m_sweep}
                out[f"{design}_D{D}_depth{depth}"] = dict(D=D, per_m=per_m, theory=theory)
    return out


def t6_degraded_cue(designs, D_targets, seeds, m=10, depth=1, ps=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9), d=4):
    out = {}
    for design in designs:
        for Dt in D_targets:
            O = make_omega(design, d, Dt)
            D = int(O.shape[0])
            curve = {}
            for p in ps:
                accs = []
                for seed in range(seeds):
                    rng = np.random.default_rng(4242 * seed + int(100 * p))
                    accs.append(_bundled_query(O, d, D, m, depth, rng, dropout=p))
                curve[str(p)] = float(np.mean(accs))
            out[f"{design}_D{D}"] = dict(D=D, m=m, depth=depth, accuracy_vs_dropout=curve)
    return out


def _half_width(O, R=1.0, n_pts=200, sweep=40.0):
    deltas = np.linspace(0.0, sweep * R, n_pts)
    k = kappa_curve(O, deltas)
    below = np.where(k < 0.5)[0]
    return float(deltas[below[0]]) if below.size else float(sweep * R)


def t7_variable_entities(designs, D_targets, seeds, Ns=(1, 2, 4, 8), d=2, grid_n=21):
    """N spatial entities bound to roles in one fixed-size vector, each retrieved individually.

    Entities are *continuous* SSP codes of 2-D positions on a grid (docs/10 section 12 rank 2: bind
    role_i (*) phi(pos_i), bundle, unbind with role_i, clean up by nearest neighbour).  Each design's
    length_scale is *resolution-matched* so that one grid step equals the design's kernel half-width
    (kappa ~ 0.5 between adjacent cells); without that matching the SSP codes of nearby cells are nearly
    identical and every design fails for a reason that says nothing about capacity.
    """
    out = {}
    g = np.linspace(0.0, 1.0, grid_n)
    pts = np.array([(a, b) for a in g for b in g])
    step = 1.0 / (grid_n - 1)
    for design in designs:
        for Dt in D_targets:
            O1 = make_omega(design, d, Dt, length_scale=1.0)
            h = _half_width(O1)
            ls = step / h
            O = make_omega(design, d, Dt, length_scale=ls)
            D = int(O.shape[0])
            pos_codes = phi(O, pts)
            per_N = {}
            for N in Ns:
                accs = []
                for seed in range(seeds):
                    rng = np.random.default_rng(555 * seed + N)
                    roles = random_codes(rng, N, D)
                    idx = rng.choice(len(pts), size=N, replace=False)
                    S = np.zeros(D, dtype=complex)
                    for i in range(N):
                        S += roles[i] * pos_codes[idx[i]]
                    pred = np.array(
                        [int(np.argmax(np.real(pos_codes @ np.conj(S * np.conj(roles[i]))))) for i in range(N)]
                    )
                    accs.append(float(np.mean(pred == idx)))
                per_N[N] = float(np.mean(accs))
            out[f"{design}_D{D}"] = dict(D=D, grid=grid_n * grid_n, length_scale=ls, half_width_at_ls1=h, per_N=per_N)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="experiments/vsa_memory_stage0/results.json")
    args = ap.parse_args()

    designs = DESIGNS
    d_targets = [201, 1601] if args.quick else D_TARGETS
    m_sweep = [1, 10, 100] if args.quick else M_SWEEP
    seeds = 2 if args.quick else args.seeds

    tests = ["T0", "T1", "T4", "T6", "T7"] if args.test == "all" else [args.test]
    results = {}
    for t in tests:
        t0 = time.perf_counter()
        if t == "T0":
            r = t0_parity(designs)
        elif t == "T1":
            r = t1_kernel_floor(designs)
        elif t == "T4":
            r = t4_query_capacity(designs, d_targets, m_sweep, DEPTHS, seeds)
        elif t == "T6":
            r = t6_degraded_cue(designs, d_targets, seeds)
        elif t == "T7":
            r = t7_variable_entities(designs, d_targets, seeds)
        else:
            raise SystemExit(f"unknown test {t}")
        el = time.perf_counter() - t0
        results[t] = dict(elapsed_s=el, data=r)
        print(f"\n=== {t} ({el:.1f}s) ===")
        print(json.dumps(r, indent=2, default=float))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    results["_meta"] = dict(peak_rss_mb=rss, quick=args.quick, seeds=seeds, designs=designs)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\npeak RSS {rss:.0f} MB; wrote {out}")


if __name__ == "__main__":
    main()
