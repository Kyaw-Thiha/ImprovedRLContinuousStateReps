"""Gate for ssp_ops: algebra parity (T0) and a hand-wired SSP-time delay readout on POPGym RepeatPrevious.

Run:  OMP_NUM_THREADS=1 ../../.venv/bin/python selftest.py

Two checks, both must pass before any learning code is trusted.

1. **T0 parity** (RUN_005 measured 3e-15): ``phi(x)*phi(y) == phi(x+y)``, ``phi(y)*conj(phi(x)) == phi(y-x)``, and
   the kernel depends only on ``x-y``.  Threshold 1e-12.
2. **The delay readout works as a policy.**  Memory ``M <- M + item_i * pos(t)`` with real SSP time
   ``pos(t) = exp(i*Omega*t)``; unbinding at ``t - (k-1)`` and cleaning up against the item codebook must recover
   the item from ``k-1`` observation steps ago.  RUN_006 measured +0.982 policy return with a random position
   *table* at dim 256.  If the SSP-time readout is not near-oracle, the algebra or the lag convention is wrong and
   nothing downstream means anything.

The episode loop (act -> step -> observe the NEW obs) mirrors ``popgym_vsa/selftest.py`` exactly, so the oracle
return and the agreement metric are directly comparable with RUN_006.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "popgym_vsa"))  # task registry, single source of truth

import ssp_ops as S  # noqa: E402
from tasks import make_env, oracle_for, spec_from_env  # noqa: E402

OUT = HERE / "selftest.json"


def check_t0(dims=(1, 2, 4), dims_c=(201, 1001), dtype=torch.float64):
    """Returns dict of max relative error for bind, unbind, kappa-depends-only-on-difference.

    ``dtype`` defaults to float64 to match RUN_005's 3e-15.  In float32 the *algebra* is still exact but the
    rounding error is set by the phase magnitude ``|Omega @ x|`` (up to ~12 for d=4, length_scale=1), so it lands
    near 1e-6 rather than 1e-15; that number is reported separately rather than treated as a failure.
    """
    errs = {"bind": 0.0, "unbind": 0.0, "kappa": 0.0}
    for d in dims:
        for D in dims_c:
            om = S.make_omega(D, d, length_scale=1.0, seed=0, dtype=dtype)
            x, y, z = (t.to(dtype) for t in (torch.randn(16, d), torch.randn(16, d), torch.randn(16, d)))
            px, py, pxy = S.phi(x, om), S.phi(y, om), S.phi(x + y, om)
            scale = float(pxy.abs().max())
            errs["bind"] = max(errs["bind"], float((S.bind(px, py) - pxy).abs().max()) / scale)
            errs["unbind"] = max(errs["unbind"], float((S.unbind(py, px) - S.phi(y - x, om)).abs().max()) / scale)
            e = float((S.sim(px, py) - S.sim(S.phi(x - y + z, om), S.phi(z, om))).abs().max())
            errs["kappa"] = max(errs["kappa"], e)
    return errs


class SSPDelayPolicy:
    """Hard-wired delay readout.  ``M <- M + item_i * pos(t)``; read by unbinding ``pos(t - lag)`` and cleaning up."""

    def __init__(self, n_items, lag, dim, lam=1.0, lscale=1.0, seed=0):
        self.item = S.random_codes(n_items, dim, seed=seed)
        self.omega = S.make_omega(dim, 1, length_scale=lscale, seed=seed)
        self.lag = int(lag)
        self.lam = float(lam)
        self.M = torch.zeros_like(self.item[0])

    def reset(self):
        self.M = torch.zeros_like(self.item[0])

    def act(self, i, t):
        p = S.pos(torch.tensor([float(t)]), self.omega)[0]
        self.M = S.leak_write(self.M, self.item[i] * p, self.lam)
        if t < self.lag:
            return 0  # no information yet; same as the oracle's fallback action
        q = S.pos(torch.tensor([float(t - self.lag)]), self.omega)[0]
        return int(S.cleanup_scores(S.unbind(self.M, q), self.item).argmax().item())


def episode(env, spec, choose, lag, seed):
    """Runs one episode with ``choose(obs, t, oracle) -> action``.  Returns (env_return, oracle_agreement_n)."""
    obs, _ = env.reset(seed=seed)
    o = oracle_for(spec)
    o.reset()
    t, R, hits, n = 0, 0.0, 0, 0
    while True:
        a = choose(obs, t, o)
        ao = o.act(obs, t, env)
        if t >= lag:
            hits += int(a == ao)
            n += 1
        obs, r, te, tr, _ = env.step(a)
        o.observe(obs)
        R += r
        t += 1
        if te or tr:
            break
    return R, hits / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=256, help="unused default; --dims drives the sweep")
    p.add_argument("--dims", type=int, nargs="*", default=[128, 256, 512])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--diff", default="Easy")
    p.add_argument("--lag-offset", type=int, default=-1, help="lag = k + offset; -1 is the fixed convention (k-1)")
    args = p.parse_args()

    res = {}
    res["t0_f64"] = check_t0()                                   # the gate (RUN_005 used float64)
    res["t0_f32"] = check_t0(dtype=torch.float32)                 # reported: set by phase magnitude |Omega x|
    res["t0_pass"] = all(e < 1e-12 for e in res["t0_f64"].values())

    env = make_env("RepeatPrevious", args.diff)
    spec = spec_from_env("RepeatPrevious", args.diff, env)
    spec["lag"] = max(spec["k"] + args.lag_offset, 0)
    lag, k = spec["lag"], spec["k"]
    res["env"] = {kk: spec[kk] for kk in ("fam", "diff", "k", "lag", "n_items", "ep_len", "act_dim")}
    res["lag_convention"] = f"lag = k{args.lag_offset:+d} = {lag} (k counts back including the current card)"

    # Reference points: the scripted oracle (+1.000 exactly) and a random policy (about -0.5), so the readout's
    # number is interpretable.  Mirrors RUN_006 headroom.
    seeds = range(args.episodes)
    orc = [episode(env, spec, lambda o, t, ora: ora.act(o, t, env), lag, 1000 + s)[0] for s in seeds]
    rnd = []
    for s in seeds:
        rng = np.random.default_rng(2000 + s)
        rnd.append(episode(env, spec, lambda o, t, ora: int(rng.integers(spec["act_dim"])), lag, 1000 + s)[0])
    res["oracle_return"], res["random_return"] = float(np.mean(orc)), float(np.mean(rnd))

    rows = []
    for D in args.dims:
        pol = SSPDelayPolicy(spec["n_items"], lag, D)
        pol.reset()
        rr, ag = [], []
        for s in seeds:
            pol.reset()
            r, a = episode(env, spec, lambda o, t, ora, pol=pol: pol.act(int(np.atleast_1d(o)[0]), t), lag, 1000 + s)
            rr.append(r)
            ag.append(a)
        rows.append({"dim": D, "policy": float(np.mean(rr)), "policy_std": float(np.std(rr)),
                     "agreement": float(np.mean(ag))})
        print(f"dim={D:5d}  ssp-delay={np.mean(rr):+.3f}+-{np.std(rr):.3f}  agreement={np.mean(ag):.3f}", flush=True)
    res["delay"] = rows
    res["delay_pass"] = bool(rows[0]["agreement"] > 0.9) if rows else False

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k_: v for k_, v in res.items() if not isinstance(v, list)}, indent=2))
    print(f"-> {OUT.name}")
    assert res["t0_pass"], f"T0 parity FAILED: {res['t0']}"
    assert res["oracle_return"] > 0.9, f"oracle baseline is broken: {res['oracle_return']}"
    assert res["delay_pass"], f"SSP delay readout is not near-oracle: {rows}"


if __name__ == "__main__":
    main()
