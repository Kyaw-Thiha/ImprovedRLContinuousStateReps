"""Encoder-only sufficiency check: run the VSA readouts *directly as policies* and compare to the scripted oracle.

This is the cheapest falsifier in the folder.  If the running bundle's readout cannot answer the task when it is
plugged straight into an argmax policy, no amount of RL will rescue it; if it can, then any later RL deficit is a
learning problem rather than a representation problem (the role T0/T4 played in RUN_005).

Two findings this file is built to expose:

1. **The count readout is only as exact as the codebook.**  Random phasors have pairwise overlap
   ``rho_ij ~ N(0, 1/(2D))``, so the estimate carries error ``sum_{i != q} counts_i * rho_iq`` -- it grows with the
   total number of items ever bundled.  A block-orthogonal codebook (``--code block``) has ``rho = 0`` exactly, and
   the bundle *is* the count vector.  Both arms are reported, and the random arm is swept over ``D`` so the
   ``1/sqrt(D)`` law is visible rather than asserted.
2. **RepeatPrevious needs lag k-1, not k.**  See ``tasks.spec_from_env``.

    python experiments/popgym_vsa/selftest.py
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from memory import TorchMemory
from tasks import make_env, oracle_for, spec_from_env

OUT = pathlib.Path(__file__).with_name("selftest.json")
ROWS = [("RepeatPrevious", "Easy"), ("RepeatPrevious", "Medium"), ("RepeatPrevious", "Hard"),
        ("CountRecall", "Easy"), ("CountRecall", "Medium"), ("CountRecall", "Hard")]


def _mem_for(spec, dim, code):
    k_max = spec["ep_len"] + spec["lag"] + 2
    return TorchMemory(spec["n_items"], dim, k_max, lag=spec["lag"], code=code, seed=0)


def scripted_returns(fam, diff, dim, code, episodes, seed0=0):
    """Returns (vsa_policy_return, its std, oracle return, action agreement)."""
    spec = spec_from_env(fam, diff, make_env(fam, diff))
    mem = _mem_for(spec, dim, code)
    vsa, agree, orc = [], [], []
    for ep in range(episodes):
        env = make_env(fam, diff)
        obs, _ = env.reset(seed=seed0 + ep)
        o = oracle_for(spec)
        o.reset()
        state = mem.zero_state(1)
        t, Rv, hits, n = 0, 0.0, 0, 0
        while True:
            obs_t = torch.as_tensor(np.atleast_1d(obs).reshape(1, -1)).long()
            t_vec = torch.as_tensor([t]).long()
            with torch.no_grad():
                if fam == "CountRecall":
                    est, state = mem.count_step(state, obs_t)
                    av = int(round(float(est)))
                else:
                    scores, state = mem.delay_step(state, obs_t, t_vec)
                    av = int(scores.argmax(-1)) if t >= spec["lag"] else 0
            av = max(0, min(av, spec["act_dim"] - 1))
            ao = o.act(obs, t, env)
            if fam == "RepeatPrevious":
                if t >= spec["lag"]:
                    hits += int(av == ao)
                    n += 1
            else:
                hits += int(av == ao)
                n += 1
            obs, r, te, tr, _ = env.step(av)
            o.observe(obs)
            Rv += r
            t += 1
            if te or tr:
                break
        vsa.append(Rv)
        agree.append(hits / max(n, 1))
    # the oracle needs its own env pass (its actions differ)
    for ep in range(episodes):
        env = make_env(fam, diff)
        obs, _ = env.reset(seed=seed0 + ep)
        o = oracle_for(spec)
        o.reset()
        t, R = 0, 0.0
        while True:
            a_o = o.act(obs, t, env)
            obs, r, te, tr, _ = env.step(a_o)
            o.observe(obs)
            R += r
            t += 1
            if te or tr:
                break
        orc.append(R)
    return float(np.mean(vsa)), float(np.std(vsa)), float(np.mean(orc)), float(np.mean(agree))


def count_readout_error(n_items, dim, code, stream=30, trials=400, seed=0):
    """Env-free: bundle `stream` random items, then measure the rounding error of the count readout."""
    rng = np.random.default_rng(seed)
    mem = TorchMemory(n_items, dim, k_max=1, code=code, seed=seed)
    m = mem.zero_state(trials)
    counts = np.zeros((trials, n_items), dtype=np.int64)
    for _ in range(stream):
        v = torch.as_tensor(rng.integers(0, n_items, trials)).long()
        m = m + mem.item[v]
        counts[np.arange(trials), v.numpy()] += 1
    errs, exact = [], 0
    for q in range(n_items):
        qt = torch.full((trials,), q).long()
        est = (m * mem.item[qt].conj()).real.sum(-1) / mem.norm
        errs.append(np.abs(est.numpy() - counts[:, q]))
        exact += int((est.round().long().numpy() == counts[:, q]).sum())
    e = np.concatenate(errs)
    return float(exact / (trials * n_items)), float(e.mean())


def delay_readout_error(n_items, dim, lag, ep_len, episodes=200, code="random", seed=0):
    """Env-free: stream random items, check that unbinding at t-lag recovers the item from lag steps ago."""
    rng = np.random.default_rng(seed)
    mem = TorchMemory(n_items, dim, ep_len + lag + 2, lag=lag, code=code, seed=seed)
    hits = tot = 0
    for _ in range(episodes):
        items = rng.integers(0, n_items, ep_len)
        state = mem.zero_state(1)
        for t in range(ep_len):
            it = torch.as_tensor([[items[t]]]).long()
            scores, state = mem.delay_step(state, it, torch.as_tensor([t]).long())
            if t >= lag:
                hits += int(int(scores.argmax(-1)) == items[t - lag])
                tot += 1
    return hits / max(tot, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--dims", type=int, nargs="+", default=[256])
    p.add_argument("--codes", nargs="+", default=["random", "block"])
    p.add_argument("--sweep", type=int, nargs="+", default=[64, 256, 1024, 4096])
    a = p.parse_args()

    res = {"scripted": {}, "count_readout": {}, "delay_readout": {}}
    for code in a.codes:
        for dim in a.dims:
            for fam, diff in ROWS:
                key = f"{fam}{diff}/d{dim}/{code}"
                v, sd, o, agree = scripted_returns(fam, diff, dim, code, a.episodes)
                res["scripted"][key] = {"vsa": v, "vsa_std": sd, "oracle": o, "action_agreement": agree}
                print(f"{code:6s} dim {dim:5d}  {fam + diff:22s} vsa-policy {v:+.3f}±{sd:.3f}  "
                      f"oracle {o:+.3f}  agreement {agree:.3f}")

    print("\ncount readout: exactness vs D (env-free, 30 streamed items, n_items=13 -> counts ~2.3 each)")
    for dim in a.sweep:
        for code in a.codes:
            ex, mae = count_readout_error(13, dim, code)
            res["count_readout"][f"n13_d{dim}_{code}"] = {"exact": ex, "mean_abs_err": mae}
            print(f"  d={dim:5d} {code:6s} exact={ex:.3f} mean|err|={mae:.3f}")
    for n_items in (2, 4, 13):
        for code in a.codes:
            ex, mae = count_readout_error(n_items, 256, code, stream=25)
            res["count_readout"][f"n{n_items}_d256_{code}_s25"] = {"exact": ex, "mean_abs_err": mae}
            print(f"  n_items={n_items:3d} d=256 {code:6s} (25 items, ~{25 / n_items:.1f} per item) "
                  f"exact={ex:.3f} mean|err|={mae:.3f}")

    print("\ndelay readout: hit rate vs lag (env-free, k=lag)")
    for k in (4, 32, 64):
        for code in a.codes:
            h = delay_readout_error(4, 256, k - 1, 60 if k == 4 else 64 + 30, code=code)
            res["delay_readout"][f"k{k}_d256_{code}"] = h
            print(f"  k={k:3d} (lag {k - 1:2d}) d=256 {code:6s} hit={h:.3f}")

    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
