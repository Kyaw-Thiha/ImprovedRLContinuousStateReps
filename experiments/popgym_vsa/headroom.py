"""Headroom check for POPGym tasks: random, oracle and best-feedforward returns. Pure numpy, no learning.

Answers "is this task solvable without memory?" before any training budget is spent.  The feedforward ceiling is a
tabular majority vote of the oracle's action given only what a feedforward policy sees (the step index and/or the
current observation), fitted on ``--train-episodes`` episodes, so it lower-bounds what a trained positional MLP can
reach.  Oracles and space metadata come from ``tasks.py``, which is the single source of truth.

    python experiments/popgym_vsa/headroom.py
"""

import argparse
import collections
import json
import pathlib

import numpy as np

from tasks import make_env, oracle_for, spec_from_env

OUT = pathlib.Path(__file__).with_name("headroom.json")
ROWS = [("RepeatPrevious", "Easy"), ("RepeatPrevious", "Medium"), ("RepeatPrevious", "Hard"),
        ("CountRecall", "Easy"), ("CountRecall", "Medium"), ("CountRecall", "Hard")]


class Random:
    """Uniform-random baseline.

    Uses its own Generator rather than ``env.action_space.sample()``: the *space's* RNG is not pinned by
    ``env.reset(seed=...)``, which made this baseline wobble by ~0.02 between runs of identical code and seeds
    (measured across 3 runs; the oracle was bit-identical at +1.000 throughout).
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def observe(self, o):
        pass

    def act(self, o, t, env):
        return int(self.rng.integers(0, env.action_space.n))


def run(env, policy, n_eps, seed0=0):
    rets = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed0 + ep)
        policy.reset()
        t, R = 0, 0.0
        while True:
            a = policy.act(obs, t, env)
            obs, r, te, tr, _ = env.step(a)
            R += r
            t += 1
            policy.observe(obs)
            if te or tr:
                break
        rets.append(R)
    return np.array(rets)


def feedforward_ceiling(fam, diff, keyfn, n_train, n_eval):
    env = make_env(fam, diff)
    spec = spec_from_env(fam, diff, env)
    tab = collections.defaultdict(collections.Counter)
    for ep in range(n_train):
        obs, _ = env.reset(seed=10_000 + ep)
        o = oracle_for(spec)
        o.reset()
        t = 0
        while True:
            a = o.act(obs, t, env)
            tab[keyfn(obs, t)][a] += 1
            obs, r, te, tr, _ = env.step(a)
            o.observe(obs)
            t += 1
            if te or tr:
                break

    class FF:
        def __init__(self):
            self.rng = np.random.default_rng(0)

        def reset(self):
            pass

        def observe(self, o):
            pass

        def act(self, o, t, env):
            c = tab.get(keyfn(o, t))
            return c.most_common(1)[0][0] if c else int(self.rng.integers(0, env.action_space.n))

    return run(env, FF(), n_eval, seed0=500_000)


def key_t_obs(fam):
    """Feedforward key: (step index, the field the answer is about)."""
    if fam == "RepeatPrevious":
        return lambda o, t: (t, int(o))
    return lambda o, t: (t, int(o[1]))


def key_obs(fam):
    if fam == "RepeatPrevious":
        return lambda o, t: int(o)
    return lambda o, t: int(o[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--train-episodes", type=int, default=800)
    a = p.parse_args()

    res = {}
    print(f"{'env':24s} {'random':>14s} {'FF(t,obs)':>14s} {'FF(obs)':>14s} {'oracle':>14s}")
    for fam, diff in ROWS:
        spec = spec_from_env(fam, diff, make_env(fam, diff))
        assert spec["fam"] == fam
        out = {
            "random": run(make_env(fam, diff), Random(), a.episodes),
            "ff_t_obs": feedforward_ceiling(fam, diff, key_t_obs(fam), a.train_episodes, a.episodes),
            "ff_obs": feedforward_ceiling(fam, diff, key_obs(fam), a.train_episodes, a.episodes),
            "oracle": run(make_env(fam, diff), oracle_for(spec), a.episodes),
        }
        res[fam + diff] = {k: [float(v.mean()), float(v.std())] for k, v in out.items()}
        f = lambda v: f"{v.mean():+.3f}±{v.std():.3f}"  # noqa: E731
        print(f"{fam + diff:24s} {f(out['random']):>14s} {f(out['ff_t_obs']):>14s} "
              f"{f(out['ff_obs']):>14s} {f(out['oracle']):>14s}")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
