"""Single source of truth for the POPGym tasks used here: environment construction, space metadata, and the
scripted oracles.  Imported by headroom.py, selftest.py and train.py so the observation fields, the action width,
the episode length and the oracle semantics are defined exactly once.

All four target environments emit *discrete* observations, so the VSA machinery being tested here is the
bundle/unbind memory over random item codes, not the continuous SSP encoding.

    RepeatPrevious(k) : output the suit seen k steps ago.  Fields = [suit].
    CountRecall       : output how many times the queried colour has been seen. Fields = [value, query].
"""

import collections

import gymnasium as gym


def make_env(fam, diff, **kw):
    import popgym.envs as E

    return getattr(E, fam + diff)(**kw)


def spec_from_env(fam, diff, env):
    """Space/episode metadata read at runtime, never hardcoded (action widths differ across difficulties)."""
    sp = env.observation_space
    if isinstance(sp, gym.spaces.MultiDiscrete):
        cards = [int(n) for n in sp.nvec]
    elif isinstance(sp, gym.spaces.Discrete):
        cards = [int(sp.n)]
    else:  # pragma: no cover - only the two shapes above are expected
        raise TypeError(f"unsupported observation space for {fam}{diff}: {sp}")
    return {
        "fam": fam,
        "diff": diff,
        "cards": cards,  # per-field cardinality; cards[0] is the item codebook size
        "n_items": cards[0],
        "act_dim": int(env.action_space.n),
        "ep_len": int(env.max_episode_length),
        "k": int(getattr(env, "k", 0)),
        # POPGym's k counts the card from k back *including the current one*, so the observation-index lag is k-1.
        # Measured: RepeatPrevious hit rate is chance (0.25) at lag=k and 0.99 at lag=k-1 for k=4.
        "lag": max(int(getattr(env, "k", 0)) - 1, 0),
    }


class RPOracle:
    """RepeatPrevious: output the suit seen k steps ago (history includes the current observation)."""

    def __init__(self, k):
        self.k = k

    def reset(self):
        self.h = []

    def observe(self, o):
        self.h.append(int(o))

    def act(self, o, t, env):
        if not self.h:
            self.h.append(int(o))
        return self.h[-self.k] if len(self.h) >= self.k else 0


class CROracle:
    """CountRecall: answer counts[query], counting the current value first."""

    def reset(self):
        self.c = collections.Counter()
        self.first = True

    def observe(self, o):
        self.c[int(o[0])] += 1

    def act(self, o, t, env):
        if self.first:
            self.c[int(o[0])] += 1
            self.first = False
        return self.c[int(o[1])]


def oracle_for(spec):
    return RPOracle(spec["k"]) if spec["fam"] == "RepeatPrevious" else CROracle()
