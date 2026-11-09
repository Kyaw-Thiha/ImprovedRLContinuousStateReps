"""Random and optimal (oracle) episode return of POPGym RepeatPrevious{Easy,Medium,Hard} (reference venv)."""
import json
import sys

import numpy as np
import popgym
from popgym import wrappers

out = {}
for name in ("RepeatPreviousEasy", "RepeatPreviousMedium", "RepeatPreviousHard"):
    cls = getattr(popgym.envs, name)
    env = wrappers.Antialias(wrappers.PreviousAction(cls()))
    base = env
    while hasattr(base, "env"):
        base = base.env
    k = base.k
    rng = np.random.default_rng(0)
    rand, orac, lens = [], [], []
    for mode in ("random", "oracle"):
        for ep in range(300):
            obs, _ = env.reset(seed=ep)
            hist = [obs[0]]
            tot, n, done = 0.0, 0, False
            while not done:
                a = int(rng.integers(4)) if mode == "random" or len(hist) < k else hist[-k]
                obs, r, term, trunc, _ = env.step(a)
                hist.append(obs[0])
                tot += r
                n += 1
                done = term or trunc
            (rand if mode == "random" else orac).append(tot)
            lens.append(n)
    out[name] = {"k": k, "episode_len": int(np.median(lens)), "random_mean": float(np.mean(rand)),
                 "random_std": float(np.std(rand)), "optimal_mean": float(np.mean(orac)),
                 "optimal_std": float(np.std(orac))}
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "../results/rl_env_returns.json", "w"), indent=1)
