"""
Fixed probe states for eval_v3: Q-value diagnostics during training and the observation-noise scale
for the robustness suite.

Why a shared file: every policy of every cell must be probed on the SAME states, and the noise sigma of
the obs_noise_* conditions must be identical for all of them (common random numbers). The file is generated
deterministically (fixed seeds, no dependence on any trained policy), so regenerating it anywhere gives the
same bytes of content; R2 just saves the work.

Contents of probe_states.npz:
  states       float32 (256, 4)  128 uniform in |x|<2, |v|<2, |theta|<0.2, |omega|<2
                                 + 128 states sampled from noisy rollouts of a scripted balancing controller
  noise_sigma  float64 (4,)      per-dimension std of the observations pooled over 50 controller episodes run
                                 WITH 10% random actions (a perfectly clean controller balances so tightly,
                                 theta std ~0.008, that sigma-scaled noise would be meaningless)
  source       str               short description of how the file was generated

Usage: python scripts/make_probe_states.py [--local-root DIR] [--no-push]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import eval_v3_layout as layout  # noqa: E402

N_UNIFORM = 128
N_CONTROLLER = 128
UNIFORM_SEED = 20261006
CONTROLLER_SEEDS = range(0, 16)       # env seeds for the state-collection rollouts
SIGMA_SEEDS = range(100, 150)         # 50 exploring controller episodes for noise_sigma
# Linear balancing rule: push right iff w.obs > 0. Survives 500 steps from every seed tried.
CONTROLLER_W = np.array([0.05, 0.1, 1.0, 1.0])
EXPLORE_P = 0.1                       # random-action probability for probe-state collection AND for noise_sigma
SOURCE = ("128 uniform (|x|<2,|v|<2,|theta|<0.2,|omega|<2; rng seed 20261006) + 128 states from "
          "16 seeded PD-controller rollouts with 10% random actions; noise_sigma pooled over 50 "
          "controller episodes with 10% random actions (env seeds 100..149, rng seed 20261006+1000+seed)")


def _controller_action(obs) -> int:
    return int(np.dot(CONTROLLER_W, obs) > 0)


def _rollout(env, seed: int, rng=None, explore_p: float = 0.0):
    obs = env.reset(seed=int(seed))[0]
    states = [obs.copy()]
    for _ in range(500):
        if rng is not None and rng.random() < explore_p:
            action = int(rng.integers(2))
        else:
            action = _controller_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        states.append(obs.copy())
    return states


def generate_probe_arrays() -> dict:
    import gymnasium as gym

    rng = np.random.default_rng(UNIFORM_SEED)
    half = np.array([2.0, 2.0, 0.2, 2.0])
    uniform = rng.uniform(-half, half, size=(N_UNIFORM, 4))

    env = gym.make("CartPole-v1")
    try:
        explore_rng = np.random.default_rng(UNIFORM_SEED + 1)
        pool = []
        for s in CONTROLLER_SEEDS:
            pool.extend(_rollout(env, s, explore_rng, EXPLORE_P))
        pool = np.asarray(pool, dtype=np.float64)
        pick = np.random.default_rng(UNIFORM_SEED + 2).choice(len(pool), size=N_CONTROLLER, replace=False)
        ctrl = pool[np.sort(pick)]

        noisy = []
        for s in SIGMA_SEEDS:
            noisy.extend(_rollout(env, s, np.random.default_rng(UNIFORM_SEED + 1000 + s), EXPLORE_P))
        sigma = np.asarray(noisy, dtype=np.float64).std(axis=0)
    finally:
        env.close()

    states = np.concatenate([uniform, ctrl], axis=0).astype(np.float32)
    return {"states": states, "noise_sigma": sigma.astype(np.float64), "source": np.array(SOURCE)}


def ensure_probe_file(local_root: str = None, push: bool = True) -> str:
    """Return the local probe path, creating it when needed.

    Order: local file -> (push=True only, i.e. R2 allowed) pull from R2 -> generate. With push=True and
    R2 credentials the generated file is uploaded. push=False never touches R2 at all.
    """
    path = layout.local_path(layout.PROBE_KEY, local_root)
    if os.path.exists(path):
        return path
    import parallel_jobs as pj

    if push and pj.pull_file(layout.PROBE_KEY, path) == "ok" and os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **generate_probe_arrays())
    os.replace(tmp, path)
    if push:
        s3, _ = pj.r2_client()
        if s3 is not None and pj.push_file(layout.PROBE_KEY, path):
            print(f"[probe] uploaded {layout.PROBE_KEY}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--no-push", action="store_true", help="never touch R2 (no pull, no upload)")
    args = ap.parse_args()
    path = ensure_probe_file(args.local_root, push=not args.no_push)
    d = np.load(path)
    print(f"[probe] {path} states={d['states'].shape} noise_sigma={np.round(d['noise_sigma'], 4).tolist()}",
          flush=True)


if __name__ == "__main__":
    main()
