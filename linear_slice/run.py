"""Grid runner for the linear slice (NumPy, CPU, parallel over cores).

    python run.py --exp td --seeds 20 --out results_td.jsonl
    python run.py --exp ac --seeds 20 --out results_ac.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp

import numpy as np

import algos
import features
from catch import Catch, exact_policy_quantities

FEATURES = ("raw", "raw_bias", "tile", "hexssp", "rff")
GAMMAS = (0.9, 0.99, 0.999)
OFFSETS = (0.0, 1.0)
CENTERS = ("none", "value")


def _td_job(args):
  feat, gamma, offset, center, seed, steps, alpha0, eta = args
  env = Catch()
  X = env.state_x()
  Phi = features.make(feat, X, seed=seed)
  pi = env.chase_policy()
  q = exact_policy_quantities(env, pi, gamma, offset)
  w, r_bar, snaps = algos.td0_fixed_policy(env, Phi, pi, gamma, offset, center, steps, alpha0, eta, seed)
  target = q["v_gamma"] - r_bar / (1.0 - gamma) if center == "value" else q["v_gamma"]  # centred critic fits r - r_bar_hat
  stats = algos.critic_error_stats(Phi @ w, target, q["d"])
  # the best any weights could do (least squares under d): the representational floor
  Dh = np.sqrt(q["d"])[:, None]
  w_star = np.linalg.lstsq(Phi * Dh, target * Dh[:, 0], rcond=None)[0]
  floor = algos.critic_error_stats(Phi @ w_star, target, q["d"])
  return {"exp": "td", "feat": feat, "gamma": gamma, "offset": offset, "center": center, "seed": seed,
          "dim": int(Phi.shape[1]), "r_bar": r_bar, "r_bar_true": q["rbar"], **stats,
          "floor_mse": floor["mse"], "floor_const_frac": floor["const_frac"],
          "v_mag": float(np.mean(np.abs(Phi @ w)))}


def _ac_job(args):
  feat, gamma, offset, center, seed, steps, alpha_c, alpha_a, eta, r_bar_init = args
  env = Catch()
  Phi = features.make(feat, env.state_x(), seed=seed)
  out = algos.actor_critic(env, Phi, gamma, offset, center, steps, alpha_c, alpha_a, eta, seed,
                           r_bar_init=r_bar_init)
  return {"exp": "ac", "feat": feat, "gamma": gamma, "offset": offset, "center": center, "seed": seed,
          "r_bar_init": r_bar_init, "dim": int(Phi.shape[1]),
          "reward_last": float(np.mean(out["reward"][-3:])), **out}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--exp", choices=["td", "ac"], required=True)
  ap.add_argument("--seeds", type=int, default=20)
  ap.add_argument("--seed0", type=int, default=0)
  ap.add_argument("--steps", type=int, default=300_000)
  ap.add_argument("--alpha", type=float, default=0.1)
  ap.add_argument("--alpha-actor", type=float, default=0.02)
  ap.add_argument("--eta", type=float, default=0.1)
  ap.add_argument("--r-bar-init", default="zero", choices=["zero", "first_window"])
  ap.add_argument("--procs", type=int, default=0)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  seeds = range(a.seed0, a.seed0 + a.seeds)
  if a.exp == "td":
    jobs = [(f, g, o, c, s, a.steps, a.alpha, a.eta) for f, g, o, c, s in itertools.product(FEATURES, GAMMAS, OFFSETS, CENTERS, seeds)]
    fn = _td_job
  else:
    jobs = [(f, g, o, c, s, a.steps, a.alpha, a.alpha_actor, a.eta, a.r_bar_init)
            for f, g, o, c, s in itertools.product(FEATURES, GAMMAS, OFFSETS, CENTERS, seeds)]
    fn = _ac_job
  with mp.Pool(a.procs or mp.cpu_count()) as pool, open(a.out, "w") as f:
    for i, row in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
      f.write(json.dumps(row, default=float) + "\n")
      if i % 50 == 0:
        print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
  main()
