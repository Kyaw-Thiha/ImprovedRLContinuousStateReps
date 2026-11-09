"""In-pod profiler: where does the wall clock actually go?

The campaign stalls in a way the loop's own timers do not explain: `iter.jsonl` reports 1.7-1.9 s per iteration inside a
25-iteration window, yet a 300-iteration run needs far longer than that. This script runs a short job and records, per
iteration, the *wall* delta plus every phase and the GPU/CPU memory and RSS, so a gap can be attributed to a phase
rather than guessed at. It writes one line per iteration to stdout (visible in `runpodctl pod logs`) and to a local
file, so a partial run is still diagnosable.

    # inside the pod (after the bootstrap has installed the package)
    python analysis/profile_pod.py --iters 60 --num-envs 2048

Compare `gap_s` with the phases: a gap that is not inside any phase points at the mirror threads; a gap inside
`collect_s` that grows with iteration number points at the env or the allocator; a flat profile with a huge total
points at the container being descheduled (compare `wall` against `time.process_time()`).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import torch

from mjlab_centering.algo import CenteredPPO
from mjlab_centering.env_cfg import build_agent_cfg
from mjlab_centering.train import RunSpec, evaluate, make_env


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=60)
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--eval-envs", type=int, default=256)
  ap.add_argument("--eval-every", type=int, default=25)
  ap.add_argument("--eval-steps", type=int, default=300)
  ap.add_argument("--out", default="/tmp/profile.jsonl")
  ap.add_argument("--num-steps-per-env", type=int, default=24, dest="num_steps_per_env")
  a = ap.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  spec = RunSpec(job_id="profile", center="value", eta=0.003, gamma=0.999, offset=1.0, seed=0,
                 iters=a.iters, num_envs=a.num_envs, eval_envs=a.eval_envs)
  env, vec = make_env(spec, a.num_envs)
  agent_cfg = build_agent_cfg(robot="go1", max_iterations=a.iters, seed=0, center="value", eta=0.003)
  agent_cfg["algorithm"]["gamma"] = 0.999
  obs = vec.get_observations()
  alg = CenteredPPO.construct_algorithm(obs, vec, agent_cfg, device)
  policy = alg.get_policy()
  eval_env, eval_vec = make_env(spec, a.eval_envs, seed_off=7_000_000)

  fh = open(a.out, "a", buffering=1)
  prev_wall = time.time()
  prev_cpu = time.process_time()
  for it in range(a.iters):
    vec.iter = it
    t0 = time.time()
    with torch.inference_mode():
      for _ in range(a.num_steps_per_env):
        actions = alg.act(obs)
        obs, rew, dones, extras = vec.step(actions)
        obs = obs.to(device)
        alg.process_env_step(obs, rew.to(device), dones.to(device), extras)
    t_collect = time.time() - t0
    t0 = time.time()
    alg.compute_returns(obs)
    loss = alg.update()
    t_update = time.time() - t0
    t_eval = 0.0
    if (it + 1) % a.eval_every == 0:
      t0 = time.time()
      evaluate(eval_vec, policy, a.eval_steps)
      t_eval = time.time() - t0
    now = time.time()
    cpu = time.process_time()
    row = {
      "iter": it,
      "wall_s": now - prev_wall,
      "cpu_s": cpu - prev_cpu,
      "collect_s": t_collect,
      "update_s": t_update,
      "eval_s": t_eval,
      "other_s": (now - prev_wall) - (t_collect + t_update + t_eval),
      "gpu_mem_gb": (torch.cuda.memory_allocated() / 2**30) if device == "cuda" else 0.0,
      "gpu_reserved_gb": (torch.cuda.memory_reserved() / 2**30) if device == "cuda" else 0.0,
      "rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
      "loss": float(loss.get("value", float("nan"))),
    }
    print("[profile] " + json.dumps(row), flush=True)
    fh.write(json.dumps(row) + "\n")
    prev_wall, prev_cpu = now, cpu
  print("[profile] END", flush=True)
  try:
    from mjlab_centering import r2

    r2.retry(lambda: r2.upload_file(a.out, f"_profiles/profile_{int(time.time())}.jsonl"), attempts=3)
  except Exception as e:
    print(f"[profile] upload failed: {e!r}", flush=True)


if __name__ == "__main__":
  main()
