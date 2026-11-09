"""One POPGym-native PPO run (POPGym's own PPO config + ray_models + env), one arm, one seed.

    cd experiments/ssp_algebra/rl
    OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_pilot.py --arm gru --seed 0 --steps 1000000 \
        --workers 4 --envs-per-worker 8 --out ../results/rl_pilot_RepeatPreviousEasy_gru_s0.jsonl

Curve rows are appended to ``--out`` (JSON lines) after **every** PPO iteration, so a killed run keeps its data.
The reported return is RLlib's ``episode_reward_mean`` over the last 100 episodes (POPGym's own metric; its paper
reports the max-mean-episode-reward over training, also stored as ``mmer`` in the summary row).

The loop is ``run_ref.py``'s (``PPOConfig().update_from_dict(popgym config)``, old API stack because POPGym's
models are ModelV2); the only thing that varies between arms is ``custom_model`` (+ SSPModel's own keys).
"""

import argparse
import json
import os
import pathlib
import sys
import time

import common  # noqa: F401  (sys.path, env)
from common import ARMS, popgym_config, n_params

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import popgym
from popgym import wrappers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="RepeatPreviousEasy")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--envs-per-worker", type=int, default=8)
    p.add_argument("--bptt-cutoff", type=int, default=1024)
    p.add_argument("--num-minibatch", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--max-iters", type=int, default=10**9)
    p.add_argument("--ckpt", default=None, help="directory: save an RLlib checkpoint at the end of the run")
    p.add_argument("--restore", default=None, help="checkpoint directory: continue that run (rows are appended to --out)")
    args = p.parse_args()

    def wrap(env):
        return wrappers.Antialias(wrappers.PreviousAction(env))

    for cls, info in popgym.envs.ALL.items():
        register_env(info["id"], lambda x, cls=cls: wrap(cls()))
    env_id = common.run_ref.env_name_for(args.env)

    ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=args.workers + 2, logging_level="ERROR")
    cfg = popgym_config(env_id, args.arm, args.workers, args.envs_per_worker, args.bptt_cutoff, args.num_minibatch,
                        seed=args.seed)
    config = PPOConfig().api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    config = config.update_from_dict(cfg)
    algo = config.build()
    model = algo.get_policy().model
    meta = {"env": env_id, "arm": args.arm, "seed": args.seed, "steps_requested": args.steps,
            "workers": args.workers, "envs_per_worker": args.envs_per_worker, "bptt_cutoff": args.bptt_cutoff,
            "num_minibatch": args.num_minibatch, "train_batch_size": cfg["train_batch_size"],
            "minibatch_size": algo.config.minibatch_size, "num_epochs": algo.config.num_epochs,
            "lr": algo.config.lr, "clip_param": algo.config.clip_param, "gamma": algo.config.gamma,
            "params_total": n_params(model),
            "state_floats": sum(int(__import__("numpy").prod(s.shape)) for s in model.initial_state()),
            "custom_model": type(model).__name__, "custom_model_config_extra": ARMS[args.arm][1]}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wall0, ts, it, rets = 0.0, 0, 0, []
    if args.restore:
        algo.restore(args.restore)
        for l in out.read_text().splitlines():
            d = json.loads(l)
            if d["row"] == "iter":
                wall0, ts, it = d["wall_s"], d["timesteps"], d["iter"]
                if d["episode_return_mean"] is not None:
                    rets.append(d["episode_return_mean"])
        with out.open("a") as f:
            f.write(json.dumps({"row": "restored", "from_timesteps": ts, "ckpt": args.restore}) + "\n")
    else:
        out.write_text(json.dumps({"row": "meta", **meta}) + "\n")
    print(meta, flush=True)

    t0 = time.time() - wall0
    while ts < args.steps and it < args.max_iters:
        res = algo.train()
        it += 1
        ts = int(res.get("timesteps_total", res.get("num_env_steps_sampled_lifetime", ts)))
        ret = res.get("episode_reward_mean", res.get("env_runners", {}).get("episode_return_mean"))
        timers = res.get("timers", {})
        learner = res.get("info", {}).get("learner", {}).get("default_policy", {}).get("learner_stats", {})
        pm = algo.get_policy().model
        addr = pm.address_summary() if hasattr(pm, "address_summary") else None
        row = {"row": "iter", "iter": it, "timesteps": ts, "episode_return_mean": None if ret is None else float(ret),
               "episodes_this_iter": res.get("episodes_this_iter"), "wall_s": time.time() - t0,
               "time_this_iter_s": res.get("time_this_iter_s"),
               "sample_time_s": timers.get("sample_time_ms", 0) / 1e3 if "sample_time_ms" in timers else None,
               "learn_time_s": timers.get("learn_time_ms", 0) / 1e3 if "learn_time_ms" in timers else None,
               "total_loss": learner.get("total_loss"), "vf_loss": learner.get("vf_loss"),
               "policy_loss": learner.get("policy_loss"), "entropy": learner.get("entropy"),
               "model_metrics": learner.get("model"), "addr": addr}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if ret is not None:
            rets.append(float(ret))
        print(f"  iter {it:3d} steps={ts:8d} ret={ret} wall={row['wall_s']:.0f}s sample={row['sample_time_s']} "
              f"learn={row['learn_time_s']}", flush=True)
    wall = time.time() - t0
    if args.ckpt:
        os.makedirs(args.ckpt, exist_ok=True)
        algo.save_checkpoint(os.path.abspath(args.ckpt))
    algo.stop()
    ray.shutdown()
    with out.open("a") as f:
        f.write(json.dumps({"row": "summary", "timesteps": ts, "iters": it, "wall_s": wall,
                            "mmer": max(rets) if rets else None}) + "\n")
    print(f"done {args.arm} s{args.seed}: {ts} steps in {wall:.0f}s -> {out}")


if __name__ == "__main__":
    main()
