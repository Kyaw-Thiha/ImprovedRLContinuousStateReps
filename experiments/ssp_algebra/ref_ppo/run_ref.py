"""Step 1 of the plan (docs/11 section 5): POPGym's own PPO reference run.

Purpose: settle whether the in-repo A2C loop or the *task* is why no arm learned CountRecall (RUN_006 section 4).
This uses POPGym's own PPO algorithm, its own model classes (``popgym.baselines.ray_models``) and the config from
its own ``baselines/ppo.py``.  It is a throwaway venv outside the repo (see ``README`` in the venv): python 3.14,
ray 2.59.0, torch 2.14.1+cpu, popgym 1.0.7, plus the un-declared baseline deps (dnc, einops, opt_einsum, scipy,
pillow, matplotlib, imageio, scikit-learn, pandas, tqdm).

Deviations from ``baselines/ppo.py``, all API drift rather than protocol changes, recorded here so they are not
rediscovered:
  * it drives ``PPOConfig.update_from_dict(<POPGym's own config dict>)`` instead of ``ray.tune.run("PPO", ...)``,
    because ray 2.59 (a) hard-errors on ``local_dir`` and (b) resolves a single-element ``grid_search`` to a bare
    string, which breaks POPGym's ``trial_name_creator``;
  * the *old* API stack is requested explicitly (``enable_rl_module_and_learner=False``), since the custom_model
    (ModelV2) classes POPGym ships are the old-stack API;
  * metrics are read from the training result dict and written to JSON.

    OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python run_ref.py --env RepeatPreviousEasy --model GRU --steps 300000
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import ray  # noqa: E402
import torch  # noqa: E402
from ray.rllib.algorithms.ppo import PPOConfig  # noqa: E402
from ray.tune.registry import register_env  # noqa: E402

import popgym  # noqa: E402
from popgym import wrappers  # noqa: E402
from popgym.baselines.ray_models.ray_gru import GRU  # noqa: E402
from popgym.baselines.ray_models.ray_lstm import LSTM  # noqa: E402
from popgym.baselines.ray_models.ray_mlp import BasicMLP, MLP  # noqa: E402

MODELS = {"GRU": GRU, "LSTM": LSTM, "MLP": MLP, "BasicMLP": BasicMLP}


def env_name_for(short):
    """Maps e.g. 'RepeatPreviousEasy' to the registered POPGym id, exactly as baselines/ppo.py does."""
    E = popgym.envs
    all_envs = popgym.envs.ALL
    res = getattr(E, short)
    keys = list(res.keys()) if isinstance(res, dict) else [res]
    return [all_envs[k]["id"] for k in keys][0]


def build_config(env_id, model_cls, bptt_cutoff, workers, envs_per_worker, num_minibatch, max_seq_len_state):
    h, h_memory = 128, 256
    train_batch_size = bptt_cutoff * max(workers, 1) * envs_per_worker
    cfg = {
        "env": env_id,
        "framework": "torch",
        "num_env_runners": workers,
        "num_envs_per_env_runner": envs_per_worker,
        "num_gpus": 0,
        "vf_loss_coeff": 1.0,
        "train_batch_size": train_batch_size,
        "rollout_fragment_length": bptt_cutoff,
        "sgd_minibatch_size": num_minibatch * bptt_cutoff,
        "gamma": 0.99,
        "batch_mode": "complete_episodes",
        "model": {
            "max_seq_len": bptt_cutoff,
            "custom_model": model_cls,
            "custom_model_config": {
                "preprocessor_input_size": h,
                "preprocessor": torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.LeakyReLU(inplace=True)),
                "preprocessor_output_size": h,
                "hidden_size": h_memory,
                "postprocessor": torch.nn.Identity(),
                "actor": torch.nn.Sequential(torch.nn.Linear(h_memory, h), torch.nn.LeakyReLU(inplace=True),
                                             torch.nn.Linear(h, h), torch.nn.LeakyReLU(inplace=True)),
                "critic": torch.nn.Sequential(torch.nn.Linear(h_memory, h), torch.nn.LeakyReLU(inplace=True),
                                              torch.nn.Linear(h, h), torch.nn.LeakyReLU(inplace=True)),
                "postprocessor_output_size": h,
            },
        },
    }
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="RepeatPreviousEasy")
    p.add_argument("--model", default="GRU")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--envs-per-worker", type=int, default=4)
    p.add_argument("--bptt-cutoff", type=int, default=1024)
    p.add_argument("--num-minibatch", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    def wrap(env):
        return wrappers.Antialias(wrappers.PreviousAction(env))

    all_envs = popgym.envs.ALL
    for cls, info in all_envs.items():
        register_env(info["id"], lambda x, cls=cls: wrap(cls()))

    env_id = env_name_for(args.env)
    model_cls = MODELS[args.model]
    print(f"env={env_id} model={args.model} steps={args.steps}", flush=True)

    ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=max(2, 2 * args.workers + 2),
             logging_level="ERROR")
    cfg = build_config(env_id, model_cls, args.bptt_cutoff, args.workers, args.envs_per_worker, args.num_minibatch,
                       None)
    config = PPOConfig().api_stack(enable_rl_module_and_learner=False,
                                   enable_env_runner_and_connector_v2=False)
    config = config.update_from_dict(cfg)
    algo = config.build()

    history, timesteps = [], 0
    while timesteps < args.steps:
        res = algo.train()
        ts = res.get("timesteps_total", res.get("num_env_steps_sampled_lifetime", timesteps))
        ret = res.get("episode_reward_mean", res.get("env_runners", {}).get("episode_return_mean"))
        if ret is None:
            ret = res.get("sampler_results", {}).get("episode_reward_mean")
        history.append({"timesteps": int(ts), "episode_return_mean": None if ret is None else float(ret)})
        print(f"  iter {len(history):3d} steps={int(ts):8d} return_mean={ret}", flush=True)
        timesteps = int(ts)
    algo.stop()
    ray.shutdown()

    rets = [h["episode_return_mean"] for h in history if h["episode_return_mean"] is not None]
    out = {"env": env_id, "model": args.model, "steps_requested": args.steps, "history": history,
           "mmer": max(rets) if rets else None,
           "last5": sum(rets[-5:]) / max(len(rets[-5:]), 1) if rets else None,
           "n_iters_with_return": len(rets), "deviations": "PPOConfig.update_from_dict + explicit old API stack"}
    path = pathlib.Path(args.out or f"/tmp/popgym_ref_results/ref_{args.env}_{args.model}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"MMER={out['mmer']} last5={out['last5']} -> {path}")


if __name__ == "__main__":
    main()
