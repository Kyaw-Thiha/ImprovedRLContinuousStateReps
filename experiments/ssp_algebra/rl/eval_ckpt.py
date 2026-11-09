"""Memory-zeroed ablation (handoff 11 step 0g) on saved RLlib checkpoints, outside RLlib's sampler.

    OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python eval_ckpt.py gru:/tmp/rl_ckpt/gru_s0 ssp:/tmp/rl_ckpt2/ssp_s0 ...

For each ``arm:ckpt_dir`` it loads the policy weights into the arm's POPGym/SSP model, then plays ``--episodes``
RepeatPreviousEasy episodes sampling actions from the policy logits (as PPO's rollout does), once with the memory
carried normally and once with the *memory state zeroed before every step* (POPGym's time counter kept).  The
difference is how much of the return the memory carries.  The normal-memory number should match the training curve.
"""

import argparse
import json
import pickle

import numpy as np
import torch

import common
from common import ARMS, popgym_config
from ray.rllib.models.preprocessors import get_preprocessor
import popgym
from popgym import wrappers
import gymnasium as gym


def build(arm, env_id, obs_space):
    cfg = popgym_config(env_id, arm, 0, 1)
    pre = get_preprocessor(obs_space)(obs_space)
    model = cfg["model"]["custom_model"](pre.observation_space, gym.spaces.Discrete(4), 4, cfg["model"], "m",
                                         **cfg["model"]["custom_model_config"])
    return model, pre


def play(model, pre, env, episodes, zero_memory, seed):
    g = torch.Generator().manual_seed(seed)
    rets = []
    model.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        state = [s.unsqueeze(0).clone() for s in model.get_initial_state()]
        tot, done = 0.0, False
        while not done:
            o = torch.as_tensor(pre.transform(obs), dtype=torch.float32).reshape(1, -1)
            if zero_memory:
                for i in range(len(state) - 1):
                    state[i] = torch.zeros_like(state[i])
            with torch.no_grad():
                logits, state = model({"obs": o, "obs_flat": o}, state, torch.ones(1, dtype=torch.int32))
            a = int(torch.distributions.Categorical(logits=logits[0]).sample())
            obs, r, term, trunc, _ = env.step(a)
            tot += r
            done = term or trunc
        rets.append(tot)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("targets", nargs="+")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    env = wrappers.Antialias(wrappers.PreviousAction(popgym.envs.RepeatPreviousEasy()))
    out = {}
    for t in args.targets:
        arm, d = t.split(":")
        arm_key = arm.rsplit("_s", 1)[0] if arm not in ARMS else arm
        model, pre = build(arm_key, "popgym-RepeatPreviousEasy-v0", env.observation_space)
        w = pickle.load(open(f"{d}/policies/default_policy/policy_state.pkl", "rb"))["weights"]
        model.load_state_dict({k: torch.as_tensor(v) for k, v in w.items()})
        full = play(model, pre, env, args.episodes, False, 0)
        zero = play(model, pre, env, args.episodes, True, 0)
        out[t] = {"memory": full, "memory_zeroed_each_step": zero}
        print(t, f"memory {full[0]:+.3f}±{full[1]:.3f}  zeroed {zero[0]:+.3f}±{zero[1]:.3f}", flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
