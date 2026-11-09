"""Reinforcement-learning check: does the VSA memory actually reduce to a better *learner* on POPGym?

Small self-contained advantage-actor-critic over the whole episode (truncated BPTT, ``--window``), because the
POPGym RLlib baselines would drag in a conflicting dependency stack (ray 2.59 + gymnasium 1.2 on py3.14) and their
model registry is the deprecated old-style API.  Score is the mean of the last k updates and a final greedy
evaluation -- **not** POPGym's MMER, which is selection-inflated (docs/10 section 13.1).

    python experiments/popgym_vsa/train.py --fam CountRecall --diff Easy --model vsa_count --updates 200
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from models import MODELS, build_model
from tasks import make_env
from vec_env import Vec

RESULTS = pathlib.Path(__file__).with_name("results")


def rollout(vec, model, gamma, lam, window, device):
    logps, ents, vals, rews, dones = [], [], [], [], []
    state = model.initial_state(vec.n, device)
    for step in range(vec.ep_len):
        logits, value, state = model.step(vec.obs_tensor(), vec.t_tensor(), state)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        vals.append(value)
        _, r, d = vec.step(a.cpu().numpy())
        rews.append(torch.as_tensor(r, dtype=torch.float32))
        dones.append(torch.as_tensor(d, dtype=torch.float32))
        state = model.reset_state(state, torch.as_tensor(d))
        if (step + 1) % window == 0:
            state = model.detach_state(state)

    # POPGym's memory tasks have fixed-length episodes, so every env terminates on the last step and the GAE below
    # can bootstrap from zero without per-env truncation bookkeeping. If that ever stops holding, this fires.
    assert vec.all_done, f"episodes did not all terminate at ep_len={vec.ep_len}"
    v = torch.stack(vals)  # (T, B)
    r = torch.stack(rews)
    done = torch.stack(dones)
    adv = torch.zeros_like(v)
    last = torch.zeros(vec.n)
    for i in range(vec.ep_len - 1, -1, -1):
        v_next = v[i + 1] if i + 1 < vec.ep_len else torch.zeros(vec.n)
        nonterm = 1.0 - done[i]
        delta = r[i] + gamma * v_next * nonterm - v[i]
        last = delta + gamma * lam * nonterm * last
        adv[i] = last
    adv = adv.detach()
    returns = (adv + v.detach()).detach()
    return adv, returns, torch.stack(logps), torch.stack(ents), v


def evaluate(fam, diff, model, n_eps, seed0, device):
    rets = []
    for ep in range(n_eps):
        env = make_env(fam, diff)
        obs, _ = env.reset(seed=seed0 + ep)
        state = model.initial_state(1, device)
        t, R = 0, 0.0
        while True:
            obs_t = torch.as_tensor(np.atleast_1d(obs)).long().view(1, -1)
            t_vec = torch.as_tensor([t]).long()
            with torch.no_grad():
                logits, _, state = model.step(obs_t, t_vec, state)
            obs, r, te, tr, _ = env.step(int(logits.argmax(-1)))
            R += r
            t += 1
            if te or tr:
                break
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fam", default="CountRecall", choices=["CountRecall", "RepeatPrevious"])
    p.add_argument("--diff", default="Easy", choices=["Easy", "Medium", "Hard"])
    p.add_argument("--model", default="vsa_count", choices=MODELS)
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--code", default="random", choices=["random", "block"])
    p.add_argument("--cleanup", action="store_true", help="add the VSA nearest-neighbour decode features (vsa_count)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-eps", type=int, default=20)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    vec = Vec(a.fam, a.diff, a.envs, seed0=1000 * (1 + a.seed))
    model = build_model(a.model, vec.spec, dim=a.dim, hidden=a.hidden, k_max=vec.ep_len, code=a.code,
                        cleanup=a.cleanup, seed=a.seed)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=a.lr)

    curve, t0 = [], time.time()
    for u in range(a.updates):
        n_before = len(vec.ep_returns)
        adv, returns, logps, ents, v = rollout(vec, model, a.gamma, a.lam, a.window, "cpu")
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(adv_n * logps).mean() + 0.5 * ((returns - v) ** 2).mean() - a.ent * ents.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        new = vec.ep_returns[n_before:] or vec.ep_returns[-vec.n :]
        curve.append(float(np.mean(new)))
        if (u + 1) % 25 == 0:
            print(f"  [{a.model}/{a.fam}{a.diff}] update {u + 1:4d} loss {loss.item():+.4f} "
                  f"train-return {np.mean(curve[-10:]):+.3f} ({time.time() - t0:.0f}s)")

    last_k = float(np.mean(curve[-10:]))
    ev_mean, ev_std = evaluate(a.fam, a.diff, model, a.eval_eps, seed0=10_000 + 1000 * a.seed, device="cpu")
    out = {
        "args": vars(a),
        "spec": vec.spec,
        "last_k_train_return": last_k,
        "eval_return": [ev_mean, ev_std],
        "curve": curve,
        "wall_s": round(time.time() - t0, 2),
    }
    RESULTS.mkdir(exist_ok=True)
    tag = a.tag or f"{a.fam}{a.diff}_{a.model}_s{a.seed}"
    (RESULTS / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[{a.model}/{a.fam}{a.diff}] last-10 train {last_k:+.3f} | greedy eval {ev_mean:+.3f}±{ev_std:.3f} "
          f"| {out['wall_s']}s -> results/{tag}.json")


if __name__ == "__main__":
    main()
