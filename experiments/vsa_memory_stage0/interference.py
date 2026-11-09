#!/usr/bin/env python
"""H1 from docs/08: does interference explain the SSP deficit?

Liu et al. (arXiv:1811.06626 / PMLR v232) define interference as the change in squared TD error on a
fixed probe set caused by an update.  This implements that with the repo's own
`rl.networks.dqnBasic.DQN.update`, so the "update" is the real training update.

Three design choices, each fixing a way the measurement can go wrong:

  1. **In-distribution data.** The transitions come from a hand-coded CartPole controller, not from a
     uniform-random policy.  On random-policy states the trained nets are far from their Bellman fixed
     point, so the probe TD^2 is dominated by genuine learning progress rather than by representation
     crosstalk.  The controller is arm-neutral: it is neither arm's policy, so neither is privileged.
  2. **A fixed target.** The regression target `r + gamma*max Q0(s')` is computed once from the loaded
     net and then held fixed for the whole loop.  With a moving target, a change in probe TD^2 mixes
     "the predictor moved" with "the target moved"; with a fixed target it is purely the predictor.
  3. **Interference per unit parameter step.** The two winners have different learning rates and
     different parameter counts; a raw TD^2 change is not comparable across them.  The headline number
     is |d TD^2| / ||dTheta||.  The same experiment is also run at a common lr as a control.

Shared across arms: the same raw transitions in the same order, the same probe subset, and the same
RNG seed before every update, so `ReplayBuffer.sample` draws identical batch indices.

Usage:
    .venv/bin/python experiments/vsa_memory_stage0/interference.py
"""

import json
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from rl.networks.dqnBasic import DQN  # noqa: E402
from rl.policy_io import load_policy  # noqa: E402

EVAL = REPO / "outputs" / "eval_v3"
PAIRS = [
    ("linear", "dqn_ssp_none_v4_hparam_search", "dqn_tile_coding_none_v4_hparam_search"),
    ("deep", "dqn_ssp_value_deep_v4_hparam_search", "dqn_tile_coding_value_deep_v4_hparam_search"),
]
SEED = 3000
N_TRANS = 20000
PROBE = 256
BATCH = 64
K_UPDATES = 200
GAMMA = 0.99


def heuristic_action(obs):
    """Standard CartPole bang-bang controller: arm-neutral, reaches the time limit."""
    _, _, theta, theta_dot = obs
    return 1 if (theta + 0.5 * theta_dot) > 0 else 0


def collect_transitions(n, seed=12345, use_heuristic=True):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rows, ep_len, lengths = [], 0, []
    while len(rows) < n:
        a = heuristic_action(obs) if use_heuristic else int(rng.integers(2))
        nxt, r, term, trunc, _ = env.step(a)
        rows.append((np.asarray(obs, dtype=np.float64), a, float(r), np.asarray(nxt, dtype=np.float64), bool(term or trunc)))
        obs, ep_len = nxt, ep_len + 1
        if term or trunc:
            lengths.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()
    env.close()
    return rows, float(np.mean(lengths)) if lengths else 0.0


def encode_all(policy, rows):
    phi = np.stack([np.asarray(policy.encode(o)) for (o, a, r, no, d) in rows])
    phi_next = np.stack([np.asarray(policy.encode(no)) for (o, a, r, no, d) in rows])
    return phi, phi_next


def build_dqn(policy, lr, capacity):
    dqn = DQN(
        policy.rep,
        state_dim=policy.dim,
        n_actions=policy.n_actions,
        lr=lr,
        gamma=GAMMA,
        buffer_size=capacity,
        batch_size=BATCH,
        target_update_freq=10,
        reward_center_mode="none",
        hidden_dims=policy.hidden_dims,
        init_seed=0,
    )
    dqn.q_net.load_state_dict(policy.net.state_dict())
    dqn.q_target.load_state_dict(policy.net.state_dict())
    dqn.lr = lr
    return dqn


def run_arm(policy, rows, probe_idx, lr, label):
    phi, phi_next = encode_all(policy, rows)
    dqn = build_dqn(policy, lr, capacity=len(rows))
    for i, (o, a, r, no, d) in enumerate(rows):
        dqn.push(phi[i], a, r, phi_next[i], d)

    tf = torch.float32 if not dqn.sparse else torch.int64

    def sel(x, dt):
        return torch.as_tensor(x[probe_idx], dtype=dt)

    phi_p, phi_next_p = sel(phi, tf), sel(phi_next, tf)
    act_p = sel(np.array([r[1] for r in rows]), torch.long)
    rew_p = sel(np.array([r[2] for r in rows]), torch.float32)
    done_p = sel(np.array([r[4] for r in rows]), torch.bool)

    fwd = dqn.q_net.forward_sparse if dqn.sparse else dqn.q_net.forward_dense

    # fixed regression target from the loaded net (design choice 2)
    with torch.no_grad():
        q0_next = fwd(phi_next_p).max(dim=1).values
        target = rew_p + GAMMA * q0_next * (~done_p)

    def probe_residual2():
        with torch.no_grad():
            cur = fwd(phi_p).gather(1, act_p.unsqueeze(1)).squeeze(1)
            return (cur - target) ** 2

    deltas, signed, levels, steps = [], [], [], []
    for _ in range(K_UPDATES):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        before = probe_residual2()
        theta_before = torch.cat([p.detach().reshape(-1) for p in dqn.q_net.parameters()]).clone()
        dqn.update()
        after = probe_residual2()
        theta_after = torch.cat([p.detach().reshape(-1) for p in dqn.q_net.parameters()])
        d = (after - before).detach().cpu().numpy()
        deltas.append(float(np.mean(np.abs(d))))
        signed.append(float(np.mean(d)))
        levels.append(float(before.mean()))
        steps.append(float(torch.linalg.vector_norm(theta_after - theta_before)))

    return {
        "label": label,
        "lr": lr,
        "sparse": bool(dqn.sparse),
        "n_params": int(sum(p.numel() for p in dqn.q_net.parameters())),
        "interference_meanabs": float(np.mean(deltas)),
        "interference_signed": float(np.mean(signed)),
        "probe_residual2_level": float(np.mean(levels)),
        "param_step_norm": float(np.mean(steps)),
        "interference_per_unit_step": float(np.mean(deltas) / max(np.mean(steps), 1e-12)),
        "interference_per_unit_step_signed": float(np.mean(signed) / max(np.mean(steps), 1e-12)),
    }


def main():
    rows, mean_ep = collect_transitions(N_TRANS)
    rng = np.random.default_rng(7)
    probe_idx = rng.choice(len(rows), size=PROBE, replace=False)
    print(f"shared pool: {len(rows)} transitions, behaviour-policy mean episode length {mean_ep:.1f}\n")

    results = {}
    for name, ssp_dir, tile_dir in PAIRS:
        ssp = load_policy(str(EVAL / ssp_dir / f"policies/seed_{SEED}.pt"))
        tile = load_policy(str(EVAL / tile_dir / f"policies/seed_{SEED}.pt"))
        for arm_name, pol in (("ssp", ssp), ("tile", tile)):
            lr = float(pol.meta["params"]["lr"])
            results[f"{name}/{arm_name}/own_lr"] = run_arm(pol, rows, probe_idx, lr, f"{name}/{arm_name}/own_lr")
        common = 1e-3
        for arm_name, pol in (("ssp", ssp), ("tile", tile)):
            results[f"{name}/{arm_name}/lr1e-3"] = run_arm(pol, rows, probe_idx, common, f"{name}/{arm_name}/lr1e-3")

    dest = pathlib.Path(__file__).with_name("interference.json")
    dest.write_text(json.dumps({
        "config": {"n_trans": N_TRANS, "probe": PROBE, "batch": BATCH, "k": K_UPDATES,
                   "behaviour": "hand-coded CartPole controller", "mean_episode_len": mean_ep,
                   "target": "fixed at initialisation", "metric": "squared residual change on fixed probe set"},
        "arms": results,
    }, indent=2))

    hdr = f"{'arm':26s} {'lr':>8s} {'|dR^2|':>9s} {'dR^2':>9s} {'R^2 level':>10s} {'|dTheta|':>9s} {'|dR^2|/step':>12s} {'dR^2/step':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k, v in results.items():
        print(f"{k:26s} {v['lr']:8.5f} {v['interference_meanabs']:9.4f} {v['interference_signed']:9.4f} "
              f"{v['probe_residual2_level']:10.4f} {v['param_step_norm']:9.5f} "
              f"{v['interference_per_unit_step']:12.4f} {v['interference_per_unit_step_signed']:10.4f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
