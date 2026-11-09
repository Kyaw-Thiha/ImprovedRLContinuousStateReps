"""Diagnose the stuck pointer seeds: role gate, query map, per-hop accuracy, and a role-gate basin sweep.

    OMP_NUM_THREADS=1 ../../.venv/bin/python diag_pointer.py [--seeds 6] [--out results/diag_pointer.json]

Why (handoff 09 section 15.4 step 4): plain training of ``SSPPointerLearned`` succeeds on some seeds and stalls around
0.5 on others, and the stuck seeds are undiagnosed.  The pointer pass cannot be called non-provisional until that is
understood, because optimisation failures and genuine fragility look identical from the outside.

What is measured per seed:

  * **final accuracy** on the in-distribution and held-out-hop splits (the thing that stalls);
  * **role-gate accuracy** -- does the gate learn which stream token is a value (parity ``t % 2 == 1``)?  A gate stuck
    at 0.5 accuracy writes key/value pairs in the wrong role, which makes the bundle unreadable whatever else works;
  * **query-map alignment** -- is ``qmap`` the identity (start node -> its own code)?  A non-identity map reads a
    different node than the query asks for;
  * **per-hop accuracy** -- where in the chain the arm fails, taken from its own per-hop scores;
  * **basin sweep** -- role-gate weights initialised at the correct solution plus noise ``sigma``: the width of that
    basin says whether the failure is a bad basin (fixable by initialisation or restarts) or something else.

The output is a diagnosis, not a result: it tells us *which* fix to try, and the fix then has to be tested separately.
"""

import argparse
import json
import pathlib

import numpy as np
import torch

import arms_learned as L
import harness as H
import tasks_b

OUT = pathlib.Path(__file__).with_name("results")


def _task(n_nodes=6, n_query=6, train_hops=(1, 2), max_hops=8):
    return tasks_b.PointerTask(n_nodes=n_nodes, n_query=n_query, train_hops=train_hops, max_hops=max_hops)


def role_gate_accuracy(arm, task, B=32, seed=0):
    """Fraction of pair-stream tokens whose role gate agrees with parity (the hand-wired rung-0 roles)."""
    b = task.batch(B, task.ep_len, rng=np.random.default_rng(seed))
    feat = task.features(b)
    with torch.no_grad():
        arm(b, feat)
        r = torch.sigmoid(arm.role(feat))[..., 0]                      # (B, T): P(token is a value)
    n2 = 2 * task.n_nodes
    want = (b["t"][:, :n2] % 2 == 1).float()
    pred = (r[:, :n2] > 0.5).float()
    return float((pred == want).float().mean())


def query_map_alignment(arm, task, seed=0):
    """Is the start->code map the identity, and how peaked is it?  Returns (accuracy, peak mass, entropy)."""
    if getattr(arm, "cue", "batch") != "batch":
        return None
    b = task.batch(4, task.ep_len, rng=np.random.default_rng(seed))
    with torch.no_grad():
        logits = arm.qmap(torch.eye(task.n_nodes))                     # (n, n): row s = map from node s
        p = torch.softmax(logits, -1)
        identity = float((p.argmax(-1) == torch.arange(task.n_nodes)).float().mean())
        ent = float(-(p * (p + 1e-9).log()).sum(-1).mean())
    return {"identity_frac": identity, "peak_mass": float(p.max(-1).values.mean()), "mean_entropy": ent}


def per_hop_accuracy(arm, task, B=64, seed=0):
    """Accuracy by hop count, read from the arm's own per-hop scores (which hop does it get wrong)."""
    out = {}
    for h in range(1, task.max_hops + 1):
        b = task.batch(B, task.ep_len, rng=np.random.default_rng(100 + h), hops=(h,))
        feat = task.features(b)
        with torch.no_grad():
            arm(b, feat)
            sc = arm._hop_scores[:, :, h - 1]                          # (B, nq, n) after h iterations
        pred = sc.argmax(-1)
        out[str(h)] = float((pred == b["target"][:, 2 * task.n_nodes:2 * task.n_nodes + task.n_query]).float().mean())
    return out


def run_seed(seed, steps=1500, lr=1e-2, n_nodes=6, dim=128, hidden=64, init="random", aux_weight=0.0,
             role_noise=None):
    task = _task(n_nodes=n_nodes)
    sp = H.splits_for("pointer", task)
    arm, _ = L.build_learned("ssp_pointer_learned", task, budget=dim * 2, seed=seed, rung=1, init=init,
                             hidden=hidden, lam_init=0.99, temp=0.1)
    if role_noise is not None:
        with torch.no_grad():                                          # basin sweep around the correct role gate
            arm.role.weight.zero_()
            arm.role.bias.zero_()
            arm.role.weight[0, task.n_nodes + 1 + 3 + 2] = -8.0
            arm.role.weight.add_(torch.randn_like(arm.role.weight) * role_noise)
    curve, _ = H.train_arm(arm, task, sp, seed=seed, steps=steps, batch=64, lr=lr, eval_every=steps,
                           aux_weight=aux_weight)
    row = {"seed": int(seed), "id": float(curve[-1]["id"]), "extrap_lag": float(curve[-1]["extrap_lag"]),
           "role_acc": role_gate_accuracy(arm, task, seed=seed), "per_hop": per_hop_accuracy(arm, task)}
    row["qmap"] = query_map_alignment(arm, task)
    row["stuck"] = bool(row["id"] < 0.9)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--n-nodes", type=int, default=6)
    p.add_argument("--basin-sigmas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--basin-seeds", type=int, default=3)
    p.add_argument("--out", default=str(OUT / "diag_pointer.json"))
    args = p.parse_args()

    rows = []
    for s in range(args.seeds):
        row = run_seed(s, steps=args.steps, lr=args.lr, n_nodes=args.n_nodes)
        rows.append(row)
        q = row["qmap"] or {}
        print(f"  seed {s}: id {row['id']:.3f} extrap {row['extrap_lag']:.3f} role_acc {row['role_acc']:.3f} "
              f"qmap_identity {q.get('identity_frac', float('nan')):.2f} stuck={row['stuck']}", flush=True)

    basin = []
    for sigma in args.basin_sigmas:
        accs = []
        for k in range(args.basin_seeds):
            r = run_seed(1000 + k, steps=max(args.steps // 3, 300), lr=args.lr, n_nodes=args.n_nodes,
                         role_noise=sigma)
            accs.append(r["id"])
        basin.append({"sigma": float(sigma), "mean_id": float(np.mean(accs)), "ids": accs})
        print(f"  basin sigma={sigma}: mean id {np.mean(accs):.3f} {[round(a, 2) for a in accs]}", flush=True)

    stuck = [r for r in rows if r["stuck"]]
    summary = {
        "steps": args.steps, "lr": args.lr, "n_nodes": args.n_nodes, "seeds": rows, "basin": basin,
        "n_stuck": len(stuck),
        "role_acc_stuck": float(np.mean([r["role_acc"] for r in stuck])) if stuck else None,
        "role_acc_ok": float(np.mean([r["role_acc"] for r in rows if not r["stuck"]])) if len(stuck) < len(rows) else None,
    }
    OUT.mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"-> {args.out}: {len(stuck)}/{len(rows)} seeds stuck")


if __name__ == "__main__":
    main()
