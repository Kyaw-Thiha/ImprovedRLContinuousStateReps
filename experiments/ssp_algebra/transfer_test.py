"""Transfer test for interface pretraining: is the learned interface reused, or is it just more training data?

    OMP_NUM_THREADS=1 ../../.venv/bin/python transfer_test.py [--seeds 10] [--smoke]

Why (handoff 09 section 15.4 step 3, docs/11 4.7): the adoption criterion for the pretraining stage is *not* only
"it aligns reliably" -- RUN_010 measures 19/20 seeds aligned -- but "the interface is reused".  Pretraining counts as
amortised only if an interface learned on one task shape is reused on another (a different stream length, or a
different alphabet); otherwise those tokens are simply more training data for the same problem.

Protocol, per seed:

  1. **stage 1**: build the source arm for the *source* shape (T=16, 8 items) and pretrain its interface with
     ``pretrain_mode="probe"`` on fresh synthetic streams, exactly as the recipe does;
  2. **stage 2**: build a destination arm for a *different* shape (T=32, or 16 items), copy only the interface
     (``load_address_params``: write phase, read head, kernel width) and freeze it;
  3. train **only the readout** on the destination task and measure converged accuracy there.

Reported next to a control that needs no transfer argument: the same destination shape trained end to end by the
frozen recipe (measured in RUN_010's unseen-variant run: 1.000 at T=48/64, 0.995 at T=24), and the no-transfer
destination arm, whose interface is random (the N=64-style collapse, RUN_010 section 6).

A transfer *pass* means: the destination readout reaches >= 0.9 on >= 9 of 10 seeds, with the pretraining tokens
reported separately from the task tokens.
"""

import argparse
import json
import pathlib

import numpy as np

import arms_learned as L
import harness as H
import tasks_c

OUT = pathlib.Path(__file__).with_name("results")


def _src_cfg(pretrain_steps, pretrain_mode="probe"):
    return {"rung": 1, "init": "random", "length_scale": 1.0, "ls_start": 8.0, "ls_mode": "sched",
            "width_steps": 375, "lr": 0.02, "lam_init": 0.99, "temp": 0.1,
            "pretrain_steps": pretrain_steps, "pretrain_mode": pretrain_mode, "pretrain_lr": 0.02}


def run_seed(seed, src_task, dst_task, pretrain_steps=500, readout_steps=800, lr=0.02, budget=256, hidden=32,
             pretrain_shapes=None):
    """One seed of the two-stage protocol.  Returns a dict with the destination accuracy and the token costs."""
    sp_src, sp_dst = H.splits_for("index", src_task), H.splits_for("index", dst_task)
    src, _ = L.build_learned("ssp_index_learned", src_task, budget=budget, seed=seed, hidden=hidden,
                             **_src_cfg(pretrain_steps))
    # stage 1: pretraining only (one task step afterwards is harmless and keeps the harness contract)
    _, extra = H.train_arm(src, src_task, sp_src, seed=seed, steps=1, batch=64, lr=lr, eval_every=1,
                           pretrain_steps=pretrain_steps, pretrain_mode="probe", pretrain_lr=0.02,
                           width_steps=375, pretrain_shapes=pretrain_shapes)
    # stage 2: a *different* shape, interface copied and frozen, readout trained on the destination task
    dst, _ = L.build_learned("ssp_index_learned", dst_task, budget=budget, seed=seed, hidden=hidden,
                            **{k: v for k, v in _src_cfg(0).items() if k not in ("pretrain_steps", "pretrain_mode")})
    dst.load_address_params(src)
    dst.freeze_address()
    curve, txtra = H.train_arm(dst, dst_task, sp_dst, seed=seed, steps=readout_steps, batch=64, lr=lr,
                               eval_every=readout_steps)
    return {"seed": int(seed), "dst_id": float(curve[-1]["id"]), "dst_extrap_T": float(curve[-1]["extrap_T"]),
            "pretrain_tokens": int(extra["pretrain_tokens"]), "task_tokens": int(txtra["tokens"])}


def run_axis(name, src_kw, dst_kw, seeds, pretrain_steps, readout_steps, budget, out, pretrain_shapes=None):
    rows = []
    for s in range(seeds):
        row = run_seed(s, tasks_c.IndexTask(**src_kw), tasks_c.IndexTask(**dst_kw), pretrain_steps=pretrain_steps,
                       readout_steps=readout_steps, budget=budget, pretrain_shapes=pretrain_shapes)
        rows.append(row)
        print(f"  [{name}] seed {s}: dst id {row['dst_id']:.3f} (src pretrain {row['pretrain_tokens']} tok, "
              f"dst task {row['task_tokens']} tok)", flush=True)
    acc = np.array([r["dst_id"] for r in rows])
    n_ok = int((acc >= 0.9).sum())
    lo, hi = _wilson(n_ok, len(acc))
    need = int(np.ceil(0.9 * len(acc)))
    summary = {"axis": name, "src": src_kw, "dst": dst_kw, "seeds": rows, "conv": float(acc.mean()),
               "min": float(acc.min()), "n_ge_0.9": n_ok, "alignment_wilson": [lo, hi],
               "pretrain_tokens": int(np.mean([r["pretrain_tokens"] for r in rows])),
               "task_tokens": int(np.mean([r["task_tokens"] for r in rows])),
               "need_ge_0.9": need, "transfer_pass": bool(n_ok >= need)}
    print(f"  [{name}] conv {summary['conv']:.3f} min {summary['min']:.3f} n>=0.9 {n_ok}/{len(acc)} "
          f"-> transfer {'PASS' if summary['transfer_pass'] else 'FAIL'}", flush=True)
    out.append(summary)
    return summary


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--pretrain-steps", type=int, default=500)
    p.add_argument("--readout-steps", type=int, default=800)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--control", action="store_true",
                   help="no transfer: skip pretraining, so the destination starts from a random *frozen* interface and "
                        "only the readout trains at the same budget.  Without this the transfer number cannot be "
                        "attributed to transfer rather than to the readout alone (RUN_010 section 7 item 8)")
    p.add_argument("--smoke", action="store_true", help="2 seeds, tiny budget: a mechanics check, not a measurement")
    p.add_argument("--pretrain-shapes", default=None,
                   help="phase 2.1: 'LO,HI' stream lengths drawn per pretraining step, so the interface cannot fit one "
                        "shape (e.g. '8,64').  Without it the interface is trained at the source shape only")
    p.add_argument("--out", default=str(OUT / "transfer.json"))
    args = p.parse_args()
    if args.smoke:
        args.seeds, args.pretrain_steps, args.readout_steps = 2, 30, 60
    if args.control:
        args.pretrain_steps = 0
        if args.out == str(OUT / "transfer.json"):
            args.out = str(OUT / "transfer_control.json")
    shapes = None
    if args.pretrain_shapes:
        lo, hi = (int(x) for x in args.pretrain_shapes.split(","))
        shapes = (lo, hi)

    out = []
    # axis 1: same alphabet, longer stream (the interface must carry the address scale)
    run_axis("length_T16_to_T32", {"n_items": 8, "ep_len": 16}, {"n_items": 8, "ep_len": 32},
             args.seeds, args.pretrain_steps, args.readout_steps, args.budget, out, shapes)
    # axis 2: same length, bigger alphabet (a different codebook size: the interface must not encode 8 items)
    run_axis("alphabet_8_to_16", {"n_items": 8, "ep_len": 16}, {"n_items": 16, "ep_len": 16},
             args.seeds, args.pretrain_steps, args.readout_steps, args.budget, out, shapes)
    OUT.mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({"smoke": args.smoke, "protocol": vars(args), "axes": out}, indent=2))
    print(f"-> {args.out}")
    print("  both axes pass -> the interface is reused; one axis fails -> report it as more training data")


if __name__ == "__main__":
    main()
