"""Parallel, resumable sweep and ablation runner.

Two modes:
  * ``sweep``     -- random search (equal config count per arm) then hold-out seeds, per variant; this is the
                    protocol in docs/11 section 4.2-4.3.
  * ``ablation``  -- one change at a time from a reference SSP configuration (docs/11 section 4.4), 5 hold-out seeds.

Every run is one JSONL row keyed by ``(variant, arm, cfg_idx, seed)``, so a restart resumes instead of redoing.

    OMP_NUM_THREADS=1 ../../.venv/bin/python run_sweep.py --mode sweep --variants delay --workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np
import torch

OUT = pathlib.Path(__file__).with_name("results")
OUT.mkdir(exist_ok=True)

# One reference configuration per variant; the ablations change exactly one entry of it.  `lr` is not part of the
# search space here: the ablation compares *within* a reference, so a fixed, sane rate is the right choice.
LR_REF = 5e-3
REFERENCES = {
    "delay": dict(lr=LR_REF, pos_kind="ssp", cleanup=True, learn_codes=False, decay=True, gate=False,
                  length_scale=1.0, temp=0.1, hidden=64, lam_init=0.9),
    "index": dict(lr=LR_REF, pos_kind="ssp", cleanup=True, learn_codes=False, decay=True, gate=False,
                  length_scale=1.0, temp=0.1, hidden=64, lam_init=0.99),
    "pointer": dict(lr=LR_REF, decay=True, hard=False, temp=0.1, hidden=64, lam_init=0.99),
    "path": dict(lr=LR_REF, shift_mode="read_relative", decay=True, length_scale=4.0, hidden=64, lam_init=0.99),
}

ABLATIONS = {
    "delay": {"reference": {}, "pos_table": dict(pos_kind="table"), "no_cleanup": dict(cleanup=False),
              "learn_codes": dict(learn_codes=True), "no_decay": dict(decay=False), "gated": dict(gate=True),
              "slow_forget": dict(lam_init=0.99)},
    "index": {"reference": {}, "pos_table": dict(pos_kind="table"), "no_cleanup": dict(cleanup=False),
              "learn_codes": dict(learn_codes=True), "no_decay": dict(decay=False),
              "fast_forget": dict(lam_init=0.9)},
    "pointer": {"reference": {}, "no_decay": dict(decay=False), "hard_cleanup": dict(hard=True)},
    "path": {"reference": {}, "shift_memory": dict(shift_mode="shift_memory"), "no_decay": dict(decay=False)},
}


def make_task(variant, ep_len=32, n_items=8, train_lags=(1, 2, 3, 4), n_nodes=8, n_query=8, train_hops=(1, 2),
              n_landmarks=6, chain=False, max_hops=8):
    import tasks_a
    import tasks_b
    import tasks_c

    if variant == "delay":
        return tasks_c.DelayTask(n_items=n_items, ep_len=ep_len, train_lags=train_lags)
    if variant == "index":
        return tasks_c.IndexTask(n_items=n_items, ep_len=ep_len)
    if variant == "pointer":
        return tasks_b.PointerTask(n_nodes=n_nodes, n_query=n_query, train_hops=train_hops, chain=chain,
                                   max_hops=max_hops)
    if variant == "delay_distract":
        import tasks_f
        return tasks_f.DelayDistractTask(n_items=n_items, ep_len=ep_len, train_lags=train_lags)
    if variant == "delay_nolag":
        import tasks_f
        return tasks_f.DelayNoLagTask(n_items=n_items, ep_len=ep_len, train_lags=train_lags)
    if variant == "assoc":
        import tasks_d
        return tasks_d.AssocTask(n_pairs=max(int(n_items) // 2, 2), ep_len=ep_len)
    if variant == "assoc_mixed":
        import tasks_d
        return tasks_d.AssocMixedTask(n_pairs=max(int(n_items) // 2, 2), ep_len=ep_len)
    if variant == "path":
        return tasks_a.LandmarkTask(n_landmarks=n_landmarks, ep_len=ep_len)
    if variant == "spatial":
        import tasks_e
        return tasks_e.SpatialRecallTask(n_landmarks=n_landmarks, ep_len=ep_len)
    if variant == "spatial_easy":
        # Trainability-gate variant (docs/11 4.2): ONE landmark and NO motion, so the answer is simply the position
        # shown at step 0 and the only requirement is to carry a 2-vector across the episode.  A baseline that cannot
        # clear this cannot have its number used on the real spatial task.
        import tasks_e
        return tasks_e.SpatialRecallTask(n_landmarks=1, ep_len=ep_len, step_scale=0.0)
    raise ValueError(variant)


def run_job(job):
    """One training run in a worker process.  Returns a JSON-serialisable record."""
    torch.set_num_threads(1)
    import arms as A
    import harness as H

    variant, cfg, seed = job["variant"], job["cfg"], job["seed"]
    arm_name = job.get("arm_real", job["arm"])             # `arm` may be a condition label (e.g. ssp_index_learned_r1)
    task = make_task(variant, **job.get("task_kw", {}))
    sp = H.splits_for(variant, task, robust=bool(job.get("robust")))
    row = H.run_one(arm_name, task, sp, job["budget"], cfg, seed, job["steps"], job["batch"], job["eval_every"],
                    eval_steps=job.get("eval_steps"), pool=job.get("pool"))
    row["arm"] = job["arm"]                                 # the label, so conditions never merge in analysis
    row.update({"arm_real": arm_name, "variant": variant, "cfg_idx": job["cfg_idx"], "phase": job["phase"], "tag": job["tag"],
                "eval_steps": job.get("eval_steps"), "pool": job.get("pool"), "budget": job["budget"],
                "steps": job["steps"]})
    return row


def jsonl_rows(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def run_jobs(jobs, path, workers, label):
    # the phase is part of the key: otherwise every ablation arm (cfg_idx 0, same seeds) would look "already done"
    done = {(r["variant"], r["arm"], r["cfg_idx"], r["seed"], r.get("phase")) for r in jsonl_rows(path)}
    todo = [j for j in jobs if (j["variant"], j["arm"], j["cfg_idx"], j["seed"], j["phase"]) not in done]
    print(f"[{label}] {len(jobs)} jobs, {len(todo)} to run ({len(done)} already done)", flush=True)
    if not todo:
        return
    t0 = time.time()
    with path.open("a") as fh, mp.Pool(workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(run_job, todo, chunksize=1), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            s = row["summary"]
            print(f"[{label}] {i}/{len(todo)} {row['variant']}/{row['arm']} cfg{row['cfg_idx']} s{row['seed']} "
                  f"id={s['last5_id']:.3f} aulc={s['aulc_id']:.3f} "
                  f"extrap={max(s[k] for k in s if k.startswith('last5_extrap')):.3f} "
                  f"({row['wall_s']:.0f}s, {time.time() - t0:.0f}s total)", flush=True)


def _run_common(args):
    """Fields every job carries, so search / final / ablation agree on the measurement protocol."""
    return {"eval_steps": getattr(args, "eval_steps", None), "pool": getattr(args, "pool", 0) or None,
            "robust": bool(getattr(args, "robust", False))}


def build_search_jobs(args, n_configs=8):
    """Deterministic config draw per (variant, arm) so the search and the final phase agree on the grid.

    ``--search-seeds K`` (default 1) runs every config on K search seeds, numbered ``ci + 100*k``, so a config's score
    is a mean over seeds rather than one draw (handoff 09 section 15.4 step 1).  The config itself is drawn once per
    (variant, arm), *independent* of K, so adding seeds never changes which configs are on the grid and rows already
    on disk stay reusable.
    """
    import harness as H

    k_seeds = max(1, int(getattr(args, "search_seeds", 1)))
    jobs = []
    for variant in args.variants:
        task = make_task(variant, **args.task_kw)
        for arm_name in args.arms:
            label = getattr(args, "label", None) or arm_name
            rng = np.random.default_rng(H.stable_seed(arm_name, variant, args.budget))
            for ci in range(n_configs):
                cfg = H.sample_config(arm_name, rng, args.budget, variant)
                cfg.update(getattr(args, "arm_cfg", None) or {})                       # condition (rung / init / aids), not searched
                for k in range(k_seeds):
                    jobs.append({"variant": variant, "arm": label, "arm_real": arm_name, "cfg": cfg, "cfg_idx": ci,
                                 "seed": ci + 100 * k,
                                 "phase": "search", "tag": args.tag, "budget": args.budget, "steps": args.steps,
                                 "batch": args.batch, "eval_every": args.eval_every, "task_kw": args.task_kw,
                                 **_run_common(args)})
    return jobs


def pick_winners(rows, n_configs=8, select="last5", min_conv=0.0):
    """Best config per (variant, arm).

    ``select="last5"`` is the historical rule: rank by the short learning-curve area, ties to the first config drawn.
    With one search seed per config it reproduces the old pick exactly, so existing tags are unaffected.

    ``select="aulc"`` ranks by the *mean area under the learning curve* over every search seed of the config, ties
    broken by mean converged accuracy, and compares only configs that carry the full set of seeds (an interrupted
    search must not win on one lucky seed).  This is the fix for RUN_009's finding that selection on converged
    accuracy is blind to speed under ties: a "faster" claim may only be restated from a speed-aware selection.

    ``min_conv`` is a *floor*, not a second ranking key (docs/11 step 2 asks for "AULC subject to converged >= x"):
    among configs whose mean converged score clears the floor the fastest wins; if none clears it, the best mean
    converged score wins instead, so a run never picks a config that cannot reach the accuracy under test.  It has no
    effect on the ``last5`` rule.
    """
    search = [r for r in rows if r["phase"] == "search"]
    best = {}
    for key in {(r["variant"], r["arm"]) for r in search}:
        groups = {}
        for r in search:
            if (r["variant"], r["arm"]) == key:
                groups.setdefault(r["cfg_idx"], []).append(r)
        if not groups:
            continue
        full = max(len(g) for g in groups.values())
        cand = [ci for ci, g in groups.items() if len(g) == full]

        def mean(ci, k):
            return float(np.mean([float(r["summary"][k]) for r in groups[ci]]))

        if select == "aulc":
            qualifying = [ci for ci in cand if mean(ci, "conv_id") >= min_conv]
            pool = qualifying or cand

            def score(ci):
                return (mean(ci, "conv_id"), -ci) if not qualifying else (mean(ci, "aulc_id"), mean(ci, "conv_id"), -ci)
        else:
            pool = cand

            def score(ci):
                return (mean(ci, "last5_id"), -ci)

        ci = max(pool, key=score)
        winner = dict(groups[ci][0])
        winner["cfg_idx"] = ci
        winner["n_search_seeds"] = len(groups[ci])
        best[key] = winner
    return best


def build_final_jobs(args, winners):
    jobs = []
    for (variant, arm), w in sorted(winners.items()):
        for k in range(args.seeds):
            seed = 1000 + k
            jobs.append({"variant": variant, "arm": arm, "arm_real": w.get("arm_real", arm), "cfg": w["cfg"],
                         "cfg_idx": w["cfg_idx"], "seed": seed,
                         "phase": "final", "tag": args.tag, "budget": args.budget, "steps": args.steps,
                         "batch": args.batch, "eval_every": args.eval_every, "task_kw": args.task_kw,
                         **_run_common(args)})
    return jobs


def build_ablation_jobs(args):
    jobs = []
    for variant in args.variants:
        for name, change in ABLATIONS[variant].items():
            cfg = dict(REFERENCES[variant])
            cfg.update(change)
            for k in range(args.seeds):
                jobs.append({"variant": variant, "arm": "ssp", "cfg": cfg, "cfg_idx": 0, "seed": 1000 + k,
                             "phase": f"abl_{name}", "tag": args.tag, "budget": args.budget, "steps": args.steps,
                             "batch": args.batch, "eval_every": args.eval_every, "task_kw": args.task_kw,
                             **_run_common(args)})
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="sweep", choices=["sweep", "ablation"])
    p.add_argument("--variants", nargs="*", default=["delay", "index", "pointer", "path"])
    p.add_argument("--arms", nargs="*", default=["mlp", "posmlp", "gru", "lru", "tf_full", "tf_window",
                                                 "kernel_attn", "ssp"])
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--configs", type=int, default=8)
    p.add_argument("--search-seeds", type=int, default=1,
                   help="search seeds per config (seeds ci + 100*k); 1 = the historical single-seed behaviour")
    p.add_argument("--select", default="last5", choices=["last5", "aulc"],
                   help="how the search winner is chosen: 'last5' = historical short-curve area, 'aulc' = mean area "
                        "under the learning curve over all search seeds, ties by converged accuracy (speed-aware)")
    p.add_argument("--min-conv", type=float, default=0.0,
                   help="floor on the mean converged score for --select aulc (docs/11 step 2: 'AULC subject to "
                        "converged >= x'); 0 = no constraint")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--dense-eval", action="store_true",
                   help="checkpoint every 25 updates up to 400, then every 100 (handoff 09 section 5.2)")
    p.add_argument("--robust", action="store_true",
                   help="also evaluate the robustness splits (index: 8x stream; pointer: cue drop, input noise)")
    p.add_argument("--pool", type=int, default=0,
                   help="low-data regime: train on a fixed pool of N sequences (0 = fresh sequences each step)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--ep-len", type=int, default=16)
    p.add_argument("--chain", action="store_true",
                   help="pointer only: expand each query into max_hops supervised sub-steps (the equal-aid protocol "
                        "for the baseline arms, handoff 09 step 4); accuracy is still measured on the answer")
    p.add_argument("--n-items", type=int, default=8)
    p.add_argument("--train-lags", type=int, nargs="*", default=[1, 2, 3, 4], help="delay: lag range trained on")
    p.add_argument("--n-nodes", type=int, default=8, help="pointer: nodes in the random function (trainability gate "
                   "uses 4 with --train-hops 1 --max-hops 1)")
    p.add_argument("--n-query", type=int, default=8, help="pointer: queries per episode")
    p.add_argument("--train-hops", type=int, nargs="*", default=[1, 2], help="pointer: hop counts trained on")
    p.add_argument("--max-hops", type=int, default=8, help="pointer: largest hop count any split asks for")
    p.add_argument("--n-landmarks", type=int, default=6)
    p.add_argument("--tag", default="main")
    p.add_argument("--arm-cfg", default="{}", help="JSON merged into every sampled config: the condition under test, "
                   "e.g. '{\"rung\": 1, \"init\": \"random\", \"aux_weight\": 0.5}'")
    p.add_argument("--label", default=None, help="arm label written to the rows (default: the arm name); needed when "
                   "one learned arm is run under several conditions")
    args = p.parse_args()
    args.arm_cfg = json.loads(args.arm_cfg)
    if args.label and len(args.arms) != 1:
        raise SystemExit("--label needs exactly one --arms entry")
    args.task_kw = {"ep_len": args.ep_len, "n_items": args.n_items, "n_landmarks": args.n_landmarks,
                    "chain": bool(args.chain), "n_nodes": args.n_nodes, "n_query": args.n_query,
                    "train_hops": tuple(args.train_hops), "max_hops": args.max_hops,
                    "train_lags": tuple(args.train_lags)}
    args.eval_steps = None
    if args.dense_eval:
        import harness as H

        args.eval_steps = H.eval_schedule(args.steps)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    path = OUT / f"{args.mode}_{args.tag}.jsonl"
    if args.mode == "sweep":
        jobs = build_search_jobs(args, args.configs)
        run_jobs(jobs, path, args.workers, "search")
        winners = pick_winners(jsonl_rows(path), args.configs, select=args.select, min_conv=args.min_conv)
        (OUT / f"winners_{args.tag}.json").write_text(json.dumps(
            {f"{k[0]}/{k[1]}": {"cfg": v["cfg"], "cfg_idx": v["cfg_idx"], "n_search_seeds": v["n_search_seeds"],
                                 "select": args.select, "min_conv": args.min_conv} for k, v in winners.items()},
            indent=2))
        jobs = build_final_jobs(args, winners)
        run_jobs(jobs, path, args.workers, "final")
    else:
        import harness  # noqa: F401  (import check only)
        jobs = build_ablation_jobs(args)
        run_jobs(jobs, path, args.workers, "ablation")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
