"""
eval_v3 policy pass: retrain the winner config of one DQN cell on fresh seeds and SAVE each final policy.

Normal pass  : 20 seeds (3000..3019), every policy saved, reloaded from disk and replay-verified EXACTLY,
               uploaded to R2 under <study>/policies/seed_<s>.pt, records in <study>/train_runs.jsonl.
Variant pass : any of --variant-buffer-size / --variant-target-update-freq (the DQN-settings test): 10 seeds,
               NO policy is saved, results go to dqn_settings/<study>/<variant>/.

Resumable (parallel_jobs.run_jobs): a seed is DONE only when its record is error-free, i.e. the policy
reloaded exactly and the upload succeeded (or --no-r2 and the file exists locally).

    python scripts/run_policy_pass.py --cell ssp/none/deep --workers 4
    python scripts/run_policy_pass.py --cell ssp/none/deep --variant-buffer-size 50000
    python scripts/run_policy_pass.py --cell discrete/none/linear --tiny --params-yaml w.yaml --no-r2 \
        --local-root /tmp/x     # fully offline smoke budget
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))

import eval_v3_layout as layout  # noqa: E402
import parallel_jobs as pj  # noqa: E402

SOLVED_THRESHOLD = 475.0
LAST_K = 5
N_EVALUATIONS = 15
N_EVAL_EPISODES = 10
FINAL_EVAL_EPISODES = 100
VERIFY_EPISODES = 20
TINY = dict(trials=24, steps=80, n_evaluations=6, n_eval_episodes=2, final_eval_episodes=4,
            verify_episodes=3, learning_starts=50)


# ---------------------------------------------------------------------------------------------
# Pure logic (unit-tested; no training, no network)
# ---------------------------------------------------------------------------------------------
def base_params() -> dict:
    """BASE_PARAMS + DQN_DEFAULTS exactly as final_evaluate() starts from (single source of truth)."""
    import run_hparam_search_dqn as m
    p = dict(m.BASE_PARAMS)
    p.update(m.DQN_DEFAULTS)
    return p


def build_trial_params(cell: str, search_params: dict, buffer_size: int = None,
                       target_update_freq: int = None, tiny: bool = False) -> dict:
    """Identical to final_evaluate(): BASE_PARAMS, DQN_DEFAULTS, then the winner's params converted with
    trial_params_from_search_params (idempotent on an already-converted v2 yaml). Variant overrides last."""
    import run_hparam_search_dqn as m
    rep, cen, arch = layout.parse_cell(cell)
    params = base_params()
    params.update(m.trial_params_from_search_params(rep, cen, dict(search_params), arch))
    if tiny:
        params.update(trials=TINY["trials"], steps=TINY["steps"], learnTrials=TINY["trials"])
        # With 24 x <=80 steps the default learning_starts (1000) would mean no update ever happens.
        params["learning_starts"] = TINY["learning_starts"]
    if buffer_size is not None:
        params["buffer_size"] = int(buffer_size)
    if target_update_freq is not None:
        params["target_update_freq"] = int(target_update_freq)
    return params


def variant_of(buffer_size, target_update_freq):
    """None for the normal pass, else the layout's variant folder name."""
    if buffer_size is None and target_update_freq is None:
        return None
    return layout.variant_name(buffer_size, target_update_freq)


def read_winner_yaml(path: str) -> dict:
    """Returns {"params": search params, "winner_trial": int|None}. Accepts the payload written by
    save_best_params ({params: {...}, ...}) or a bare params mapping (hand-written test files)."""
    with open(path) as f:
        payload = yaml.safe_load(f) or {}
    params = payload["params"] if isinstance(payload.get("params"), dict) else payload
    wt = payload.get("winner_search_trial", payload.get("winner_trial"))
    return {"params": dict(params), "winner_trial": None if wt is None else int(wt)}


def build_jobs(cell: str, params: dict, winner_trial: int, n_seeds: int, seed_offset: int, local_root: str,
               save: bool, upload: bool, eval_freq: int, n_eval_episodes: int, final_eval_episodes: int,
               verify_episodes: int, probe_path: str, data_dir: str, variant: str = None,
               last_k: int = LAST_K) -> list:
    study = layout.study_name(cell)
    jobs = []
    for i in range(n_seeds):
        seed = int(seed_offset) + i
        pkey = layout.policy_key(study, seed)
        jobs.append(dict(
            kind="policy", trial=int(winner_trial), rank=0, seed_idx=i, params=params, data_dir=data_dir,
            seed_offset=int(seed_offset), eval_freq=int(eval_freq), n_eval_episodes=int(n_eval_episodes),
            last_k=int(last_k), final_eval_episodes=int(final_eval_episodes),
            verify_episodes=int(verify_episodes), save=bool(save), upload=bool(upload and save),
            policy_key=pkey, policy_path=layout.local_path(pkey, local_root), probe_path=probe_path))
    return jobs


def _col(runs, k):
    return [float(r[k]) for r in runs if r.get(k) is not None]


def _stats(xs):
    return ({"mean": float(np.mean(xs)), "std": float(np.std(xs)), "min": float(np.min(xs)),
             "max": float(np.max(xs))} if xs else None)


def summarize_runs(runs: list) -> dict:
    """Aggregate per-seed records (np.std convention, like final_evaluate)."""
    scores = _col(runs, "score")
    last1 = _col(runs, "last1")
    fin = _col(runs, "final_eval_reward")
    probe_q, probe_gap = [], []
    for r in runs:
        ph = r.get("probe_history") or []
        if ph:
            probe_q.append(float(ph[-1]["mean_max_q"]))
            probe_gap.append(float(ph[-1]["mean_action_gap"]))
    out = {
        "n_seeds": len(runs),
        "score_mean": float(np.mean(scores)) if scores else None,
        "score_std": float(np.std(scores)) if scores else None,
        "score_min": float(np.min(scores)) if scores else None,
        "score_max": float(np.max(scores)) if scores else None,
        "per_seed_scores": scores,
        "share_solved_lastk": float(np.mean([s >= SOLVED_THRESHOLD for s in scores])) if scores else None,
        "last1_mean": float(np.mean(last1)) if last1 else None,
        "last1_std": float(np.std(last1)) if last1 else None,
        "per_seed_last1": last1,
        "final_policy_eval_mean": float(np.mean(fin)) if fin else None,
        "final_policy_eval_std": float(np.std(fin)) if fin else None,
        "per_seed_final_policy_eval": fin,
        "share_solved_final_policy": float(np.mean([x >= SOLVED_THRESHOLD for x in fin])) if fin else None,
        "online_return_mean": float(np.mean(_col(runs, "terminal_reward"))) if _col(runs, "terminal_reward") else None,
        "probe": {"per_seed_final_mean_max_q": probe_q, "per_seed_final_mean_action_gap": probe_gap,
                  "mean_max_q": _stats(probe_q), "mean_action_gap": _stats(probe_gap)},
        "cost": {
            "train_time_mean_s": float(np.mean(_col(runs, "train_time"))) if _col(runs, "train_time") else None,
            "eval_time_mean_s": float(np.mean(_col(runs, "eval_time"))) if _col(runs, "eval_time") else None,
            "total_time_mean_s": float(np.mean(_col(runs, "total_time"))) if _col(runs, "total_time") else None,
            "note": "run times are measured with `workers` runs sharing one pod, i.e. contended",
            "dimensionality": runs[0].get("dimensionality") if runs else None,
            "n_params": runs[0].get("n_params") if runs else None,
            "env_steps_mean": float(np.mean(_col(runs, "env_steps"))) if _col(runs, "env_steps") else None,
            "n_updates_mean": float(np.mean(_col(runs, "n_updates"))) if _col(runs, "n_updates") else None,
        },
    }
    return out


def policy_files(runs: list) -> list:
    return [{"seed": r["seed"], "key": r["policy_key"], "sha256": r["policy_sha256"],
             "bytes": r["policy_bytes"], "verify_max_abs_diff": r.get("verify_max_abs_diff")}
            for r in runs if r.get("policy_key")]


# ---------------------------------------------------------------------------------------------
def _resolve_winner(cell, args, study):
    """(search params, winner trial number)."""
    if args.params_yaml:
        path = args.params_yaml
    else:
        path = layout.local_path(layout.shared_key(f"winner_{study}.yaml"), args.local_root)
        st = pj.pull_file(layout.winner_params_key(cell), path)
        if st != "ok":
            sys.exit(f"cannot fetch winner params {layout.winner_params_key(cell)}: {st}")
    w = read_winner_yaml(path)
    trial = args.winner_trial if args.winner_trial is not None else w["winner_trial"]
    if trial is None and not args.no_r2 and not args.params_yaml:
        tmp = layout.local_path(layout.shared_key(f"final_eval_summary_{study}.json"), args.local_root)
        key = f"{layout.STAGE}/refilter_v2/{study}/final_eval_summary.json"
        if pj.pull_file(key, tmp) == "ok":
            try:
                with open(tmp) as f:
                    trial = int(json.load(f)["winner_search_trial"])
            except Exception:  # noqa: BLE001
                trial = None
    if trial is None:
        print("[policy-pass] WARNING: winner search trial unknown, using 0 (pass --winner-trial)", flush=True)
        trial = 0
    return w["params"], int(trial)


def _versions():
    import platform
    import gymnasium
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "gymnasium": gymnasium.__version__, "numpy": np.__version__}


def main():
    # A2C-port cells ("rep/centering/a2c") have their own runner; delegating here keeps the pod launcher
    # (`launch_cell_pod.py --job policy_pass --cells ssp/none/a2c`) and the documented command unchanged.
    if any(a == "--cell" and i + 1 < len(sys.argv) and sys.argv[i + 1].endswith("/a2c")
           or a.startswith("--cell=") and a.endswith("/a2c") for i, a in enumerate(sys.argv)):
        import run_policy_pass_a2c
        return run_policy_pass_a2c.main(sys.argv[1:])
    ap = argparse.ArgumentParser(description="eval_v3 policy pass / DQN-settings variant pass")
    ap.add_argument("--cell", required=True, help="rep/centering/arch, e.g. ssp/none/deep")
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--seed-offset", type=int, default=layout.POLICY_SEED_OFFSET)
    ap.add_argument("--variant-buffer-size", type=int, default=None)
    ap.add_argument("--variant-target-update-freq", type=int, default=None)
    ap.add_argument("--params-yaml", default=None, help="local winner yaml instead of fetching from R2")
    ap.add_argument("--winner-trial", type=int, default=None, help="winner search trial (default: from yaml/R2)")
    ap.add_argument("--n-evaluations", type=int, default=N_EVALUATIONS)
    ap.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES)
    ap.add_argument("--final-eval-episodes", type=int, default=FINAL_EVAL_EPISODES)
    ap.add_argument("--verify-episodes", type=int, default=VERIFY_EPISODES)
    ap.add_argument("--last-k", type=int, default=LAST_K)
    ap.add_argument("--tiny", action="store_true", help="smoke budget (24 trials x 80 steps, 6 checkpoints)")
    args = ap.parse_args()

    if args.no_r2:
        # Blank R2 creds so parallel_jobs neither reads .env nor talks to R2 (also in spawned workers).
        for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            os.environ[k] = ""
    import time
    t0 = time.time()
    cell = args.cell
    study = layout.study_name(cell)
    variant = variant_of(args.variant_buffer_size, args.variant_target_update_freq)
    n_seeds = args.n_seeds if args.n_seeds is not None else (
        layout.VARIANT_N_SEEDS if variant else layout.POLICY_N_SEEDS)
    if args.tiny:
        args.n_evaluations, args.n_eval_episodes = TINY["n_evaluations"], TINY["n_eval_episodes"]
        args.final_eval_episodes, args.verify_episodes = TINY["final_eval_episodes"], TINY["verify_episodes"]
    use_r2 = not args.no_r2

    search_params, winner_trial = _resolve_winner(cell, args, study)
    params = build_trial_params(cell, search_params, args.variant_buffer_size,
                                args.variant_target_update_freq, tiny=args.tiny)
    eval_freq = max(1, params["trials"] // args.n_evaluations)

    import make_probe_states
    probe_path = make_probe_states.ensure_probe_file(args.local_root, push=use_r2)

    runs_path = layout.local_path(layout.train_runs_key(study, variant), args.local_root)
    folder = os.path.dirname(runs_path)
    data_dir = os.path.join(folder, "_trial_data")
    os.makedirs(data_dir, exist_ok=True)
    save = variant is None
    jobs = build_jobs(cell, params, winner_trial, n_seeds, args.seed_offset, args.local_root, save=save,
                      upload=use_r2, eval_freq=eval_freq, n_eval_episodes=args.n_eval_episodes,
                      final_eval_episodes=args.final_eval_episodes, verify_episodes=args.verify_episodes,
                      probe_path=probe_path, data_dir=data_dir, variant=variant, last_k=args.last_k)
    print(f"[policy-pass] cell={cell} study={study} variant={variant} winner_trial={winner_trial} "
          f"seeds={args.seed_offset}..{args.seed_offset + n_seeds - 1} save_policies={save} "
          f"r2={use_r2} tiny={args.tiny}", flush=True)

    from policy_pass_worker import policy_worker
    recs = pj.run_jobs(jobs, policy_worker, pj.job_key, runs_path, args.workers,
                       r2_key=layout.train_runs_key(study, variant) if use_r2 else None, label="policy-pass")
    runs = [recs[pj.job_key(j)] for j in jobs]

    if save:
        # Belt and braces: a record without its policy file locally (and --no-r2) is not a finished seed.
        missing = [r["seed"] for r in runs if not r.get("policy_verified")
                   or (not use_r2 and not os.path.exists(layout.local_path(r["policy_key"], args.local_root)))]
        if missing:
            sys.exit(f"[policy-pass] seeds without a verified policy: {missing} "
                     f"(delete their lines from {runs_path} and re-run)")

    summary = {
        "cell": cell, "study": study, "variant": variant, "winner_search_trial": winner_trial,
        "params": params, "seed_offset": args.seed_offset, "last_k": args.last_k,
        "n_eval_episodes": args.n_eval_episodes, "final_eval_episodes": args.final_eval_episodes,
        "tiny": args.tiny, **summarize_runs(runs),
        "pod": pj.pod_info(args.workers), "wall_min": (time.time() - t0) / 60,
    }
    manifest = {
        "cell": cell, "study": study, "variant": variant, "winner_params": search_params,
        "trial_params": params, "winner_search_trial": winner_trial,
        "seeds": [args.seed_offset + i for i in range(n_seeds)],
        "policy_files": policy_files(runs) if save else [],
        "versions": _versions(), "pod": pj.pod_info(args.workers),
        "code_key": os.environ.get("EVAL_CODE_KEY") or None,
        "probe_key": layout.PROBE_KEY,
    }
    for key, obj in ((layout.train_summary_key(study, variant), summary),
                     (layout.manifest_key(study, variant), manifest)):
        path = layout.local_path(key, args.local_root)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        if use_r2 and not pj.push_file(key, path):
            sys.exit(f"[policy-pass] upload of {key} failed")
    print(f"[policy-pass] DONE score_mean={summary['score_mean']:.1f} "
          f"final_policy_eval_mean={summary['final_policy_eval_mean']} wall={summary['wall_min']:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
