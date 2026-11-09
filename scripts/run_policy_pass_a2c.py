"""
eval_v3 policy pass for the A2C port: retrain the legacy winner config of one A2C cell on fresh seeds with the
NumPy port (experiments/trial_cartpole_a2c.py) and SAVE each final policy. Same protocol, layout, resumability
and output files as scripts/run_policy_pass.py (which delegates here for cells ending in /a2c):

  20 seeds (3000..3019), env seeded; 15 greedy checkpoints x 10 episodes (every 1000//15 = 66 episodes), score =
  mean of the last 5; a 100-episode greedy eval of the final policy; probe diagnostics (mean V, max preference,
  action gap, centering R-bar) at every checkpoint; the policy is saved, reloaded and replay-verified EXACTLY,
  then uploaded to <study>/policies/seed_<s>.pt. <study> = a2c_<rep>_<centering>_port.

NO hyperparameters are tuned: the winner is conf/a2c_winners/<rep>_<centering>.yaml (tracked copies of the legacy
outputs/hparam_search/best files) plus the legacy constants A2C_BASE_PARAMS.

    python scripts/run_policy_pass_a2c.py --cell ssp/none/a2c --workers 8
    python scripts/run_policy_pass_a2c.py --cell tile_coding/value/a2c --tiny --no-r2 --local-root /tmp/x
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (HERE, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_v3_layout as layout  # noqa: E402
import parallel_jobs as pj  # noqa: E402
import run_policy_pass as rpp  # noqa: E402  (shared constants and summaries)

TINY = dict(trials=24, steps=80, n_evaluations=6, n_eval_episodes=2, final_eval_episodes=4, verify_episodes=3)


def build_params(cell: str, params_yaml: str = None, tiny: bool = False) -> dict:
    from experiments.trial_cartpole_a2c import load_winner_yaml
    params, _old = load_winner_yaml(params_yaml or layout.a2c_winner_path(cell))
    if tiny:
        params.update(trials=TINY["trials"], steps=TINY["steps"], learnTrials=TINY["trials"])
    return params


def build_jobs(cell, params, n_seeds, seed_offset, local_root, upload, eval_every, n_eval_episodes,
               final_eval_episodes, verify_episodes, probe_path, last_k=rpp.LAST_K, save=True) -> list:
    study = layout.study_name(cell)
    jobs = []
    for i in range(n_seeds):
        seed = int(seed_offset) + i
        pkey = layout.policy_key(study, seed)
        jobs.append(dict(
            kind="policy", trial=0, rank=0, seed_idx=i, params=params, seed_offset=int(seed_offset),
            eval_every=int(eval_every), n_eval_episodes=int(n_eval_episodes), last_k=int(last_k),
            final_eval_episodes=int(final_eval_episodes), verify_episodes=int(verify_episodes), save=bool(save),
            upload=bool(upload and save), policy_key=pkey, policy_path=layout.local_path(pkey, local_root),
            probe_path=probe_path))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description="eval_v3 A2C-port policy pass")
    ap.add_argument("--cell", required=True, help="rep/centering/a2c, e.g. ssp/none/a2c")
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=layout.POLICY_N_SEEDS)
    ap.add_argument("--seed-offset", type=int, default=layout.POLICY_SEED_OFFSET)
    ap.add_argument("--params-yaml", default=None, help="winner yaml instead of conf/a2c_winners/<cell>.yaml")
    ap.add_argument("--n-evaluations", type=int, default=rpp.N_EVALUATIONS)
    ap.add_argument("--n-eval-episodes", type=int, default=rpp.N_EVAL_EPISODES)
    ap.add_argument("--final-eval-episodes", type=int, default=rpp.FINAL_EVAL_EPISODES)
    ap.add_argument("--verify-episodes", type=int, default=rpp.VERIFY_EPISODES)
    ap.add_argument("--last-k", type=int, default=rpp.LAST_K)
    ap.add_argument("--tiny", action="store_true", help="smoke budget (24 episodes x <=80 steps, 6 checkpoints)")
    args = ap.parse_args(argv)
    if not layout.is_a2c(args.cell):
        sys.exit(f"{args.cell}: not an A2C cell (expected rep/centering/a2c)")
    if args.no_r2:
        for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            os.environ[k] = ""
    t0 = time.time()
    cell, study, use_r2 = args.cell, layout.study_name(args.cell), not args.no_r2
    if args.tiny:
        args.n_evaluations, args.n_eval_episodes = TINY["n_evaluations"], TINY["n_eval_episodes"]
        args.final_eval_episodes, args.verify_episodes = TINY["final_eval_episodes"], TINY["verify_episodes"]
    params = build_params(cell, args.params_yaml, args.tiny)
    eval_every = max(1, params["trials"] // args.n_evaluations)

    import make_probe_states
    probe_path = make_probe_states.ensure_probe_file(args.local_root, push=use_r2)

    runs_path = layout.local_path(layout.train_runs_key(study), args.local_root)
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    jobs = build_jobs(cell, params, args.n_seeds, args.seed_offset, args.local_root, use_r2, eval_every,
                      args.n_eval_episodes, args.final_eval_episodes, args.verify_episodes, probe_path, args.last_k)
    print(f"[policy-pass-a2c] cell={cell} study={study} seeds={args.seed_offset}..{args.seed_offset + args.n_seeds - 1}"
          f" eval_every={eval_every} r2={use_r2} tiny={args.tiny}", flush=True)

    from policy_pass_worker_a2c import policy_worker_a2c
    recs = pj.run_jobs(jobs, policy_worker_a2c, pj.job_key, runs_path, args.workers,
                       r2_key=layout.train_runs_key(study) if use_r2 else None, label="policy-pass-a2c")
    runs = [recs[pj.job_key(j)] for j in jobs]
    missing = [r["seed"] for r in runs if not r.get("policy_verified")
               or (not use_r2 and not os.path.exists(layout.local_path(r["policy_key"], args.local_root)))]
    if missing:
        sys.exit(f"[policy-pass-a2c] seeds without a verified policy: {missing} "
                 f"(delete their lines from {runs_path} and re-run)")

    summary = {"cell": cell, "study": study, "variant": None, "winner_search_trial": None, "params": params,
               "seed_offset": args.seed_offset, "last_k": args.last_k, "n_eval_episodes": args.n_eval_episodes,
               "final_eval_episodes": args.final_eval_episodes, "tiny": args.tiny, **rpp.summarize_runs(runs),
               "pod": pj.pod_info(args.workers), "wall_min": (time.time() - t0) / 60}
    manifest = {"cell": cell, "study": study, "variant": None, "winner_params": params, "trial_params": params,
                "winner_yaml": os.path.relpath(args.params_yaml or layout.a2c_winner_path(cell), REPO_ROOT),
                "seeds": [args.seed_offset + i for i in range(args.n_seeds)],
                "policy_files": rpp.policy_files(runs), "versions": rpp._versions(), "pod": pj.pod_info(args.workers),
                "code_key": os.environ.get("EVAL_CODE_KEY") or None, "probe_key": layout.PROBE_KEY}
    for key, obj in ((layout.train_summary_key(study), summary), (layout.manifest_key(study), manifest)):
        path = layout.local_path(key, args.local_root)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
        if use_r2 and not pj.push_file(key, path):
            sys.exit(f"[policy-pass-a2c] upload of {key} failed")
    print(f"[policy-pass-a2c] DONE score_mean={summary['score_mean']:.1f} "
          f"final_policy_eval_mean={summary['final_policy_eval_mean']} wall={summary['wall_min']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
