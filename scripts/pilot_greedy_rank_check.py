"""
Pilot for docs/handoffs/02-dqn-v4-greedy-refilter-plan.md.

Question: does a search trial's last-K greedy-checkpoint mean (from its single search run)
predict the mean greedy score over fresh seeds? And how fast does the multi-seed score fall
with search rank (is it worth going below the top ranks)?

Part A (known candidates, 10 seeds each):
    ssp/simple/deep trials 125, 306, 285, 457, 42, 56, 78
    ssp/none/deep trials 154, 344 + the top-ranked trial by the new sort key
Part B (rank-band calibration, 5 seeds each) in ssp/simple/deep:
    5 random trials from each of: top 1%, 5-10%, 10-25%, around the median (by last-5 mean)

Every run goes through run_hparam_search_dqn._run_reeval_seed — the exact code path the
production staged reeval uses — so this also exercises it. Results are appended to a JSONL
as each run finishes, so an interrupted pilot resumes without redoing finished runs.

The Optuna study DBs are read from outputs/pilot/<study>.db (download them first from
stage0_cartpole/optuna_dbs/optuna_studies_<study>.db). Nothing is written to R2.

Usage:
    python scripts/pilot_greedy_rank_check.py --dry-run
    python scripts/pilot_greedy_rank_check.py --workers 12
    python scripts/pilot_greedy_rank_check.py --analyze
"""

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys

import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments"))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_DIR = os.path.join(REPO_ROOT, "outputs", "pilot")
OUT_JSONL = os.path.join(PILOT_DIR, "pilot_runs.jsonl")

STUDIES = {
    "ssp_simple_deep": dict(rep="ssp", centering="simple", arch="deep",
                            name="dqn_ssp_simple_deep_v4_hparam_search"),
    "ssp_none_deep": dict(rep="ssp", centering="none", arch="deep",
                          name="dqn_ssp_none_deep_v4_hparam_search"),
}
KNOWN_A = {
    "ssp_simple_deep": [125, 306, 285, 457, 42, 56, 78],
    "ssp_none_deep": [154, 344],
}
N_EVALUATIONS = 15
N_EVAL_EPISODES = 10
LAST_K = 5
SEEDS_A = 10
SEEDS_B = 5


def _load_study(key):
    s = STUDIES[key]
    path = os.path.join(PILOT_DIR, f"{s['name']}.db")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — download it from R2 first (see docstring).")
    return optuna.load_study(study_name=s["name"], storage=f"sqlite:///{path}")


def build_jobs(data_root):
    import run_hparam_search_dqn as m
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    eval_freq = max(1, m.BASE_PARAMS["trials"] // N_EVALUATIONS)
    rng = random.Random(1)
    plan = {}   # (study_key, trial_number) -> dict(group, band, n_seeds, row)

    for key in STUDIES:
        study = _load_study(key)
        ranked = m.rank_trials_by_checkpoints(study, N_EVALUATIONS, LAST_K)
        by_num = {r["number"]: (i, r) for i, r in enumerate(ranked)}
        n = len(ranked)

        a_trials = list(KNOWN_A[key])
        if key == "ssp_none_deep":
            top = ranked[0]["number"]
            if top not in a_trials:
                a_trials.append(top)
        for num in a_trials:
            i, r = by_num[num]
            plan[(key, num)] = dict(group="A", band="known" if num in KNOWN_A[key] else "top_ranked",
                                    n_seeds=SEEDS_A, rank=i, row=r)

        if key == "ssp_simple_deep":
            bands = {
                "top1pct": (0, math.ceil(0.01 * n)),
                "5-10pct": (math.ceil(0.05 * n), math.ceil(0.10 * n)),
                "10-25pct": (math.ceil(0.10 * n), math.ceil(0.25 * n)),
                "median": (int(0.45 * n), int(0.55 * n)),
            }
            for band, (lo, hi) in bands.items():
                for i in rng.sample(range(lo, hi), min(5, hi - lo)):
                    r = ranked[i]
                    k = (key, r["number"])
                    if k in plan:       # already a Part A trial (more seeds already)
                        plan[k]["band"] = plan[k]["band"] + f"+{band}"
                        continue
                    plan[k] = dict(group="B", band=band, n_seeds=SEEDS_B, rank=i, row=r)

    jobs = []
    trials_by_study = {key: {tt.number: tt for tt in _load_study(key).get_trials(deepcopy=False)}
                       for key in STUDIES}
    for (key, num), p in plan.items():
        s = STUDIES[key]
        t = trials_by_study[key][num]
        params = dict(m.BASE_PARAMS)
        params.update(m.DQN_DEFAULTS)
        params.update(m.trial_params_from_search_params(s["rep"], s["centering"], t.params, s["arch"]))
        for seed_idx in range(p["n_seeds"]):
            jobs.append(dict(
                study=key, trial=num, seed_idx=seed_idx, group=p["group"], band=p["band"],
                search_rank=p["rank"], search_last5=p["row"]["last_k"],
                search_last1=p["row"]["last1"], search_stable_from=p["row"]["stable_from"],
                params=params, eval_freq=eval_freq, n_eval_episodes=N_EVAL_EPISODES, last_k=LAST_K,
                tag=f"pilot study={key} search_trial={num} band={p['band']}",
                data_root=data_root,
            ))
    # Slower cell first so the tail of the schedule is short runs.
    jobs.sort(key=lambda j: (j["study"] != "ssp_none_deep", j["trial"], j["seed_idx"]))
    return jobs


def _init_worker():
    os.environ["TQDM_DISABLE"] = "1"


def _worker(job):
    import run_hparam_search_dqn as m
    meta = {k: job[k] for k in ("study", "trial", "seed_idx", "group", "band", "search_rank",
                                "search_last5", "search_last1", "search_stable_from")}
    data_dir = os.path.join(job["data_root"], f"{job['study']}_{job['trial']}_{job['seed_idx']}")
    os.makedirs(data_dir, exist_ok=True)
    try:
        res = m._run_reeval_seed(job["params"], data_dir, job["seed_idx"], job["tag"],
                                 job["eval_freq"], job["n_eval_episodes"], job["last_k"])
        return {**meta, **res}
    except Exception as e:  # keep the pilot going; failures are visible in the JSONL
        return {**meta, "error": repr(e)}
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _done_keys():
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        done.add((r["study"], r["trial"], r["seed_idx"]))
    return done


R2_PREFIX = "stage0_cartpole/pilot/"   # NEW prefix: nothing existing is ever overwritten


def _r2():
    import boto3
    env = dict(os.environ)
    if "R2_ACCOUNT_ID" not in env:   # local use (e.g. --analyze --pull-results): read the repo .env
        try:
            with open(os.path.join(REPO_ROOT, ".env")) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env.setdefault(k, v.strip().strip('"').strip("'"))
        except FileNotFoundError:
            pass
    s3 = boto3.client(
        "s3", endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"])
    return s3, env["R2_BUCKET"]


def fetch_inputs():
    """Pod mode: read-only download of the production study DBs, plus any earlier pilot results
    (resume after a pod restart)."""
    s3, bucket = _r2()
    for s in STUDIES.values():
        path = os.path.join(PILOT_DIR, f"{s['name']}.db")
        if not os.path.exists(path):
            s3.download_file(bucket, f"stage0_cartpole/optuna_dbs/optuna_studies_{s['name']}.db", path)
            print(f"fetched {s['name']}.db", flush=True)
    try:
        s3.download_file(bucket, R2_PREFIX + "pilot_runs.jsonl", OUT_JSONL)
        print("resumed earlier results from R2", flush=True)
    except Exception:
        print("no earlier pilot results on R2 — starting fresh", flush=True)


def push_results():
    s3, bucket = _r2()
    s3.upload_file(OUT_JSONL, bucket, R2_PREFIX + "pilot_runs.jsonl")


def run(args):
    os.makedirs(PILOT_DIR, exist_ok=True)
    if args.r2:
        fetch_inputs()
    data_root = os.path.join(args.data_root)
    os.makedirs(data_root, exist_ok=True)
    jobs = build_jobs(data_root)
    done = _done_keys()
    todo = [j for j in jobs if (j["study"], j["trial"], j["seed_idx"]) not in done]
    n_trials = len({(j["study"], j["trial"]) for j in jobs})
    print(f"{len(jobs)} runs over {n_trials} trials planned; {len(done)} already done; "
          f"{len(todo)} to run with {args.workers} workers.", flush=True)
    for key in STUDIES:
        c = {}
        for j in jobs:
            if j["study"] == key:
                c.setdefault((j["group"], j["band"]), set()).add(j["trial"])
        print(f"  {key}: " + "; ".join(f"{g}/{b}: {len(t)} trials" for (g, b), t in sorted(c.items())))
    if args.dry_run:
        return
    n_done = 0
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                initializer=_init_worker) as ex, open(OUT_JSONL, "a") as out:
        futs = [ex.submit(_worker, j) for j in todo]
        for fut in cf.as_completed(futs):
            r = fut.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            n_done += 1
            tag = "ERROR " + r["error"][:80] if "error" in r else f"score={r['score']:.1f} last1={r['last1']:.0f}"
            print(f"[{n_done}/{len(todo)}] {r['study']} trial {r['trial']} seed {r['seed_idx']}: {tag}",
                  flush=True)
            if args.r2 and n_done % 5 == 0:
                push_results()
    if args.r2:
        push_results()
    print("pilot finished.", flush=True)
    if args.idle_after:
        import time
        print("idling (pod mode) — fetch results from R2, then delete the pod.", flush=True)
        while True:
            time.sleep(3600)


def analyze(_args):
    from scipy.stats import spearmanr
    recs = []
    with open(OUT_JSONL) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)} finished runs, {len(errs)} errors")
    per = {}
    for r in recs:
        per.setdefault((r["study"], r["trial"]), []).append(r)
    rows = []
    for (study, trial), rs in per.items():
        rs = sorted(rs, key=lambda r: r["seed_idx"])
        sc = [r["score"] for r in rs]
        sc5 = sc[:SEEDS_B]
        rows.append(dict(
            study=study, trial=trial, n=len(rs), band=rs[0]["band"], rank=rs[0]["search_rank"],
            search_last5=rs[0]["search_last5"], search_last1=rs[0]["search_last1"],
            mean=float(np.mean(sc)), std=float(np.std(sc)), mean5=float(np.mean(sc5)),
            last1_mean=float(np.mean([r["last1"] for r in rs])),
            online=float(np.nanmean([r["terminal_reward"] for r in rs])),
            min_s=float(np.min(sc)), max_s=float(np.max(sc)),
            t=float(np.mean([r["total_time"] for r in rs])),
        ))
    for study in STUDIES:
        sub = sorted([r for r in rows if r["study"] == study], key=lambda r: r["rank"])
        if not sub:
            continue
        print(f"\n== {study} ==")
        print(f"{'trial':>5} {'rank':>4} {'band':<18} {'n':>2} {'search_last5':>12} {'search_last1':>12} "
              f"{'reeval greedy last5 mean':>24} {'std':>6} {'min':>5} {'max':>5} {'reeval last1':>12} {'online':>7} {'min/run':>7}")
        for r in sub:
            print(f"{r['trial']:5d} {r['rank']:4d} {r['band']:<18} {r['n']:2d} {r['search_last5']:12.1f} "
                  f"{r['search_last1']:12.1f} {r['mean']:24.1f} {r['std']:6.1f} {r['min_s']:5.0f} {r['max_s']:5.0f} "
                  f"{r['last1_mean']:12.1f} {r['online']:7.1f} {r['t']/60:7.2f}")
        if len(sub) >= 4:
            rho, p = spearmanr([r["search_last5"] for r in sub], [r["mean5"] for r in sub])
            rho_r, p_r = spearmanr([-r["rank"] for r in sub], [r["mean5"] for r in sub])
            print(f"Spearman(search last-5, reeval mean over first {SEEDS_B} seeds) over {len(sub)} trials: "
                  f"rho={rho:.2f} p={p:.3f}; vs -rank: rho={rho_r:.2f} p={p_r:.3f}")
    sub = [r for r in rows if r["study"] == "ssp_simple_deep"]
    bands = {}
    for r in sub:
        for b in r["band"].replace("known", "known").split("+"):
            if b not in ("known", "top_ranked"):
                bands.setdefault(b, []).append(r["mean5"])
    if bands:
        print("\nssp_simple_deep: mean over trials of (reeval greedy last-5 mean, first 5 seeds), by search-rank band:")
        for b in ("top1pct", "5-10pct", "10-25pct", "median"):
            if b in bands:
                v = bands[b]
                print(f"  {b:<9} n={len(v)}  mean={np.mean(v):6.1f}  min={np.min(v):6.1f}  max={np.max(v):6.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12,
                    help="Parallel runs. Pass this explicitly on pods (os.cpu_count() there reports the host).")
    ap.add_argument("--data-root", default="/tmp/pilot_data")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--r2", action="store_true",
                    help="Pod mode: fetch the study DBs + any earlier results from R2 first, push "
                         f"results to R2 (prefix {R2_PREFIX}) as runs finish.")
    ap.add_argument("--idle-after", action="store_true",
                    help="Pod mode: sleep forever after finishing, so RunPod's restart-on-exit does not "
                         "re-launch the entrypoint in a loop.")
    ap.add_argument("--pull-results", action="store_true",
                    help="With --analyze: download pilot_runs.jsonl from R2 first.")
    a = ap.parse_args()
    if a.analyze:
        if a.pull_results:
            os.makedirs(PILOT_DIR, exist_ok=True)
            s3, bucket = _r2()
            s3.download_file(bucket, R2_PREFIX + "pilot_runs.jsonl", OUT_JSONL)
        analyze(a)
    else:
        run(a)
