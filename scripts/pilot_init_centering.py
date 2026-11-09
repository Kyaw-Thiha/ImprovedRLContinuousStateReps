"""
Two follow-up experiments to the greedy-rank pilot (docs/handoffs/02-dqn-v4-greedy-refilter-plan.md).
Both reuse the pilot's fresh seeds (1000..1009), so each run pairs with a pilot run at the same seed.

Experiment "init" — does varying the initial weights change the results?
    Same configs as the pilot (ssp/none/deep trial 154, ssp/simple/deep trial 285), but with
    init_seed = run seed (instead of the legacy fixed init=0 the pilot used).
    Compare with the pilot's fixed-init numbers (154: 497 +- 4, 285: 353 +- 104).

Experiment "cent" — is simple centering the cause of ssp/simple/deep's learn-then-decay curve?
    Same hyperparameters as ssp/simple/deep trials 285 and 42, but reward_center_mode="none",
    fixed init (as in the pilot). Compare with the pilot's simple-centering numbers for the same
    trials and seeds.

Usage:
    python scripts/pilot_init_centering.py --dry-run
    python scripts/pilot_init_centering.py --r2 --idle-after --workers 16     # on the pod
    python scripts/pilot_init_centering.py --analyze --pull-results            # locally
"""

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np
import optuna

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_greedy_rank_check as P  # noqa: E402  (R2 helpers, study list, paths)

OUT = os.path.join(P.PILOT_DIR, "init_centering_runs.jsonl")
R2_KEY = P.R2_PREFIX + "init_centering_runs.jsonl"
SEEDS = 10
EXPERIMENTS = [
    # (exp, variant, study, trial, param overrides)
    ("init", "init_vary", "ssp_none_deep", 154, {"init_seed": -1}),
    ("init", "init_vary", "ssp_simple_deep", 285, {"init_seed": -1}),
    ("cent", "cent_none", "ssp_simple_deep", 285, {"reward_center_mode": "none", "init_seed": 0}),
    ("cent", "cent_none", "ssp_simple_deep", 42, {"reward_center_mode": "none", "init_seed": 0}),
]
EXPECTED = SEEDS * len(EXPERIMENTS)

# Reverse centering test: a GOOD ssp/none/deep config (#154) with simple centering switched ON at
# three step sizes (search range was 1e-5..1e-1; ssp/simple/deep's top trials sat near 8e-5).
# Fixed init, so each run pairs with the pilot's centering-none runs of #154 (496.9 +- 4.3).
REVERSE = [("rev", f"rev_b{b:g}", "ssp_none_deep", 154,
            {"reward_center_mode": "simple", "reward_center_beta": b, "init_seed": 0})
           for b in (1e-5, 1e-4, 1e-2)]


def build_jobs(data_root):
    import run_hparam_search_dqn as m
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    eval_freq = max(1, m.BASE_PARAMS["trials"] // P.N_EVALUATIONS)
    trials = {k: {t.number: t for t in P._load_study(k).get_trials(deepcopy=False)} for k in P.STUDIES}
    jobs = []
    for exp, variant, study, trial, over in EXPERIMENTS:
        s = P.STUDIES[study]
        params = dict(m.BASE_PARAMS)
        params.update(m.DQN_DEFAULTS)
        params.update(m.trial_params_from_search_params(s["rep"], s["centering"], trials[study][trial].params, s["arch"]))
        params.update(over)
        for i in range(SEEDS):
            jobs.append(dict(exp=exp, variant=variant, study=study, trial=trial, seed_idx=i, params=params,
                             eval_freq=eval_freq, data_root=data_root,
                             tag=f"{exp}/{variant} study={study} search_trial={trial}"))
    # slowest first (none-centering / ssp_none_deep runs are the longest)
    jobs.sort(key=lambda j: (j["variant"] != "cent_none", j["study"] != "ssp_none_deep", j["trial"], j["seed_idx"]))
    return jobs


def _worker(job):
    import run_hparam_search_dqn as m
    meta = {k: job[k] for k in ("exp", "variant", "study", "trial", "seed_idx")}
    data_dir = os.path.join(job["data_root"], f"{job['variant']}_{job['study']}_{job['trial']}_{job['seed_idx']}")
    os.makedirs(data_dir, exist_ok=True)
    try:
        res = m._run_reeval_seed(job["params"], data_dir, job["seed_idx"], job["tag"],
                                 job["eval_freq"], P.N_EVAL_EPISODES, P.LAST_K)
        return {**meta, **res}
    except Exception as e:
        return {**meta, "error": repr(e)}
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def _done():
    out = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    out.add((r["variant"], r["study"], r["trial"], r["seed_idx"]))
    return out


def run(a):
    os.makedirs(P.PILOT_DIR, exist_ok=True)
    if a.r2:
        s3, bucket = P._r2()
        for s in P.STUDIES.values():
            path = os.path.join(P.PILOT_DIR, f"{s['name']}.db")
            if not os.path.exists(path):
                s3.download_file(bucket, f"stage0_cartpole/optuna_dbs/optuna_studies_{s['name']}.db", path)
                print("fetched", s["name"], flush=True)
        try:
            s3.download_file(bucket, R2_KEY, OUT)
            print("resumed earlier results from R2", flush=True)
        except Exception:
            print("no earlier results on R2 — fresh start", flush=True)
    os.makedirs(a.data_root, exist_ok=True)
    jobs = build_jobs(a.data_root)
    done = _done()
    todo = [j for j in jobs if (j["variant"], j["study"], j["trial"], j["seed_idx"]) not in done]
    print(f"{len(jobs)} runs planned ({EXPECTED} expected); {len(done)} done; {len(todo)} to run, "
          f"{a.workers} workers.", flush=True)
    if a.dry_run:
        return
    n = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                initializer=P._init_worker) as ex, open(OUT, "a") as out:
        for fut in cf.as_completed([ex.submit(_worker, j) for j in todo]):
            r = fut.result()
            out.write(json.dumps(r) + "\n")
            out.flush()
            n += 1
            msg = ("ERROR " + r["error"][:100]) if "error" in r else f"score={r['score']:.1f}"
            print(f"[{n}/{len(todo)}] {r['variant']} {r['study']} #{r['trial']} seed {r['seed_idx']}: {msg}", flush=True)
            if a.r2 and n % 3 == 0:
                s3, bucket = P._r2()
                s3.upload_file(OUT, bucket, R2_KEY)
    if a.r2:
        s3, bucket = P._r2()
        s3.upload_file(OUT, bucket, R2_KEY)
    print("finished.", flush=True)
    if a.idle_after:
        import time
        while True:
            time.sleep(3600)


def _load(path):
    return [r for r in (json.loads(l) for l in open(path) if l.strip()) if "error" not in r]


def analyze(a):
    if a.pull_results:
        s3, bucket = P._r2()
        os.makedirs(P.PILOT_DIR, exist_ok=True)
        s3.download_file(bucket, R2_KEY, OUT)
        s3.download_file(bucket, P.R2_PREFIX + "pilot_runs.jsonl", P.OUT_JSONL)
    new = _load(OUT)
    base = _load(P.OUT_JSONL)
    key = lambda r: (r["study"], r["trial"], r["seed_idx"])
    base_by = {key(r): r for r in base}
    print(f"{len(new)} new runs; baseline = pilot runs (fixed init, original centering)\n")

    def stats(vs):
        v = np.array(vs, float)
        return f"{v.mean():6.1f} +- {v.std():5.1f} (n={len(v)})" if len(v) else "   n/a"

    print("== Experiment init: fixed init (pilot) vs init follows seed, same configs, same seeds ==")
    for (study, trial) in sorted({(r["study"], r["trial"]) for r in new if r["variant"] == "init_vary"}):
        nv = sorted([r for r in new if r["variant"] == "init_vary" and r["study"] == study and r["trial"] == trial],
                    key=lambda r: r["seed_idx"])
        pairs = [(base_by[key(r)]["score"], r["score"]) for r in nv if key(r) in base_by]
        print(f"{study} #{trial}: fixed-init {stats([p[0] for p in pairs])} | vary-init {stats([r['score'] for r in nv])}")
        if pairs:
            d = np.array([b - a_ for a_, b in pairs])
            print(f"    paired diff (vary - fixed): mean {d.mean():+.1f}, sd {d.std():.1f}")
    print("\n== Experiment cent: simple centering (pilot) vs centering none, same hyperparameters/seeds, fixed init ==")
    for (study, trial) in sorted({(r["study"], r["trial"]) for r in new if r["variant"] == "cent_none"}):
        nv = sorted([r for r in new if r["variant"] == "cent_none" and r["study"] == study and r["trial"] == trial],
                    key=lambda r: r["seed_idx"])
        pairs = [(base_by[key(r)]["score"], r["score"]) for r in nv if key(r) in base_by]
        print(f"{study} #{trial}: simple {stats([p[0] for p in pairs])} | none {stats([r['score'] for r in nv])}")
        if pairs:
            d = np.array([b - a_ for a_, b in pairs])
            print(f"    paired diff (none - simple): mean {d.mean():+.1f}, sd {d.std():.1f}")
        # shape of the learning curve: does 'none' hold up through training?
        H = np.array([r["eval_history"] for r in nv]); Hb = np.array([base_by[key(r)]["eval_history"] for r in nv if key(r) in base_by])
        if len(H) and len(Hb):
            print("    mean eval curve simple:", np.round(Hb.mean(0)).astype(int))
            print("    mean eval curve none  :", np.round(H.mean(0)).astype(int))
    rev = [r for r in new if r["variant"].startswith("rev_")]
    if rev:
        print("\n== Reverse test: ssp/none/deep #154 (good config) with simple centering ON vs the pilot's centering none ==")
        base154 = [base_by[k] for k in sorted(base_by) if k[0] == "ssp_none_deep" and k[1] == 154]
        print(f"centering none (pilot, fixed init): {stats([r['score'] for r in base154])}")
        for v in sorted({r["variant"] for r in rev}):
            nv = sorted([r for r in rev if r["variant"] == v], key=lambda r: r["seed_idx"])
            pairs = [(base_by[key(r)]["score"], r["score"]) for r in nv if key(r) in base_by]
            line = f"{v:12s} simple centering: {stats([r['score'] for r in nv])}"
            if pairs:
                d = np.array([b - a_ for a_, b in pairs]); line += f" | paired diff vs none: {d.mean():+.1f} (sd {d.std():.1f})"
            print(line)
            print("    mean eval curve:", np.round(np.mean([r["eval_history"] for r in nv], 0)).astype(int))
        if base154:
            print("    none (pilot)   :", np.round(np.mean([r["eval_history"] for r in base154], 0)).astype(int))
    t = np.array([r["total_time"] for r in new]) / 60
    print(f"\nrun time (16 parallel): median {np.median(t):.1f} min, max {t.max():.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--data-root", default="/tmp/init_cent_data")
    ap.add_argument("--r2", action="store_true")
    ap.add_argument("--idle-after", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--pull-results", action="store_true")
    ap.add_argument("--exp-set", choices=["init_cent", "reverse"], default="init_cent")
    a = ap.parse_args()
    if a.exp_set == "reverse":
        EXPERIMENTS = REVERSE
        EXPECTED = SEEDS * len(EXPERIMENTS)
        OUT = os.path.join(P.PILOT_DIR, "centering_reverse_runs.jsonl")
        R2_KEY = P.R2_PREFIX + "centering_reverse_runs.jsonl"
    analyze(a) if a.analyze else run(a)
