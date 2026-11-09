"""
Robustness suite on the SAVED final policies of one eval_v3 cell (test-time perturbations only; nothing
is retrained).

One job = (policy seed, condition): the policy is rolled out for `--episodes` greedy episodes under that
perturbation (see rl/perturbations.py; episode seeds depend only on the condition name, so every policy of
every cell sees the same initial states = common random numbers). Records go to
<local-root>/<study>/robustness_runs.jsonl (R2 mirror: layout.robustness_key(study)) and the run is
resumable: a second invocation skips every (seed, condition) already recorded.

Usage:
    python scripts/run_robustness.py --cell ssp/none/deep --workers 4
    python scripts/run_robustness.py --cell tile_coding/value/linear --no-r2 --local-root /tmp/ev3 \
        --conditions baseline,gravity_x1.2 --episodes 5
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_v3_layout as layout  # noqa: E402
import parallel_jobs as pj  # noqa: E402


class MissingPolicyError(RuntimeError):
    """Raised when saved policies are absent: robustness is only meaningful on the exact saved weights,
    so we refuse to continue instead of silently evaluating a subset."""


def rob_key(x: dict) -> tuple:
    """Same key for a job dict and the record its worker returns."""
    return ("rob", int(x["seed_idx"]), x["condition"])


def policy_paths(study: str, local_root: str, n_seeds: int, seed_offset: int) -> list:
    return [layout.local_path(layout.policy_key(study, seed_offset + i), local_root) for i in range(n_seeds)]


def ensure_policies(study: str, local_root: str, n_seeds: int, seed_offset: int, no_r2: bool) -> list:
    """Local paths of all policies; downloads the missing ones from R2 unless no_r2. Raises
    MissingPolicyError listing what is absent."""
    paths = policy_paths(study, local_root, n_seeds, seed_offset)
    for i, path in enumerate(paths):
        if os.path.exists(path):
            continue
        if not no_r2:
            key = layout.policy_key(study, seed_offset + i)
            st = pj.pull_file(key, path)
            print(f"[rob] pull {key}: {st}", flush=True)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        where = "locally (--no-r2)" if no_r2 else "locally and on R2"
        raise MissingPolicyError(
            f"{len(missing)} of {len(paths)} saved policies for {study} not found {where}; "
            f"first missing: {missing[0]}. Run the retrain pass first.")
    return paths


def build_jobs(study: str, local_root: str, n_seeds: int, seed_offset: int, conditions: list,
               episodes: int, noise_sigma) -> list:
    """Seed-major order so a worker handling consecutive jobs reuses the loaded policy."""
    sigma = None if noise_sigma is None else [float(v) for v in np.asarray(noise_sigma).reshape(-1)]
    jobs = []
    for i in range(n_seeds):
        seed = seed_offset + i
        path = layout.local_path(layout.policy_key(study, seed), local_root)
        for cond in conditions:
            jobs.append({"kind": "rob", "study": study, "seed_idx": i, "seed": seed, "condition": cond,
                         "n_episodes": int(episodes), "policy_path": path, "noise_sigma": sigma})
    return jobs


# ---------------------------------------------------------------------------------------------
# Worker (top level so spawn can pickle it)
# ---------------------------------------------------------------------------------------------
_POLICY_CACHE = {}  # path -> Policy; one entry only (jobs are seed-major) to bound memory


def _get_policy(path: str):
    if path not in _POLICY_CACHE:
        from rl.policy_io import load_policy
        _POLICY_CACHE.clear()
        _POLICY_CACHE[path] = load_policy(path)
    return _POLICY_CACHE[path]


def robustness_worker(job: dict) -> dict:
    """One (seed, condition) rollout set. Never raises: failures come back as {..., "error": ...}."""
    meta = {"kind": "rob", "study": job["study"], "seed_idx": int(job["seed_idx"]), "seed": int(job["seed"]),
            "condition": job["condition"]}
    try:
        for p in (HERE, os.path.dirname(HERE)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from rl.perturbations import family_of, run_condition
        policy = _get_policy(job["policy_path"])
        sigma = None if job.get("noise_sigma") is None else np.asarray(job["noise_sigma"], dtype=np.float64)
        returns = [float(r) for r in run_condition(policy, job["condition"], n_episodes=job["n_episodes"],
                                                    noise_sigma=sigma)]
        arr = np.asarray(returns, dtype=np.float64)
        return {**meta, "family": family_of(job["condition"]), "n_episodes": int(job["n_episodes"]),
                "returns": returns, "mean": float(arr.mean()), "std": float(np.std(arr)),
                "min": float(arr.min()), "p10": float(np.percentile(arr, 10))}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {**meta, "error": repr(e) + " | " + traceback.format_exc()[-400:]}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="rep/centering/arch, e.g. ssp/none/deep")
    ap.add_argument("--local-root", default=None, help="default: <repo>/outputs/eval_v3")
    ap.add_argument("--no-r2", action="store_true", help="never touch R2; policies must exist locally")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=layout.POLICY_N_SEEDS)
    ap.add_argument("--seed-offset", type=int, default=layout.POLICY_SEED_OFFSET)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--conditions", default=None, help="comma separated subset; default all conditions")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from rl.perturbations import ALL_CONDITIONS
    layout.parse_cell(args.cell)
    study = layout.study_name(args.cell)
    conditions = ALL_CONDITIONS if not args.conditions else [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}; valid: {ALL_CONDITIONS}")

    ensure_policies(study, args.local_root, args.n_seeds, args.seed_offset, args.no_r2)

    from make_probe_states import ensure_probe_file
    probe = ensure_probe_file(args.local_root, push=not args.no_r2)
    with np.load(probe) as z:
        noise_sigma = np.asarray(z["noise_sigma"], dtype=np.float64)

    jobs = build_jobs(study, args.local_root, args.n_seeds, args.seed_offset, conditions, args.episodes,
                      noise_sigma)
    jsonl = layout.local_path(layout.robustness_key(study), args.local_root)
    print(f"[rob] {args.cell}: {len(jobs)} jobs ({args.n_seeds} seeds x {len(conditions)} conditions, "
          f"{args.episodes} episodes) -> {jsonl}", flush=True)
    pj.run_jobs(jobs, robustness_worker, rob_key, jsonl, args.workers,
                r2_key=None if args.no_r2 else layout.robustness_key(study), label=f"rob {args.cell}")
    print(f"[rob] {args.cell}: done", flush=True)


if __name__ == "__main__":
    main()
