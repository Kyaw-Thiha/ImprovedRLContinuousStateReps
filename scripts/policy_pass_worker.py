"""
Worker for the eval_v3 policy pass: trains ONE seed of a cell winner, saves the final policy, reloads it
from disk and checks that the reloaded policy replays the in-memory verify episodes EXACTLY, uploads the
policy file to R2 and returns the `train_runs.jsonl` record.

Lives in its own module (not in run_policy_pass.py) so a spawned process imports it by name without
re-running the CLI; same never-raise convention as parallel_jobs.eval_worker: any failure comes back as
{..., "error": ...} so run_jobs records and retries it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parallel_jobs as pj  # noqa: E402

JOB_META_KEYS = ("kind", "trial", "rank", "seed_idx")


def _jsonable(v):
    import math
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def policy_worker(job: dict) -> dict:
    """Job keys: kind, trial, rank, seed_idx, params, data_dir, seed_offset, eval_freq, n_eval_episodes,
    last_k, final_eval_episodes, verify_episodes, save (bool), policy_path, policy_key, probe_path,
    upload (bool)."""
    meta = {k: job[k] for k in JOB_META_KEYS}
    try:
        import numpy as np
        from experiments.trial_cartpole_dqn import DQNTrial
        pj._silence_tqdm()  # no-op until the trial module is imported, so import first (above)
        data_dir = os.path.join(job["data_dir"], f"{job['kind']}_t{job['trial']}_s{job['seed_idx']}")
        os.makedirs(data_dir, exist_ok=True)
        seed = int(job["seed_offset"]) + int(job["seed_idx"])
        save = bool(job.get("save", True))

        extra = dict(seed_env=True, final_eval_episodes=int(job["final_eval_episodes"]))
        if job.get("probe_path"):
            extra["probe_path"] = job["probe_path"]
        if save:
            os.makedirs(os.path.dirname(job["policy_path"]), exist_ok=True)
            extra["save_policy_path"] = job["policy_path"]
            extra["verify_episodes"] = int(job["verify_episodes"])
        md = DQNTrial().run(
            seed=seed, data_dir=data_dir,
            pre_comment=f"policy_pass trial={job['trial']} seed_idx={job['seed_idx']}",
            eval_freq=job["eval_freq"], n_eval_episodes=job["n_eval_episodes"],
            **{"init_seed": -1, **job["params"], **extra})

        hist = [float(x) for x in md["eval_history"]]
        k = int(job.get("last_k", 5))
        rec = {**meta, "seed": seed, "score": float(np.mean(hist[-k:])), "last1": hist[-1],
               "eval_history": hist, "terminal_reward": float(md["terminal_reward"]),
               "total_time": float(md["total_time"]), "train_time": float(md["train_time"]),
               "eval_time": float(md["eval_time"])}
        for key in ("build_time", "env_steps", "n_updates", "n_params", "n_active", "dimensionality",
                    "final_eval_reward", "final_eval_returns", "probe_history", "avg_reward_final"):
            if md.get(key) is not None:
                rec[key] = _jsonable(md[key])

        if not save:
            return rec

        saved = md.get("policy_saved")
        if not saved or not os.path.exists(job["policy_path"]):
            return {**meta, "error": "trial did not report a saved policy"}
        from rl.policy_io import verify_saved_policy
        vsteps = int(md.get("verify_max_steps") or job["params"].get("steps", 500))
        ver = verify_saved_policy(job["policy_path"], int(md["verify_env_seed"]), len(md["verify_returns"]),
                                  [float(x) for x in md["verify_returns"]], max_steps=vsteps)
        if not ver["ok"]:
            return {**meta, "error": f"policy reload verification FAILED max_abs_diff={ver['max_abs_diff']}"}
        rec.update(policy_key=job["policy_key"], policy_sha256=saved["sha256"], policy_bytes=int(saved["bytes"]),
                   policy_verified=True, verify_max_abs_diff=float(ver["max_abs_diff"]),
                   verify_returns=[float(x) for x in md["verify_returns"]],
                   verify_env_seed=int(md["verify_env_seed"]))
        if job.get("upload", True):
            if not pj.push_file(job["policy_key"], job["policy_path"]):
                return {**meta, "error": f"upload of {job['policy_key']} failed"}
            rec["policy_uploaded"] = True
        return rec
    except Exception as e:  # noqa: BLE001
        import traceback
        return {**meta, "error": repr(e) + " | " + traceback.format_exc()[-400:]}
