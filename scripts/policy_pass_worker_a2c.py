"""
Worker for the eval_v3 A2C policy pass: trains ONE seed of a legacy A2C winner with the NumPy port, saves the
final policy, reloads it from disk and checks that the reloaded policy replays the in-memory verify episodes
EXACTLY, uploads the policy file to R2 and returns the `train_runs.jsonl` record.

Same contract as policy_pass_worker.policy_worker (never raises: failures come back as {..., "error": ...} so
run_jobs records and retries them) and the same record fields, so `run_policy_pass.summarize_runs` and the
robustness / deployment scripts treat A2C and DQN records alike.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parallel_jobs as pj  # noqa: E402
from policy_pass_worker import JOB_META_KEYS, _jsonable  # noqa: E402


def policy_worker_a2c(job: dict) -> dict:
    """Job keys: kind, trial, rank, seed_idx, params, seed_offset, eval_every, n_eval_episodes, last_k,
    final_eval_episodes, verify_episodes, save, policy_path, policy_key, probe_path, upload."""
    meta = {k: job[k] for k in JOB_META_KEYS}
    try:
        import numpy as np
        from experiments.trial_cartpole_a2c import train_a2c

        seed = int(job["seed_offset"]) + int(job["seed_idx"])
        save = bool(job.get("save", True))
        probe = None
        if job.get("probe_path"):
            probe = np.load(job["probe_path"])["states"].astype(np.float32)
        kw = dict(eval_every=int(job["eval_every"]), n_eval_episodes=int(job["n_eval_episodes"]),
                  final_eval_episodes=int(job["final_eval_episodes"]), probe_states=probe)
        if save:
            os.makedirs(os.path.dirname(job["policy_path"]), exist_ok=True)
            kw.update(save_policy_path=job["policy_path"], verify_episodes=int(job["verify_episodes"]))
        md = train_a2c(job["params"], seed=seed, seed_env=True, **kw)

        hist = [float(x) for x in md["eval_history"]]
        k = int(job.get("last_k", 5))
        rec = {**meta, "seed": seed, "score": float(np.mean(hist[-k:])), "last1": hist[-1],
               "eval_history": hist, "terminal_reward": float(md["terminal_reward"]),
               "total_time": float(md["total_time"]), "train_time": float(md["train_time"]),
               "eval_time": float(md["eval_time"]), "final_eps": float(md["final_eps"])}
        for key in ("build_time", "env_steps", "n_params", "n_active", "dimensionality", "final_eval_reward",
                    "final_eval_returns", "probe_history", "avg_reward_final"):
            if md.get(key) is not None:
                rec[key] = _jsonable(md[key])
        if not save:
            return rec

        saved = md.get("policy_saved")
        if not saved or not os.path.exists(job["policy_path"]):
            return {**meta, "error": "trial did not report a saved policy"}
        from rl.policy_io import verify_saved_policy
        ver = verify_saved_policy(job["policy_path"], int(md["verify_env_seed"]), len(md["verify_returns"]),
                                  [float(x) for x in md["verify_returns"]], max_steps=int(md["verify_max_steps"]))
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
