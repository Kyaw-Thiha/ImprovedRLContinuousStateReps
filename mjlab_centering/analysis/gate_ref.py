"""Hard gate: the plain reference arm must learn before any tuning or core result is trusted.

Compares, on the common continuing evaluation env, the reference arm trained through the continuing wrapper
(`wrapped`) against the same arm trained on mjlab's unmodified episodic env (`episodic`).

Thresholds (fixed here and in docs/mjlab_centering/02-decisions.md entry 39, before any gate run):
  LEARN_FRAC     final track_lin_frac (mean of the last 5 evals) >= 0.25, i.e. at least a quarter of the best
                 possible per-step tracking reward. A standing-still policy scores ~0.28 (measured, decision 45).
  LEARN_X_STAND  and >= 3x the first-eval value of the same run (the near-initial policy).
  MAX_FALLS_1K   final falls per 1k steps <= 5.
  MATCH          wrapped final track_lin_frac >= 0.8 x episodic final (means over seeds).
  PLATEAU        reported, not gated: first iteration whose next-100-iteration track_lin_frac moves < 3%.
J_task is NOT used as the learning signal: a standing policy already scores ~0.078 on it (measured), so it cannot
distinguish "stands still" from "tracks velocity".

    python analysis/gate_ref.py --wrapped results/gate/w_* --episodic results/gate/e_*   (dirs with eval.jsonl)
Exit code 0 = pass, 1 = fail.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np

LEARN_FRAC = 0.25
LEARN_X_STAND = 3.0
MAX_FALLS_1K = 5.0
MATCH = 0.8


def load_evals(run_dir: str, legacy_env_scale: int = 0) -> list[dict]:
  """`legacy_env_scale`: logs written before decision 43 divided every term (and track_*_frac) by the number of eval
  envs once too often; pass that number (read from config.json eval_envs) to undo it. Never use on new logs."""
  rows = [json.loads(l) for l in open(f"{run_dir}/eval.jsonl") if l.strip()]
  if legacy_env_scale:
    for r in rows:
      for k in list(r):
        if k.startswith("term/") or k in ("track_lin_frac", "track_ang_frac"):
          r[k] *= legacy_env_scale
  return sorted(rows, key=lambda r: r["iter"])


def run_summary(rows: list[dict]) -> dict:
  tl = np.array([r.get("track_lin_frac", np.nan) for r in rows], dtype=float)
  fl = np.array([r.get("fall_rate_per_1k", np.nan) for r in rows], dtype=float)
  it = np.array([r["iter"] for r in rows], dtype=float)
  first = float(tl[0]) if len(tl) else float("nan")
  return {
    "n_evals": int(len(rows)),
    "last_iter": int(it[-1]) if len(it) else -1,
    "track_lin_first": first,
    "track_lin_final": float(np.nanmean(tl[-5:])) if len(tl) else float("nan"),
    "falls_final": float(np.nanmean(fl[-5:])) if len(fl) else float("nan"),
    "J_task_final": float(np.mean([r["J_task"] for r in rows[-5:]])) if rows else float("nan"),
    "curve": tl.tolist(),
    "iters": it.tolist(),
  }


def plateau_iter(it: np.ndarray, y: np.ndarray, window: int = 100, tol: float = 0.03) -> int:
  for i in range(len(it)):
    t = it[i]
    prev = y[(it >= t - window) & (it <= t)]
    nxt = y[(it > t) & (it <= t + window)]
    if len(prev) < 2 or len(nxt) < 2:
      continue
    if abs(float(np.mean(nxt)) - float(np.mean(prev))) / (abs(float(np.mean(prev))) + 1e-9) < tol:
      return int(t)
  return -1


def verdict(wrapped: list[dict], episodic: list[dict]) -> dict:
  def learns(s: dict) -> bool:
    return bool(
      s["track_lin_final"] >= LEARN_FRAC
      and s["track_lin_final"] >= LEARN_X_STAND * max(s["track_lin_first"], 1e-6)
      and s["falls_final"] <= MAX_FALLS_1K
    )

  w_final = float(np.mean([s["track_lin_final"] for s in wrapped])) if wrapped else float("nan")
  e_final = float(np.mean([s["track_lin_final"] for s in episodic])) if episodic else float("nan")
  checks = {
    "wrapped_learns_all_seeds": bool(wrapped) and all(learns(s) for s in wrapped),
    "episodic_learns_all_seeds": bool(episodic) and all(learns(s) for s in episodic),
    "wrapped_matches_episodic": bool(w_final >= MATCH * e_final) if episodic and wrapped else False,
  }
  mean_curve = None
  plateau = -1
  if wrapped and len({len(s["curve"]) for s in wrapped}) == 1:
    mean_curve = np.mean([s["curve"] for s in wrapped], axis=0)
    plateau = plateau_iter(np.array(wrapped[0]["iters"]), mean_curve)
  return {
    "pass": all(checks.values()),
    "checks": checks,
    "wrapped_track_lin_final": w_final,
    "episodic_track_lin_final": e_final,
    "plateau_iter_track_lin": plateau,
    "thresholds": {"LEARN_FRAC": LEARN_FRAC, "LEARN_X_STAND": LEARN_X_STAND, "MAX_FALLS_1K": MAX_FALLS_1K, "MATCH": MATCH},
    "wrapped": [{k: v for k, v in s.items() if k not in ("curve", "iters")} for s in wrapped],
    "episodic": [{k: v for k, v in s.items() if k not in ("curve", "iters")} for s in episodic],
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--wrapped", nargs="+", required=True)
  ap.add_argument("--episodic", nargs="+", required=True)
  ap.add_argument("--out", default=None)
  ap.add_argument("--legacy-env-scale", type=int, default=0, help="undo the pre-decision-43 per-env division (e.g. 128)")
  a = ap.parse_args()
  dirs = lambda pats: sorted(d for p in pats for d in glob.glob(p))  # noqa: E731
  w = [run_summary(load_evals(d, a.legacy_env_scale)) for d in dirs(a.wrapped)]
  e = [run_summary(load_evals(d, a.legacy_env_scale)) for d in dirs(a.episodic)]
  v = verdict(w, e)
  print(json.dumps(v, indent=1))
  if a.out:
    json.dump(v, open(a.out, "w"), indent=1)
  return 0 if v["pass"] else 1


if __name__ == "__main__":
  sys.exit(main())
