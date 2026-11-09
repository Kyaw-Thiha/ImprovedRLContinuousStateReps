"""G1 convergence rule (pre-registered, d126): choose the claim-set length from the probe's tracking slope.

Rule, fixed before the probe launches:
  * metric: eval `track_lin_frac`; a run's x axis is `iter + 1` (completed iterations).
  * slope at t: OLS slope over the evals with t-250 <= iters <= t, in tracking per 100 iterations (se reported beside it).
  * walks at t: mean tracking over the evals in (t-100, t] is >= 0.5.
  * a seed is FLAT at t iff it walks at t AND slope(t) < +0.01 per 100 iterations. A seed that has not walked is NOT flat.
  * an arm passes at t iff >= 2 of its 3 seeds are flat; t passes iff BOTH arms (centered, bias-init) pass.
  * if t = 2000 passes -> the claim set runs at 2000 iterations.
  * else the first t in 2000..2750 (step 25) that passes gives length min(3000, ceil((t + 250) / 250) * 250);
    if no t passes, the length is 3000 and the claim set is labelled "not converged at 3000".

    python analysis/g1_slope.py --campaign <id> [--arms centered=<job,job,job>;bias=<job,job,job>]
    python analysis/g1_slope.py --dry-run     # same rule on the existing centered G1 pilot curves (free)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SLOPE_TOL = 0.01      # tracking per 100 iterations
WALK_LEVEL = 0.5
WINDOW = 250
WALK_WINDOW = 100
GRID = tuple(range(2000, 2751, 25))


def newest(rows: list[dict]) -> list[dict]:
  """Keep the rows of the newest attempt (it starts at the last row whose iter is the run's first eval)."""
  rows = [r for r in rows if "iter" in r]
  if not rows:
    return rows
  first = min(r["iter"] for r in rows)
  starts = [i for i, r in enumerate(rows) if r["iter"] == first]
  return rows[starts[-1]:]


def series(rows: list[dict], key: str = "track_lin_frac") -> tuple[np.ndarray, np.ndarray]:
  rows = newest(rows)
  it = np.array([r["iter"] + 1 for r in rows], float)
  y = np.array([r[key] for r in rows], float)
  o = np.argsort(it)
  return it[o], y[o]


def slope_per100(it: np.ndarray, y: np.ndarray, t: float, width: float = WINDOW) -> tuple[float, float, int]:
  m = (it >= t - width) & (it <= t)
  n = int(m.sum())
  if n < 4:
    return float("nan"), float("nan"), n
  x, z = it[m] / 100.0, y[m]
  xc = x - x.mean()
  sxx = float((xc ** 2).sum())
  b = float((xc * (z - z.mean())).sum() / sxx)
  resid = z - (z.mean() + b * xc)
  s2 = float((resid ** 2).sum() / max(n - 2, 1))
  return b, math.sqrt(s2 / sxx), n


def walks_at(it: np.ndarray, y: np.ndarray, t: float) -> tuple[bool, float]:
  m = (it > t - WALK_WINDOW) & (it <= t)
  if not m.any():
    return False, float("nan")
  v = float(y[m].mean())
  return v >= WALK_LEVEL, v


def seed_state(it: np.ndarray, y: np.ndarray, t: float) -> dict:
  b, se, n = slope_per100(it, y, t)
  w, lvl = walks_at(it, y, t)
  flat = bool(w and not math.isnan(b) and b < SLOPE_TOL)
  return {"t": t, "slope_per100": b, "se_per100": se, "n_evals": n, "walks": w, "level": lvl, "flat": flat}


def passes(arms: dict[str, list[tuple[np.ndarray, np.ndarray]]], t: float) -> tuple[bool, dict]:
  detail = {}
  ok = True
  for arm, runs in arms.items():
    st = [seed_state(it, y, t) for it, y in runs]
    n_flat = sum(s["flat"] for s in st)
    detail[arm] = {"flat": n_flat, "of": len(st), "seeds": st}
    ok = ok and n_flat >= 2
  return ok, detail


def decide(arms: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> dict:
  last = min(float(it.max()) for runs in arms.values() for it, _ in runs)
  out: dict = {"max_common_iter": last}
  ok, det = passes(arms, 2000)
  out["at_2000"] = det
  if last < 2000:
    out.update(length=None, verdict="probe has not reached iteration 2000")
    return out
  if ok:
    out.update(length=2000, verdict="flat at 2000: run the claim set at 2000")
    return out
  for t in GRID[1:]:
    if t > last:
      break
    ok, det = passes(arms, t)
    if ok:
      out.update(length=min(3000, int(math.ceil((t + 250) / 250) * 250)), first_flat_t=t, at_first_flat=det,
                 verdict=f"first flat at {t}: claim length {min(3000, int(math.ceil((t + 250) / 250) * 250))}")
      return out
  out.update(length=3000, verdict="never flat by %d: run at 3000, label 'not converged at 3000'" % int(last))
  return out


def load_local(job: str, root: Path) -> list[dict]:
  p = root / job / "eval.jsonl"
  return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def load_r2(campaign: str, job: str) -> list[dict]:
  from mjlab_centering import r2
  body = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{campaign}/runs/{job}/eval.jsonl"))
  return [json.loads(ln) for ln in body["Body"].read().decode().splitlines() if ln.strip()]


DRY = {  # existing centered G1 curves, gamma .99, offset 1, penalty 5.7, 6 seeds, 50-iteration evals
  "centered": [f"w2g1c57_g1_raw_value_g0.99_o1_s{s}" for s in (1000, 1001, 1002)]
              + [f"w2g1c57b_g1_raw_value_g0.99_o1_s{s}" for s in (1003, 1004, 1005)],
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign")
  ap.add_argument("--arms", help="name=job,job,job;name=job,job,job")
  ap.add_argument("--probe", action="store_true", help="read the round-6 probe jobs (d127) and apply the rule")
  ap.add_argument("--dry-run", action="store_true")
  ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "results" / "merged_r2"))
  a = ap.parse_args()
  if a.probe:
    camp = a.campaign or "r6a-ad66ecf2"
    arms = {
      "centered": [f"r6probe_g1_g0.99_o1_value_s{s}" for s in (1000, 1001, 1002)],
      "bias": [f"r6probe_g1_g0.99_o1_bias_none_s{s}" for s in (1000, 1001, 1002)],
    }
    series_ = {k: [series(load_r2(camp, j)) for j in v] for k, v in arms.items()}
    print(json.dumps(decide(series_), indent=1, default=float))
    return
  if a.dry_run:
    runs = [series(load_local(j, Path(a.root))) for j in DRY["centered"]]
    print("dry run: centered G1 pilot, 6 seeds, evals every 50 (the probe evaluates every 25)")
    for j, (it, y) in zip(DRY["centered"], runs):
      for t in (2000,):
        s = seed_state(it, y, t)
        print(f"{j}: t={t} slope/100={s['slope_per100']:+.4f} se={s['se_per100']:.4f} n={s['n_evals']} "
              f"level={s['level']:.3f} walks={s['walks']} flat={s['flat']}")
    # resolution of the slope on this data: residual sd about a straight line in the last 250 iterations
    sds = []
    for it, y in runs:
      m = (it >= 1750) & (it <= 2000)
      if m.sum() >= 4:
        c = np.polyfit(it[m], y[m], 1)
        sds.append(float(np.std(y[m] - np.polyval(c, it[m]), ddof=2)))
    print(f"eval noise (residual sd about a line, last 250 its): mean {np.mean(sds):.4f}, max {np.max(sds):.4f}")
    # same rule on every consecutive 3-seed subset, to see how often 'most seeds flat' passes
    import itertools
    npass = tot = 0
    for combo in itertools.combinations(range(6), 3):
      ok, _ = passes({"centered": [runs[i] for i in combo]}, 2000)
      npass += ok
      tot += 1
    print(f"2-of-3 flat at t=2000 in {npass} of {tot} three-seed subsets of the pilot")
    return
  assert a.campaign and a.arms
  arms = {}
  for part in a.arms.split(";"):
    name, jobs = part.split("=")
    arms[name] = [series(load_r2(a.campaign, j)) for j in jobs.split(",")]
  res = decide(arms)
  print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
  main()
