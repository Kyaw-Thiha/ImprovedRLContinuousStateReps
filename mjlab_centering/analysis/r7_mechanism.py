"""Mechanism readouts from the round-7 logs (free: cached per-iteration rows, no compute).

    python analysis/r7_mechanism.py [--json results/r7_mechanism.json]

Two quantities, fixed here before either is computed:

(a) **VN's state-dependent spread of the normalised return targets**, iterations 100-300. For a VN run the critic regresses
    z = (R - mu) / sigma, so the spread of its targets within a batch is std(R) / sigma. std(R) is recovered exactly from two
    logged keys by the identity d153 used: ev = 1 - err_mse / var_ret => var_ret = err_mse / (1 - ev); sigma is the logged
    `vn_std`. So S(t) = sqrt(var_ret(t)) / vn_std(t). Per run, S is averaged over t in [100, 300]; the table gives the mean
    and a 95 % t-interval over the 20 seeds, for c = 0.1 and c = 1. Note what S is and is not: it is the *batch* spread of
    the normalised targets across env-steps, i.e. how much of the target is state-dependent rather than a constant level. It
    is not a critic error.

(b) **r_bar fluctuation**, iterations 100-500. For each run, F = standard deviation of r_bar over t in [100, 500] (inclusive),
    reported as the mean and 95 % t-interval over seeds, with the mean level of r_bar in the same window. Computed for SC and
    for value centering at both offsets. The window starts at 100 to skip r_bar's first-batch initialisation and ends at 500
    because that is where the centred arms take off.

Both are descriptive: they say what the arms' internal estimates look like, not whether the arms learn (that is the claim
set's job).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "merged_r4"
MERGED_OLD = ROOT / "results" / "merged_r2"  # the c = 1 reference cells live in the REPORT_003 mirror


def rows(jid: str) -> list[dict]:
  for base in (MERGED, MERGED_OLD):
    p = base / jid / "iter.jsonl"
    if p.exists():
      out = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
      if not out:
        continue
      starts = [i for i, r in enumerate(out) if r.get("iter") == 0]
      return out[starts[-1]:] if len(starts) > 1 else out
  return []


def jid(arm: str, off: float, seed: int) -> str:
  """The new arms live in the round-7 claim campaign; the value-centring reference cells come from the earlier campaigns on
  the same seeds (night1 at c = 1, the round-4/5 sweep at c = 0.1), exactly as the paired readout pairs them."""
  if arm == "value":
    return (f"w2main_go1_raw_value_g0.999_o1_s{seed}" if off == 1.0
            else f"r4sweep_go1_g0.999_o{off:g}_value_s{seed}")
  return f"r7vnsc_go1_{arm}_g0.999_o{off:g}_s{seed}"


def ci95(x: np.ndarray) -> tuple[float, float]:
  n = len(x)
  if n < 2:
    return float("nan"), float("nan")
  se = x.std(ddof=1) / math.sqrt(n)
  h = 2.093 * se  # t(0.975, 19); with other n this is conservative by a few percent
  return float(x.mean() - h), float(x.mean() + h)


def vn_spread(off: float, lo: int = 100, hi: int = 300) -> dict:
  per_seed = []
  for seed in range(3000, 3020):
    rs = [r for r in rows(jid("vn", off, seed)) if lo <= r["iter"] <= hi]
    vals = []
    for r in rs:
      ev, mse, sd = r.get("ev"), r.get("err_mse"), r.get("vn_std")
      if ev is None or mse is None or not sd or ev >= 1.0 or not math.isfinite(ev):
        continue
      var_ret = mse / (1.0 - ev)
      if var_ret <= 0 or not math.isfinite(var_ret):
        continue
      vals.append(math.sqrt(var_ret) / sd)
    if vals:
      per_seed.append(float(np.mean(vals)))
  a = np.array(per_seed)
  lo95, hi95 = ci95(a)
  return {"n_seeds": len(per_seed), "mean": float(a.mean()) if len(a) else float("nan"),
          "ci": [lo95, hi95], "per_seed": [round(x, 4) for x in per_seed]}


def rbar_fluctuation(arm: str, off: float, lo: int = 100, hi: int = 500) -> dict:
  fluc, level = [], []
  for seed in range(3000, 3020):
    rs = [r["r_bar"] for r in rows(jid(arm, off, seed)) if lo <= r["iter"] <= hi and "r_bar" in r]
    if len(rs) < 10:
      continue
    fluc.append(float(np.std(rs)))
    level.append(float(np.mean(rs)))
  f, l = np.array(fluc), np.array(level)
  flo, fhi = ci95(f)
  return {"n_seeds": len(fluc), "std_mean": float(f.mean()) if len(f) else float("nan"),
          "std_median": float(np.median(f)) if len(f) else float("nan"), "std_ci": [flo, fhi],
          "level_mean": float(l.mean()) if len(l) else float("nan"),
          "per_seed_std": [round(x, 4) for x in fluc]}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default=str(ROOT / "results" / "r7_mechanism.json"))
  a = ap.parse_args()
  out = {"vn_spread": {f"{o:g}": vn_spread(o) for o in (0.1, 1.0)},
         "rbar_fluctuation": {f"{arm}_{o:g}": rbar_fluctuation(arm, o)
                              for arm in ("sc", "value") for o in (0.1, 1.0)}}
  Path(a.json).write_text(json.dumps(out, indent=1))
  print("(a) VN normalised-target spread, iterations 100-300 (std of targets = std(R)/sigma)")
  for o in ("0.1", "1"):
    d = out["vn_spread"][o]
    print(f"   c = {o:>3}: n={d['n_seeds']}, mean {d['mean']:.4f}  95% CI [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}]")
  print("\n(b) r_bar fluctuation, std over iterations 100-500")
  for k, d in out["rbar_fluctuation"].items():
    print(f"   {k:12s}: n={d['n_seeds']}, std {d['std_mean']:.4f} [{d['std_ci'][0]:.4f}, {d['std_ci'][1]:.4f}] "
          f"(median {d['std_median']:.4f}), level {d['level_mean']:.4f}")
  print("\nwrote", a.json)


if __name__ == "__main__":
  main()
