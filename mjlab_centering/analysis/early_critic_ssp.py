"""Free mechanism check from the Stage-B pilot logs (round 7, step 1).

Compare the early critic diagnostics of SSP (2x2, ls 1.0) against raw input, both uncentered at offset 0.1, over
iterations 100-300: the constant share of the advantage (`const_frac`), the explained variance (`ev`), and the value level
(`v_mean` against the analytic c/(1-gamma) = 100). The question: does SSP's critic learn the offset level sooner, which
would be a mechanism for the walk-count difference rather than seed luck.

    python analysis/early_critic_ssp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SSP_CAMP = "r6sspp-e8af4cb0"
RAW_CAMP = "r4sweep-42a0af34"
SSP_JOB = "r6sspp_go1_ssp_low2x2_ls1_g0.999_o0.1_none_s{}"
RAW_JOB = "r4sweep_go1_g0.999_o0.1_none_s{}"
SEEDS = (3000, 3001, 3002, 3003, 3004)
LO, HI = 100, 300


def rows(camp: str, job: str) -> list[dict]:
  from mjlab_centering import r2
  try:
    body = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{camp}/runs/{job}/iter.jsonl"))["Body"].read()
  except Exception:  # noqa: BLE001
    p = ROOT / "results" / "merged_r2" / job / "iter.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
  return [json.loads(l) for l in body.decode().splitlines() if l.strip()]


def window(rs: list[dict]) -> dict:
  w = [r for r in rs if LO <= r["iter"] <= HI]
  if not w:
    return {}
  g = lambda k: float(np.mean([r.get(k, float("nan")) for r in w]))  # noqa: E731
  return {"n": len(w), "const_frac": g("const_frac"), "ev": g("ev"), "v_mean": g("v_mean"),
          "err_mean": g("err_mean"), "action_std": g("action_std")}


def final_track(rs: list[dict]) -> float:
  last = [r for r in rs if r["iter"] >= 1900]
  return float(np.mean([r.get("const_frac", 0) for r in last])) if last else float("nan")


def boot_ci(d, n: int = 5000, seed: int = 0):
  d = np.asarray(d, float)
  if len(d) < 2:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
  out = {}
  for arm, camp, pat in (("ssp", SSP_CAMP, SSP_JOB), ("raw", RAW_CAMP, RAW_JOB)):
    per = {}
    for s in SEEDS:
      w = window(rows(camp, pat.format(s)))
      if w:
        per[s] = w
    out[arm] = per
  print(f"early critic diagnostics, iterations {LO}-{HI}, uncentered, offset 0.1\n")
  print(f"{'arm':4s} {'seed':5s} {'const_frac':>10s} {'ev':>7s} {'v_mean':>8s} {'err_mean':>9s} {'action_std':>10s}")
  for arm in ("ssp", "raw"):
    for s, w in out[arm].items():
      print(f"{arm:4s} {s:5d} {w['const_frac']:10.3f} {w['ev']:7.3f} {w['v_mean']:8.2f} {w['err_mean']:9.3f} "
            f"{w['action_std']:10.3f}")
  print()
  for key in ("const_frac", "ev", "v_mean", "action_std"):
    a = [out["ssp"][s][key] for s in SEEDS if s in out["ssp"]]
    b = [out["raw"][s][key] for s in SEEDS if s in out["raw"]]
    common = [s for s in SEEDS if s in out["ssp"] and s in out["raw"]]
    d = [out["ssp"][s][key] - out["raw"][s][key] for s in common]
    lo, hi = boot_ci(d)
    print(f"{key:11s} ssp {np.mean(a):8.3f}  raw {np.mean(b):8.3f}  paired delta {np.mean(d):+8.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]  (n={len(common)})")


if __name__ == "__main__":
  main()
