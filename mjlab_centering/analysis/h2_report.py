"""H2 reading rules (pre-registered, d131). Reads the three cells' `final_eval_summary.json` from R2 (or a local
directory) and applies R1-R3 mechanically.

    python analysis/h2_report.py                     # from R2, current round
    python analysis/h2_report.py --local results/h2  # from `<local>/<arm>/final_eval_summary.json`

Metric: final-eval `score_mean` (mean of the last 5 of 15 greedy checkpoints), 20 claim seeds 3000-3019 per arm,
paired by seed index. Bootstrap 95% CI over seeds (5000 resamples, seed 0). Equivalence margin = 3% of the tile mean
(about 15 points at 500). R1 "the loss was tuning": lower bound of (ssp_h2 - tile_h2) above -margin. R2 "inherent":
upper bound below -margin. R3: (ssp_h2raw - ssp_h2) inside +-margin means normalisation was not the fixing factor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("ssp_h2", "ssp_h2raw", "tile_h2")
PREFIX = "stage0_cartpole/refilter_v2/dqn_{}_none_v4_hparam_search/final_eval_summary.json"


def load(arm: str, local: Path | None = None) -> dict | None:
  if local is not None:
    p = local / arm / "final_eval_summary.json"
    return json.loads(p.read_text()) if p.exists() else None
  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root / "src"))
  from mjlab_centering import r2
  # these live at the bucket root (stage0_cartpole/...), not under the mjlab_ssp_centering prefix that r2.key adds
  try:
    body = r2.client().get_object(Bucket=r2.bucket(), Key=PREFIX.format(arm))["Body"].read()
  except Exception:  # noqa: BLE001 - missing key = that arm has not finished
    return None
  return json.loads(body)


def boot_ci(d, n: int = 5000, seed: int = 0) -> tuple[float, float]:
  d = np.asarray(d, float)
  if len(d) < 2:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def read(summaries: dict[str, dict]) -> dict:
  out: dict = {"arms": {}, "rules": {}}
  fe = {}
  for a in ARMS:
    s = summaries.get(a)
    if not s:
      out["arms"][a] = None
      continue
    f = s["final_eval"]
    fe[a] = np.asarray(f["per_seed_scores"], float)
    ls = (s.get("winner_search_params") or {}).get("length_scale")
    out["arms"][a] = {"n": f["n_seeds"], "mean": f["score_mean"], "std": f["score_std"], "min": f["score_min"],
                      "max": f["score_max"], "solved": f["share_solved_lastk"], "length_scale": ls,
                      "per_seed": f["per_seed_scores"], "filter_mean": (s.get("filter_winner") or {}).get("mean")}
  if "ssp_h2" not in fe or "tile_h2" not in fe:
    out["verdict"] = "H2 incomplete: need ssp_h2 and tile_h2 final evals"
    return out
  n = min(len(fe["ssp_h2"]), len(fe["tile_h2"]))
  margin = 0.03 * float(np.mean(fe["tile_h2"][:n]))
  d = fe["ssp_h2"][:n] - fe["tile_h2"][:n]
  lo, hi = boot_ci(d)
  out["vs_tile"] = {"n": n, "mean": float(d.mean()), "lo": lo, "hi": hi, "margin": margin,
                    "seeds_positive": int((d > 0).sum())}
  if lo > -margin:
    r1 = "R1 supported: SSP (normalised, length scale searched 0.01-1) is within the margin of tile coding"
  elif hi < -margin:
    r1 = "R2 supported: SSP trails tile coding by more than the margin, so the loss is not (only) tuning"
  else:
    r1 = "inconclusive: interval spans the margin"
  out["rules"]["R1"] = r1
  if "ssp_h2raw" in fe:
    m = min(len(fe["ssp_h2raw"]), len(fe["ssp_h2"]))
    dr = fe["ssp_h2raw"][:m] - fe["ssp_h2"][:m]
    lo2, hi2 = boot_ci(dr)
    out["raw_vs_norm"] = {"n": m, "mean": float(dr.mean()), "lo": lo2, "hi": hi2}
    if hi2 < -margin:
      out["rules"]["R3"] = "normalisation is what fixed it: the unnormalised wide-range arm is worse by more than the margin"
    elif lo2 > margin:
      out["rules"]["R3"] = "normalisation is not necessary: the unnormalised wide-range arm is better by more than the margin"
    else:
      out["rules"]["R3"] = "normalisation was not the fixing factor; the wider length-scale range was (inside the margin)"
  out["verdict"] = out["rules"]["R1"] + ("; " + out["rules"]["R3"] if "R3" in out["rules"] else "")
  return out


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--local", default=None, help="directory holding <arm>/final_eval_summary.json")
  a = ap.parse_args()
  local = Path(a.local) if a.local else None
  print(json.dumps(read({arm: load(arm, local) for arm in ARMS}), indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
