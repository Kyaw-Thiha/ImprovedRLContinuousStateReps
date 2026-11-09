"""Round-7 readout (d1020-d1021): the Stage-B escalation for one cell, and the length-scale boundary check.

    python analysis/r7_readout.py [--root results/merged_r2]

Escalation: SSP 2x2 @ ls 1.0, uncentered, offset 0.1, on 20 seeds (3000-3004 from the Stage-B pilot campaign
`r6sspp-e8af4cb0`, 3005-3019 from `r7queue-...`), paired with the existing 20-seed raw cell (seeds 3000-3019).
The claim rule is d1021: "SSP-specific advantage" iff the paired dtrack 95% interval excludes 0 **and** the walk counts
differ by at least 3 of 20.

Boundary: SSP 2x2, centered, offset 1, length scales 2.0 and 4.0 on search seeds 1000-1002, against the measured 0.880 at
ls 1.0. Reported as a description of the sweep, with the rule "the optimum lies above 1.0 if either is higher by >= 0.02".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import ssp_readout  # noqa: E402

PILOT_CAMP = "r6sspp-e8af4cb0"
R7_CAMP = "r7queue-9e67be96"
SEEDS = tuple(range(3000, 3020))
WALK = 0.5
LS1_MEAN = 0.880  # measured in Stage A at 2x2, three search seeds


def ssp_job(seed: int, camp: str) -> str:
  return f"r6ssp{'p' if camp == PILOT_CAMP else 'e'}_go1_ssp_low2x2_ls1_g0.999_o0.1_none_s{seed}"


def boot_ci(d, n: int = 5000, seed: int = 0):
  d = np.asarray(d, float)
  if len(d) < 2:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def resolved(ci_lo: float, walks_ssp: int, walks_raw: int, seeds: int = 20) -> bool:
  """d1021: an SSP-specific advantage needs the paired interval above zero AND >= 3 more walks out of 20."""
  return bool(ci_lo > 0 and (walks_ssp - walks_raw) >= 3 and seeds >= 20)


def escalation(root: Path) -> dict:
  per = {}
  for s in SEEDS:
    if s <= 3004:
      a = ssp_readout.track(ssp_job(s, PILOT_CAMP), root)
    else:
      a = ssp_readout.track(ssp_job(s, R7_CAMP), root)
    b = ssp_readout.track(f"r4sweep_go1_g0.999_o0.1_none_s{s}", root)
    if a is not None and b is not None:
      per[s] = (a, b)
  out: dict = {"n_pairs": len(per), "ssp": {}, "raw": {}}
  if not per:
    out["verdict"] = "no paired runs yet"
    return out
  ssp = [v[0] for v in per.values()]
  raw = [v[1] for v in per.values()]
  d = [a - b for a, b in per.values()]
  lo, hi = boot_ci(d)
  ws, wr = sum(x >= WALK for x in ssp), sum(x >= WALK for x in raw)
  out.update({
      "ssp": {"mean": float(np.mean(ssp)), "walks": ws, "n": len(ssp)},
      "raw": {"mean": float(np.mean(raw)), "walks": wr, "n": len(raw)},
      "paired": {"mean": float(np.mean(d)), "ci": [lo, hi], "seeds_positive": int(sum(x > 0 for x in d))},
      "per_seed": {str(k): [round(v[0], 3), round(v[1], 3)] for k, v in per.items()},
  })
  advantage = resolved(lo, ws, wr)
  out["rule_d170_advantage"] = bool(advantage)
  out["verdict"] = ("SSP-specific advantage" if advantage else "no resolved difference")
  return out


def boundary(root: Path) -> dict:
  out = {}
  for ls in (2.0, 4.0):
    jobs = [f"r6lss_go1_ssp_low2x2_ls{ls:g}_value_s{s}" for s in (1000, 1001, 1002)]
    vals = [ssp_readout.track(j, root) for j in jobs]
    ok = [v for v in vals if v is not None]
    out[f"{ls:g}"] = {"n": len(ok), "mean": float(np.mean(ok)) if ok else None,
                      "walks": int(sum(v >= WALK for v in ok))}
  complete = all(v["n"] == 3 for v in out.values())
  out["ls_1.0_reference"] = LS1_MEAN
  if complete:
    higher = [k for k, v in out.items() if isinstance(v, dict) and v.get("mean") is not None
              and v["mean"] >= LS1_MEAN + 0.02]
    out["optimum_above_1.0"] = bool(higher)
    out["verdict"] = (f"optimum lies above the swept range ({', '.join(higher)})" if higher
                      else "1.0 remains the best measured value (within 0.03-4.0)")
  else:
    out["verdict"] = "boundary check incomplete"
  return out


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", default=str(ROOT / "results" / "merged_r2"))
  a = ap.parse_args()
  root = Path(a.root)
  print(json.dumps({"escalation": escalation(root), "boundary": boundary(root)}, indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
