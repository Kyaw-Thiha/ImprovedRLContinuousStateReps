"""G1 claim-set readout (d128). Reads the 60 `r6g1` runs from R2 and applies the pre-registered reading.

    python analysis/g1_claim_report.py [--campaign r6g1queue-0ede5636]

Per arm (no centering / value centering / bias-init): walks = last-5-eval mean `track_lin_frac` >= 0.5; tracking and
J_task are the per-seed last-5 means. Contrasts are paired by seed with bootstrap 95% CIs (5000 resamples):
value - none and bias - none. The bias-vs-value equivalence test uses the d117 margins (dtrack within +-0.03, dJ within
+-3% of the value arm's mean, walk counts within 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(3000, 3020))
WALK = 0.5
ARMS = {"none": "r6g1_g1_g0.99_o1_none_s{}", "value": "r6g1_g1_g0.99_o1_value_s{}",
        "bias": "r6g1_g1_g0.99_o1_bias_none_s{}"}


def crossing(rows: list[dict], level: float) -> float | None:
  """Iteration at which the eval tracking curve first reaches `level` (linear interpolation between evals).

  Reported beside the plateau because the probe showed the centered arm still climbing at 2000 while bias-init was flat:
  same destination, different speed. `None` means the run never got there.
  """
  if not rows:
    return None
  prev_it = prev_v = None
  for r in sorted(rows, key=lambda r: r["iter"]):
    v, it = r["track_lin_frac"], r["iter"] + 1
    if v >= level:
      if prev_v is None or prev_v >= level:
        return float(it)
      f = (level - prev_v) / (v - prev_v) if v > prev_v else 1.0
      return float(prev_it + f * (it - prev_it))
    prev_it, prev_v = it, v
  return None


def newest(rows: list[dict]) -> list[dict]:
  if not rows:
    return rows
  first = min(r["iter"] for r in rows)
  starts = [i for i, r in enumerate(rows) if r["iter"] == first]
  return rows[starts[-1]:]


def read_arm(campaign: str, pattern: str) -> dict[int, dict]:
  from mjlab_centering import r2
  c, b = r2.client(), r2.bucket()
  out = {}
  for s in SEEDS:
    job = pattern.format(s)
    # only finished runs: a run that is still training has eval rows too, and its last five come from a policy that has
    # not converged (that mistake made an interim readout report 10 seeds per arm when only 6 had finished)
    try:
      c.get_object(Bucket=b, Key=r2.key(f"{campaign}/runs/{job}/final.json"))
    except Exception:  # noqa: BLE001
      continue
    try:
      body = c.get_object(Bucket=b, Key=r2.key(f"{campaign}/runs/{job}/eval.jsonl"))["Body"].read()
    except Exception:  # noqa: BLE001
      continue
    rows = newest(sorted([json.loads(l) for l in body.decode().splitlines() if l.strip()], key=lambda r: r["iter"]))
    if len(rows) < 5:
      continue
    last = rows[-5:]
    tr = float(np.mean([r["track_lin_frac"] for r in last]))
    out[s] = {"track": tr, "J": float(np.mean([r["J_task"] for r in last])), "walks": tr >= WALK,
              "t50": crossing(rows, 0.5), "t90": crossing(rows, 0.9),
              # G1's plateau sits near 0.81 tracking, so an absolute 0.90 is never reached; time to 90% of the seed's
              # own final level is the comparable "how fast does it get there" measure
              # only for a seed that actually walks: 90% of a floor-level final value is reached in minutes and would
              # make a failing arm look fastest
              "t90rel": crossing(rows, 0.9 * tr) if tr >= WALK else None, "done": True}
  return out


def boot_ci(d, n: int = 5000, seed: int = 0):
  d = np.asarray(d, float)
  if len(d) < 2:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _speed(rec: dict[int, dict], key: str) -> dict:
  """Median/mean iterations to a threshold, and how many seeds reached it."""
  vals = [v[key] for v in rec.values() if v.get(key) is not None]
  return {"reached": len(vals), "of": len(rec),
          "median": float(np.median(vals)) if vals else None,
          "mean": float(np.mean(vals)) if vals else None}


def summarise(arms: dict[str, dict[int, dict]]) -> dict:
  out: dict = {"arms": {}, "contrasts": {}}
  for name, rec in arms.items():
    if not rec:
      out["arms"][name] = None
      continue
    t = [v["track"] for v in rec.values()]
    j = [v["J"] for v in rec.values()]
    out["arms"][name] = {"n": len(rec), "track": float(np.mean(t)), "J": float(np.mean(j)),
                         "walks": int(sum(v["walks"] for v in rec.values())),
                         "t50": _speed(rec, "t50"), "t90": _speed(rec, "t90"), "t90rel": _speed(rec, "t90rel"),
                         "per_seed": {str(k): v for k, v in rec.items()}}
  for a, b_ in (("value", "none"), ("bias", "none"), ("bias", "value")):
    if not arms.get(a) or not arms.get(b_):
      continue
    common = sorted(set(arms[a]) & set(arms[b_]))
    if len(common) < 2:
      continue
    dt = [arms[a][s]["track"] - arms[b_][s]["track"] for s in common]
    dj = [arms[a][s]["J"] - arms[b_][s]["J"] for s in common]
    dw = sum(arms[a][s]["walks"] for s in common) - sum(arms[b_][s]["walks"] for s in common)
    out["contrasts"][f"{a} - {b_}"] = {"n_pairs": len(common), "dtrack": float(np.mean(dt)),
                                       "dtrack_ci": boot_ci(dt), "dJ": float(np.mean(dj)), "dJ_ci": boot_ci(dj),
                                       "d_walks": int(dw),
                                       "seeds_positive_track": int(sum(x > 0 for x in dt))}
  for a, b_ in (("bias", "value"),):
    if arms.get(a) and arms.get(b_):
      for key in ("t50", "t90rel"):
        pairs = [(arms[a][s][key], arms[b_][s][key]) for s in sorted(set(arms[a]) & set(arms[b_]))
                 if arms[a][s].get(key) is not None and arms[b_][s].get(key) is not None]
        if pairs:
          d = [x - y for x, y in pairs]
          out.setdefault("speed_contrasts", {})[f"{a} - {b_} {key}"] = {
            "n_pairs": len(pairs), "median_delta": float(np.median(d)), "mean_delta": float(np.mean(d)),
            "ci": boot_ci(d), "seeds_faster": int(sum(x < 0 for x in d))}
  # d117 equivalence for bias vs value
  c = out["contrasts"].get("bias - value")
  v = out["arms"].get("value")
  if c and v:
    ok = (abs(c["dtrack"]) <= 0.03 and abs(c["dJ"]) <= 0.03 * abs(v["J"]) and abs(c["d_walks"]) <= 1)
    out["bias_vs_value_equivalent_d117"] = bool(ok)
  return out


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r6g1queue-0ede5636")
  a = ap.parse_args()
  from mjlab_centering import r2  # noqa: F401  (path setup for the import inside read_arm)
  sys.path.insert(0, str(ROOT / "src"))
  arms = {k: read_arm(a.campaign, p) for k, p in ARMS.items()}
  print(json.dumps(summarise(arms), indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
