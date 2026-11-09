"""Shock tables under the d116 filters, from branch records (the `branches` dict that report4/report5 write).

    python analysis/shock_tables.py results/report4_shock.json [more.json ...]

Each branch record carries `base` (the mother's pre-shock level: mean of its last 3 evals before the shock), `dip`,
`end` (mean of the last 5 evals) and the shock `kind`, `arm`, `seed`. Two analyses are printed side by side:
  (A) all seeds, as pre-registered;
  (B) walking mothers only: base >= WALK_MOTHER (0.8, d116). Arm rows use that arm's walking mothers; the arm contrast
      uses the seeds where both arms' mothers walk.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

WALK_MOTHER = 0.8
KINDS = ("off+0.1", "off+1.0", "actx5")


def _t(d: list[float]) -> dict:
  d = np.asarray(d, float)
  n = len(d)
  if n < 2:
    return {"n": n, "mean": float(d.mean()) if n else float("nan"), "ci": [float("nan")] * 2, "positive": int((d > 0).sum())}
  se = d.std(ddof=1) / math.sqrt(n)
  h = stats.t.ppf(0.975, n - 1) * se
  return {"n": n, "mean": float(d.mean()), "se": float(se), "ci": [float(d.mean() - h), float(d.mean() + h)],
          "positive": int((d > 0).sum())}


def load(paths: list[str]) -> list[dict]:
  rows: list[dict] = []
  for p in paths:
    rows += list(json.load(open(p))["branches"].values())
  return rows


def tables(rows: list[dict], walking_only: bool) -> dict:
  by = {(r["kind"], r["arm"], r["seed"]): r for r in rows}
  mother_ok = {}
  for (k, arm, s), r in by.items():
    if k == "ctrl":
      mother_ok[(arm, s)] = r["base"] >= WALK_MOTHER
  out: dict = {"rows": {}, "contrast": {}, "n_mothers": {}}
  for arm in ("none", "value"):
    seeds = sorted(s for (a, s) in mother_ok if a == arm)
    keep = [s for s in seeds if (mother_ok[(arm, s)] or not walking_only)]
    out["n_mothers"][arm] = {"all": len(seeds), "kept": len(keep)}
    for kind in KINDS:
      rs = [by[(kind, arm, s)] for s in keep if (kind, arm, s) in by and ("ctrl", arm, s) in by]
      if not rs:
        continue
      ctrl = [by[("ctrl", arm, r["seed"])] for r in rs]
      d = [r["end"] - c["end"] for r, c in zip(rs, ctrl)]
      out["rows"][f"{kind}|{arm}"] = {
        "n": len(rs), "pre": float(np.mean([r["base"] for r in rs])), "dip": float(np.mean([r["dip"] for r in rs])),
        "end": float(np.mean([r["end"] for r in rs])), "ctrl_end": float(np.mean([c["end"] for c in ctrl])),
        "recovered": int(sum(r["end"] >= 0.95 * r["base"] for r in rs)), "shocked_minus_ctrl": _t(d)}
  for kind in KINDS:
    seeds = sorted(s for (k, a, s) in by if k == kind and a == "none")
    use = [s for s in seeds if (kind, "value", s) in by and ("ctrl", "none", s) in by and ("ctrl", "value", s) in by
           and (not walking_only or (mother_ok[("none", s)] and mother_ok[("value", s)]))]
    d = [(by[(kind, "value", s)]["end"] - by[("ctrl", "value", s)]["end"]) -
         (by[(kind, "none", s)]["end"] - by[("ctrl", "none", s)]["end"]) for s in use]
    out["contrast"][kind] = _t(d)
  return out


def fmt(res: dict, title: str) -> str:
  L = [f"## {title}", f"mothers kept: {res['n_mothers']}", "",
       "| shock | arm | n | pre-shock | worst dip | end | control end | final >= 95% pre | shocked - control (95% CI) |",
       "|---|---|---|---|---|---|---|---|---|"]
  for key, r in res["rows"].items():
    kind, arm = key.split("|")
    t = r["shocked_minus_ctrl"]
    L.append(f"| {kind} | {arm} | {r['n']} | {r['pre']:.3f} | {r['dip']:+.3f} | {r['end']:.3f} | {r['ctrl_end']:.3f} | "
             f"{r['recovered']}/{r['n']} | {t['mean']:+.4f} [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] |")
  L += ["", "| shock | arm contrast (value - none, shocked-minus-control) | 95% CI | seeds positive |", "|---|---|---|---|"]
  for kind, t in res["contrast"].items():
    L.append(f"| {kind} | {t['mean']:+.4f} (n={t['n']}) | [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | {t['positive']}/{t['n']} |")
  return "\n".join(L)


if __name__ == "__main__":
  rows = load(sys.argv[1:])
  print(fmt(tables(rows, False), "(A) all seeds, as pre-registered"))
  print()
  print(fmt(tables(rows, True), f"(B) walking mothers only (pre-shock level >= {WALK_MOTHER})"))
