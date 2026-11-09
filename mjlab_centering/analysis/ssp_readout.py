"""SSP reading rules (pre-registered: d105/d106 as amended by d128/d129). Reads local run dirs (restore first:
`python analysis/restore_runs.py --out results/merged_r2`).

    python analysis/ssp_readout.py width      # Stage A (+ length-scale check, when present) and the Stage-B gate
    python analysis/ssp_readout.py pilot      # Stage B escalation rule

Stage A (centered, offset +1, gamma .999, seeds 1000-1002; `walks` = last-5 eval mean tracking >= 0.5):
  best SSP      = highest mean tracking over seeds; configs within 0.03 of the top prefer the narrower code, then ls 0.3.
  closes gap    = best SSP mean >= raw mean - 0.03                                                       (d105 i)
  beats RFF     = SSP mean tracking >= matched RFF + 0.05 and walks at least as often                    (d105 ii)
  8x8 pilot     = runs iff 4x4 has the highest mean of {2x2, 4x4} SSP configs at ls 0.3                  (d105 iv)
  ls check      = runs iff the best SSP trails raw by > 0.05 tracking: that width at ls 0.15 and 0.6     (d128)
  Stage-B gate  = the best SSP (over every config run, ls check included) walks on >= 2 of 3 seeds        (d106, d128)
Stage B (best SSP x {none, value} x offset {0.1, 1}, seeds 3000-3004, paired with the raw cells), per offset, with
Delta = SSP - raw on the same seeds                                                                      (d129):
  reliability   |walks_SSP - walks_raw| >= 2 of 5 in a cell                      -> escalate that cell to seeds 3000-3019
  interaction   |mean over seeds of [(SSP-raw)|value - (SSP-raw)|none]| > 0.10    -> escalate both arms at that offset
  otherwise     reported as a pilot null. Nothing in Stage B is a claim at 5 seeds.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results" / "merged_r2"
SEARCH = (1000, 1001, 1002)
PILOT_SEEDS = tuple(range(3000, 3005))
LS_SWEEP = (0.03, 0.1, 0.3, 1.0)  # d138
WALK = 0.5


def track(job: str, root: Path = RESULTS) -> float | None:
  """Last-5 mean tracking of a FINISHED run; None otherwise.

  A run that is still training has eval rows too, and counting them made an interim table report full cells for cells
  that were half-done (the same mistake as the G1 readout, d134). Only a `DONE` marker, written by the restore for a
  finished run, qualifies.
  """
  p = root / job / "eval.jsonl"
  if not p.exists() or not (root / job / "DONE").exists():
    return None
  rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  starts = [i for i, r in enumerate(rows) if r.get("iter") == min(x["iter"] for x in rows)] if rows else []
  rows = rows[starts[-1]:] if starts else rows
  rows = sorted(rows, key=lambda r: r["iter"])
  if len(rows) < 5:
    return None
  return float(np.mean([r["track_lin_frac"] for r in rows[-5:]]))


def _cell(ids: list[str], root: Path = RESULTS) -> dict:
  v = [track(j, root) for j in ids]
  ok = [x for x in v if x is not None]
  return {"n": len(ok), "of": len(ids), "mean": float(np.mean(ok)) if ok else float("nan"),
          "walks": int(sum(x >= WALK for x in ok)), "per_seed": v}


def width_ids(root: Path = RESULTS) -> dict[str, list[str]]:
  cfg: dict[str, list[str]] = {"raw": [f"r4width_go1_raw_value_s{s}" for s in SEARCH]}
  for n in (2, 4):
    cfg[f"ssp{n}x{n}@0.3"] = [f"r4width_go1_ssp_low{n}x{n}_value_s{s}" for s in SEARCH]
    cfg[f"rff{n}x{n}"] = [f"r4width_go1_rff_low{n}x{n}_value_s{s}" for s in SEARCH]
  cfg["ssp8x8@0.3"] = [f"r4width8_go1_ssp_low8x8_value_s{s}" for s in SEARCH]
  cfg["rff8x8"] = [f"r4width8_go1_rff_low8x8_value_s{s}" for s in SEARCH]
  for n in (2, 4, 8):  # d138: the unconditional log-spaced sweep, three new values per width (0.3 is the stage job)
    for ls in (0.03, 0.1, 1.0):
      cfg[f"ssp{n}x{n}@{ls:g}"] = [f"r6lss_go1_ssp_low{n}x{n}_ls{ls:g}_value_s{s}" for s in SEARCH]
  return cfg


def pick_best(cells: dict[str, dict], cores: list[str]) -> str | None:
  """Highest mean tracking; anything within 0.03 of the top prefers the narrower code, then ls closest to 0.3."""
  have = {k: cells[k] for k in cores if k in cells and cells[k]["n"] == cells[k]["of"]}
  if not have:
    return None
  top = max(c["mean"] for c in have.values())
  close = [k for k, c in have.items() if c["mean"] >= top - 0.03]

  def key(k: str):
    n = int(k[3:k.index("x")])
    ls = float(k.split("@")[1])
    return (n, abs(math.log(ls / 0.3)), k)
  return sorted(close, key=key)[0]


def width_report(root: Path = RESULTS) -> dict:
  """Width stage, then d138's unconditional sweep at the best width, then the Stage-B gate."""
  cells = {k: _cell(v, root) for k, v in width_ids(root).items()}
  cells = {k: c for k, c in cells.items() if c["n"] > 0}
  out: dict = {"cells": {k: {kk: vv for kk, vv in c.items() if kk != "per_seed"} for k, c in cells.items()}}
  raw = cells.get("raw")
  if not raw or raw["n"] < 3:
    out["verdict"] = "raw reference incomplete: " + str({k: v["n"] for k, v in cells.items()})
    return out
  s2, s4 = cells.get("ssp2x2@0.3"), cells.get("ssp4x4@0.3")
  out["run_8x8_pilot_d105iv"] = bool(s2 and s4 and s2["n"] == 3 and s4["n"] == 3 and s4["mean"] > s2["mean"])
  widths = [2, 4] + ([8] if out["run_8x8_pilot_d105iv"] else [])
  best03 = pick_best(cells, [f"ssp{n}x{n}@0.3" for n in widths])
  if best03 is None:
    out["verdict"] = "no complete SSP width at ls 0.3 yet"
    return out
  n = int(best03[3:best03.index("x")])
  out["best_at_ls0.3"] = best03
  out["gap_to_raw"] = raw["mean"] - cells[best03]["mean"]
  out["closes_gap_d105i"] = bool(out["gap_to_raw"] <= 0.03)
  for m in (2, 4, 8):
    a, r = cells.get(f"ssp{m}x{m}@0.3"), cells.get(f"rff{m}x{m}")
    if a and r and a["n"] == 3 and r["n"] == 3:
      out[f"ssp_beats_rff_{m}x{m}"] = bool(a["mean"] >= r["mean"] + 0.05 and a["walks"] >= r["walks"])
  # d138: the sweep is unconditional, at the best width only
  sweep_keys = [f"ssp{n}x{n}@{ls:g}" for ls in LS_SWEEP]
  sweep = {ls: cells.get(f"ssp{n}x{n}@{ls:g}") for ls in LS_SWEEP}
  out["ls_sweep"] = {"width": n, "values": {str(ls): (None if c is None else {"mean": c["mean"], "walks": c["walks"], "n": c["n"]}) for ls, c in sweep.items()},
                     "done": all(c is not None and c["n"] == 3 for c in sweep.values())}
  if not out["ls_sweep"]["done"]:
    missing = [str(ls) for ls, c in sweep.items() if c is None or c["n"] < 3]
    out["verdict"] = (f"run the length-scale sweep at {n}x{n} (d138): missing {', '.join(missing)}; "
                      f"`python jobs_r4.py ssp-ls-sweep {n}` then launch it")
    return out
  best_ls = max(LS_SWEEP, key=lambda ls: (sweep[ls]["mean"], -abs(math.log(ls / 0.3))))
  out["best_ls"] = best_ls
  out["best_overall"] = f"ssp{n}x{n}@{best_ls:g}"
  out["best_walks"] = f"{sweep[best_ls]['walks']} of {sweep[best_ls]['of']}"
  out["stage_b_gate_passes"] = bool(sweep[best_ls]["walks"] >= 2)
  out["verdict"] = (f"Stage B runs on {out['best_overall']}" if out["stage_b_gate_passes"]
                    else "SSP does not walk at any width or length scale tried: skip Stage B")
  return out


def pilot_ids(n: int, ls: float, offset: float, center: str, seed: int) -> tuple[str, str]:
  o = f"{offset:g}"
  ssp = f"r6sspp_go1_ssp_low{n}x{n}_ls{ls:g}_g0.999_o{o}_{center}_s{seed}"
  raw = (f"w2main_go1_raw_{center}_g0.999_o1_s{seed}" if offset == 1.0
         else f"r4sweep_go1_g0.999_o{o}_{center}_s{seed}")
  return ssp, raw


def pilot_report(n: int, ls: float, root: Path = RESULTS) -> dict:
  out: dict = {"config": f"ssp_low {n}x{n} ls {ls:g}", "offsets": {}, "escalate": []}
  for offset in (0.1, 1.0):
    per = {c: {s: tuple(track(j, root) for j in pilot_ids(n, ls, offset, c, s)) for s in PILOT_SEEDS} for c in ("none", "value")}
    rec: dict = {}
    for c in ("none", "value"):
      pairs = [v for v in per[c].values() if v[0] is not None and v[1] is not None]
      ws = sum(a >= WALK for a, _ in pairs)
      wr = sum(b >= WALK for _, b in pairs)
      rec[c] = {"n_pairs": len(pairs), "walks_ssp": ws, "walks_raw": wr, "delta_walks": ws - wr,
                "mean_ssp": float(np.mean([a for a, _ in pairs])) if pairs else float("nan"),
                "mean_raw": float(np.mean([b for _, b in pairs])) if pairs else float("nan"),
                "mean_delta": float(np.mean([a - b for a, b in pairs])) if pairs else float("nan")}
      if len(pairs) == len(PILOT_SEEDS) and abs(ws - wr) >= 2:
        out["escalate"].append({"offset": offset, "center": c, "why": f"reliability, delta walks {ws - wr:+d} of 5"})
    seeds = [s for s in PILOT_SEEDS if all(x is not None for c in ("none", "value") for x in per[c][s])]
    did = [(per["value"][s][0] - per["value"][s][1]) - (per["none"][s][0] - per["none"][s][1]) for s in seeds]
    rec["interaction"] = {"n_seeds": len(did), "mean": float(np.mean(did)) if did else float("nan"),
                          "se": float(np.std(did, ddof=1) / math.sqrt(len(did))) if len(did) > 1 else float("nan")}
    if len(did) == len(PILOT_SEEDS) and abs(rec["interaction"]["mean"]) > 0.10:
      for c in ("none", "value"):
        out["escalate"].append({"offset": offset, "center": c, "why": f"interaction {rec['interaction']['mean']:+.3f} tracking"})
    out["offsets"][str(offset)] = rec
  # one escalation per (offset, center); keep the reasons joined
  merged: dict[tuple, dict] = {}
  for e in out["escalate"]:
    k = (e["offset"], e["center"])
    merged[k] = {**e, "why": merged[k]["why"] + "; " + e["why"]} if k in merged else e
  out["escalate"] = list(merged.values())
  n_pairs = sum(rec[c]["n_pairs"] for rec in out["offsets"].values() for c in ("none", "value"))
  if n_pairs == 0:
    out["verdict"] = "incomplete: no Stage-B run has both its SSP and raw cell yet"
  elif out["escalate"]:
    out["verdict"] = "escalate: " + ", ".join(f"o{e['offset']:g}/{e['center']}" for e in out["escalate"])
  else:
    out["verdict"] = "pilot null"
  return out


def main(argv: list[str]) -> None:
  if not argv or argv[0] not in ("width", "pilot"):
    raise SystemExit(__doc__)
  if argv[0] == "width":
    print(json.dumps(width_report(), indent=1))
  else:
    n, ls = int(argv[1]), float(argv[2])
    print(json.dumps(pilot_report(n, ls), indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
