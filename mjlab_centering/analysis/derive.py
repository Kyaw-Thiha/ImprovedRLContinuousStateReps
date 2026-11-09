"""Derive every reported quantity from the per-iteration JSONL, the eval JSONL and final.json.

Rules are frozen in docs/mjlab_centering/01-preregistration.md:
  J_task      mean per-step task reward of the deterministic eval policy; per seed, the mean of its last 5 evals
  J_ref       plateau of the reference arm (raw, no centering, gamma=0.99, offset 0), continuing wrapper
  Tq          first eval iteration at which the 3-eval running mean reaches q * J_ref and stays above it
  AULC        mean J_task over eval iterations, divided by J_ref (so 1.0 = reach the reference immediately)
  collapse    final < 0.5 * J_ref (or a NaN anywhere in iter.jsonl)
  shock       dip = min(J_task after shock)/pre ; recovery = first eval after the shock at >= 0.95*pre
  const_frac  share of critic error that is a constant offset (mean(err)^2 / mse)

    python analysis/derive.py --campaign c1-abc123 --pull      # download from R2 into results/<campaign>/
    python analysis/derive.py --campaign c1-abc123 --local      # use results/<job>/ already on disk
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("eta", "A", "fixclip", "fixnorm", "fixscale", "E", "pen", "lr", "ssp")


# ------------------------------------------------------------------ io
def pull(campaign: str, dest: Path) -> None:
  from mjlab_centering import r2

  for k in r2.list_keys(f"{campaign}/runs/"):
    rel = k.split(f"{campaign}/runs/", 1)[1]
    local = dest / rel
    if not local.exists() and not k.endswith("model_final.pt"):
      r2.download_file(f"{campaign}/runs/{rel}", str(local))


def load_jobs(campaign: str, local: bool, dest: Path) -> list[dict]:
  from mjlab_centering import r2

  manifest = r2.get_json(f"{campaign}/manifest.json")
  jobs = manifest["jobs"] if manifest else []
  if local:
    for p in dest.glob("*/config.json"):
      pass
  return jobs


def read_jsonl(p: Path) -> list[dict]:
  if not p.exists():
    return []
  out = []
  for line in p.read_text().splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      out.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return out


# ------------------------------------------------------------------ per-run metrics
def derive_run(job: dict, run_dir: Path) -> dict:
  evals = read_jsonl(run_dir / "eval.jsonl")
  iters = read_jsonl(run_dir / "iter.jsonl")
  final = json.loads((run_dir / "final.json").read_text()) if (run_dir / "final.json").exists() else {}
  evals.sort(key=lambda r: r["iter"])
  iters.sort(key=lambda r: r["iter"])
  out = {
    "job_id": job["job_id"],
    "block": job.get("block"),
    "seed": job.get("seed"),
    "center": job.get("center"),
    "gamma": job.get("gamma"),
    "offset": job.get("offset"),
    "shock_kind": job.get("shock_kind"),
    "iters": job.get("iters"),
    "n_evals": len(evals),
    "n_iters": len(iters),
    "done": (run_dir / "final.json").exists(),
  }
  if not evals:
    return out
  it = np.array([e["iter"] for e in evals])
  j = np.array([e["J_task"] for e in evals], dtype=float)
  falls = np.array([e["fall_rate_per_1k"] for e in evals], dtype=float)
  out["final"] = float(np.mean(j[-5:]))
  out["falls_final"] = float(np.mean(falls[-5:]))
  out["jcurve"] = j.tolist()
  out["itcurve"] = it.tolist()
  out["aulc"] = float(np.mean(j)) if len(j) else float("nan")
  if iters:
    cf = np.array([r.get("const_frac", np.nan) for r in iters], dtype=float)
    v = np.array([r.get("v_mean", np.nan) for r in iters], dtype=float)
    out["const_frac_early"] = float(np.nanmedian(cf[:50])) if len(cf) else float("nan")
    out["const_frac_late"] = float(np.nanmedian(cf[-50:])) if len(cf) else float("nan")
    out["v_abs_early"] = float(np.nanmedian(np.abs(v[:50]))) if len(v) else float("nan")
    out["v_abs_late"] = float(np.nanmedian(np.abs(v[-50:]))) if len(v) else float("nan")
    out["steps_since_reset_p95"] = float(np.nanmax([r.get("steps_since_reset_p95", np.nan) for r in iters]))
    out["nan"] = bool(any(not math.isfinite(r.get("r_task_rate", 0.0)) for r in iters))
  # shock metrics
  if job.get("shock_kind") not in (None, "none") and job.get("shock_at") is not None:
    at = int(job["shock_at"])
    pre = j[it < at][-3:]
    post = j[it >= at]
    out["shock_pre"] = float(np.mean(pre)) if len(pre) else float("nan")
    out["shock_dip"] = float(np.min(post) / np.mean(pre)) if len(pre) and len(post) else float("nan")
    rec = np.where(post >= 0.95 * np.mean(pre))[0] if len(pre) else []
    out["shock_recover_evals"] = int(rec[0]) * int(np.median(np.diff(it))) if len(rec) else -1
  if final:
    for old, new in (("nominal_5k", "nominal"), ("cmd_extrap_3k", "cmd_extrap"), ("delay1_3k", "delay1")):
      d = final.get(new, final.get(old))
      if isinstance(d, dict) and "J_task" in d:
        out[f"{new}_J"] = d["J_task"]
        out[f"{new}_falls"] = d["fall_rate_per_1k"]
  return out


def thresholds(run: dict, j_ref: float) -> dict:
  it = np.asarray(run.get("itcurve", []), dtype=float)
  j = np.asarray(run.get("jcurve", []), dtype=float)
  out = {}
  if len(j) < 4 or not math.isfinite(j_ref) or j_ref <= 0:
    return {"T50": -1, "T90": -1, "T95": -1}
  # edge-replicate before smoothing: zero padding would make the tail of every curve look like a drop
  sm = np.convolve(np.pad(j, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
  for q in (0.5, 0.9, 0.95):
    hit = -1
    for i in range(len(sm)):
      if sm[i] >= q * j_ref and np.all(sm[i:] >= q * j_ref * 0.98):
        hit = int(it[i])
        break
    out[f"T{int(q*100)}"] = hit
  return out


# ------------------------------------------------------------------ aggregates
def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
  x = np.asarray([v for v in x if math.isfinite(v)], dtype=float)
  if len(x) == 0:
    return float("nan"), float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
  return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_diff(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float, int]:
  seeds = sorted(set(a) & set(b))
  d = np.array([a[s] - b[s] for s in seeds], dtype=float)
  if len(d) == 0:
    return float("nan"), float("nan"), float("nan"), 0
  mean, lo, hi = boot_ci(d)
  return mean, lo, hi, len(d)


def summarise(runs: list[dict]) -> dict:
  ref = [r for r in runs if r["block"] == "A" and r["center"] == "none" and r["gamma"] == 0.99 and r["offset"] == 0.0]
  j_ref = float(np.mean([r["final"] for r in ref])) if ref else float("nan")
  for r in runs:
    r.update(thresholds(r, j_ref))
  cells: dict[str, dict] = {}
  for r in runs:
    key = f"{r['block']}|{r['center']}|g{r['gamma']}|o{r['offset']}" + (
      f"|{r['shock_kind']}" if r.get("shock_kind") not in (None, "none") else ""
    )
    cells.setdefault(key, {"final": {}, "T50": {}, "T90": {}, "T95": {}, "aulc": {}, "falls": {},
                           "const_frac_late": {}, "seeds": []})
    c = cells[key]
    if not r["done"]:
      continue
    c["seeds"].append(r["seed"])
    c["final"][r["seed"]] = r["final"] / j_ref if j_ref else float("nan")
    c["aulc"][r["seed"]] = r["aulc"]
    c["falls"][r["seed"]] = r.get("falls_final", float("nan"))
    c["T50"][r["seed"]] = r.get("T50", -1)
    c["T90"][r["seed"]] = r.get("T90", -1)
    c["T95"][r["seed"]] = r.get("T95", -1)
    c["const_frac_late"][r["seed"]] = r.get("const_frac_late", float("nan"))
  table = {}
  for k, c in cells.items():
    m, lo, hi = boot_ci(np.array(list(c["final"].values())))
    t50 = np.array([v for v in c["T50"].values() if v >= 0], dtype=float)
    table[k] = {
      "n": len(c["seeds"]),
      "final_over_ref": m, "ci": [lo, hi],
      "aulc": float(np.nanmean(list(c["aulc"].values()))) if c["aulc"] else float("nan"),
      "falls_final": float(np.nanmean(list(c["falls"].values()))) if c["falls"] else float("nan"),
      "T50_median": float(np.median(t50)) if len(t50) else -1,
      "T90_median": float(np.median([v for v in c["T90"].values() if v >= 0] or [-1])),
      "T95_median": float(np.median([v for v in c["T95"].values() if v >= 0] or [-1])),
      "const_frac_late": float(np.nanmedian(list(c["const_frac_late"].values()))),
      "final_by_seed": c["final"],
    }
  # headline paired comparisons (preregistration P4/P5): centered vs each rival at gamma=.999, offset c
  def cell(block, center, gamma=0.999, offset=1.0, shock=None):
    return [r for r in runs if r["block"] == block and r["center"] == center and r["gamma"] == gamma
            and r["offset"] == offset and (r.get("shock_kind") in (None, "none") if shock is None else r.get("shock_kind") == shock)]

  comparisons = {}
  base = {r["seed"]: r["final"] for r in cell("A", "value")}
  for label, rs in (
    ("value_vs_none", cell("A", "none")),
    ("value_vs_fixclip", cell("fixclip", "none")),
    ("value_vs_fixnorm", cell("fixnorm", "none")),
    ("value_vs_fixscale", cell("fixscale", "none")),
  ):
    other = {r["seed"]: r["final"] for r in rs}
    comparisons[label] = paired_diff(base, other)
  for kind in ("offset", "action_rate"):
    a = {r["seed"]: r.get("shock_recover_evals", -1) for r in cell("E", "value", shock=kind)}
    b = {r["seed"]: r.get("shock_recover_evals", -1) for r in cell("E", "none", shock=kind)}
    comparisons[f"shock_recovery_iters_{kind}"] = paired_diff(a, b)
    a = {r["seed"]: r.get("shock_dip", float("nan")) for r in cell("E", "value", shock=kind)}
    b = {r["seed"]: r.get("shock_dip", float("nan")) for r in cell("E", "none", shock=kind)}
    comparisons[f"shock_dip_{kind}"] = paired_diff(a, b)
  return {"j_ref": j_ref, "n_runs": len(runs), "cells": table, "paired_diffs": comparisons}


def to_markdown(res: dict) -> str:
  lines = [f"J_ref = {res['j_ref']:.5f}  (n_runs = {res['n_runs']})", "",
           "| cell | n | final/J_ref | 95% CI | T50 | T90 | T95 | AULC | falls/1k | const_frac |",
           "|---|---|---|---|---|---|---|---|---|---|"]
  for k, v in sorted(res["cells"].items()):
    ci = f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if math.isfinite(v["ci"][0]) else "-"
    lines.append(
      f"| {k} | {v['n']} | {v['final_over_ref']:.3f} | {ci} | {v['T50_median']:.0f} | {v['T90_median']:.0f} | "
      f"{v['T95_median']:.0f} | {v['aulc']:.4f} | {v['falls_final']:.2f} | {v['const_frac_late']:.3f} |"
    )
  lines += ["", "Paired differences (later minus earlier), 95% bootstrap CI:", ""]
  for k, (m, lo, hi, n) in res["paired_diffs"].items():
    lines.append(f"- {k}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={n})")
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True, help="one campaign id, or several comma-separated")
  ap.add_argument("--j-ref", type=float, default=None, help="override the reference plateau (absolute J_task)")
  ap.add_argument("--pull", action="store_true")
  ap.add_argument("--local", action="store_true")
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  camps = [c for c in a.campaign.split(",") if c]
  dest = ROOT / "results" / camps[0]
  dest.mkdir(parents=True, exist_ok=True)
  jobs: list[dict] = []
  for c in camps:
    d = ROOT / "results" / c
    d.mkdir(parents=True, exist_ok=True)
    if a.pull:
      pull(c, d)
    jobs += load_jobs(c, a.local, d)
  runs = []
  for j in jobs:
    d = ROOT / "results" / camps[0] / j["job_id"]
    if not d.exists():
      for c in camps[1:]:
        alt = ROOT / "results" / c / j["job_id"]
        if alt.exists():
          d = alt
          break
    runs.append(derive_run(j, d))
  # prune giant curves from the report file but keep them in the per-run block
  res = summarise(runs)
  if a.j_ref is not None:
    res["j_ref_computed"] = res["j_ref"]
    res["j_ref"] = float(a.j_ref)
  out = Path(a.out) if a.out else ROOT / "results" / f"derived_{camps[0]}.json"
  out.write_text(json.dumps({"summary": res, "runs": runs}, indent=1, default=float))
  md = to_markdown(res)
  (out.with_suffix(".md")).write_text(md)
  print(md)


if __name__ == "__main__":
  main()
