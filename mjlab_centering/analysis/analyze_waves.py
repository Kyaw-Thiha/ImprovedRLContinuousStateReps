"""Analysis for the wave runs (entries 35-47): per-run summaries, paired bootstrap contrasts, equivalence against the
frozen margin, shock dip/recovery, branching validation, and wave-1 selection.

    python analysis/analyze_waves.py --results /project/results/w2 --manifest manifests/w2.json --margin 0.03 --out w2.json

Unit of analysis is the seed; every contrast is paired on shared seeds. Primary metric: J_task (mean of the last 5
evals) unless `--metric track_lin_frac`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOOT = 5000


def _rows(path: Path) -> list[dict]:
  return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def load_runs(results: str, manifest: str) -> dict[str, dict]:
  jobs = {j["job_id"]: j for j in json.loads(Path(manifest).read_text())["jobs"]}
  out = {}
  for jid, j in jobs.items():
    d = Path(results) / jid
    ev = sorted(_rows(d / "eval.jsonl"), key=lambda r: r["iter"])
    out[jid] = {"job": j, "eval": ev, "iter": _rows(d / "iter.jsonl"), "done": (d / "final.json").exists()}
  return out


def final(run: dict, metric: str = "J_task", k: int = 5) -> float:
  v = [r[metric] for r in run["eval"] if metric in r]
  return float(np.mean(v[-k:])) if v else float("nan")


def time_to(run: dict, frac: float, metric: str = "track_lin_frac") -> float:
  """First eval iteration whose 3-eval smoothed metric reaches `frac` of its own final value (edge-replicated)."""
  v = np.array([r[metric] for r in run["eval"] if metric in r], dtype=float)
  it = np.array([r["iter"] for r in run["eval"] if metric in r], dtype=float)
  if len(v) < 3:
    return float("nan")
  sm = np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")
  tgt = frac * float(np.mean(v[-5:]))
  hit = np.nonzero(sm >= tgt)[0]
  return float(it[hit[0]]) if len(hit) else float("inf")


def const_frac_late(run: dict, k: int = 50) -> float:
  v = [r["const_frac"] for r in run["iter"][-k:] if "const_frac" in r]
  return float(np.mean(v)) if v else float("nan")


def boot_ci(d: np.ndarray, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
  """Mean and 95% percentile bootstrap CI of paired differences d (one per seed)."""
  rng = rng or np.random.default_rng(0)
  d = np.asarray(d, dtype=float)
  d = d[np.isfinite(d)]
  if len(d) < 2:
    return float("nan"), float("nan"), float("nan")
  idx = rng.integers(0, len(d), size=(BOOT, len(d)))
  m = d[idx].mean(1)
  return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def by_seed(runs: dict, pick, metric: str = "J_task") -> dict[int, float]:
  return {r["job"]["seed"]: final(r, metric) for r in runs.values() if pick(r["job"]) and r["done"]}


def paired(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
  common = sorted(set(a) & set(b))
  return np.array([a[s] - b[s] for s in common])


def equivalence(ci: tuple[float, float, float], margin: float) -> dict:
  m, lo, hi = ci
  return {"mean": m, "lo": lo, "hi": hi, "margin": margin, "half_width": (hi - lo) / 2,
          "equivalent": bool(lo > -margin and hi < margin), "precision_ok": bool((hi - lo) / 2 <= margin)}


def core_contrasts(runs: dict, robot: str, gamma: float, margin_frac: float, metric: str = "J_task",
                   reference_scale: float | None = None) -> dict:
  def cell(rep, center):
    return by_seed(
      runs,
      lambda j: j["robot"] == robot and j["gamma"] == gamma and j["block"].startswith(("core", "tie"))
      and (j["extra"].get("encoder", "raw") == rep) and j["center"] == center,
      metric,
    )

  raw_n, raw_v, ssp_n, ssp_v = cell("raw", "none"), cell("raw", "value"), cell("ssp_low", "none"), cell("ssp_low", "value")
  # The margin must scale with the *task*, not with whichever cell happens to be the comparator: with an offset present
  # the uncentered arms sit near the standing floor, and 3% of that is a meaningless bar. Use the best observed cell
  # mean (or an explicit reference scale, e.g. the gate's 0.10) as the unit.
  cell_means = [float(np.nanmean(list(v.values()))) for v in (raw_n, raw_v, ssp_n, ssp_v) if v]
  ref = reference_scale if reference_scale is not None else (max(cell_means) if cell_means else float("nan"))
  margin = margin_frac * abs(ref)
  out = {"robot": robot, "gamma": gamma, "reference_scale": ref, "margin": margin,
         "n": {k: len(v) for k, v in dict(raw_none=raw_n, raw_value=raw_v, ssp_none=ssp_n, ssp_value=ssp_v).items()}}
  out["centering_main_raw"] = equivalence(boot_ci(paired(raw_v, raw_n)), margin)
  if ssp_n and ssp_v:
    out["centering_main_ssp"] = equivalence(boot_ci(paired(ssp_v, ssp_n)), margin)
    out["ssp_main_none"] = equivalence(boot_ci(paired(ssp_n, raw_n)), margin)
    common = sorted(set(raw_n) & set(raw_v) & set(ssp_n) & set(ssp_v))
    inter = np.array([(ssp_v[s] - raw_v[s]) - (ssp_n[s] - raw_n[s]) for s in common])
    out["interaction"] = equivalence(boot_ci(inter), margin)
  return out


def shock_metrics(runs: dict, shock_at: int, metric: str = "J_task") -> dict:
  """Dip depth and recovery time after the reward shock, relative to each seed's own pre-shock level.

  The pre-shock baseline must come from the **mother** run (the unbranched core run of the same seed): a shock branch
  is resumed at `shock_at`, so its own evals all start after the shock and cannot provide a baseline. `dip` is
  baseline - min(after); `recovery` is the first post-shock eval within 5% of the baseline (censored at the run end).
  """
  res: dict = {}
  for center in ("none", "value"):
    for kind in ("none", "offset", "action_rate"):  # "none" = resumed-unshocked control: its dip must be ~0
      dips, recs, rec50, bases = [], [], [], []
      for r in runs.values():
        j = r["job"]
        if j["block"] != "shock" or j["center"] != center or j["shock_kind"] != kind or not r["done"]:
          continue
        mother = runs.get((j.get("extra") or {}).get("after"))
        if not mother:
          continue
        pre = [e[metric] for e in mother["eval"] if e["iter"] < shock_at][-3:]
        post = [e for e in r["eval"] if e["iter"] >= shock_at and metric in e]
        if not pre or not post:
          continue
        base = float(np.mean(pre))
        bases.append(base)
        dip = base - min(e[metric] for e in post)
        dips.append(dip)
        hit = [e["iter"] for e in post if e[metric] >= 0.95 * base]
        recs.append(float(hit[0] - shock_at) if hit else float(post[-1]["iter"] - shock_at + 1))  # censored
        # Baseline-free recovery: iterations until half of THIS arm's own dip is recovered. The 95%-of-baseline measure
        # depends on where the arm started (a centred arm starts 3x higher, so 95% of its baseline is a much harder bar),
        # which is why both are reported.
        half = [e["iter"] for e in post if e[metric] >= base - 0.5 * dip]
        rec50.append(float(half[0] - shock_at) if half else float(post[-1]["iter"] - shock_at + 1))
      res[f"{center}/{kind}"] = {"n": len(dips), "dip_mean": float(np.mean(dips)) if dips else float("nan"),
                                 "dip_relative": float(np.mean([d / b for d, b in zip(dips, bases)])) if dips else float("nan"),
                                 "recovery_mean": float(np.mean(recs)) if recs else float("nan"),
                                 "recovery50_mean": float(np.mean(rec50)) if rec50 else float("nan"),
                                 "dips": dips, "recoveries": recs, "recoveries50": rec50}
  return res


def shock_contrast(sm: dict, kind: str) -> dict:
  a, b = sm[f"none/{kind}"], sm[f"value/{kind}"]
  n = min(len(a["dips"]), len(b["dips"]))
  return {"dip_none_minus_value": boot_ci(np.array(a["dips"][:n]) - np.array(b["dips"][:n])),
          "recovery_none_minus_value": boot_ci(np.array(a["recoveries"][:n]) - np.array(b["recoveries"][:n]))}


def branch_validation(runs: dict, shock_at: int, metric: str = "J_task") -> dict:
  """Resumed-unshocked control vs the unbranched mother over the post-branch window, per (center, seed)."""
  diffs, mothers = [], []
  for r in runs.values():
    j = r["job"]
    if j["block"] != "shock" or j["shock_kind"] != "none" or not r["done"]:
      continue
    m = runs.get((j.get("extra") or {}).get("after"))
    if not m or not m["done"]:
      continue
    w = lambda run: [e[metric] for e in run["eval"] if e["iter"] >= shock_at]  # noqa: E731
    if w(r) and w(m):
      diffs.append(float(np.mean(w(r)[-3:]) - np.mean(w(m)[-3:])))
      mothers.append(float(np.mean(w(m)[-3:])))
  sd = float(np.std(mothers, ddof=1)) if len(mothers) > 2 else float("nan")
  ci = boot_ci(np.array(diffs))
  return {"n": len(diffs), "mean_diff": ci[0], "ci": ci[1:], "between_seed_sd": sd,
          "pass": bool(len(diffs) > 2 and ci[1] <= 0 <= ci[2] and abs(ci[0]) < sd)}


def select_best(runs: dict, block: str, key, metric: str = "J_task") -> dict:
  cells: dict = {}
  for r in runs.values():
    if r["job"]["block"] == block and r["done"]:
      cells.setdefault(key(r["job"]), []).append(final(r, metric))
  return {str(k): {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "n": len(v)}
          for k, v in sorted(cells.items(), key=lambda kv: -np.mean(kv[1]))}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--results", required=True)
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--margin", type=float, default=0.03)
  ap.add_argument("--metric", default="J_task")
  ap.add_argument("--shock-at", type=int, default=600)
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  runs = load_runs(a.results, a.manifest)
  out: dict = {"n_runs": len(runs), "n_done": sum(r["done"] for r in runs.values())}
  blocks = {r["job"]["block"] for r in runs.values()}
  if {"eta", "sslen", "g1eta"} & blocks:
    out["eta_go1"] = select_best(runs, "eta", lambda j: (j["eta"], j["r_bar_init"]), a.metric)
    out["ssp_length_scale"] = select_best(runs, "sslen", lambda j: j["extra"]["ssp_length_scale"], a.metric)
    out["eta_g1"] = select_best(runs, "g1eta", lambda j: j["eta"], a.metric)
  for robot, gammas in (("go1", (0.99, 0.999)), ("g1", (0.99, 0.999))):
    for g in gammas:
      c = core_contrasts(runs, robot, g, a.margin, a.metric)
      if c["n"]["raw_none"]:
        out[f"core/{robot}/g{g}"] = c
  if "shock" in blocks:
    sm = shock_metrics(runs, a.shock_at, a.metric)
    out["shock"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("dips", "recoveries", "recoveries50")}
                    for k, v in sm.items()}
    out["shock_contrast"] = {k: shock_contrast(sm, k) for k in ("offset", "action_rate")}
    out["branch_validation"] = branch_validation(runs, a.shock_at, a.metric)
  txt = json.dumps(out, indent=1, default=float)
  print(txt)
  if a.out:
    Path(a.out).write_text(txt)


if __name__ == "__main__":
  main()
