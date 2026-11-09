"""EXPLORATORY, POST HOC: does kappa = L / S line up Go1 and G1 offset failures better than the level error L alone?

    python analysis/kappa_analysis.py --fetch      # first time: read-only fetch of the G1 claim files that are not cached
    python analysis/kappa_analysis.py              # everything else: tables, regressions, figures, results.json

Run from mjlab_centering/ with its venv (.venv/bin/python). Outputs go to docs/analysis/kappa/ (figures, tables.md,
results.json); the write-up docs/analysis/kappa/kappa.md is hand-written around tables.md.

Nothing here launches compute. R2 is touched only by `--fetch`, and only with read calls: the DONE list, small files of
finished G1 claim jobs (eval.jsonl, final.json, config.json) and RANGED reads of iter.jsonl (head and, for the centered arm,
tail), written to results/kappa_cache/g1/. Nothing is written to or deleted from R2, and the supervisor cache is not modified.

Definitions (user brief, 2026-11-05; all of it post hoc, not pre-registered):
  r_bar   mean per-step reward the learner sees = logged `r_seen_rate` (task + offset - penalty * falls) of the UNCENTERED arm,
          per seed averaged over iterations 0..100, then over seeds.   (Verified: r_seen = r_task + offset - P * fall_rate/1000.)
  L       |r_bar| / (1 - gamma)                      level error of the critic at start (V starts ~0). |.| because r_bar < 0 in some cells.
  S       mean over seeds of the per-seed mean of logged `v_std` of the CENTERED arm of the same cell, iterations 100..300
          ("mid", primary) and the last 250 iterations ("late", robustness). Where a cell has no centered arm, S comes from the
          uncentered arm and the table says so.
  kappa   L / S.
  L_nopen robustness: |r_task + offset| / (1 - gamma), i.e. r_bar with the fall penalty added back (r_task = `r_task_rate`).
  bias-init reference arms: L_eff = |r_bar - c| / (1 - gamma) using the bias arm's own early r_seen_rate (the critic starts at
          c/(1-gamma), so the level that remains is the task-plus-penalty part).
Outcome = "walks" = mean of the last 5 evals of `track_lin_frac` (newest attempt) >= 0.5, exactly as REPORT_004/supervisor figs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MJ = HERE.parent
REPO = MJ.parent
sys.path[:0] = [str(HERE), str(MJ / "src"), str(REPO / "scripts")]

RES = MJ / "results"
CACHE_SUP = RES / "supervisor_cache" / "g1"
CACHE = RES / "kappa_cache" / "g1"
OUT = REPO / "docs" / "analysis" / "kappa"
G1_CAMPAIGN = "r6g1queue-0ede5636"
WALK = 0.5
EARLY = (0, 100)  # r_bar window
MID = (100, 300)  # S window (primary)
LATE_N = 250  # S window (robustness): last 250 iterations
HEAD_BYTES, TAIL_BYTES = 450_000, 400_000  # ~1.01 kB per iter row: head covers iters 0..~440, tail the last ~390


# ================================================================================================ fetch (read-only)
def fetch_g1() -> None:
  from mjlab_centering import r2

  done = sorted(r2.done_set(G1_CAMPAIGN))
  c, bucket = r2.client(), r2.bucket()
  CACHE.mkdir(parents=True, exist_ok=True)
  log = {"campaign": G1_CAMPAIGN, "n_done": len(done), "small_files": 0, "ranged": 0, "full_iter": 0, "r2_calls": "list, get, head only"}
  for jid in done:
    d = CACHE / jid
    d.mkdir(exist_ok=True)
    sup = CACHE_SUP / jid
    for f in ("eval.jsonl", "final.json", "config.json"):
      if (d / f).exists() or ((sup / f).exists() and (sup / "DONE").exists()):
        continue  # (a supervisor-cache copy of a job that had no DONE marker then may be a partial snapshot: re-read it)
      try:
        c.download_file(bucket, r2.key(f"{G1_CAMPAIGN}/runs/{jid}/{f}"), str(d / f))
        log["small_files"] += 1
      except Exception:  # noqa: BLE001 - config.json may be absent
        pass
    if (sup / "iter.jsonl").exists() and sum(1 for _ in open(sup / "iter.jsonl")) >= 2500:
      continue  # full iter file already cached by someone else
    key = r2.key(f"{G1_CAMPAIGN}/runs/{jid}/iter.jsonl")
    size = c.head_object(Bucket=bucket, Key=key)["ContentLength"]
    arm = jid.split("_o1_")[1].rsplit("_s", 1)[0]  # none | value | bias_none
    need_tail = arm == "value"
    if size > 2.2e6 * 1.2 or size < 2.0e6:  # unexpected size (restart appended, or short): take the whole file
      if not (d / "iter_full.jsonl").exists():
        c.download_file(bucket, key, str(d / "iter_full.jsonl"))
        log["full_iter"] += 1
      continue
    if not (d / "iter_head.jsonl").exists():
      (d / "iter_head.jsonl").write_bytes(c.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{HEAD_BYTES - 1}")["Body"].read())
      log["ranged"] += 1
    if need_tail and not (d / "iter_tail.jsonl").exists():
      (d / "iter_tail.jsonl").write_bytes(c.get_object(Bucket=bucket, Key=key, Range=f"bytes={size - TAIL_BYTES}-{size - 1}")["Body"].read())
      log["ranged"] += 1
  (CACHE / "fetch_log.json").write_text(json.dumps(log, indent=1))
  print(json.dumps(log))


# ================================================================================================ data access
def _rows(path: Path, partial_first: bool = False, partial_last: bool = False) -> list[dict]:
  lines = path.read_text().splitlines()
  if partial_first:
    lines = lines[1:]
  if partial_last:
    lines = lines[:-1]
  return [json.loads(l) for l in lines if l.strip()]


def newest(rows: list[dict]) -> list[dict]:
  starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]
  return rows[starts[-1]:] if len(starts) > 1 else rows


def go1_dir(jid: str) -> Path | None:
  for base in (RES / "merged_r4", RES / "merged_r2"):
    if (base / jid / "eval.jsonl").exists() and (base / jid / "iter.jsonl").exists():
      return base / jid
  return None


def iter_rows(jid: str, robot: str) -> dict:
  """{'rows': {iter: row}, 'n_iters': int|None, 'source': str}. G1 claim jobs may be head/tail slices (see --fetch)."""
  if robot == "go1" or not jid.startswith("r6g1_"):
    d = go1_dir(jid)
    if d is None:
      return {"rows": {}, "n_iters": None, "source": "missing"}
    rows = newest(_rows(d / "iter.jsonl"))
    return {"rows": {r["iter"]: r for r in rows}, "n_iters": max(r["iter"] for r in rows) + 1, "source": "full"}
  rows: dict[int, dict] = {}
  src = []
  for base in (CACHE_SUP, CACHE):
    d = base / jid
    if (d / "iter.jsonl").exists():
      full = newest(_rows(d / "iter.jsonl"))
      rows.update({r["iter"]: r for r in full}); src.append("full")
    if (d / "iter_full.jsonl").exists():
      full = newest(_rows(d / "iter_full.jsonl"))
      rows.update({r["iter"]: r for r in full}); src.append("full")
    if (d / "iter_head.jsonl").exists():
      rows.update({r["iter"]: r for r in _rows(d / "iter_head.jsonl", partial_last=True)}); src.append("head")
    if (d / "iter_tail.jsonl").exists():
      rows.update({r["iter"]: r for r in _rows(d / "iter_tail.jsonl", partial_first=True)}); src.append("tail")
  return {"rows": rows, "n_iters": (max(rows) + 1) if rows else None, "source": "+".join(src) or "missing"}


def eval_track(jid: str, robot: str) -> dict | None:
  """Last-5-eval mean of track_lin_frac / J_task (newest attempt), as report4.cell()."""
  paths = [RES / "merged_r4" / jid / "eval.jsonl", RES / "merged_r2" / jid / "eval.jsonl"]
  if jid.startswith("r6g1_"):
    paths = [CACHE_SUP / jid / "eval.jsonl", CACHE / jid / "eval.jsonl"]
  best = None
  for p in paths:  # two G1 caches may hold different snapshots of one job: keep the longest
    if p.exists():
      ev = newest(_rows(p))
      if ev and (best is None or ev[-1]["iter"] > best[-1]["iter"]):
        best = ev
  if best is None:
    return None
  ev = best
  v = [r["track_lin_frac"] for r in ev[-5:] if "track_lin_frac" in r]
  j = [r["J_task"] for r in ev[-5:] if "J_task" in r]
  return {"track": float(np.mean(v)), "J": float(np.mean(j)), "last_iter": ev[-1]["iter"], "kinds": Counter(r.get("kind") for r in ev)}


def window_mean(rows: dict[int, dict], key: str, lo: int, hi: int) -> float | None:
  v = [rows[i][key] for i in range(lo, hi + 1) if i in rows and key in rows[i]]
  need = hi - lo + 1
  return float(np.mean(v)) if len(v) == need else None  # only complete windows count


# ================================================================================================ cell registry
def seeds(lo: int, hi: int, tmpl: str) -> dict[int, str]:
  return {s: tmpl.format(s=s) for s in range(lo, hi + 1)}


def go1_cells() -> list[dict]:
  def two(t1: str, t2: str) -> dict[int, str]:
    return {**seeds(3000, 3009, t1), **seeds(3010, 3019, t2)}

  cells = []
  # c = 0 is E12 (3000-3009) + the round-5 top-up (3010-3019); c = 1 is E11 (night1); 0.1 / 0.3 are the round-4 sweep.
  spec = {
    0.0: (two("w2off0_go1_raw_none_g0.999_o0_s{s}", "r4off0_go1_g0.999_o0_none_s{s}"), two("w2off0_go1_raw_value_g0.999_o0_s{s}", "r4off0_go1_g0.999_o0_value_s{s}"), None),
    0.1: (seeds(3000, 3019, "r4sweep_go1_g0.999_o0.1_none_s{s}"), seeds(3000, 3019, "r4sweep_go1_g0.999_o0.1_value_s{s}"), seeds(3000, 3019, "r5bias01_go1_g0.999_o0.1_none_s{s}")),
    0.3: (seeds(3000, 3019, "r4sweep_go1_g0.999_o0.3_none_s{s}"), seeds(3000, 3019, "r4sweep_go1_g0.999_o0.3_value_s{s}"), None),
    1.0: (seeds(3000, 3019, "w2main_go1_raw_none_g0.999_o1_s{s}"), seeds(3000, 3019, "w2main_go1_raw_value_g0.999_o1_s{s}"), seeds(3000, 3019, "r5bias_go1_g0.999_o1_none_s{s}")),
  }
  for c, (unc, cen, bias) in spec.items():
    cells.append(dict(key=f"Go1 g.999 c={c:g}", robot="go1", gamma=0.999, c=c, penalty=10.0, kind="claim", unc=unc, cen=cen, bias=bias, iters=2000))
  cells.append(dict(key="Go1 g.99 c=0 (pilot)", robot="go1", gamma=0.99, c=0.0, penalty=10.0, kind="pilot",
                    unc=seeds(3000, 3002, "w2off0g99_go1_raw_none_g0.99_o0_s{s}"), cen=seeds(3000, 3002, "w2off0g99_go1_raw_value_g0.99_o0_s{s}"), bias=None, iters=2000))
  return cells


def g1_cells() -> list[dict]:
  man = json.loads((CACHE_SUP / "manifest.json").read_text())
  unc, cen, bias = {}, {}, {}
  for j in man["jobs"]:
    jid = j["job_id"]
    if not (3000 <= j["seed"] <= 3019):
      continue
    arm = "bias" if (j.get("extra") or {}).get("critic_bias_init") else j["center"]
    {"none": unc, "value": cen, "bias": bias}[arm][j["seed"]] = jid
  cells = [dict(key="G1 g.99 c=1", robot="g1", gamma=0.99, c=1.0, penalty=5.7, kind="claim", unc=unc, cen=cen, bias=bias, iters=2500)]
  cells.append(dict(key="G1 g.99 c=0 (pilot)", robot="g1", gamma=0.99, c=0.0, penalty=0.0, kind="pilot",
                    unc=seeds(1000, 1002, "fix_g1_raw_none_o0_s{s}"), cen=None, bias=None, iters=2000,
                    note="penalty 0, search seeds 1000-1002, no centered counterpart: S from the uncentered arm"))
  cells.append(dict(key="G1 g.999 c=0 (pilot)", robot="g1", gamma=0.999, c=0.0, penalty=5.7, kind="pilot",
                    unc=seeds(1000, 1002, "w2g1c999_57_g1_raw_none_g0.999_o0_s{s}"), cen=seeds(1000, 1002, "w2g1c999_57_g1_raw_value_g0.999_o0_s{s}"), bias=None, iters=2000,
                    note="search seeds 1000-1002"))
  return cells


# ================================================================================================ statistics
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
  if n == 0:
    return float("nan"), float("nan")
  p, d = k / n, 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - h, c + h


def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def loglik(X, y, b) -> float:
  p = np.clip(sigmoid(X @ b), 1e-12, 1 - 1e-12)
  return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def firth_logit(X: np.ndarray, y: np.ndarray, iters: int = 200) -> dict:
  """Firth-penalised logistic regression (Heinze & Schemper 2002). Walking is (quasi-)separated by construction here, so the
  ordinary MLE diverges; the Firth estimate is finite. Returns coefficients, SEs from the penalised information, the ordinary
  log-likelihood AT the Firth estimate, AIC from it, and McFadden pseudo-R2 against the intercept-only Firth fit."""
  n, k = X.shape
  b = np.zeros(k)
  for _ in range(iters):
    p = sigmoid(X @ b)
    W = p * (1 - p)
    I = X.T @ (X * W[:, None]) + 1e-12 * np.eye(k)
    Iinv = np.linalg.inv(I)
    h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], Iinv, X * np.sqrt(W)[:, None])
    U = X.T @ (y - p + h * (0.5 - p))
    step = Iinv @ U
    m = np.max(np.abs(step))
    if m > 5:
      step *= 5 / m
    b_new = b + step
    if np.max(np.abs(b_new - b)) < 1e-9:
      b = b_new
      break
    b = b_new
  p = sigmoid(X @ b)
  W = p * (1 - p)
  se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * W[:, None]) + 1e-12 * np.eye(k))))
  ll = loglik(X, y, b)
  return {"beta": b.tolist(), "se": se.tolist(), "ll": ll, "aic": 2 * k - 2 * ll, "k": k}


def plain_mle_diverges(X, y) -> bool:
  b = np.zeros(X.shape[1])
  for _ in range(60):
    p = sigmoid(X @ b)
    W = np.clip(p * (1 - p), 1e-10, None)
    try:
      step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-10 * np.eye(len(b)), X.T @ (y - p))
    except np.linalg.LinAlgError:
      return True
    b = b + step
  return bool(np.max(np.abs(b)) > 15)


def auc(score: np.ndarray, y: np.ndarray, higher_is_failure: bool = True) -> float:
  """P(a failing seed has a higher score than a walking seed), ties = 0.5."""
  s_fail, s_walk = score[y == 0], score[y == 1]
  if not len(s_fail) or not len(s_walk):
    return float("nan")
  cmp = s_fail[:, None] - s_walk[None, :]
  a = (cmp > 0).mean() + 0.5 * (cmp == 0).mean()
  return float(a if higher_is_failure else 1 - a)


def single_threshold_separable(x_by_cell: list[tuple[float, float]]) -> bool:
  """Cells as (x, walking rate): is there one threshold on x with every cell below it walking >= 50% and every cell above it < 50%
  (or the mirror image)? The descriptive version of 'one rule serves both robots'."""
  pts = sorted(x_by_cell)
  ok = [r >= 0.5 for _, r in pts]
  flips = sum(1 for a, b in zip(ok, ok[1:]) if a != b)
  return flips <= 1


# ================================================================================================ per-cell computation
def arm_series(jobs: dict[int, str] | None, robot: str, iters: int) -> dict[int, dict]:
  out = {}
  for s, jid in (jobs or {}).items():
    d = iter_rows(jid, robot)
    ev = eval_track(jid, robot)
    ok_ev = ev is not None and ev["last_iter"] >= iters - 50  # a finished run reaches its last eval
    if not d["rows"] or not ok_ev:
      continue
    late_lo = iters - LATE_N
    r = d["rows"]
    pen = None
    out[s] = dict(job=jid, source=d["source"], ev=ev,
                  seen=window_mean(r, "r_seen_rate", *EARLY), task=window_mean(r, "r_task_rate", *EARLY),
                  off=window_mean(r, "offset_now", *EARLY), fall=window_mean(r, "fall_rate_per_1k", *EARLY),
                  vstd_mid=window_mean(r, "v_std", *MID), vstd_late=window_mean(r, "v_std", late_lo, iters - 1))
  return out


def mean_of(series: dict[int, dict], key: str) -> tuple[float, int]:
  v = [x[key] for x in series.values() if x.get(key) is not None]
  return (float(np.mean(v)) if v else float("nan")), len(v)


def compute_cell(cell: dict) -> dict:
  g, c, P, iters = cell["gamma"], cell["c"], cell["penalty"], cell["iters"]
  unc = arm_series(cell["unc"], cell["robot"], iters)
  cen = arm_series(cell["cen"], cell["robot"], iters) if cell["cen"] else {}
  bias = arm_series(cell["bias"], cell["robot"], iters) if cell["bias"] else {}
  src = cen if cen else unc  # S source
  s_source = "centered arm" if cen else "UNCENTERED arm (no centered counterpart)"
  rbar, n_r = mean_of(unc, "seen")
  rtask, _ = mean_of(unc, "task")
  off, _ = mean_of(unc, "off")
  fall, _ = mean_of(unc, "fall")
  S_mid, n_smid = mean_of(src, "vstd_mid")
  S_late, n_slate = mean_of(src, "vstd_late")
  h = 1.0 - g
  L = abs(rbar) / h
  L_nopen = abs(rtask + off) / h
  walks = [int(x["ev"]["track"] >= WALK) for x in unc.values()]
  n, k = len(walks), int(sum(walks))
  lo, hi = wilson(k, n)
  out = dict(
    key=cell["key"], robot=cell["robot"], gamma=g, c=c, kind=cell["kind"], penalty=P, n=n, n_centered=len(cen), n_S=n_smid, n_Slate=n_slate,
    s_source=s_source, note=cell.get("note", ""), rbar=rbar, rbar_signed_negative=bool(rbar < 0), r_task=rtask, offset=off, fall_per_1k=fall,
    S_mid_sd=float(np.std([x["vstd_mid"] for x in src.values() if x["vstd_mid"] is not None], ddof=1)) if len(src) > 1 else float("nan"),
    S_late_sd=float(np.std([x["vstd_late"] for x in src.values() if x["vstd_late"] is not None], ddof=1)) if len(src) > 1 else float("nan"),
    L=L, L_nopen=L_nopen, S_mid=S_mid, S_late=S_late, kappa_mid=L / S_mid, kappa_late=L / S_late,
    kappa_nopen_mid=L_nopen / S_mid, kappa_nopen_late=L_nopen / S_late,
    walks=k, rate=k / n if n else float("nan"), wilson=[lo, hi],
    track_mean=float(np.mean([x["ev"]["track"] for x in unc.values()])) if unc else float("nan"),
    track_centered=float(np.mean([x["ev"]["track"] for x in cen.values()])) if cen else float("nan"),
    walks_centered=int(sum(x["ev"]["track"] >= WALK for x in cen.values())) if cen else None,
    seeds_walk={s: int(x["ev"]["track"] >= WALK) for s, x in unc.items()},
    seeds_L={s: abs(x["seen"]) / h for s, x in unc.items()}, seeds_L_nopen={s: abs(x["task"] + x["off"]) / h for s, x in unc.items()},
    sources=sorted({x["source"] for x in list(unc.values()) + list(cen.values()) + list(bias.values())}),
    eval_kinds=dict(sum((x["ev"]["kinds"] for x in unc.values()), Counter())),
  )
  if bias:
    rb, _ = mean_of(bias, "seen")
    rbt, _ = mean_of(bias, "task")
    L_eff = abs(rb - c) / h
    out["bias"] = dict(n=len(bias), rbar=rb, L_eff=L_eff, L_eff_nopen=abs(rbt) / h, kappa_mid=L_eff / S_mid, kappa_late=L_eff / S_late,
                       kappa_nopen_mid=abs(rbt) / h / S_mid, kappa_nopen_late=abs(rbt) / h / S_late,
                       walks=int(sum(x["ev"]["track"] >= WALK for x in bias.values())), track_mean=float(np.mean([x["ev"]["track"] for x in bias.values()])))
    out["bias"]["wilson"] = list(wilson(out["bias"]["walks"], len(bias)))
  return out


# ================================================================================================ regressions
VARIANTS = {
  # name: (x column, label)
  "logL": ("L", "log10 L"),
  "logK_mid": ("kappa_mid", "log10 kappa (S mid, it 100-300)"),
  "logK_late": ("kappa_late", "log10 kappa (S late, last 250)"),
  "logLnp": ("L_nopen", "log10 L, no fall penalty"),
  "logKnp_mid": ("kappa_nopen_mid", "log10 kappa, L no fall penalty, S mid"),
  "logKnp_late": ("kappa_nopen_late", "log10 kappa, L no fall penalty, S late"),
}


def regress(cells: list[dict]) -> dict:
  claim = [c for c in cells if c["kind"] == "claim"]
  rows = []
  for c in claim:
    for s, w in c["seeds_walk"].items():
      rows.append((c, s, w))
  y = np.array([w for _, _, w in rows], float)
  robot = np.array([1.0 if c["robot"] == "g1" else 0.0 for c, _, _ in rows])
  res = {"n_seeds": len(y), "n_cells": len(claim), "n_walk": int(y.sum()), "variants": {}}
  for name, (col, label) in VARIANTS.items():
    x = np.log10(np.array([c[col] for c, _, _ in rows]))
    xc = x - x.mean()
    one = np.ones_like(xc)
    Xa = np.column_stack([one, xc])
    Xb = np.column_stack([one, xc, robot])
    X0 = np.column_stack([one])
    f0, fa, fb = firth_logit(X0, y), firth_logit(Xa, y), firth_logit(Xb, y)
    cell_pts = [(float(np.log10(c[col])), c["rate"]) for c in claim]
    res["variants"][name] = dict(
      label=label, x_mean=float(x.mean()),
      null_ll=f0["ll"], alone=fa, with_robot=fb,
      mcfadden_alone=1 - fa["ll"] / f0["ll"], mcfadden_robot=1 - fb["ll"] / f0["ll"],
      d_ll_robot=fb["ll"] - fa["ll"], d_aic_robot=fb["aic"] - fa["aic"],
      lr_robot=2 * (fb["ll"] - fa["ll"]),
      auc_seed=auc(x, y, True), plain_mle_diverges_alone=plain_mle_diverges(Xa, y),
      single_threshold=single_threshold_separable(cell_pts),
      cell_x=[(c["key"], float(np.log10(c[col])), c["rate"]) for c in claim],
    )
  return res


def within_cell(cells: list[dict]) -> list[dict]:
  """Descriptive check inside cells whose outcome is mixed: do the seeds that fail have the higher per-seed early level |r_bar|/(1-gamma)?
  (S is a cell constant, so within a cell kappa and L order the seeds identically.) AUC 0.5 = no information."""
  out = []
  for c in cells:
    y = np.array([c["seeds_walk"][s] for s in sorted(c["seeds_walk"])])
    if 0 < y.sum() < len(y):
      L = np.array([c["seeds_L"][s] for s in sorted(c["seeds_walk"])])
      Ln = np.array([c["seeds_L_nopen"][s] for s in sorted(c["seeds_walk"])])
      out.append(dict(key=c["key"], kind=c["kind"], n=len(y), walks=int(y.sum()), auc_L=auc(L, y, True), auc_L_nopen=auc(Ln, y, True),
                      L_walk=float(L[y == 1].mean()), L_fail=float(L[y == 0].mean())))
  return out


# ================================================================================================ output tables
def fnum(x: float, nd: int = 3) -> str:
  if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
    return "n/a"
  if abs(x) >= 1000:
    return f"{x:,.0f}"
  if abs(x) >= 100:
    return f"{x:.0f}"
  if abs(x) >= 10:
    return f"{x:.1f}"
  return f"{x:.{nd}g}"


def cell_table(cells: list[dict]) -> str:
  hdr = ("| cell | robot | γ | c | n (unc / cen) | r̄ (it 0-100) | L | S mid (± SD over seeds) | S late (± SD) | κ mid | κ late | walks k/n [Wilson 95%] | mean tracking (unc) |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
  lines = []
  for c in cells:
    star = "" if c["s_source"].startswith("centered") else " †"
    lines.append(
      f"| {c['key']}{'' if c['kind'] == 'claim' else ' (pilot)' if 'pilot' not in c['key'] else ''} | {c['robot']} | {c['gamma']:g} | {c['c']:g} | {c['n']} / {c['n_centered']} | "
      f"{c['rbar']:+.4f} | {fnum(c['L'])} | {fnum(c['S_mid'])} ± {fnum(c['S_mid_sd'], 2)}{star} | {fnum(c['S_late'])} ± {fnum(c['S_late_sd'], 2)}{star} | {fnum(c['kappa_mid'])} | {fnum(c['kappa_late'])} | "
      f"{c['walks']}/{c['n']} [{c['wilson'][0]:.2f}, {c['wilson'][1]:.2f}] | {c['track_mean']:.3f} |")
  return hdr + "\n".join(lines)


def robust_table(cells: list[dict]) -> str:
  hdr = ("| cell | r_task + c | L (no penalty) | fall rate /1k (it 0-100) | κ mid, L no-penalty | κ late, L no-penalty |\n|---|---|---|---|---|---|\n")
  lines = [f"| {c['key']} | {c['r_task'] + c['offset']:+.4f} | {fnum(c['L_nopen'])} | {c['fall_per_1k']:.2f} | {fnum(c['kappa_nopen_mid'])} | {fnum(c['kappa_nopen_late'])} |" for c in cells]
  return hdr + "\n".join(lines)


def bias_table(cells: list[dict]) -> str:
  hdr = ("| cell | n | critic starts at | r̄ (bias arm, it 0-100) | L_eff = abs(r̄ − c)/(1−γ) | κ_eff mid | κ_eff late | walks k/n | mean tracking |\n|---|---|---|---|---|---|---|---|---|\n")
  lines = []
  for c in cells:
    b = c.get("bias")
    if b:
      lines.append(f"| {c['key']} bias-init | {b['n']} | c/(1−γ) = {c['c'] / (1 - c['gamma']):g} | {b['rbar']:+.4f} | {fnum(b['L_eff'])} | {fnum(b['kappa_mid'])} | {fnum(b['kappa_late'])} | "
                   f"{b['walks']}/{b['n']} | {b['track_mean']:.3f} |")
  return hdr + "\n".join(lines)


def within_table(rows: list[dict]) -> str:
  hdr = "| cell | walks | mean L, walking seeds | mean L, failing seeds | AUC (fail has higher L) | AUC, L without penalty |\n|---|---|---|---|---|---|\n"
  return hdr + "\n".join(f"| {r['key']} | {r['walks']}/{r['n']} | {fnum(r['L_walk'])} | {fnum(r['L_fail'])} | {r['auc_L']:.2f} | {r['auc_L_nopen']:.2f} |" for r in rows)


def regression_table(res: dict) -> str:
  hdr = ("| x | model | slope on x (per decade) | SE | robot (G1) coef | log-lik | AIC | McFadden R² | seed-level AUC | one threshold serves all 5 cells? |\n|---|---|---|---|---|---|---|---|---|---|\n")
  lines = []
  for name, v in res["variants"].items():
    a, b = v["alone"], v["with_robot"]
    lines.append(f"| {v['label']} | x alone | {a['beta'][1]:+.2f} | {a['se'][1]:.2f} | – | {a['ll']:.1f} | {a['aic']:.1f} | {v['mcfadden_alone']:.2f} | {v['auc_seed']:.2f} | {'yes' if v['single_threshold'] else 'no'} |")
    lines.append(f"| | x + robot | {b['beta'][1]:+.2f} | {b['se'][1]:.2f} | {b['beta'][2]:+.2f} (SE {b['se'][2]:.2f}) | {b['ll']:.1f} | {b['aic']:.1f} | {v['mcfadden_robot']:.2f} | | |")
  return hdr + "\n".join(lines)


# ================================================================================================ figures
def figures(cells: list[dict], res: dict) -> None:
  import matplotlib.pyplot as plt
  import make_supervisor_figs as SF

  SF.apply_style()
  FIG = {"go1": dict(marker="o", color=SF.ARM["none"]["color"], name="Go1"), "g1": dict(marker="s", color=SF.ARM["value"]["color"], name="G1")}
  BIAS_C = SF.ARM["bias"]["color"]
  OUT.mkdir(parents=True, exist_ok=True)

  def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.png", "+ .pdf")

  # per-cell label offsets (points, ha) to keep labels apart; default (6, 6)
  def short(c):
    return f"{c['robot'].capitalize()} γ{c['gamma']:g} c={c['c']:g}" + (" pilot" if c["kind"] == "pilot" else "")

  def panel(ax, xcol, xlab, bias_col=None, title="", labels=True):
    texts = []
    for c in cells:
      st = FIG[c["robot"]]
      hollow = c["kind"] == "pilot"
      x, y = c[xcol], c["rate"]
      ax.errorbar(x, y, yerr=[[y - c["wilson"][0]], [c["wilson"][1] - y]], fmt="none", ecolor=st["color"], alpha=0.45, lw=1.2, capsize=2.5)
      ax.plot(x, y, marker=st["marker"], ms=9, ls="none", mfc="white" if hollow else st["color"], mec=st["color"], mew=1.8, zorder=3)
      texts.append((x, y, short(c) + f"  {c['walks']}/{c['n']}", st["color"]))
      b = c.get("bias")
      if b and bias_col:
        xb, yb = b[bias_col], b["walks"] / b["n"]
        ax.plot(xb, yb, marker="*", ms=12, ls="none", mfc=BIAS_C, mec="#2b2b2b", mew=0.6, zorder=3, alpha=0.9)
        texts.append((xb, yb, f"{c['robot'].capitalize()} c={c['c']:g} bias-init", BIAS_C))
    # greedy vertical nudging of labels that sit close together in axis space
    placed = []
    for x, y, t, col in ([] if not labels else sorted(texts, key=lambda a: (np.log10(a[0]), a[1]))):
      dy = 0
      while any(abs(np.log10(x) - px) < 0.55 and abs((y + dy * 0.055) - py) < 0.05 for px, py in placed):
        dy += 1
      placed.append((np.log10(x), y + dy * 0.055))
      ax.annotate(t, (x, y), xytext=(7, 7 + 10 * dy), textcoords="offset points", fontsize=7.4, color=col, ha="left", zorder=2.5,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylim(-0.08, 1.14)
    ax.grid(alpha=0.25, which="both")
    ax.set_title(title, fontsize=9.5)

  # ---- figure 1: rate vs log L / log kappa
  fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
  panel(axes[0], "L", "level error L = |r̄| / (1−γ)", None, "walking rate vs L")
  panel(axes[1], "kappa_mid", "κ = L / S   (S = V_std of the centered arm, it 100-300)", "kappa_mid", "walking rate vs κ")
  axes[0].set_ylabel("uncentered walking rate (k/n, Wilson 95%)")
  hs = [plt.Line2D([], [], marker="o", ls="none", mfc=FIG["go1"]["color"], mec=FIG["go1"]["color"], ms=8, label="Go1 (claim, n=20)"),
        plt.Line2D([], [], marker="s", ls="none", mfc=FIG["g1"]["color"], mec=FIG["g1"]["color"], ms=8, label="G1 (claim)"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=8, mew=1.8, label="hollow = pilot (n=3), not in the fit"),
        plt.Line2D([], [], marker="*", ls="none", mfc=BIAS_C, mec="#2b2b2b", ms=11, label="bias-init arm (L_eff, κ_eff)")]
  axes[1].legend(handles=hs, loc="center left", fontsize=7.8)
  fig.suptitle("EXPLORATORY, POST HOC: does κ = L/S line up Go1 and G1 better than L? (corrected protocol, uncentered arm = outcome)", fontsize=10.2)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  save(fig, "fig1_rate_vs_L_and_kappa")

  # ---- figure 2: robustness (3 variants of kappa + L without penalty)
  variants = [("L", "L (primary)"), ("L_nopen", "L, fall penalty added back"), ("kappa_mid", "κ: S mid (primary)"), ("kappa_late", "κ: S late (last 250)"),
              ("kappa_nopen_mid", "κ: L no penalty, S mid"), ("kappa_nopen_late", "κ: L no penalty, S late")]
  fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), sharey=True)
  order = ["L", "kappa_mid", "kappa_late", "L_nopen", "kappa_nopen_mid", "kappa_nopen_late"]
  vmap = dict(variants)
  for ax, col in zip(axes.flat, order):
    bcol = {"L": None, "L_nopen": None, "kappa_mid": "kappa_mid", "kappa_late": "kappa_late", "kappa_nopen_mid": "kappa_nopen_mid", "kappa_nopen_late": "kappa_nopen_late"}[col]
    panel(ax, col, vmap[col], bcol, vmap[col], labels=False)
  for ax in axes[:, 0]:
    ax.set_ylabel("uncentered walking rate")
  fig.legend(handles=hs, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
  fig.suptitle("EXPLORATORY, POST HOC: robustness of the ordering (hollow = pilot, star = bias-init arm where defined)", fontsize=10.2)
  fig.tight_layout(rect=(0, 0.04, 1, 0.96))
  save(fig, "fig2_robustness")

  # ---- figure 3: the (S, L) plane; kappa is the slope of a line through the origin of the log-log plane
  fig, ax = plt.subplots(figsize=(7.4, 5.6))
  Ss = np.array([c["S_mid"] for c in cells])
  Ls = np.array([c["L"] for c in cells])
  xs = np.logspace(np.log10(Ss.min()) - 0.6, np.log10(Ss.max()) + 0.6, 50)
  for k_ in (10, 100, 1000):
    ax.plot(xs, k_ * xs, color="#999999", lw=0.8, ls=":")
    ax.text(xs[-1], k_ * xs[-1], f"κ={k_}", fontsize=7.5, color="#777777", ha="right", va="bottom")
  cmap = plt.get_cmap("RdYlGn")
  for c in cells:
    st = FIG[c["robot"]]
    ax.scatter(c["S_mid"], c["L"], s=150, marker=st["marker"], facecolor=cmap(c["rate"]) if c["kind"] == "claim" else "white", edgecolor=st["color"], linewidth=2.0, zorder=3)
    ax.annotate(f"{short(c)} {c['walks']}/{c['n']}", (c["S_mid"], c["L"]), xytext=(7, 6), textcoords="offset points", fontsize=7.4, color=st["color"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
  ax.set_xscale("log"); ax.set_yscale("log")
  ax.set_xlabel("S = V_std of the centered arm (it 100-300)")
  ax.set_ylabel("L = |r̄| / (1−γ)")
  ax.set_title("EXPLORATORY: cells in the (S, L) plane; fill = uncentered walking rate (green walks), hollow = pilot\nκ = L/S is the position across the dotted diagonals", fontsize=9)
  ax.grid(alpha=0.25, which="both")
  fig.tight_layout()
  save(fig, "fig3_S_vs_L_plane")


# ================================================================================================ main
def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--fetch", action="store_true", help="read-only R2 fetch of the missing G1 claim files (small files + ranged iter.jsonl)")
  a = ap.parse_args()
  if a.fetch:
    fetch_g1()
    return
  cells = [compute_cell(c) for c in go1_cells() + g1_cells()]
  order = {"claim": 0, "pilot": 1}
  cells.sort(key=lambda c: (order[c["kind"]], c["robot"] != "go1", c["gamma"] != 0.999, c["c"]))
  res = regress(cells)
  res["within"] = within_cell(cells)
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "results.json").write_text(json.dumps({"cells": cells, "regression": res}, indent=1, default=str))
  md = ["## Cell table (generated)\n", cell_table(cells), "\n† S taken from the uncentered arm (no centered counterpart exists).\n",
        "## Robustness inputs: L with the fall penalty added back (generated)\n", robust_table(cells), "\n",
        "## Bias-init reference arms (generated)\n", bias_table(cells), "\n",
        f"## Logistic regressions, claim cells only: {res['n_cells']} cells, {res['n_seeds']} seeds, {res['n_walk']} walk (generated)\n", regression_table(res), "\n",
        "## Within-cell check, mixed-outcome cells only (generated)\n", within_table(res["within"]), "\n"]
  (OUT / "tables.md").write_text("\n".join(md))
  print("\n".join(md))
  figures(cells, res)


if __name__ == "__main__":
  main()
