"""Readouts for REPORT_007 (d155 onward): VN, RS and SC against the existing uncentered and value cells.

    python analysis/r7_report.py stage1 --campaigns r7pilot-xxxxxxxx
    python analysis/r7_report.py claim  --campaigns r7vnsc-xxxxxxxx[,r7rs-yyyyyyyy]
    python analysis/r7_report.py claim  --campaigns ... --fig      # also the tracking-curve figure

Conventions are REPORT_003-006's: metric = last-5-eval mean of `track_lin_frac`; "walks" = that mean >= 0.5; paired
contrasts are per-seed differences with a 95% t-interval; fewer than 20 complete pairs is a pilot. The paired reference
cells are the existing Go1 cells on the same seeds: `w2main_go1_raw_{none,value}_g0.999_o1_s*` at c = 1 (night1) and
`r4sweep_go1_g0.999_o<off>_{none,value}_s*` at c = 0.1 (round-4 sweep + round-5 top-ups), read from the local mirrors.

The pre-registered reading (d157) is applied mechanically:
  fixes    - the arm is equivalent to value centering: paired dtrack CI inside +-0.03 AND dJ CI inside +-3% of the
             value-cell mean AND walking counts within 1;
  partial  - not equivalent, but the paired CI on dtrack against uncentered excludes 0;
  no help  - the paired CI on dtrack against uncentered includes 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report4 as R4  # noqa: E402
from report4 import MERGED, ROOT as MJROOT, WALK, cell, mean_last5, paired, pull, seed_of  # noqa: E402
from mjlab_centering import r2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "docs" / "mjlab_centering" / "figures" / "report7"
TOL_TRACK, TOL_J_REL, TOL_WALKS = 0.03, 0.03, 1
ARMS = ("none", "value", "vn", "vn_noclamp", "sc", "rs")
LABEL = {"none": "no centering", "value": "value centering", "vn": "VN: value normalization (rl_games)",
         "vn_noclamp": "VN, no +-5 clamp (diagnostic)", "sc": "SC: simple centering (raw reward)",
         "rs": "RS: reward / return std (SB3)"}
COLOR = {"none": "#c0392b", "value": "#1f77b4", "vn": "#2ca02c", "vn_noclamp": "#98df8a",
         "sc": "#9467bd", "rs": "#8c564b"}


def evals(jid: str) -> list[dict]:
  """Eval rows of one run, from either cache: the round-4/5/6 mirrors (merged_r4) or the REPORT_003 mirror (merged_r2),
  which is where the c = 1 reference cells live."""
  for base in (MERGED, MJROOT / "results" / "merged_r2"):
    p = base / jid / "eval.jsonl"
    if p.exists():
      rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
      if not rows:
        continue  # an empty mirror file is a missing run, not a run with no evals
      starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]
      return rows[starts[-1]:] if len(starts) > 1 else rows
  return []


def existing(arm: str, offset: float, seed: int) -> str:
  if offset == 1.0:
    return f"w2main_go1_raw_{arm}_g0.999_o1_s{seed}"
  return f"r4sweep_go1_g0.999_o{offset:g}_{arm}_s{seed}"


def new_jobs(campaigns: list[str]) -> dict[str, dict]:
  import tempfile
  jobs: dict[str, dict] = {}
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or r2.get_json(f"{c}/queue.json") or {"jobs": []}
    ids = [j["job_id"] for j in man["jobs"]]
    pull(c, ids)
    for jid in ids:  # the torch build of every cell, for d122/d161: cells are never mixed silently
      cfg = MERGED / jid / "config.json"
      if cfg.exists():
        continue
      with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "config.json"
        try:
          if r2.download_file(f"{c}/runs/{jid}/config.json", str(t)):
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_bytes(t.read_bytes())
        except Exception:
          pass
    for j in man["jobs"]:
      jobs[j["job_id"]] = j
  return jobs


def builds(jobs: dict[str, dict]) -> dict[str, int]:
  """torch build -> number of runs of this round that ran on it (read from each run's config.json)."""
  out: dict[str, int] = {}
  for jid in jobs:
    cfg = MERGED / jid / "config.json"
    try:
      b = f"torch {json.loads(cfg.read_text()).get('torch')}"
    except Exception:
      b = "build unknown (config.json not mirrored)"
    out[b] = out.get(b, 0) + 1
  return out


def arm_of(jid: str) -> str:
  for a in ("vn_noclamp", "vn", "sc", "rs"):
    if f"_go1_{a}_" in jid:
      return a
  return "?"


def offset_of(jid: str) -> float:
  return float(jid.split("_g0.999_o")[1].split("_")[0])


def collect(campaigns: list[str]) -> dict[tuple[float, str], dict[int, dict]]:
  """(offset, arm) -> seed -> {track, J, walks} for the new arms, plus the existing none/value cells on the same seeds."""
  jobs = new_jobs(campaigns)
  out: dict[tuple[float, str], dict[int, dict]] = {}
  for jid in jobs:
    arm, off = arm_of(jid), offset_of(jid)
    if arm == "?":
      continue
    c = cell(jid)
    if not c["n_evals"]:
      continue  # the manifest lists all 80 jobs; only finished runs are cells (a nan would poison every paired CI)
    out.setdefault((off, arm), {})[seed_of(jid)] = c
  for (off, arm) in list(out):
    seeds = set(out[(off, arm)])
    for ref in ("none", "value"):
      for s in seeds:
        c = cell(existing(ref, off, s))
        if c["n_evals"]:
          out.setdefault((off, ref), {})[s] = c
  return out


def summary(cells: dict[int, dict]) -> dict:
  tr = [c["track"] for c in cells.values() if c["n_evals"]]
  j = [c["J"] for c in cells.values() if c["n_evals"]]
  return {"n": len(tr), "track": float(np.mean(tr)) if tr else float("nan"),
          "J": float(np.mean(j)) if j else float("nan"),
          "walks": int(sum(1 for c in cells.values() if c["walks"]))}


def verdict(cells: dict[tuple[float, str], dict[int, dict]], off: float, arm: str) -> dict:
  a = {s: c["track"] for s, c in cells[(off, arm)].items()}
  ja = {s: c["J"] for s, c in cells[(off, arm)].items()}
  an = {s: c["track"] for s, c in cells[(off, "none")].items()}
  av = {s: c["track"] for s, c in cells[(off, "value")].items()}
  jn = {s: c["J"] for s, c in cells[(off, "none")].items()}
  jv = {s: c["J"] for s, c in cells[(off, "value")].items()}
  pn, pv, pjn, pjv = paired(a, an), paired(a, av), paired(ja, jn), paired(ja, jv)
  wn = summary(cells[(off, "none")])["walks"]
  wv = summary(cells[(off, "value")])["walks"]
  wa = summary(cells[(off, arm)])["walks"]
  n = summary(cells[(off, arm)])["n"]
  vmean = float(np.mean([c["J"] for c in cells[(off, "value")].values()]))
  parts = {"track": bool(pv["ci"][0] > -TOL_TRACK and pv["ci"][1] < TOL_TRACK),
           "J": bool(pjv["ci"][0] > -TOL_J_REL * abs(vmean) and pjv["ci"][1] < TOL_J_REL * abs(vmean)),
           "walks": bool(abs(wa - wv) <= TOL_WALKS)}
  equiv = all(parts.values())
  reading = ("fixes the offset failure (equivalent to value centering)" if equiv else
             "partial (beats uncentered, paired CI excludes 0, not equivalent to value centering)" if pn["ci"][0] > 0 else
             "doesn't help, and is worse (paired CI against uncentered excludes 0 below)" if pn["ci"][1] < 0 else
             "doesn't help (paired CI against uncentered includes 0)")
  return {"arm": arm, "offset": off, "n": n, "walks": wa, "walks_none": wn, "walks_value": wv, "equiv_parts": parts,
          "vs_none": pn, "vs_value": pv, "dJ_vs_none": pjn, "dJ_vs_value": pjv, "reading": reading,
          "pilot": n < 20}


def fmt(p: dict) -> str:
  if p["n"] < 2:
    return f"n={p['n']} (too few)"
  return f"{p['mean']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]"


def table(cells: dict, offsets: list[float], arms: list[str], title: str) -> list[str]:
  lines = [f"### {title}", "",
           "| arm | n | walking k/n | tracking | J_task | dtrack vs uncentered (95% CI) | dtrack vs value (95% CI) "
           "| dJ vs uncentered (95% CI) | dJ vs value (95% CI) | reading |",
           "|---|---|---|---|---|---|---|---|---|---|"]
  for off in offsets:
    for arm in arms:
      if (off, arm) not in cells:
        continue
      s = summary(cells[(off, arm)])
      v = verdict(cells, off, arm) if arm not in ("none", "value") else None
      lines.append(
        f"| {LABEL[arm]} @ c={off:g} | {s['n']} | {s['walks']}/{s['n']} | {s['track']:.3f} | {s['J']:.4f} | "
        f"{fmt(v['vs_none']) if v else '—'} | {fmt(v['vs_value']) if v else '—'} | "
        f"{fmt(v['dJ_vs_none']) if v else '—'} | {fmt(v['dJ_vs_value']) if v else '—'} | "
        f"{(v['reading'] + ' [' + ', '.join(k + ('=yes' if ok else '=no') for k, ok in v['equiv_parts'].items()) + ']') if v else '—'} |")
  return lines


DIAG_KEYS = ("r_task_rate", "r_bar", "r_bar_gap", "vn_mean", "vn_std", "vn_target_clamped", "rs_ret_std",
             "task_rew_std", "raw_rew_mean", "ev", "action_std")


def diagnostics(jid: str, arm: str) -> dict:
  """Last-iteration diagnostics of one run, so Stage 1 can check that r_bar / the normalizer stats behave as expected."""
  p = MERGED / jid / "iter.jsonl"
  if not p.exists():
    return {}
  rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]
  rows = rows[starts[-1]:] if len(starts) > 1 else rows
  if not rows:
    return {}
  last = rows[-1]
  out = {k: last.get(k) for k in DIAG_KEYS if k in last}
  if arm == "sc":
    out["r_bar_minus_task"] = (last.get("r_bar", float("nan")) - last.get("r_task_rate", float("nan")))
  return out


def diag_table(jobs: dict[str, dict], campaigns: list[str]) -> list[str]:
  lines = ["", "| seed | arm | tracking | J_task | walks | last-iteration diagnostics |", "|---|---|---|---|---|---|"]
  for jid in sorted(jobs, key=lambda j: (offset_of(j), seed_of(j), arm_of(j))):
    arm = arm_of(jid)
    if arm == "?":
      continue
    c = cell(jid)
    d = diagnostics(jid, arm)
    shown = {k: round(v, 4) for k, v in d.items() if isinstance(v, (int, float))}
    lines.append(f"| {seed_of(jid)} | {arm} | {c['track']:.3f} | {c['J']:.4f} | {'yes' if c['walks'] else 'no'} | "
                 f"{json.dumps(shown)} |")
  return lines


# ---------------------------------------------------------------------------------------------------- timing
def timing(jid: str) -> dict:
  """Take-off speed of one run: first eval at or above 0.5 tracking (t50) and at or above 90 % of that run's own final
  level (t90rel). t90rel exists only for runs that walk, as in REPORT_006/d149."""
  ev = evals(jid)
  if not ev:
    return {}
  its = [r["iter"] for r in ev]
  tr = [r["track_lin_frac"] for r in ev]
  final = float(np.mean(tr[-5:]))
  t50 = next((it for it, v in zip(its, tr) if v >= WALK), float("nan"))
  t90 = next((it for it, v in zip(its, tr) if v >= 0.9 * final), float("nan")) if final >= WALK else float("nan")
  return {"t50": t50, "t90rel": t90, "final": final}


def timing_table(jobs: dict[str, dict], offsets: list[float]) -> list[str]:
  rows, names = [], []
  seeds = sorted({seed_of(k) for k in jobs}) or list(range(3000, 3020))  # reference arms on the same seeds only
  for off in offsets:
    for arm in ("none", "value", "vn", "sc", "rs"):
      ids = [k for k in jobs if arm_of(k) == arm and offset_of(k) == off]
      if not ids and arm in ("none", "value"):
        ids = [existing(arm, off, s) for s in seeds if evals(existing(arm, off, s))]
      t = [timing(i) for i in ids]
      t = [x for x in t if x]
      if not t:
        continue
      t50 = np.array([x["t50"] for x in t], dtype=float)
      t90 = np.array([x["t90rel"] for x in t], dtype=float)
      med = lambda x: (f"{np.nanmedian(x[np.isfinite(x)]):.0f}" if np.isfinite(x).any() else "—")
      rows.append(f"| {LABEL[arm]} @ c={off:g} | {len(t)} | {med(t50)} | {int(np.isfinite(t50).sum())} | "
                  f"{med(t90)} | {int(np.isfinite(t90).sum())} |")
  if not rows:
    return []
  return ["", "**Take-off speed** (median over seeds; t50 = first eval at tracking >= 0.5, t90rel = first eval at 90 % of "
          "that run's own final level, only for runs that walk)", "",
          "| arm | n | t50 (median) | seeds with t50 | t90rel (median) | seeds with t90rel |", "|---|---|---|---|---|---|"] + rows


# ---------------------------------------------------------------------------------------------------- figure
def curves(jobs: dict[str, dict], arm: str, off: float, seeds=range(3000, 3020)) -> tuple[list[int], np.ndarray]:
  """Per-seed tracking curves for one arm at one offset. The two reference arms are read from the existing cells on the
  same seeds (d155 pairing), so the figure shows every arm on one axis."""
  ids = [k for k in jobs if arm_of(k) == arm and offset_of(k) == off]
  if not ids and arm in ("none", "value"):
    ids = [existing(arm, off, s) for s in seeds if evals(existing(arm, off, s))]
  per = []
  for jid in ids:
    ev = evals(jid)
    if ev:
      per.append({r["iter"]: r["track_lin_frac"] for r in ev})
  if not per:
    return [], np.empty((0, 0))
  its = sorted(set.intersection(*[set(p) for p in per]))
  return its, np.array([[p[i] for i in its] for p in per])


def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
  if len(x) < 3:
    return float("nan"), float("nan")
  rng = np.random.default_rng(seed)
  m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def figure(cells: dict, jobs: dict[str, dict], offsets: list[float], stem: str) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({"savefig.bbox": "tight", "axes.titlesize": 10.5, "axes.labelsize": 10,
                       "legend.fontsize": 8.0, "pdf.fonttype": 42, "figure.dpi": 110})
  arms = [a for a in ("none", "value", "vn", "sc", "rs")
          if a in ("none", "value") or any(arm_of(k) == a for k in jobs)]
  # the reference arms must be drawn on the *same seeds* as the new arms: a pilot (search seeds) has no paired
  # reference cells, and plotting the claim-seed reference curves over pilot curves would mix two seed blocks
  seeds = sorted({seed_of(k) for k in jobs}) or list(range(3000, 3020))
  fig, axes = plt.subplots(1, len(offsets), figsize=(12.0, 4.5), sharey=True)
  for ax, off in zip(np.atleast_1d(axes), offsets):
    xmax = 0
    for k, arm in enumerate(arms):
      its, Y = curves(jobs, arm, off, seeds=seeds)
      if len(its) == 0:
        continue
      xmax = max(xmax, int(its[-1]))
      mean = Y.mean(0)
      lo = np.array([boot_ci(Y[:, i])[0] for i in range(Y.shape[1])])
      hi = np.array([boot_ci(Y[:, i])[1] for i in range(Y.shape[1])])
      for row in Y:  # thin per-seed lines, as in the supervisor figures
        ax.plot(its, row, color=COLOR[arm], lw=0.5, alpha=0.18)
      ax.fill_between(its, lo, hi, color=COLOR[arm], alpha=0.16, lw=0)
      st = summary(cells[(off, arm)])
      ax.plot(its, mean, color=COLOR[arm], lw=1.8 if arm in ("none", "value") else 1.9,
              ls="--" if arm == "none" else ("-" if arm in ("value", "vn", "sc") else ":"),
              label=f"{LABEL[arm]}  n={st['n']}, walks {st['walks']}/{st['n']}, final {st['track']:.3f}")
    if not xmax:
      raise RuntimeError(f"no tracking curves for offset {off}: nothing to plot")
    ax.axhline(WALK, color="grey", ls=":", lw=1)
    ax.text(xmax, WALK + 0.012, "walks = 0.5", ha="right", fontsize=8, color="grey")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("PPO iteration")
    ax.set_title(f"offset c = {off:g}")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
  np.atleast_1d(axes)[0].set_ylabel("tracking (track_lin_frac)")
  fig.suptitle("Go1, gamma 0.999, corrected protocol: mean over seeds, 95% bootstrap CI, thin lines per seed", fontsize=10.5)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  OUT.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUT / f"{stem}.png", dpi=300)
  fig.savefig(OUT / f"{stem}.pdf")
  plt.close(fig)
  print("wrote", OUT / f"{stem}.png", "+ .pdf")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["stage1", "claim"])
  ap.add_argument("--campaigns", required=True)
  ap.add_argument("--offsets", default=None)
  ap.add_argument("--fig", action="store_true")
  ap.add_argument("--diag", action="store_true", help="also print the last-iteration diagnostics per run (Stage 1)")
  ap.add_argument("--timing", action="store_true", help="also print take-off speed (t50 / t90rel) per arm and offset")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  campaigns = [c for c in a.campaigns.split(",") if c]
  jobs = new_jobs(campaigns)
  cells = collect(campaigns)
  offsets = sorted({off for (off, _) in cells}) if a.offsets is None else [float(x) for x in a.offsets.split(",")]
  present = [arm for arm in ARMS if any((off, arm) in cells for off in offsets)]
  lines = table(cells, offsets, present, f"Round 7 {a.mode}: per arm and offset")
  print("\n".join(lines))
  if a.diag:
    print("\n".join(diag_table(jobs, campaigns)))
  if a.timing:
    print("\n".join(timing_table(jobs, offsets)))
  print("\nbuilds:", json.dumps(builds(jobs)))
  rec = {"mode": a.mode, "campaigns": campaigns, "offsets": offsets,
         "verdicts": [verdict(cells, o, arm) for o in offsets for arm in present if arm not in ("none", "value")],
         "summary": {f"{o:g}_{arm}": summary(cells[(o, arm)]) for o in offsets for arm in present}}
  if a.json:
    Path(a.json).write_text(json.dumps(rec, indent=1, default=float))
    print("wrote", a.json)
  if a.fig:
    figure(cells, jobs, offsets, f"r7_01_tracking_{a.mode}")


if __name__ == "__main__":
  main()
