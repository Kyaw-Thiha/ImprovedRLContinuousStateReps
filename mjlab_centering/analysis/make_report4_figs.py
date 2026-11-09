"""Figures for REPORT_004. One figure per experiment, regenerated from the R2 mirrors.

    python analysis/make_report4_figs.py --out ../docs/mjlab_centering/figures/report4

Requires the campaigns to have been pulled once (`analysis/report4.py <cmd> --campaign ...`), then reads
results/merged_r4 (+ results/merged_r2 for the E11/E12 baselines) and the report4_*.json records.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report4 as R  # noqa: E402

C_NONE, C_VAL, C_CTRL = "#c0392b", "#1f77b4", "#7f7f7f"
W = R.MERGED


def _traj(jid: str, key: str = "track_lin_frac") -> tuple[np.ndarray, np.ndarray]:
  ev = R.evals(jid)
  if not ev:
    p2 = R.ROOT / "results" / "merged_r2" / jid / "eval.jsonl"
    ev = R.newest(p2) if p2.exists() else []
  return np.array([e["iter"] for e in ev]), np.array([e.get(key, np.nan) for e in ev])


def _itercol(jid: str, key: str) -> tuple[np.ndarray, np.ndarray]:
  p = W / jid / "iter.jsonl"
  if not p.exists():
    return np.array([]), np.array([])
  rows = R.newest(p)
  return np.array([r["iter"] for r in rows]), np.array([r.get(key, np.nan) for r in rows])


def fig_sweep(offsets=(0.0, 0.1, 0.3, 1.0), out: Path | None = None) -> None:
  man = json.loads((R.ROOT / "results" / "report4_sweep.json").read_text()) if (R.ROOT / "results" / "report4_sweep.json").exists() else None
  if man is None:
    print("[skip] sweep: run analysis/report4.py sweep first")
    return
  # sweep cells only: merged_r4 also holds shock mothers/branches (offset 0 too), which must not leak into c = 0
  jids = ([p.name for p in W.iterdir() if p.is_dir() and p.name.startswith("r4sweep_")] if W.exists() else []) + R.baseline_jids(R.ROOT / "results" / "merged_r2")
  per: dict[tuple[float, str], list[float]] = defaultdict(list)
  walks: dict[tuple[float, str], int] = defaultdict(int)
  for j in jids:
    o, arm = R.offset_of(j), ("value" if "_value_" in j else "none")
    if o is None or o not in offsets or "g0.999" not in j:
      continue
    c = R.cell(j)
    if c["track"] != c["track"]:
      continue
    per[(o, arm)].append(c["track"])
    walks[(o, arm)] += int(c["walks"])
  fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
  xs = [o for o in offsets]
  for arm, col in (("none", C_NONE), ("value", C_VAL)):
    m = [np.mean(per[(o, arm)]) if per[(o, arm)] else np.nan for o in xs]
    ax[0].plot(xs, m, "o-", color=col, label=arm, lw=1.8)
    for o in xs:
      v = per[(o, arm)]
      if v:
        ax[0].scatter(np.full(len(v), o) + np.random.default_rng(0).uniform(-0.012, 0.012, len(v)), v, s=16, color=col, alpha=0.7)
    w = [walks[(o, arm)] / max(len(per[(o, arm)]), 1) for o in xs]
    ax[1].plot(xs, w, "o-", color=col, label=arm, lw=1.8)
  ax[0].axhline(0.5, ls="--", color="grey", lw=1)
  ax[0].set_xlabel("constant reward offset c"); ax[0].set_ylabel("tracking (last-5 eval mean)")
  ax[0].set_xscale("symlog", linthresh=0.1); ax[0].legend(); ax[0].grid(alpha=0.25)
  ax[1].set_xlabel("constant reward offset c"); ax[1].set_ylabel("walking rate"); ax[1].set_ylim(-0.05, 1.05)
  ax[1].set_xscale("symlog", linthresh=0.1); ax[1].legend(); ax[1].grid(alpha=0.25)
  ax[0].set_title("Go1, gamma .999, corrected protocol: does offset size matter", fontsize=9.5)
  ax[1].set_title("walking rate (last-5 tracking >= 0.5)", fontsize=9.5)
  fig.tight_layout()
  (out / "r4_01_offset_sweep.png").parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r4_01_offset_sweep.png", dpi=140)
  print("wrote", out / "r4_01_offset_sweep.png")


def fig_shock(campaign: str, out: Path) -> None:
  rec = R.ROOT / "results" / "report4_shock.json"
  if not rec.exists():
    print("[skip] shock: run analysis/report4.py shock first")
    return
  meta = json.loads(R.r2.get_json(f"{campaign}/manifest.json") and json.dumps({"x": 1})) if False else None
  d = json.loads(rec.read_text())
  kinds = sorted({v["kind"] for v in d["branches"].values()} - {"ctrl"})
  fig, ax = plt.subplots(len(kinds), 2, figsize=(12, 3.1 * len(kinds)), squeeze=False)
  for i, k in enumerate(kinds):
    for arm, col in (("none", C_NONE), ("value", C_VAL)):
      js = [j for j, v in d["branches"].items() if v["kind"] == k and v["arm"] == arm]
      cs = [j for j, v in d["branches"].items() if v["kind"] == "ctrl" and v["arm"] == arm]
      for group, ls, lab in ((js, "-", f"{arm}, shocked"), (cs, "--", f"{arm}, unshocked control")):
        its, vals = [], []
        for j in group:
          it, v = _traj(j)
          m = it >= 1150
          its.append(it[m]); vals.append(v[m])
        if not vals:
          continue
        n = min(len(v) for v in vals)
        M = np.stack([v[:n] for v in vals])
        ax[i, 0].plot(its[0][:n], np.median(M, 0), ls, color=col, lw=1.6 if ls == "-" else 1.1, label=lab)
        ax[i, 0].fill_between(its[0][:n], np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=col, alpha=0.12 if ls == "-" else 0.05)
      rbars = []
      for j in js + cs:
        it, rb = _itercol(j, "r_bar")
        m = it >= 1100
        if m.any():
          rbars.append((f"{arm}{' shock' if j in js else ' ctrl'}", it[m], rb[m]))
      for lab, it, rb in rbars:
        ax[i, 1].plot(it, rb, color=C_NONE if lab.startswith("none") else C_VAL, lw=1.4, ls="-" if "shock" in lab else "--")
    ax[i, 0].axvline(1200, color="k", lw=0.8, ls=":")
    ax[i, 0].set_title(f"{k}: tracking around the shock at iteration 1200", fontsize=9.5)
    ax[i, 0].set_ylabel("tracking"); ax[i, 0].set_xlabel("iteration"); ax[i, 0].grid(alpha=0.25); ax[i, 0].legend(fontsize=7)
    ax[i, 1].axvline(1200, color="k", lw=0.8, ls=":")
    ax[i, 1].set_title(f"{k}: running average reward r_bar (centered arm)", fontsize=9.5)
    ax[i, 1].set_xlabel("iteration"); ax[i, 1].set_ylabel("r_bar"); ax[i, 1].grid(alpha=0.25)
  fig.suptitle("Step 3: proper shock test. Go1, offset 0 mothers, both arms walking; branches 1200 -> 1600", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  fig.savefig(out / "r4_03_shock.png", dpi=140)
  print("wrote", out / "r4_03_shock.png")


def fig_width(out: Path) -> None:
  rec = R.ROOT / "results" / "report4_width.json"
  if not rec.exists():
    print("[skip] width: run analysis/report4.py width first")
    return
  d = json.loads(rec.read_text())["arms"]
  order = [k for k in ("raw", "ssp_low2x2", "rff_low2x2", "ssp_low4x4", "rff_low4x4", "ssp_low8x8", "rff_low8x8") if k in d]
  fig, ax = plt.subplots(figsize=(9.5, 4.4))
  for i, k in enumerate(order):
    v = d[k]["track_per_seed"]
    ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.09, 0.09, len(v)), v, s=34,
               color=C_VAL if k.startswith("ssp") else (C_NONE if k == "raw" else C_CTRL), alpha=0.85, zorder=3)
    ax.hlines(d[k]["track"], i - 0.3, i + 0.3, color="k", lw=2, zorder=4)
  ax.axhline(d["raw"]["track"], color=C_NONE, ls="--", lw=1, alpha=0.6)
  ax.axhline(0.5, color="grey", ls=":", lw=1)
  ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=15)
  ax.set_ylabel("tracking (last-5 eval mean)")
  ax.set_title("Step 4: SSP width vs matched random Fourier vs raw input (3 search seeds, centered, offset +1)", fontsize=9.5)
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out / "r4_04_width.png", dpi=140)
  print("wrote", out / "r4_04_width.png")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="../docs/mjlab_centering/figures/report4")
  ap.add_argument("--which", default="all")
  ap.add_argument("--shock-campaign", default=None)
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  if a.which in ("all", "sweep"):
    fig_sweep(out=out)
  if a.which in ("all", "shock") and a.shock_campaign:
    fig_shock(a.shock_campaign, out)
  if a.which in ("all", "width"):
    fig_width(out)


if __name__ == "__main__":
  main()
