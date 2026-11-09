"""Figures for REPORT_005 (round 5). Regenerated from the R2 mirrors after the report5.py commands have pulled the runs.

    python analysis/make_report5_figs.py --out ../docs/mjlab_centering/figures/report5

Each figure is skipped (with a printed note) when its record or runs are absent, so the script is safe to run at any
point of the round.
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
import mechanism_e11 as MECH  # noqa: E402
import report4 as R  # noqa: E402

C_NONE, C_VAL, C_CTRL, C_BIAS = "#c0392b", "#1f77b4", "#7f7f7f", "#2ca02c"
W = R.MERGED


def _traj(jid: str, key: str = "track_lin_frac") -> tuple[np.ndarray, np.ndarray]:
  ev = R.evals(jid)
  if not ev:
    p2 = R.ROOT / "results" / "merged_r2" / jid / "eval.jsonl"
    ev = R.newest(p2) if p2.exists() else []
  return np.array([e["iter"] for e in ev]), np.array([e.get(key, np.nan) for e in ev])


def fig_bias(out: Path) -> None:
  rec = R.ROOT / "results" / "report5_bias.json"
  if not rec.exists():
    print("[skip] bias: run analysis/report5.py bias first")
    return
  d = json.loads(rec.read_text())
  fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
  for c in d["cells"]:
    it, v = _traj(c["job_id"])
    if len(v):
      ax[0].plot(it, v, color=C_BIAS, lw=1.3, alpha=0.8,
                 label=f"bias={c['job_id'].rsplit('_s', 1)[1]} ({c['track']:.2f})")
  # E11 uncentered band on the same seeds for reference
  import glob
  ref = [p for s in [int(c["job_id"].rsplit("_s", 1)[1]) for c in d["cells"]]
         for p in glob.glob(str(R.ROOT / "results" / "merged_r2" / f"w2main_go1_raw_none_g0.999_o1_s{s}"))]
  if ref:
    T = np.stack([np.interp(np.arange(0, 2000, 25), *_traj(Path(p).name)) for p in ref])
    x = np.arange(0, 2000, 25)
    ax[0].plot(x, T.mean(0), color=C_NONE, lw=1.6, ls="--", label="E11 uncentered (same seeds)")
    ax[0].fill_between(x, np.percentile(T, 10, 0), np.percentile(T, 90, 0), color=C_NONE, alpha=0.12)
  ax[0].axhline(0.5, color="grey", ls=":", lw=1)
  ax[0].set_xlabel("iteration"); ax[0].set_ylabel("tracking"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25)
  ax[0].set_title("(a) bias-initialised critic, tracking by seed", fontsize=9.5)

  # critic windows: value/target and ev_state, bias arm vs E11 plain vs clip-off, as in the E11 mechanism figure
  sets = {"bias": [R.ROOT / "results" / "merged_r4" / c["job_id"] for c in d["cells"]]}
  sets["plain (E11)"] = [Path(p) for p in ref]
  clip = sorted(glob.glob(str(R.ROOT / "results" / "merged_r4" / "clip1_go1_raw_none_clipoff_g0.999_o1_s*")))
  if clip:
    sets["clip-off"] = [Path(p) for p in clip]
  cols = {"bias": C_BIAS, "plain (E11)": C_NONE, "clip-off": "#e67e22"}
  for name, dirs in sets.items():
    if not dirs:
      continue
    S = [MECH.series(MECH.load(str(p / "iter.jsonl"))) for p in dirs if (p / "iter.jsonl").exists()]
    if not S:
      continue
    for key, a_ in (("ratio", ax[1]), ("ev_state", ax[2])):
      M = np.stack([MECH.roll(s[key]) for s in S])
      it = S[0]["it"]
      a_.plot(it, np.median(M, 0), color=cols[name], lw=1.7, label=name)
      a_.fill_between(it, np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=cols[name], alpha=0.15)
  ax[1].axhline(1.0, color="grey", ls="--", lw=1); ax[1].axhline(0.93, color="grey", ls=":", lw=1)
  ax[1].set_ylim(-0.05, 1.15); ax[1].set_ylabel("V / (r_seen / (1-gamma))")
  ax[1].set_title("(b) critic level vs analytic target", fontsize=9.5)
  ax[2].axhline(0.0, color="grey", ls="--", lw=1); ax[2].set_ylim(-0.3, 1.02)
  ax[2].set_ylabel("ev_state"); ax[2].set_title("(c) state-explained share of the advantage", fontsize=9.5)
  for a_ in ax[1:]:
    a_.set_xlabel("iteration"); a_.grid(alpha=0.25); a_.legend(fontsize=8)
  fig.suptitle(f"Step 0: critic output bias initialised to c/(1-gamma) = 1000 (Go1, offset +1, gamma .999, n={d['n']}). "
               f"d114 prediction: 0 of 3 walk. Result: {d['walks']}/{d['n']}.", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  fig.savefig(out / "r5_01_bias.png", dpi=140)
  print("wrote", out / "r5_01_bias.png")


def fig_sweep(out: Path, offsets=(0.0, 0.1, 0.3, 1.0)) -> None:
  jids = ([p.name for p in W.iterdir() if p.is_dir() and (p.name.startswith("r4sweep_") or p.name.startswith("r5sweep")
           or p.name.startswith("r4off0_"))] if W.exists() else []) + R.baseline_jids(R.ROOT / "results" / "merged_r2")
  if not jids:
    print("[skip] sweep: no sweep runs pulled")
    return
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
  rng = np.random.default_rng(0)
  for arm, col in (("none", C_NONE), ("value", C_VAL)):
    ax[0].plot(offsets, [np.mean(per[(o, arm)]) if per[(o, arm)] else np.nan for o in offsets], "o-", color=col,
               label=arm, lw=1.8)
    for o in offsets:
      v = per[(o, arm)]
      if v:
        ax[0].scatter(np.full(len(v), o) + rng.uniform(-0.012, 0.012, len(v)), v, s=14, color=col, alpha=0.55)
    w = [walks[(o, arm)] / max(len(per[(o, arm)]), 1) for o in offsets]
    ax[1].plot(offsets, w, "o-", color=col, label=arm, lw=1.8)
    for o, y in zip(offsets, w):
      n = len(per[(o, arm)])
      ax[1].annotate(f"{int(round(y * n))}/{n}", (o, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
  ax[0].axhline(0.5, ls="--", color="grey", lw=1)
  for a_ in ax:
    a_.set_xscale("symlog", linthresh=0.1); a_.set_xlabel("constant reward offset c"); a_.grid(alpha=0.25); a_.legend()
  ax[0].set_ylabel("tracking (last-5 eval mean)")
  ax[1].set_ylabel("walking rate"); ax[1].set_ylim(-0.05, 1.08)
  ns = sorted({len(per[(o, a)]) for o in offsets for a in ("none", "value") if per[(o, a)]})
  ax[0].set_title(f"Go1, gamma .999, corrected protocol: offset size ({ns[0]}-{ns[-1]} seeds per cell; 20 = claim grade)", fontsize=9.5)
  ax[1].set_title("walking rate, n per cell annotated", fontsize=9.5)
  fig.tight_layout()
  fig.savefig(out / "r5_02_offset_sweep.png", dpi=140)
  print("wrote", out / "r5_02_offset_sweep.png")


def fig_shock(out: Path) -> None:
  rec = R.ROOT / "results" / "report5_shock.json"
  pooled = R.ROOT / "results" / "report4_shock.json"
  if not rec.exists():
    print("[skip] shock: run analysis/report5.py shock first")
    return
  rows = json.loads(rec.read_text())["branches"]
  if pooled.exists():
    rows = {**json.loads(pooled.read_text())["branches"], **rows}
  kinds = ["off+0.1", "off+1.0", "actx5"]
  fig, ax = plt.subplots(len(kinds), 1, figsize=(10.5, 3.0 * len(kinds)))
  for i, k in enumerate(kinds):
    for arm, col in (("none", C_NONE), ("value", C_VAL)):
      js = [j for j, v in rows.items() if v["kind"] == k and v["arm"] == arm]
      cs = [j for j, v in rows.items() if v["kind"] == "ctrl" and v["arm"] == arm]
      for group, ls, lab, alpha in ((js, "-", f"{arm}, shocked", 0.25), (cs, "--", f"{arm}, unshocked control", 0.10)):
        its, vals = [], []
        for j in group:
          it, v = _traj(j)
          m = it >= 1150
          its.append(it[m]); vals.append(v[m])
        if not vals:
          continue
        n = min(len(v) for v in vals)
        M = np.stack([v[:n] for v in vals])
        ax[i].plot(its[0][:n], np.median(M, 0), ls, color=col, lw=1.7 if ls == "-" else 1.1, label=lab)
        ax[i].fill_between(its[0][:n], np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=col, alpha=alpha)
    ax[i].axvline(1200, color="k", lw=0.8, ls=":")
    ax[i].set_title(f"{k}: pooled 20-seed mothers, branches 1200 -> 1600", fontsize=9.5)
    ax[i].set_ylabel("tracking"); ax[i].set_xlabel("iteration"); ax[i].grid(alpha=0.25); ax[i].legend(fontsize=7)
  fig.suptitle("Step 2: shock test at 20 seeds (rounds 4+5 pooled; identical branch settings and walking rule)", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(out / "r5_03_shock.png", dpi=140)
  print("wrote", out / "r5_03_shock.png")


def fig_width(out: Path) -> None:
  rec = R.ROOT / "results" / "report5_width.json"
  if not rec.exists():
    print("[skip] width: run analysis/report5.py width first")
    return
  d = json.loads(rec.read_text())["arms"]
  order = [k for k in ("raw", "ssp_low2x2", "rff_low2x2", "ssp_low4x4", "rff_low4x4", "ssp_low8x8", "rff_low8x8") if k in d]
  fig, ax = plt.subplots(figsize=(9.5, 4.4))
  for i, k in enumerate(order):
    v = d[k]["track_per_seed"]
    ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.09, 0.09, len(v)), v, s=34,
               color=C_VAL if k.startswith("ssp") else (C_NONE if k == "raw" else C_CTRL), alpha=0.85, zorder=3)
    ax.hlines(d[k]["track"], i - 0.3, i + 0.3, color="k", lw=2, zorder=4)
  if "raw" in d:
    ax.axhline(d["raw"]["track"], color=C_NONE, ls="--", lw=1, alpha=0.6)
  ax.axhline(0.5, color="grey", ls=":", lw=1)
  ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=15)
  ax.set_ylabel("tracking (last-5 eval mean)")
  ax.set_title("Step 4: SSP width vs matched random Fourier vs raw (search seeds, centered, offset +1). Pilot.", fontsize=9.5)
  ax.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out / "r5_04_width.png", dpi=140)
  print("wrote", out / "r5_04_width.png")


def fig_sspc(out: Path) -> None:
  rec = R.ROOT / "results" / "report5_sspc.json"
  if not rec.exists():
    print("[skip] sspc: run analysis/report5.py sspc first")
    return
  d = json.loads(rec.read_text())
  fig, ax = plt.subplots(figsize=(9, 4.2))
  arms = [a for a in ("none", "value") if a in d.get("per_seed", {})]
  xs = []
  for i, arm in enumerate(arms):
    ssp = d["per_seed"][arm]
    raw = (d.get("vs_ref", {}).get(arm, {}) or {}).get("ref_per_seed", {})
    ax.scatter(np.full(len(ssp), 2 * i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(ssp)),
               list(ssp.values()), s=30, color=C_VAL, label="SSP (best config)" if i == 0 else None, zorder=3)
    if raw:
      ax.scatter(np.full(len(raw), 2 * i + 1) + np.random.default_rng(1).uniform(-0.08, 0.08, len(raw)),
                 list(raw.values()), s=30, color=C_CTRL, label="raw input" if i == 0 else None, zorder=3)
      for s_, v in ssp.items():
        if s_ in raw:
          ax.plot([2 * i, 2 * i + 1], [v, raw[s_]], color="k", lw=0.5, alpha=0.4)
    xs += [f"{arm}\nSSP", f"{arm}\nraw"]
  ax.axhline(0.5, color="grey", ls=":", lw=1)
  ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8)
  ax.set_ylabel("tracking (last-5 eval mean)")
  ax.set_title("Step 5: best SSP config vs raw, {none, value} centering, 20 seeds (paired lines = same seed)", fontsize=9.5)
  ax.grid(alpha=0.25); ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(out / "r5_05_sspc.png", dpi=140)
  print("wrote", out / "r5_05_sspc.png")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="../docs/mjlab_centering/figures/report5")
  ap.add_argument("--which", default="all")
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  if a.which in ("all", "bias"):
    fig_bias(out)
  if a.which in ("all", "sweep"):
    fig_sweep(out)
  if a.which in ("all", "shock"):
    fig_shock(out)
  if a.which in ("all", "width"):
    fig_width(out)
  if a.which in ("all", "sspc"):
    fig_sspc(out)


if __name__ == "__main__":
  main()
