"""Figures for REPORT_002, built from run files (no hand-entered numbers).

    python analysis/make_report_figs.py --merged results/merged_r2 --out ../docs/mjlab_centering/figures/report
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"raw": "#1f77b4", "ssp": "#d62728", "rff": "#7f7f7f", "none": "#888888", "value": "#2ca02c", "stand": "#bbbbbb"}


def evals(path):
  try:
    return sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["iter"])
  except FileNotFoundError:
    return []


def final(run_dir, metric="J_task", k=5):
  ev = evals(f"{run_dir}/eval.jsonl")
  return float(np.mean([e[metric] for e in ev[-k:]])) if ev else float("nan")


def boot(d, n=5000, seed=0):
  d = np.asarray(d, float)
  d = d[np.isfinite(d)]
  rng = np.random.default_rng(seed)
  m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
  return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def curve(run_dirs, metric):
  xs = None
  ys = []
  for d in run_dirs:
    ev = evals(f"{d}/eval.jsonl")
    if not ev:
      continue
    x = np.array([e["iter"] for e in ev])
    y = np.array([e[metric] for e in ev])
    xs = x if xs is None or len(x) < len(xs) else xs
    ys.append(y)
  n = min(len(y) for y in ys)
  Y = np.array([y[:n] for y in ys])
  return xs[:n], Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else np.zeros(n)


def fig_gate(out):
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
  for arm, col, lab in (("w_s*", C["raw"], "wrapped (continuing)"), ("e_s*", "#ff7f0e", "mjlab episodic")):
    dirs = sorted(glob.glob(f"results/gate/{arm}"))
    dirs = [d for d in dirs if Path(d).is_dir()]
    for metric, a, scale in (("track_lin_frac", ax[0], 128), ("J_task", ax[1], 1)):
      ys = []
      for d in dirs:
        ev = evals(f"{d}/eval.jsonl")
        x = [e["iter"] for e in ev]
        ys.append([e[metric] * scale for e in ev])
      n = min(map(len, ys))
      Y = np.array([y[:n] for y in ys])
      a.plot(x[:n], Y.mean(0), color=col, label=lab, lw=2)
      a.fill_between(x[:n], Y.min(0), Y.max(0), color=col, alpha=0.2)
  ax[0].axhline(0.283, color=C["stand"], ls="--", label="standing policy (0.283)")
  ax[1].axhline(0.0771, color=C["stand"], ls="--", label="standing policy (0.077)")
  ax[0].set_title("Velocity tracking (fraction of max)")
  ax[1].set_title("J_task (per-step task reward)")
  for a in ax:
    a.set_xlabel("PPO iteration")
    a.grid(alpha=0.3)
  ax[0].legend(fontsize=8)
  fig.suptitle("Gate: the continuing wrapper does not change learning (2 seeds each, band = min/max)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "01_gate.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_eta(out):
  manifest = {j["job_id"]: j for j in json.load(open("manifests/w1.json"))["jobs"]}
  manifest.update({j["job_id"]: j for f in ("manifests/w6.json", "manifests/w7.json") for j in json.load(open(f))["jobs"]})
  cells = collections.defaultdict(list)
  for jid, j in manifest.items():
    if not jid.startswith("eta_go1"):
      continue
    for base in ("results/w1", "results/w6", "results/w7", "results/from_r2/pod3/results/w6", "results/from_r2/pod3/results/w7"):
      d = f"{base}/{jid}"
      if glob.glob(d + "/final.json"):
        cells[(j["eta"], j["r_bar_init"])].append((final(d), final(d, "track_lin_frac")))
        break
  keys = sorted(cells, key=lambda k: (k[1] != "zero", k[0]))
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
  for a, idx, title in ((ax[0], 0, "J_task"), (ax[1], 1, "Velocity tracking")):
    for i, k in enumerate(keys):
      v = [x[idx] for x in cells[k]]
      col = C["value"] if k[1] == "first_batch" else C["raw"]
      a.bar(i, np.mean(v), color=col, alpha=0.8)
      a.scatter([i] * len(v), v, color="k", s=14, zorder=3)
    a.set_xticks(range(len(keys)))
    a.set_xticklabels([f"eta {k[0]:g}\n{'first-batch' if k[1] == 'first_batch' else 'zero'}" for k in keys], fontsize=8)
    a.set_title(title)
    a.grid(axis="y", alpha=0.3)
  ax[0].axhline(0.0771, color=C["stand"], ls="--")
  ax[1].axhline(0.283, color=C["stand"], ls="--")
  fig.suptitle("Centering step size eta (Go1, gamma .999, offset c, search seeds 1000-1002; dots = seeds, dashed = standing policy)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "02_eta.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_width(merged, out):
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
  for rep, col, lab in (("raw", C["raw"], "raw (48-wide)"), ("rff_low", C["rff"], "random Fourier, 1068-wide"),
                        ("ssp_low", C["ssp"], "HexSSP, 1068-wide")):
    dirs = sorted(glob.glob(f"{merged}/diagwc_go1_{rep}_s*"))
    dirs = [d for d in dirs if Path(d, "final.json").exists()]
    for metric, a in (("track_lin_frac", ax[0]), ("J_task", ax[1])):
      x, m, se = curve(dirs, metric)
      a.plot(x, m, color=col, lw=2, label=f"{lab} (n={len(dirs)})")
      a.fill_between(x, m - se, m + se, color=col, alpha=0.2)
  ax[0].axhline(0.283, color=C["stand"], ls="--")
  ax[1].axhline(0.0771, color=C["stand"], ls="--")
  ax[0].set_title("Velocity tracking")
  ax[1].set_title("J_task")
  ax[0].legend(fontsize=8)
  for a in ax:
    a.set_xlabel("PPO iteration")
    a.grid(alpha=0.3)
  fig.suptitle("Equal-width control (centered, gamma .99): HexSSP beats a random code of the same width", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "03_equal_width.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def cell_runs(merged, rep, center, gamma):
  pat = f"core_go1_{'ssp_low_' if rep == 'ssp' else 'raw_'}{center}_g{gamma}_s*"
  return {int(Path(d).name.split("_s")[-1]): d for d in glob.glob(f"{merged}/{pat}") if Path(d, "final.json").exists()}


def fig_factorial(merged, out):
  fig, ax = plt.subplots(1, 3, figsize=(14, 4))
  stats = {}
  for gi, gamma in enumerate((0.999, 0.99)):
    cells = {(r, c): cell_runs(merged, r, c, gamma) for r in ("raw", "ssp") for c in ("none", "value")}
    stats[gamma] = cells
  # panel 1: gamma .999 cell means with seed dots
  a = ax[0]
  x = 0
  ticks, labels = [], []
  for rep in ("raw", "ssp"):
    for center in ("none", "value"):
      runs = stats[0.999][(rep, center)]
      v = [final(d) for d in runs.values()]
      a.bar(x, np.mean(v), color=C[rep], alpha=0.35 if center == "none" else 0.9, edgecolor="k")
      a.scatter([x] * len(v), v, color="k", s=12, zorder=3)
      ticks.append(x)
      labels.append(f"{'raw' if rep == 'raw' else 'SSP'}\n{'no center' if center == 'none' else 'centered'}\n(n={len(v)})")
      x += 1
    x += 0.5
  a.set_xticks(ticks)
  a.set_xticklabels(labels, fontsize=8)
  a.axhline(0.0771, color=C["stand"], ls="--")
  a.set_title("J_task, gamma 0.999 (dashed = standing policy)")
  a.grid(axis="y", alpha=0.3)
  # panel 2: tracking
  a = ax[1]
  x = 0
  for rep in ("raw", "ssp"):
    for center in ("none", "value"):
      runs = stats[0.999][(rep, center)]
      v = [final(d, "track_lin_frac") for d in runs.values()]
      a.bar(x, np.mean(v), color=C[rep], alpha=0.35 if center == "none" else 0.9, edgecolor="k")
      a.scatter([x] * len(v), v, color="k", s=12, zorder=3)
      x += 1
    x += 0.5
  a.set_xticks(ticks)
  a.set_xticklabels(labels, fontsize=8)
  a.axhline(0.283, color=C["stand"], ls="--")
  a.set_title("Velocity tracking, gamma 0.999")
  a.grid(axis="y", alpha=0.3)
  # panel 3: contrasts with CI
  a = ax[2]
  rows = []
  for gamma in (0.999, 0.99):
    c = stats[gamma]
    def pair(rep, c1, c2):
      A, B = c[(rep, c1)], c[(rep, c2)]
      s = sorted(set(A) & set(B))
      return np.array([final(A[k]) - final(B[k]) for k in s])
    rn, rv, sn, sv = (c[("raw", "none")], c[("raw", "value")], c[("ssp", "none")], c[("ssp", "value")])
    s4 = sorted(set(rn) & set(rv) & set(sn) & set(sv))
    inter = np.array([(final(sv[k]) - final(rv[k])) - (final(sn[k]) - final(rn[k])) for k in s4])
    rows += [(f"centering | raw  (g {gamma:g})", boot(pair("raw", "value", "none"))),
             (f"centering | SSP  (g {gamma:g})", boot(pair("ssp", "value", "none"))),
             (f"interaction (g {gamma:g})", boot(inter))]
  for i, (lab, (m, lo, hi)) in enumerate(rows[::-1]):
    col = C["value"] if lab.startswith("centering") else C["ssp"]
    a.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3)
  a.set_yticks(range(len(rows)))
  a.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
  a.axvline(0, color="k", lw=0.8)
  a.set_title("Paired contrasts, 95% bootstrap CI")
  a.grid(axis="x", alpha=0.3)
  fig.suptitle("Wave 2 factorial: centering x input representation (Go1, offset c; raw gamma .99 cells have only 3 seeds)", y=1.03)
  fig.tight_layout()
  fig.savefig(out / "04_factorial.png", dpi=140, bbox_inches="tight")
  plt.close(fig)
  return rows


def fig_curves(merged, out):
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
  for rep, center, col, ls, lab in (("raw", "value", C["raw"], "-", "raw, centered"), ("raw", "none", C["raw"], ":", "raw, no centering"),
                                    ("ssp", "value", C["ssp"], "-", "SSP, centered"), ("ssp", "none", C["ssp"], ":", "SSP, no centering")):
    dirs = list(cell_runs(merged, rep, center, 0.999).values())
    for metric, a in (("track_lin_frac", ax[0]), ("J_task", ax[1])):
      x, m, se = curve(dirs, metric)
      a.plot(x, m, color=col, ls=ls, lw=2, label=f"{lab} (n={len(dirs)})")
      a.fill_between(x, m - se, m + se, color=col, alpha=0.12)
  ax[0].axhline(0.283, color=C["stand"], ls="--")
  ax[1].axhline(0.0771, color=C["stand"], ls="--")
  ax[0].set_title("Velocity tracking")
  ax[1].set_title("J_task")
  ax[0].legend(fontsize=8)
  for a in ax:
    a.set_xlabel("PPO iteration")
    a.grid(alpha=0.3)
  fig.suptitle("Learning curves, gamma 0.999, offset c (mean +- s.e.m. over 10 seeds)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "05_curves.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_rivals(merged, out):
  arms = [("plain uncentered", [final(d) for d in cell_runs(merged, "raw", "none", 0.999).values()],
           [final(d, "track_lin_frac") for d in cell_runs(merged, "raw", "none", 0.999).values()], C["none"])]
  for name, pat, col in (("value clip OFF", "rival_clipoff_s*", "#ff7f0e"), ("return normalisation", "rival_retnorm_s*", "#9467bd")):
    ds = [d for d in glob.glob(f"{merged}/{pat}") if Path(d, "final.json").exists()]
    arms.append((name, [final(d) for d in ds], [final(d, "track_lin_frac") for d in ds], col))
  arms.append(("value centering", [final(d) for d in cell_runs(merged, "raw", "value", 0.999).values()],
               [final(d, "track_lin_frac") for d in cell_runs(merged, "raw", "value", 0.999).values()], C["value"]))
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
  for a, idx, title, base in ((ax[0], 1, "J_task", 0.0771), (ax[1], 2, "Velocity tracking", 0.283)):
    for i, arm in enumerate(arms):
      v = arm[idx]
      a.bar(i, np.mean(v), color=arm[3], alpha=0.85, edgecolor="k")
      a.scatter([i] * len(v), v, color="k", s=12, zorder=3)
    a.axhline(base, color=C["stand"], ls="--")
    a.set_xticks(range(len(arms)))
    a.set_xticklabels([f"{x[0]}\n(n={len(x[1])})" for x in arms], fontsize=8)
    a.set_title(title)
    a.grid(axis="y", alpha=0.3)
  fig.suptitle("Critic-fix rivals (gamma .999, offset c): clipping off recovers reward, not behaviour", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "06_rivals.png", dpi=140, bbox_inches="tight")
  plt.close(fig)
  return arms


def fig_shock(merged, out):
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
  for a, center, title in ((ax[0], "none", "Uncentered"), (ax[1], "value", "Centered")):
    for kind, col, lab in (("none", "#2ca02c", "resumed, no shock"), ("offset", "#d62728", "+4 offset at it 600")):
      xs, ys = None, []
      for s in range(2000, 2010):
        mother = f"{merged}/core_go1_raw_{center}_g0.999_s{s}"
        br = f"{merged}/shock_{kind}_{center}_s{s}"
        if not (Path(mother, "final.json").exists() and Path(br, "final.json").exists()):
          continue
        em = [e for e in evals(f"{mother}/eval.jsonl") if e["iter"] < 600]
        eb = [e for e in evals(f"{br}/eval.jsonl") if e["iter"] >= 600]
        full = em + eb
        ys.append([e["J_task"] for e in full])
        xs = [e["iter"] for e in full]
      n = min(map(len, ys))
      Y = np.array([y[:n] for y in ys])
      a.plot(xs[:n], Y.mean(0), color=col, lw=2, label=f"{lab} (n={len(ys)})")
      a.fill_between(xs[:n], Y.mean(0) - Y.std(0, ddof=1) / np.sqrt(len(Y)), Y.mean(0) + Y.std(0, ddof=1) / np.sqrt(len(Y)),
                     color=col, alpha=0.2)
    a.axvline(600, color="k", ls=":")
    a.axhline(0.0771, color=C["stand"], ls="--")
    a.set_title(title)
    a.set_xlabel("PPO iteration")
    a.grid(alpha=0.3)
  ax[0].legend(fontsize=8)
  ax[0].set_ylabel("J_task")
  fig.suptitle("Reward-redesign shock (branched from the same checkpoint; mean +- s.e.m., 10 seeds)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "07_shock.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_linear(out):
  rows = [json.loads(l) for l in open("../linear_slice/results/td_pulled.jsonl")]
  feats = ["raw", "raw_bias", "tile", "hexssp", "rff"]
  gammas = [0.9, 0.99, 0.999]
  fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
  for a, center, title in ((ax[0], "none", "No centering"), (ax[1], "value", "Value centering")):
    M = np.zeros((len(feats), len(gammas)))
    for i, f in enumerate(feats):
      for j, g in enumerate(gammas):
        v = [r["state_part"] for r in rows if r["feat"] == f and r["gamma"] == g and r["offset"] == 1.0 and r["center"] == center]
        M[i, j] = np.mean(v)
    im = a.imshow(np.log10(M + 1e-4), cmap="viridis_r", aspect="auto")
    a.set_xticks(range(3))
    a.set_xticklabels([f"gamma {g:g}" for g in gammas])
    a.set_yticks(range(len(feats)))
    a.set_yticklabels(feats)
    for i in range(len(feats)):
      for j in range(3):
        a.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", color="w", fontsize=8)
    a.set_title(title)
  fig.suptitle("Linear TD(0), offset c: decision-relevant (state-dependent) value error, log colour scale (lower = better)", y=1.03)
  fig.tight_layout()
  fig.savefig(out / "08_linear_td.png", dpi=140, bbox_inches="tight")
  plt.close(fig)

  ac = [json.loads(l) for l in open("../linear_slice/results/ac_fw.jsonl")]
  fig, ax = plt.subplots(figsize=(8, 3.6))
  w = 0.38
  for j, g in enumerate(gammas):
    pass
  labs, vals = [], {"none": [], "value": []}
  for f in feats:
    for g in (0.99,):
      for c in ("none", "value"):
        v = [r["reward_last"] for r in ac if r["feat"] == f and r["gamma"] == g and r["offset"] == 1.0 and r["center"] == c]
        vals[c].append(np.mean(v))
    labs.append(f)
  x = np.arange(len(feats))
  ax.bar(x - w / 2, vals["none"], w, color=C["none"], label="no centering")
  ax.bar(x + w / 2, vals["value"], w, color=C["value"], label="centered")
  ax.set_xticks(x)
  ax.set_xticklabels(labs)
  ax.axhline(0, color="k", lw=0.8)
  ax.set_ylabel("task return per step")
  ax.set_title("Linear actor-critic control, gamma .99, offset c (10 seeds): a bias unit does not rescue the controller")
  ax.legend()
  ax.grid(axis="y", alpha=0.3)
  fig.tight_layout()
  fig.savefig(out / "09_linear_control.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_g1(out):
  fig, ax = plt.subplots(figsize=(8, 3.6))
  for d, lab, col in (("results/from_r2/pod3/results/w8/g1len_raw_none_g0.99_long_s1000", "G1 reference, 5000 iterations", C["raw"]),):
    ev = evals(d + "/eval.jsonl")
    ax.plot([e["iter"] for e in ev], [e["track_lin_frac"] for e in ev], color=col, label=lab)
  ax.axhline(0.267, color=C["stand"], ls="--", label="G1 standing policy (0.267)")
  ax.axhline(0.89, color=C["value"], ls=":", label="Go1 trained level (0.89)")
  ax.set_xlabel("PPO iteration")
  ax.set_ylabel("velocity tracking (fraction of max)")
  ax.set_ylim(0, 1)
  ax.set_title("G1 never leaves its standing baseline at 6x the Go1 run length")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig(out / "10_g1.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--merged", default="results/merged_r2")
  ap.add_argument("--out", default="../docs/mjlab_centering/figures/report")
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  fig_gate(out)
  fig_eta(out)
  fig_width(a.merged, out)
  rows = fig_factorial(a.merged, out)
  fig_curves(a.merged, out)
  arms = fig_rivals(a.merged, out)
  fig_shock(a.merged, out)
  fig_linear(out)
  fig_g1(out)
  print("contrasts:")
  for lab, (m, lo, hi) in rows:
    print(f"  {lab:32s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
  print("rivals:")
  for n, j, t, _ in arms:
    print(f"  {n:24s} n={len(j):2d} J {np.mean(j):.4f} track {np.mean(t):.3f}")


if __name__ == "__main__":
  main()
