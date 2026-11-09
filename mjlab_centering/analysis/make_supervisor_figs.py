"""Supervisor figures for the reward-centering study (plotting and analysis only; nothing here launches compute).

    python analysis/make_supervisor_figs.py                 # everything, from the local caches
    python analysis/make_supervisor_figs.py --pull-g1       # first refresh the G1 claim runs from R2 (read-only)
    python analysis/make_supervisor_figs.py --only fig6     # one figure (fig1a fig1b fig2 fig3 fig4 fig5 fig6 tables)

Outputs (docs/figures/supervisor/): fig*.png (300 dpi) + fig*.pdf, summary_table.md/.csv, captions.md, NOTES.md.

Rules this script follows (user brief, 2026-11-05):
  * corrected protocol only for Go1/G1 (time-outs kept, falls reset in place and hidden from PPO); claim seeds 3000-3019;
  * primary metric `track_lin_frac`; "walks" = mean of the last 5 evals >= 0.5; summary numbers are last-5-eval means,
    paired by seed where a contrast is shown;
  * learning curves = mean over seeds with a 95% bootstrap CI over seeds at every eval point (`plot_eval_v3.bootstrap_ci`);
  * R2 is READ-ONLY here: only `restore_runs.py` (list + download) and get_object on the CartPole artefacts. Nothing is
    written to or deleted from R2, and no existing figure, report or plotting script is touched.

Palette, line styles and helpers are imported, not redefined:
  mjlab arms        <- make_report5_figs (C_NONE red, C_VAL blue, C_BIAS green, C_CTRL grey) + mechanism_e11.COLORS["clipoff"]
  CartPole reps     <- scripts/plot_eval_v3 (REP_LINESTYLE, MODE_COLOR gold, bootstrap_ci, apply_style) and the pastel
                       ETA_COLOR_PALETTE read out of experiments/plotting/ablation_plotting.py (pandas is not installed
                       in this venv, so that one constant is read with `ast`, not imported)
  window statistics <- mechanism_e11 (series/load), shock records <- report5._branch_record + shock_tables
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MJ = HERE.parent
REPO = MJ.parent
sys.path[:0] = [str(HERE), str(MJ / "src"), str(REPO / "scripts")]

import h2_report as H2  # noqa: E402
import make_report5_figs as M5  # noqa: E402  (palette)
import mechanism_e11 as MECH  # noqa: E402
import plot_eval_v3 as P3  # noqa: E402  (CartPole style + bootstrap_ci)
import report4 as R4  # noqa: E402
import report5 as R5  # noqa: E402
import shock_tables as ST  # noqa: E402

OUT = REPO / "docs" / "figures" / "supervisor"
RES = MJ / "results"
CACHE = RES / "supervisor_cache"
CP_CACHE = CACHE / "cartpole"
G1_CACHE = CACHE / "g1"
G1_CAMPAIGN = "r6g1queue-0ede5636"  # the brief calls it r6g1claim-*; the campaign on R2 is the merged claim queue
SEEDS = list(range(3000, 3020))
WALK = R4.WALK  # 0.5
N_BOOT = P3.N_BOOT

# ----------------------------------------------------------------------------------------------- style
def _ast_const(path: Path, name: str):
  tree = ast.parse(path.read_text())
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
      return ast.literal_eval(node.value)
  raise KeyError(name)


PASTEL = _ast_const(REPO / "experiments" / "plotting" / "ablation_plotting.py", "ETA_COLOR_PALETTE")


def lighten(hex_color: str, f: float = 0.55) -> str:
  h = hex_color.lstrip("#")
  rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
  return "#" + "".join(f"{int(c + (255 - c) * f):02x}" for c in rgb)


# One style per arm, used by every figure that shows that arm. Colour is the existing palette; marker and line style are
# added so the figures still read in greyscale (markers differ in shape, line styles differ, bars carry hatches).
ARM = {
  "none":  dict(color=M5.C_NONE, marker="o", ls="-", hatch="", label="uncentered"),
  "value": dict(color=M5.C_VAL, marker="s", ls="--", hatch="//", label="centered"),
  "bias":  dict(color=M5.C_BIAS, marker="^", ls="-.", hatch="xx", label="bias-init (uncentered)"),
  "clip":  dict(color=MECH.COLORS["clipoff"], marker="v", ls=":", hatch="..", label="clip-off (pilot)"),
}
REP = {
  "tile":     dict(color=P3.MODE_COLOR["simple"], marker="D", ls=P3.REP_LINESTYLE["tile_coding"], mfc=None),
  "ssp":      dict(color=PASTEL[4], marker="P", ls=P3.REP_LINESTYLE["ssp"], mfc=None),
  "ssp_old":  dict(color=lighten(PASTEL[4], 0.15), marker="P", ls=(0, (1, 1.6)), mfc="white"),
  "discrete": dict(color=PASTEL[5], marker="X", ls=P3.REP_LINESTYLE["discrete"], mfc=None),
}


def apply_style() -> None:
  P3.apply_style()
  plt.rcParams.update({"savefig.bbox": "tight", "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.2,
                       "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 110})


def save(fig, stem: str) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUT / f"{stem}.png", dpi=300)
  fig.savefig(OUT / f"{stem}.pdf")
  plt.close(fig)
  print("wrote", OUT / f"{stem}.png", "+ .pdf")


# ----------------------------------------------------------------------------------------------- statistics
def boot_ci(x) -> tuple[float, float]:
  return P3.bootstrap_ci(list(x), n_boot=N_BOOT)


def band(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Mean over seeds (rows) and a 95% seed-resampling bootstrap CI at every column (eval point)."""
  mean = np.nanmean(Y, 0)
  ci = np.array([boot_ci(Y[:, k]) for k in range(Y.shape[1])])
  return mean, ci[:, 0], ci[:, 1]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
  if n == 0:
    return float("nan"), float("nan")
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - h, c + h


def contrast(a: dict[int, float], b: dict[int, float]) -> dict:
  """Paired (a - b) over the seeds both have: mean, bootstrap 95% CI, and the t-interval REPORT_005 used."""
  seeds = sorted(set(a) & set(b))
  d = np.array([a[s] - b[s] for s in seeds])
  out = {"n": len(d), "mean": float(d.mean()) if len(d) else float("nan"), "positive": int((d > 0).sum())}
  if len(d) >= 2:
    out["boot"] = list(boot_ci(d))
    t = R4.paired(a, b)
    out["t"] = t["ci"]
  else:
    out["boot"] = out["t"] = [float("nan")] * 2
  return out


def pm(c: dict, key: str = "boot", fmt: str = "+.4f") -> str:
  lo, hi = c[key]
  return f"{c['mean']:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


# ----------------------------------------------------------------------------------------------- Go1 / G1 data
def _eval_exists(jid: str) -> bool:
  return (R4.MERGED / jid / "eval.jsonl").exists() or (RES / "merged_r2" / jid / "eval.jsonl").exists()


# (seed lo, seed hi, job-id template). c = 0 is E12 (3000-3009) + the round-5 top-up (3010-3019); c = 1 is E11.
TEMPL = {
  ("none", 0.0):  [(3000, 3009, "w2off0_go1_raw_none_g0.999_o0_s{s}"), (3010, 3019, "r4off0_go1_g0.999_o0_none_s{s}")],
  ("value", 0.0): [(3000, 3009, "w2off0_go1_raw_value_g0.999_o0_s{s}"), (3010, 3019, "r4off0_go1_g0.999_o0_value_s{s}")],
  ("none", 0.1):  [(3000, 3019, "r4sweep_go1_g0.999_o0.1_none_s{s}")],
  ("value", 0.1): [(3000, 3019, "r4sweep_go1_g0.999_o0.1_value_s{s}")],
  ("bias", 0.1):  [(3000, 3019, "r5bias01_go1_g0.999_o0.1_none_s{s}")],
  ("none", 0.3):  [(3000, 3019, "r4sweep_go1_g0.999_o0.3_none_s{s}")],
  ("value", 0.3): [(3000, 3019, "r4sweep_go1_g0.999_o0.3_value_s{s}")],
  ("none", 1.0):  [(3000, 3019, "w2main_go1_raw_none_g0.999_o1_s{s}")],
  ("value", 1.0): [(3000, 3019, "w2main_go1_raw_value_g0.999_o1_s{s}")],
  ("bias", 1.0):  [(3000, 3019, "r5bias_go1_g0.999_o1_none_s{s}")],
}
OFFSETS = (0.0, 0.1, 0.3, 1.0)
_CELL_CACHE: dict = {}


def go1_jobs(arm: str, c: float) -> dict[int, str]:
  out = {}
  for s in SEEDS:
    for lo, hi, tp in TEMPL[(arm, c)]:
      if lo <= s <= hi and _eval_exists(tp.format(s=s)):
        out[s] = tp.format(s=s)
  return out


def go1_stats(arm: str, c: float) -> dict:
  """Last-5-eval means per seed (R4.cell: newest attempt only) for one Go1 cell."""
  key = (arm, c)
  if key not in _CELL_CACHE:
    jobs = go1_jobs(arm, c)
    cells = {s: R4.cell(j) for s, j in jobs.items()}
    _CELL_CACHE[key] = {"jobs": jobs, "track": {s: v["track"] for s, v in cells.items()},
                        "J": {s: v["J"] for s, v in cells.items()}, "n": len(cells),
                        "walks": sum(v["walks"] for v in cells.values())}
  return _CELL_CACHE[key]


def curves(jobs: dict[int, str]) -> tuple[list[int], np.ndarray, np.ndarray]:
  """Seeds x common-eval-points matrix of `track_lin_frac` for a {seed: job_id} map."""
  rows = {}
  for s, j in jobs.items():
    it, v = M5._traj(j)
    if len(v):
      rows[s] = dict(zip(it.tolist(), v.tolist()))
  common = sorted(set.intersection(*[set(r) for r in rows.values()]))
  seeds = sorted(rows)
  return seeds, np.array(common), np.array([[rows[s][i] for i in common] for s in seeds], float)


def plot_arm_curve(ax, x, Y, arm: str, label: str, k: int = 0, every: int = 4, **kw) -> None:
  st = ARM[arm]
  m, lo, hi = band(Y)
  ax.fill_between(x, lo, hi, color=st["color"], alpha=0.17, lw=0)
  ax.plot(x, m, color=st["color"], ls=kw.get("ls", st["ls"]), lw=1.8, marker=st["marker"], ms=4.6, markevery=(k * 2 % every, every),
          mfc=st["color"], label=label)


# ----------------------------------------------------------------------------------------------- CartPole
CP_FILES = {
  "old_ssp_none_linear": ("stage0_cartpole/eval_v3/dqn_ssp_none_v4_hparam_search/", ("train_runs.jsonl", "train_summary.json")),
  "old_discrete_none_linear": ("stage0_cartpole/eval_v3/dqn_discrete_none_v4_hparam_search/", ("train_runs.jsonl", "train_summary.json")),
  "old_tile_coding_none_linear": ("stage0_cartpole/eval_v3/dqn_tile_coding_none_v4_hparam_search/", ("train_runs.jsonl", "train_summary.json")),
  "ssp_h2": ("stage0_cartpole/refilter_v2/dqn_ssp_h2_none_v4_hparam_search/", ("final_eval_runs.jsonl", "final_eval_summary.json")),
  "ssp_h2raw": ("stage0_cartpole/refilter_v2/dqn_ssp_h2raw_none_v4_hparam_search/", ("final_eval_runs.jsonl", "final_eval_summary.json")),
  "tile_h2": ("stage0_cartpole/refilter_v2/dqn_tile_h2_none_v4_hparam_search/", ("final_eval_runs.jsonl", "final_eval_summary.json")),
}
EPISODES_PER_CHECKPOINT = 1000 // 15  # run_hparam_search_dqn: eval_freq = trials // n_evaluations = 66; evals at 66, 132, ... 990


def ensure_cartpole() -> None:
  """Download the CartPole artefacts that are missing from the cache (get_object only: R2 is read-only)."""
  from mjlab_centering import r2
  for name, (prefix, files) in CP_FILES.items():
    for f in files:
      p = CP_CACHE / name / f
      if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        r2.client().download_file(r2.bucket(), prefix + f, str(p))  # bucket-root keys: no mjlab prefix


def cp_runs(name: str) -> dict[int, dict]:
  p = CP_CACHE / name / ("train_runs.jsonl" if name.startswith("old_") else "final_eval_runs.jsonl")
  rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  rows = [r for r in rows if r.get("kind") in ("policy", "final") and r.get("rank", 0) == 0 and 3000 <= r["seed"] <= 3019]
  return {r["seed"]: r for r in rows}


def cp_matrix(name: str) -> np.ndarray:
  runs = cp_runs(name)
  assert len(runs) == 20, f"{name}: expected 20 seeds 3000-3019, found {len(runs)}"
  return np.array([runs[s]["eval_history"] for s in sorted(runs)], float)


def cp_summary(name: str) -> dict:
  runs = cp_runs(name)
  score = {s: r["score"] for s, r in runs.items()}
  v = np.array([score[s] for s in sorted(score)])
  sj = json.loads((CP_CACHE / name / ("train_summary.json" if name.startswith("old_") else "final_eval_summary.json")).read_text())
  ls = (sj.get("params", {}) or {}).get("length_scale") if name.startswith("old_") else (sj.get("winner_search_params") or {}).get("length_scale")
  return {"n": len(v), "mean": float(v.mean()), "sd0": float(v.std(ddof=0)), "sd1": float(v.std(ddof=1)), "score": score,
          "solved": float((v >= P3.SOLVED).mean()), "length_scale": ls}


def cp_curves(ax, names_styles: list, title: str) -> None:
  for k, (name, sty, label) in enumerate(names_styles):
    Y = cp_matrix(name)
    x = (np.arange(Y.shape[1]) + 1) * EPISODES_PER_CHECKPOINT
    m, lo, hi = band(Y)
    st = REP[sty]
    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.17, lw=0)
    ax.plot(x, m, color=st["color"], ls=st["ls"], lw=1.9, marker=st["marker"], ms=5.2, markevery=(k % 2, 1), mfc=st["mfc"] or st["color"],
            mec=st["color"], label=label)
  ax.axhline(P3.SOLVED, color="#222222", lw=1.2, ls="--", alpha=0.8)
  ax.text(x[0], P3.SOLVED + 5, "solved = 475", fontsize=8, color="#222222", va="bottom")
  ax.set_xlabel("training episode (greedy evaluation every 66 episodes)")
  ax.set_ylabel("greedy evaluation return\n(mean of 10 episodes per checkpoint)")
  ax.set_ylim(0, 520)
  ax.set_xlim(0, 1000)
  ax.grid(alpha=0.25)
  ax.set_title(title)
  ax.legend(loc="lower right")


def fig1a(S: dict) -> None:
  fig, ax = plt.subplots(figsize=(7.0, 4.5))
  items = []
  for name, sty, lab in (("tile_h2", "tile", "tile coding, new search (tile_h2)"), ("ssp_h2", "ssp", "SSP, new search (ssp_h2)"),
                         ("old_discrete_none_linear", "discrete", "discrete, old search pipeline")):
    s = cp_summary(name)
    S["cp"][name] = s
    items.append((name, sty, f"{lab}  n={s['n']}, final {s['mean']:.1f}"))
  cp_curves(ax, items, "CartPole, linear DQN, no centering: representations")
  fig.tight_layout()
  save(fig, "fig1a_cartpole_representations")


def fig1b(S: dict) -> None:
  fig, ax = plt.subplots(figsize=(7.0, 4.5))
  old, new = cp_summary("old_ssp_none_linear"), cp_summary("ssp_h2")
  S["cp"]["old_ssp_none_linear"], S["cp"]["ssp_h2"] = old, new
  cp_curves(ax, [("old_ssp_none_linear", "ssp_old", f"SSP, old search: length scale 0.1-1 (chosen {old['length_scale']:.3f})  n={old['n']}, final {old['mean']:.1f}"),
                 ("ssp_h2", "ssp", f"SSP, new search: length scale 0.01-1 (chosen {new['length_scale']:.3f})  n={new['n']}, final {new['mean']:.1f}")],
            f"CartPole, linear DQN, no centering: old vs new SSP search (final {old['mean']:.1f} -> {new['mean']:.1f})")
  fig.tight_layout()
  save(fig, "fig1b_cartpole_ssp_old_vs_new")


# ----------------------------------------------------------------------------------------------- Fig 2, 3
def fig2(S: dict) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
  panels = [(0.0, ("none", "value"), "offset 0"), (1.0, ("none", "value", "bias"), "offset +1")]
  for ax, (c, arms, ttl) in zip(axes, panels):
    for k, arm in enumerate(arms):
      st = go1_stats(arm, c)
      seeds, it, Y = curves(st["jobs"])
      plot_arm_curve(ax, it, Y, arm, f"{ARM[arm]['label']}  n={len(seeds)}, walks {st['walks']}/{st['n']}", k=k)
    ax.axhline(WALK, color="grey", ls=":", lw=1)
    ax.text(it[-1], WALK + 0.012, "walks = 0.5", ha="right", fontsize=8, color="grey")
    ax.set_xlabel("PPO iteration")
    ax.set_title(f"{ttl}" + (" (critic output bias initialised to c/(1-gamma) = 1000 in the bias-init arm)" if c == 1.0 else ""), fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
  axes[0].set_ylabel("tracking (track_lin_frac)")
  fig.suptitle("Go1, gamma 0.999, corrected protocol: mean over 20 seeds (3000-3019), 95% bootstrap CI over seeds", fontsize=10.5)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  save(fig, "fig2_go1_main_effect")


def fig3(S: dict) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
  dx = {"none": -0.17, "value": 0.0, "bias": 0.17}
  rng = np.random.default_rng(0)
  S["dose"] = {}
  for arm in ("none", "value", "bias"):
    xs, walk, wlo, whi, mean, mlo, mhi = [], [], [], [], [], [], []
    for i, c in enumerate(OFFSETS):
      if (arm, c) not in TEMPL:
        continue
      st = go1_stats(arm, c)
      n, k = st["n"], st["walks"]
      v = np.array([st["track"][s] for s in sorted(st["track"])])
      lo, hi = wilson(k, n)
      blo, bhi = boot_ci(v)
      S["dose"][(arm, c)] = {"n": n, "walks": k, "wilson": [lo, hi], "mean": float(v.mean()), "boot": [blo, bhi]}
      xs.append(i + dx[arm]); walk.append(k / n); wlo.append(lo); whi.append(hi); mean.append(v.mean()); mlo.append(blo); mhi.append(bhi)
      axes[1].scatter(np.full(n, i + dx[arm]) + rng.uniform(-0.035, 0.035, n), v, s=7, color=ARM[arm]["color"], alpha=0.28, lw=0, zorder=1)
      if arm == "value":
        axes[0].annotate(f"{k}/{n}", (i + dx[arm], hi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7.4, color=ARM[arm]["color"])
      else:
        axes[0].annotate(f"{k}/{n}", (i + dx[arm], k / n), textcoords="offset points", xytext=(-9, 0) if arm == "none" else (9, 0),
                         ha="right" if arm == "none" else "left", va="center", fontsize=7.4, color=ARM[arm]["color"])
    xs, walk, wlo, whi, mean, mlo, mhi = map(np.array, (xs, walk, wlo, whi, mean, mlo, mhi))
    line = ARM[arm]["ls"] if arm != "bias" else "none"
    for a, y, lo, hi in ((axes[0], walk, wlo, whi), (axes[1], mean, mlo, mhi)):
      a.errorbar(xs, y, yerr=[y - lo, hi - y], color=ARM[arm]["color"], marker=ARM[arm]["marker"], ms=7, ls=line, lw=1.6, capsize=3,
                 elinewidth=1.3, label=f"{ARM[arm]['label']}  (n={n} per point)", zorder=3)
  axes[1].axhline(WALK, color="grey", ls=":", lw=1)
  for a in axes:
    a.set_xticks(range(len(OFFSETS)))
    a.set_xticklabels([f"{c:g}" for c in OFFSETS])
    a.set_xlabel("constant reward offset c")
    a.grid(alpha=0.25)
    a.set_xlim(-0.5, len(OFFSETS) - 0.5)
  axes[0].set_ylim(-0.05, 1.12)
  axes[0].set_ylabel("fraction of seeds that walk")
  axes[0].set_title("walking rate, Wilson 95% interval", fontsize=10)
  axes[1].set_ylim(0.15, 1.0)
  axes[1].set_ylabel("final tracking (last-5-eval mean)")
  axes[1].set_title("mean final tracking, bootstrap 95% CI over seeds (dots = seeds)", fontsize=10)
  axes[0].legend(loc="lower left", fontsize=8)
  fig.suptitle("Go1 offset dose-response, gamma 0.999, corrected protocol, seeds 3000-3019 (n = 20 per arm and offset)", fontsize=10.5)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  save(fig, "fig3_go1_offset_dose_response")


# ----------------------------------------------------------------------------------------------- Fig 4
MECH_SETS = [("none", "r2", "w2main_go1_raw_none_g0.999_o1_s", "uncentered\n(E11)"),
             ("clip", "r2", "clip1_go1_raw_none_clipoff_g0.999_o1_s", "clip-off\n(clip1, pilot)"),
             ("bias", "r2", "r5bias_go1_g0.999_o1_none_s", "bias-init"),
             ("value", "r2", "w2main_go1_raw_value_g0.999_o1_s", "centered\n(E11)")]
WIN = (100, 300)  # mechanism_e11.WINDOWS[0]; window mean = iterations with lo <= it < hi, exactly as mechanism_e11's wmean


def _iter_path(jid: str) -> Path:
  for d in (R4.MERGED, RES / "merged_r2"):
    if (d / jid / "iter.jsonl").exists():
      return d / jid / "iter.jsonl"
  raise FileNotFoundError(jid)


def mech_window(prefix: str, key: str) -> dict[int, float]:
  out = {}
  for s in SEEDS:
    jid = f"{prefix}{s}"
    try:
      p = _iter_path(jid)
    except FileNotFoundError:
      continue
    ser = MECH.series(MECH.load(str(p)))
    out[s] = float(ser[key][(ser["it"] >= WIN[0]) & (ser["it"] < WIN[1])].mean())
  return out


def fig4(S: dict) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.9, 1.0]})
  S["mech"] = {}
  rng = np.random.default_rng(1)
  for ax, key, ttl, yl, ylim in ((axes[0], "ev_state", "state-explained share of the advantage (ev_state)", "ev_state, mean over iterations 100-299", (0, 1.12)),
                                 (axes[1], "const_frac", "constant share of the critic error (const_frac)", "const_frac, mean over iterations 100-299", (0, 1.12))):
    for i, (arm, _, prefix, lab) in enumerate(MECH_SETS):
      w = mech_window(prefix, key)
      v = np.array([w[s] for s in sorted(w)])
      lo, hi = boot_ci(v)
      walks = sum(R4.cell(f"{prefix}{s}")["walks"] for s in w)
      S["mech"][(arm, key)] = {"n": len(v), "mean": float(v.mean()), "boot": [lo, hi], "walks": walks}
      st = ARM[arm]
      ax.bar(i, v.mean(), 0.66, color=lighten(st["color"], 0.5), edgecolor=st["color"], hatch=st["hatch"], lw=1.2, zorder=2)
      ax.scatter(np.full(len(v), i) + rng.uniform(-0.2, 0.2, len(v)), v, s=14, marker=st["marker"], color=st["color"], alpha=0.75, lw=0, zorder=3)
      ax.errorbar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], color="k", capsize=4, lw=1.4, zorder=4)
      if key == "ev_state":
        ax.text(i, min(hi, 1.0) + 0.045, f"walks {walks}/{len(v)}", ha="center", fontsize=8.2, fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0), zorder=5)
      else:
        ax.text(i, hi + 0.04, f"{v.mean():.2f}", ha="center", fontsize=8, color="#333333")
    ax.set_xticks(range(len(MECH_SETS)))
    ax.set_xticklabels([f"{m[3]}\nn={S['mech'][(m[0], key)]['n']}" for m in MECH_SETS], fontsize=8.4)
    ax.set_ylim(*ylim)
    ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=9.8)
    ax.grid(alpha=0.25, axis="y")
    ax.xaxis.grid(False)
  fig.suptitle("Mechanism, Go1 offset +1, gamma 0.999: critic window iterations 100-299 (bars: mean over seeds, whiskers: 95% bootstrap CI, dots: seeds)", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  save(fig, "fig4_mechanism")


# ----------------------------------------------------------------------------------------------- Fig 5
KINDS = ST.KINDS  # ("off+0.1", "off+1.0", "actx5")
SHOCK_AT = 1200


def shock_records() -> list[dict]:
  rows = []
  for arm in ("none", "value"):
    for s in SEEDS:
      mother = f"r4mother_go1_g0.999_o0_{arm}_s{s}"
      for kind in ("ctrl",) + KINDS:
        job = {"job_id": f"r4branch_go1_g0.999_o0_{kind}_{arm}_s{s}", "shock_at": SHOCK_AT, "seed": s, "center": arm,
               "shock_kind": "none" if kind == "ctrl" else ("offset" if kind.startswith("off") else "actx5"),
               "extra": {"shock_delta": float(kind[3:])} if kind.startswith("off") else {}}
        rows.append(R5._branch_record(job, mother))  # report5's own record builder: base = mother's last-3 evals before 1200
  return rows


def fig5(S: dict) -> None:
  rows = shock_records()
  S["shock_rows"] = rows
  tabA, tabB = ST.tables(rows, False), ST.tables(rows, True)
  S["shock_A"], S["shock_B"] = tabA, tabB
  mother_ok = {(r["arm"], r["seed"]): r["base"] >= ST.WALK_MOTHER for r in rows if r["kind"] == "ctrl"}
  fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), sharey=True)
  titles = {"off+1.0": "+1.0 offset (level change)", "actx5": "action-rate x5 (shape change)"}
  for ax, kind in zip(axes, ("off+1.0", "actx5")):
    for k, arm in enumerate(("none", "value")):
      seeds = [s for s in SEEDS if mother_ok[(arm, s)]]
      st = ARM[arm]
      mj = {s: f"r4mother_go1_g0.999_o0_{arm}_s{s}" for s in seeds}
      _, mit, MY = curves(mj)
      sel = (mit >= 990) & (mit < SHOCK_AT)
      mit, MY = mit[sel], MY[:, sel]
      mm, mlo, mhi = band(MY)
      ax.fill_between(mit, mlo, mhi, color=st["color"], alpha=0.17, lw=0)
      ax.plot(mit, mm, color=st["color"], ls="-", lw=1.9, marker=st["marker"], ms=4.6, mfc=st["color"])
      for group, ls, lw in (("shocked", "-", 1.9), ("ctrl", "--", 1.6)):
        bj = {s: f"r4branch_go1_g0.999_o0_{kind if group == 'shocked' else 'ctrl'}_{arm}_s{s}" for s in seeds}
        _, bit, BY = curves(bj)
        x = np.concatenate([mit[-1:], bit])  # the branch resumes the mother's weights: start both from her last eval
        Y = np.concatenate([MY[:, -1:], BY], 1)
        m, lo, hi = band(Y)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.14 if group == "shocked" else 0.08, lw=0)
        ax.plot(x, m, color=st["color"], ls=ls, lw=lw, marker=st["marker"] if group == "shocked" else None, ms=4.4, markevery=(1 + k, 4),
                mfc=st["color"] if group == "shocked" else "white")
    ax.axvline(SHOCK_AT, color="k", lw=0.9, ls=":")
    ax.text(SHOCK_AT + 6, 0.965, "shock", fontsize=8.5, va="top")
    c_all, c_walk = tabA["contrast"][kind], tabB["contrast"][kind]
    ax.text(0.015, 0.03, f"centered - uncentered, shocked - control:\n walking mothers (n={c_walk['n']}): {c_walk['mean']:+.3f} [{c_walk['ci'][0]:+.3f}, {c_walk['ci'][1]:+.3f}]\n"
            f" all 20 seeds: {c_all['mean']:+.3f} [{c_all['ci'][0]:+.3f}, {c_all['ci'][1]:+.3f}]  (t 95% CI)", transform=ax.transAxes, fontsize=7.6,
            va="bottom", ha="left", bbox=dict(fc="white", ec="#cccccc", alpha=0.9, pad=2.5))
    ax.set_title(titles[kind], fontsize=10)
    ax.set_xlabel("PPO iteration")
    ax.set_xlim(990, 1605)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
  nn, nv = tabB["n_mothers"]["none"]["kept"], tabB["n_mothers"]["value"]["kept"]
  handles = [plt.Line2D([], [], color=ARM["none"]["color"], marker=ARM["none"]["marker"], ls="-", label=f"uncentered, walking mothers n={nn}/20"),
             plt.Line2D([], [], color=ARM["value"]["color"], marker=ARM["value"]["marker"], ls="-", label=f"centered, n={nv}/20"),
             plt.Line2D([], [], color="k", ls="-", label="mother, then shocked branch (solid)"),
             plt.Line2D([], [], color="k", ls="--", label="unshocked control (dashed)")]
  axes[0].legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=7.8)
  axes[0].set_ylabel("tracking (track_lin_frac)")
  fig.suptitle("Go1 shock test: offset-0 mothers, shock at iteration 1200, pooled seeds 3000-3019; mean, 95% bootstrap CI over seeds; walking mothers = mean of last 3 evals before 1200 >= 0.8",
               fontsize=9.6)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  save(fig, "fig5_shock_test")


# ----------------------------------------------------------------------------------------------- Fig 6 (G1)
def pull_g1() -> None:
  """Refresh the G1 claim runs: targeted read-only fetch (fetch_supervisor_data.py) of manifest, DONE list and, for finished
  runs only, eval.jsonl / final.json / config.json. Already-cached files are skipped. restore_runs.py is not used."""
  subprocess.run([sys.executable, str(HERE / "fetch_supervisor_data.py")], check=True, cwd=str(MJ))


def g1_data() -> dict[str, dict[int, dict]]:
  """{arm: {seed: {track, J, curve}}} for runs with a DONE marker only (so n per arm is whatever has finished)."""
  man = json.loads((G1_CACHE / "manifest.json").read_text())
  out: dict[str, dict[int, dict]] = {"none": {}, "value": {}, "bias": {}}
  for j in man["jobs"]:
    arm = "bias" if (j.get("extra") or {}).get("critic_bias_init") else j["center"]
    d = G1_CACHE / j["job_id"]
    if not ((d / "DONE").exists() and (d / "eval.jsonl").exists()):
      continue
    ev = R4.newest(d / "eval.jsonl")
    if not (3000 <= j["seed"] <= 3019) or ev[-1]["iter"] < j["iters"] - 50:
      continue  # claim seeds only; a DONE run must reach the last eval
    out[arm][j["seed"]] = {"track": R4.mean_last5(ev), "J": R4.mean_last5(ev, "J_task"), "it": [e["iter"] for e in ev],
                           "curve": [e["track_lin_frac"] for e in ev], "iters": j["iters"]}
  return out


def fig6(S: dict) -> None:
  if not (G1_CACHE / "manifest.json").exists():
    print("[skip] fig6: no G1 cache; run with --pull-g1")
    return
  data = g1_data()
  S["g1"] = data
  fig, ax = plt.subplots(figsize=(7.6, 4.7))
  for k, arm in enumerate(("none", "value", "bias")):
    runs = data[arm]
    if len(runs) < 2:
      continue
    seeds = sorted(runs)
    common = sorted(set.intersection(*[set(runs[s]["it"]) for s in seeds]))
    Y = np.array([[dict(zip(runs[s]["it"], runs[s]["curve"]))[i] for i in common] for s in seeds])
    walks = sum(runs[s]["track"] >= WALK for s in seeds)
    plot_arm_curve(ax, np.array(common), Y, arm, f"{ARM[arm]['label']}  n={len(seeds)}, walks {walks}/{len(seeds)}", k=k, every=5)
  ax.axhline(WALK, color="grey", ls=":", lw=1)
  ax.text(2495, WALK + 0.012, "walks = 0.5", ha="right", fontsize=8, color="grey")
  ax.axhline(0.267, color="grey", ls="-.", lw=0.8, alpha=0.7)
  ax.text(2495, 0.267 - 0.012, "standing 0.267", ha="right", va="top", fontsize=8, color="grey")
  n = {a: len(data[a]) for a in data}
  full = all(v == 20 for v in n.values())
  ax.set_xlabel("PPO iteration")
  ax.set_ylabel("tracking (track_lin_frac)")
  ax.set_xlim(0, 2500)
  ax.set_ylim(0, 1.0)
  ax.grid(alpha=0.25)
  ax.legend(loc="upper left")
  ax.set_title(f"G1, gamma 0.99, offset +1, fall penalty 5.7, 2500 iterations\nseeds with DONE markers so far (n per arm in legend; {'complete: 20 per arm' if full else 'PILOT: claim set still running'})",
               fontsize=9.6)
  fig.tight_layout()
  save(fig, "fig6_g1_second_robot")


# ----------------------------------------------------------------------------------------------- tables, checks, captions
def build_tables(S: dict) -> tuple[list[dict], list[str]]:
  """Summary rows (long format, for the CSV) and the markdown."""
  rows: list[dict] = []
  md: list[str] = []

  def add(section, cell, **kw):
    rows.append({"section": section, "cell": cell, **kw})

  # ---- Go1 offset sweep
  md += ["# Summary table (corrected protocol, claim seeds 3000-3019; last-5-eval means; 'walks' = last-5 tracking >= 0.5)", "",
         "CIs: 'boot' = 95% bootstrap over seeds (5000 resamples, seed 0, `plot_eval_v3.bootstrap_ci`); 't' = the 95% t-interval used in REPORT_005.",
         "", "## 1. Go1 offset dose-response, gamma 0.999, 2000 iterations: centered - uncentered, paired by seed", "",
         "| offset c | uncentered walks | centered walks | uncentered tracking | centered tracking | paired dtrack (boot 95% CI) | paired dtrack (t 95% CI) | seeds positive | n pairs |",
         "|---|---|---|---|---|---|---|---|---|"]
  S["contrast"] = {}
  for c in OFFSETS:
    a, b = go1_stats("value", c), go1_stats("none", c)
    k = contrast(a["track"], b["track"])
    kj = contrast(a["J"], b["J"])
    S["contrast"][("value-none", c)] = k
    md.append(f"| {c:g} | {b['walks']}/{b['n']} | {a['walks']}/{a['n']} | {np.mean(list(b['track'].values())):.3f} | {np.mean(list(a['track'].values())):.3f} | "
              f"{pm(k)} | {pm(k, 't')} | {k['positive']}/{k['n']} | {k['n']} |")
    add("go1_offset", f"c={c:g}", contrast="centered-uncentered", n=k["n"], walks_a=a["walks"], walks_b=b["walks"], mean_a=np.mean(list(a["track"].values())),
        mean_b=np.mean(list(b["track"].values())), dtrack=k["mean"], boot_lo=k["boot"][0], boot_hi=k["boot"][1], t_lo=k["t"][0], t_hi=k["t"][1],
        positive=k["positive"], dJ=kj["mean"], dJ_boot_lo=kj["boot"][0], dJ_boot_hi=kj["boot"][1])
  md += ["", "## 2. Bias-init (critic output bias = c/(1-gamma), uncentered), Go1, gamma 0.999", "",
         "| offset c | arm | walks | tracking | paired dtrack vs uncentered (boot) | (t) | paired dtrack vs centered (boot) | n pairs |", "|---|---|---|---|---|---|---|---|"]
  for c in (0.1, 1.0):
    bi, un, ce = go1_stats("bias", c), go1_stats("none", c), go1_stats("value", c)
    k_u, k_c = contrast(bi["track"], un["track"]), contrast(bi["track"], ce["track"])
    S["contrast"][("bias-none", c)], S["contrast"][("bias-value", c)] = k_u, k_c
    md.append(f"| {c:g} | bias-init | {bi['walks']}/{bi['n']} | {np.mean(list(bi['track'].values())):.3f} | {pm(k_u)} | {pm(k_u, 't')} | {pm(k_c)} | {k_u['n']} |")
    add("go1_bias", f"c={c:g}", contrast="bias-uncentered", n=k_u["n"], walks_a=bi["walks"], walks_b=un["walks"], mean_a=np.mean(list(bi["track"].values())),
        mean_b=np.mean(list(un["track"].values())), dtrack=k_u["mean"], boot_lo=k_u["boot"][0], boot_hi=k_u["boot"][1], t_lo=k_u["t"][0], t_hi=k_u["t"][1],
        positive=k_u["positive"])
    add("go1_bias", f"c={c:g}", contrast="bias-centered", n=k_c["n"], walks_a=bi["walks"], walks_b=ce["walks"], dtrack=k_c["mean"],
        boot_lo=k_c["boot"][0], boot_hi=k_c["boot"][1], t_lo=k_c["t"][0], t_hi=k_c["t"][1], positive=k_c["positive"])

  # ---- shocks
  md += ["", "## 3. Shock test (Go1, offset-0 mothers, shock at iteration 1200, pooled seeds 3000-3019; t 95% CIs, `shock_tables.py`)", ""]
  for name, tab in (("(A) all 20 seeds, as pre-registered", S["shock_A"]), (f"(B) walking mothers only (mother's last-3-eval tracking >= {ST.WALK_MOTHER}); uncentered n={S['shock_B']['n_mothers']['none']['kept']}, centered n={S['shock_B']['n_mothers']['value']['kept']}", S["shock_B"])):
    md += [f"**{name}**", "", "| shock | uncentered shocked - control | centered shocked - control | arm contrast (centered - uncentered) | seeds positive | n |", "|---|---|---|---|---|---|"]
    for kind in KINDS:
      r_n, r_v, ct = tab["rows"].get(f"{kind}|none"), tab["rows"].get(f"{kind}|value"), tab["contrast"][kind]
      f = lambda r: f"{r['shocked_minus_ctrl']['mean']:+.4f} [{r['shocked_minus_ctrl']['ci'][0]:+.4f}, {r['shocked_minus_ctrl']['ci'][1]:+.4f}] (n={r['n']}, end within 95%: {r['recovered']}/{r['n']})"
      md.append(f"| {kind} | {f(r_n)} | {f(r_v)} | {ct['mean']:+.4f} [{ct['ci'][0]:+.4f}, {ct['ci'][1]:+.4f}] | {ct['positive']}/{ct['n']} | {ct['n']} |")
      add("shock_A" if tab is S["shock_A"] else "shock_B", kind, contrast="centered-uncentered (shocked-control)", n=ct["n"], dtrack=ct["mean"], t_lo=ct["ci"][0], t_hi=ct["ci"][1],
          positive=ct["positive"], end_none=r_n["end"], end_value=r_v["end"], recovered_none=r_n["recovered"], n_none=r_n["n"], recovered_value=r_v["recovered"], n_value=r_v["n"])
    md.append("")

  # ---- G1
  md += ["## 4. G1, gamma 0.99, offset +1, fall penalty 5.7, 2500 iterations (seeds with DONE markers; fewer than 20 per arm = pilot)", ""]
  if "g1" in S:
    g = S["g1"]
    md += ["| arm | n | walks | tracking (last-5 mean) | J_task |", "|---|---|---|---|---|"]
    for arm in ("none", "value", "bias"):
      r = g[arm]
      if not r:
        continue
      w = sum(v["track"] >= WALK for v in r.values())
      md.append(f"| {ARM[arm]['label']} | {len(r)} | {w}/{len(r)} | {np.mean([v['track'] for v in r.values()]):.3f} | {np.mean([v['J'] for v in r.values()]):.4f} |")
      add("g1", arm, n=len(r), walks_a=w, mean_a=np.mean([v["track"] for v in r.values()]), J_a=np.mean([v["J"] for v in r.values()]))
    md += ["", "| paired contrast | n pairs | dtrack (boot 95% CI) | (t 95% CI) | seeds positive | dJ_task (boot 95% CI) |", "|---|---|---|---|---|---|"]
    S["g1_contrast"] = {}
    for a_, b_, nm in (("value", "none", "centered - uncentered"), ("bias", "none", "bias-init - uncentered"), ("bias", "value", "bias-init - centered")):
      k = contrast({s: v["track"] for s, v in g[a_].items()}, {s: v["track"] for s, v in g[b_].items()})
      kj = contrast({s: v["J"] for s, v in g[a_].items()}, {s: v["J"] for s, v in g[b_].items()})
      S["g1_contrast"][nm] = k
      if k["n"] >= 2:
        md.append(f"| {nm} | {k['n']} | {pm(k)} | {pm(k, 't')} | {k['positive']}/{k['n']} | {pm(kj, fmt='+.5f')} |")
        add("g1", nm, contrast=nm, n=k["n"], dtrack=k["mean"], boot_lo=k["boot"][0], boot_hi=k["boot"][1], t_lo=k["t"][0], t_hi=k["t"][1],
            positive=k["positive"], dJ=kj["mean"], dJ_boot_lo=kj["boot"][0], dJ_boot_hi=kj["boot"][1])
  else:
    md.append("(no G1 cache: run with --pull-g1)")

  # ---- CartPole
  md += ["", "## 5. CartPole, linear DQN, no centering: final score (mean of the last 5 of 15 greedy checkpoints), 20 seeds 3000-3019", "",
         "| arm | n | mean | sd (population, as REPORT_001/006) | sd (ddof=1) | solved (>= 475) | chosen length scale |", "|---|---|---|---|---|---|---|"]
  for name, lab in (("tile_h2", "tile coding, new search (tile_h2)"), ("ssp_h2", "SSP, new search (ssp_h2)"), ("ssp_h2raw", "SSP, new search, no input normalisation (ssp_h2raw)"),
                    ("old_tile_coding_none_linear", "tile coding, old search (RUN_004 replicate)"), ("old_ssp_none_linear", "SSP, old search (RUN_004 replicate)"),
                    ("old_discrete_none_linear", "discrete, old search pipeline (RUN_004 replicate)")):
    s = cp_summary(name)
    S["cp"][name] = s
    ls = f"{s['length_scale']:.4f}" if s["length_scale"] else "-"
    md.append(f"| {lab} | {s['n']} | {s['mean']:.2f} | {s['sd0']:.2f} | {s['sd1']:.2f} | {s['solved']:.0%} | {ls} |")
    add("cartpole", name, n=s["n"], mean_a=s["mean"], sd_pop=s["sd0"], sd_ddof1=s["sd1"], solved=s["solved"], length_scale=s["length_scale"])
  md += ["", "| paired contrast (by seed) | mean | bootstrap 95% CI (`h2_report.boot_ci`) |", "|---|---|---|"]
  S["cp_contrast"] = {}
  for a_, b_, nm in (("ssp_h2", "tile_h2", "ssp_h2 - tile_h2"), ("ssp_h2raw", "ssp_h2", "ssp_h2raw - ssp_h2"), ("ssp_h2", "old_ssp_none_linear", "ssp_h2 (new) - ssp old")):
    d = np.array([S["cp"][a_]["score"][s] - S["cp"][b_]["score"][s] for s in SEEDS])
    lo, hi = H2.boot_ci(d)
    S["cp_contrast"][nm] = {"mean": float(d.mean()), "ci": [lo, hi]}
    md.append(f"| {nm} | {d.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    add("cartpole", nm, contrast=nm, n=20, dtrack=d.mean(), boot_lo=lo, boot_hi=hi)
  return rows, md


def write_csv(rows: list[dict]) -> None:
  keys: list[str] = []
  for r in rows:
    for k in r:
      if k not in keys:
        keys.append(k)
  with open(OUT / "summary_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    for r in rows:
      w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# ---- sanity checks against the numbers the reports print -------------------------------------------------------------
def checks(S: dict) -> list[dict]:
  C: list[dict] = []

  def chk(label, source, expected, got, tol, fmt=".4f"):
    ok = (got == expected) if isinstance(expected, (int, str)) else abs(got - expected) <= tol
    C.append({"label": label, "source": source, "expected": expected, "got": got, "ok": bool(ok), "fmt": fmt})

  dose = S["dose"]
  for (arm, c), exp, src in ((("none", 0.0), 19, "R005 s3"), (("value", 0.0), 20, "R005 s3"), (("none", 0.1), 12, "R005 s3"), (("value", 0.1), 20, "R005 s3"),
                             (("none", 0.3), 2, "R005 s3"), (("value", 0.3), 18, "R005 s3"), (("none", 1.0), 0, "R005 s3 / R003"), (("value", 1.0), 19, "R005 s3 / R003"),
                             (("bias", 0.1), 20, "R005 s2"), (("bias", 1.0), 18, "R005 s2")):
    chk(f"walks, {arm}, c={c:g}", src, exp, dose[(arm, c)]["walks"], 0)
  for (arm, c), exp in ((("none", 0.1), 0.645), (("value", 0.1), 0.884), (("bias", 0.1), 0.883), (("bias", 1.0), 0.821), (("none", 1.0), 0.307), (("value", 1.0), 0.854)):
    chk(f"mean tracking, {arm}, c={c:g}", "R005", exp, dose[(arm, c)]["mean"], 0.0006, ".3f")
  # paired contrasts: REPORT_005 prints t-intervals
  for key, c, exp, src in ((("value-none"), 0.0, (+0.0143, -0.0145, +0.0430), "R005 s3"), ("value-none", 0.1, (+0.2396, +0.1060, +0.3732), "R005 s3"),
                           ("value-none", 0.3, (+0.4440, +0.3305, +0.5574), "R005 s3"), ("value-none", 1.0, (+0.5472, +0.4860, +0.6084), "R005 s3"),
                           ("bias-none", 0.1, (+0.2382, +0.1043, +0.3721), "R005 s2"), ("bias-none", 1.0, (+0.5144, +0.4309, +0.5979), "R005 s2")):
    k = S["contrast"][(key, c)]
    for nm, e, g in zip(("dtrack", "t lo", "t hi"), exp, (k["mean"], *k["t"])):
      chk(f"paired {key} {nm}, c={c:g}", src, e, g, 0.00051)
  # shocks, pooled (A) and (B), t-intervals
  A, B = S["shock_A"], S["shock_B"]
  for kind, exp in (("off+0.1", (+0.0028, -0.0107, +0.0163, 16)), ("off+1.0", (+0.5095, +0.4286, +0.5905, 20)), ("actx5", (+0.0100, -0.0096, +0.0297, 13))):
    ct = A["contrast"][kind]
    for nm, e, g in zip(("mean", "lo", "hi"), exp[:3], (ct["mean"], *ct["ci"])):
      chk(f"shock contrast (A all 20) {kind} {nm}", "R005 s4", e, g, 0.00051)
    chk(f"shock contrast (A) {kind} seeds positive", "R005 s4", exp[3], ct["positive"], 0, "d")
  for kind, exp in (("off+0.1", (+0.0114, +0.0067, +0.0162, 15)), ("off+1.0", (+0.5726, +0.5636, +0.5816, 16)), ("actx5", (+0.0034, -0.0002, +0.0069, 11))):
    ct = B["contrast"][kind]
    for nm, e, g in zip(("mean", "lo", "hi"), exp[:3], (ct["mean"], *ct["ci"])):
      chk(f"shock contrast (B walking) {kind} {nm}", "R005 s4", e, g, 0.00051)
    chk(f"shock contrast (B) {kind} seeds positive", "R005 s4", exp[3], ct["positive"], 0, "d")
  chk("walking uncentered mothers", "R005 s4", 16, B["n_mothers"]["none"]["kept"], 0, "d")
  chk("walking centered mothers", "R005 s4", 20, B["n_mothers"]["value"]["kept"], 0, "d")
  failing = sorted(r["seed"] for r in S["shock_rows"] if r["kind"] == "ctrl" and r["arm"] == "none" and r["base"] < ST.WALK_MOTHER)
  chk("uncentered mothers failing the 0.8 rule", "R005 s4", "[3005, 3008, 3018, 3019]", str(failing), 0, "s")
  r = A["rows"]
  chk("+1.0 uncentered: end", "R005 s4 (A)", 0.305, r["off+1.0|none"]["end"], 0.0006, ".3f")
  chk("+1.0 uncentered: within 95% of pre-shock", "R005 s4 (A)", 2, r["off+1.0|none"]["recovered"], 0, "d")
  chk("+1.0 uncentered: worst dip", "R005 s4 (A)", -0.524, r["off+1.0|none"]["dip"], 0.0006, ".3f")
  # shock records vs the stored report records
  stored = {**json.loads((RES / "report4_shock.json").read_text())["branches"], **json.loads((RES / "report5_shock.json").read_text())["branches"]}
  bad = 0
  for jid, rec in stored.items():
    mine = next((x for x in S["shock_rows"] if x["seed"] == rec["seed"] and x["arm"] == rec["arm"] and x["kind"] == rec["kind"]), None)
    if mine is None or any(abs(mine[k] - rec[k]) > 1e-9 for k in ("base", "min", "dip", "end")):
      bad += 1
  chk("recomputed shock branch records differ from results/report{4,5}_shock.json", "stored records", 0, bad, 0, "d")
  # mechanism
  m = S["mech"]
  for (arm, key), exp, src in ((("none", "ev_state"), 0.405, "R004 s1/s3 (20 seeds)"), (("none", "const_frac"), 0.803, "R004 s1"), (("value", "ev_state"), 0.954, "R004 s1"),
                               (("value", "const_frac"), 0.001, "R004 s1"), (("clip", "ev_state"), 0.186, "R004 s3"), (("clip", "const_frac"), 0.804, "R004 s3"),
                               (("bias", "ev_state"), 0.954, "R005 s2 (3-seed pilot; here 20)")):
    chk(f"window 100-300 {key}, {arm}", src, exp, m[(arm, key)]["mean"], 0.0006, ".3f")
  # CartPole
  cp = S["cp"]
  for name, mean, sd, src in (("ssp_h2", 486.1, 24.3, "R006 s3"), ("tile_h2", 495.8, 7.2, "R006 s3"), ("ssp_h2raw", 469.8, 19.8, "R006 s3"),
                              ("old_ssp_none_linear", 461.65, 35.88, "R001 s3.1 / R006"), ("old_tile_coding_none_linear", 499.64, 0.99, "R001 s3.1")):
    chk(f"CartPole {name} mean", src, mean, cp[name]["mean"], 0.051, ".2f")
    chk(f"CartPole {name} sd (population)", src, sd, cp[name]["sd0"], 0.051, ".2f")
  for nm, exp in (("ssp_h2 - tile_h2", (-9.66, -22.07, 0.68)), ("ssp_h2raw - ssp_h2", (-16.35, -31.19, 0.52))):
    k = S["cp_contrast"][nm]
    for lab, e, g in zip(("mean", "lo", "hi"), exp, (k["mean"], *k["ci"])):
      chk(f"CartPole paired {nm} {lab}", "R006 s3", e, g, 0.0051, ".2f")
  chk("CartPole discrete old mean within RUN_004 range 230.5-267.2", "R001 s3.1", 1, int(230.5 <= cp["old_discrete_none_linear"]["mean"] <= 267.2), 0, "d")
  chk("CartPole ssp_h2 chosen length scale", "R006 s3", 0.0220, cp["ssp_h2"]["length_scale"], 0.00005, ".4f")
  chk("CartPole ssp_h2 solved share", "R006 s3", 0.8, cp["ssp_h2"]["solved"], 1e-9, ".2f")
  return C


def fmtv(v, f):
  if isinstance(v, str):
    return v
  return f"{v:{f}}" if f != "d" else f"{int(v)}"


def write_notes(S: dict, C: list[dict]) -> None:
  bad = [c for c in C if not c["ok"]]
  L = ["# Supervisor figures: notes, sanity checks, discrepancies", "",
       f"Generated by `mjlab_centering/analysis/make_supervisor_figs.py`. {len(C)} recomputed numbers were compared with the numbers printed in REPORT_001/003/004/005/006; "
       f"**{len(C) - len(bad)} agree, {len(bad)} disagree**. Nothing was adjusted to make them agree.", "", "## Discrepancies", ""]
  if bad:
    L += ["| check | source | report says | recomputed | |", "|---|---|---|---|---|"]
    L += [f"| {c['label']} | {c['source']} | {fmtv(c['expected'], c['fmt'])} | {fmtv(c['got'], c['fmt'])} | DISAGREES |" for c in bad]
  else:
    L.append("None in the compared quantities.")
  L += ["", "## Things that are not discrepancies but would trip a reader", "",
        "- **Campaign name.** The brief says `r6g1claim-*`. The G1 claim set on R2 is `r6g1queue-0ede5636` (the claim set and its six timing jobs were merged into one queue; d137/d141). Job ids are `r6g1_g1_g0.99_o1_{none,value,bias_none}_s30xx`; the bias-init arm is identified from the manifest (`center none` + `extra.critic_bias_init = offset`), not from the id.",
        "- **sd convention.** REPORT_001/006 print the *population* sd (ddof=0) for CartPole (e.g. 24.3, 35.88); the tables here print both (`sd (ddof=1)` is about 2.5% larger).",
        "- **Mechanism, plain arm.** REPORT_004 gives ev_state 0.405 for the uncentered arm at 20 seeds; REPORT_005's 0.302 is the 3-seed subset (seeds 3000-3002) used to pair with the bias pilot. Fig. 4 uses all 20 and reproduces 0.405. The bias-init arm is also 20 seeds here (ev_state 0.954; const_frac 0.028 vs the pilot's 0.033).",
        "- **Interval type.** REPORT_004/005 and `shock_tables.py` use t-intervals for paired contrasts; REPORT_006's H2 uses a bootstrap. Fig. 3, the learning-curve bands and the summary table lead with bootstrap CIs (the brief's convention) and print the t-interval beside them, which is what the checks above compare.",
        "- **Shock mothers.** Mother and branch curves are read from `results/merged_r4` (the d116 primary copy `r4topupshock-049efe29` for seeds 3010-3019), never from `merged_r2`, whose mother directories can hold the non-identical second copy. The recomputed branch records are compared field-for-field with `results/report4_shock.json` and `report5_shock.json` (check above).",
        "- **Torch builds.** Cells come from rounds 3-5 and may have been produced on different torch builds (d122); the Go1 local mirrors do not keep `config.json`, so the build is not stated per Go1 cell here. The G1 `config.json` files are fetched into the cache (`results/supervisor_cache/g1/<job>/config.json`) but not tabulated.",
        "- **How data was fetched (read-only).** `restore_runs.py` was *not* used for the G1 claim set (it pulls every file of every run; a first attempt was stopped). Instead `analysis/fetch_supervisor_data.py` read, from R2 prefix `mjlab_ssp_centering/r6g1queue-0ede5636/`: `manifest.json`, the DONE markers (one list call), and for each finished run only `runs/<job>/eval.jsonl`, `final.json`, `config.json` (cached files skipped; log in `results/supervisor_cache/g1/fetch_log.json`; rerun with `--pull-g1`). CartPole artefacts were read (download only) from bucket-root `stage0_cartpole/eval_v3/dqn_{ssp,discrete,tile_coding}_none_v4_hparam_search/{train_runs.jsonl,train_summary.json}` and `stage0_cartpole/refilter_v2/dqn_{ssp_h2,ssp_h2raw,tile_h2}_none_v4_hparam_search/{final_eval_runs.jsonl,final_eval_summary.json}`. All Go1 data (including every `iter.jsonl` used in Fig. 4: E11, clip1, r5bias) was already in `results/merged_r2` / `merged_r4`; nothing was fetched for it. Nothing was written to or deleted from R2.",
        "- **CartPole x-axis.** Evaluation cadence is `1000 // 15 = 66` episodes, so the 15 checkpoints sit at 66, 132, ..., 990 (as in `plot_eval_v3`).",
        "- **ssp_h2 is two changes at once** (per-dimension input normalisation + length-scale range 0.01-1). Fig. 1b is therefore 'old vs new search', not 'range alone'; `ssp_h2raw` (range without normalisation, 469.8) is in the summary table for that reason (REPORT_006 R3).",
        "- **G1 completeness.** All 60 G1 claim jobs now carry DONE markers and every arm has 20 seeds, so Fig. 6 and table section 4 are claim grade on the study's 20-seed rule (fresh seeds 3000-3019, never pooled with the seed 1000-1005 pilots, d128). The figure title prints `complete: 20 per arm`; rerunning `--pull-g1` mid-campaign would print PILOT instead.",
        "- **Cached copies can be mid-upload.** One fetched G1 `eval.jsonl` (bias-init seed 3011) held 45 of the run's 50 evals while its `final.json` and the R2 copy both said 50, so the figure would have used 5 fewer evals for that seed and n would have read 19/20. `fetch_supervisor_data.py` now re-fetches any cached `eval.jsonl` whose row count is below the run's `final.json` `n_evals`; the current cache is complete. Anyone reusing the `results/merged_*` caches (built with `restore_runs.py`, which takes whatever copy it sees) should check the same thing.",
        "", "## Style decisions (conflicts between the existing scripts, resolved once)", "",
        "- Centering arms use the mjlab-report palette (uncentered red `#c0392b`, centered blue `#1f77b4`, bias-init green `#2ca02c`, clip-off orange), because REPORT_003-005 figures all use it. The CartPole scripts use the opposite red/blue assignment (none blue, value red); that was not adopted, so a reader moving from the reports to these figures sees the same colour for the same arm.",
        "- Representations (tile, SSP, old SSP, discrete) take the CartPole line styles (`REP_LINESTYLE`: discrete solid, SSP dashed, tile dash-dot) and pastel colours that do not collide with the centering colours (tile gold from `MODE_COLOR`, SSP and discrete from `ETA_COLOR_PALETTE`).",
        "- Greyscale: every arm has its own marker (uncentered o, centered s, bias-init ^, clip-off v, tile D, SSP P, discrete X), centering arms have their own line style on the main-effect figures and hatches on bars; in Fig. 5 the line style encodes shocked (solid) vs control (dashed) instead.",
        "- Bands: mean line with the arm colour at alpha 0.17 for the 95% bootstrap CI (seed resampling, 5000 draws, `plot_eval_v3.bootstrap_ci`), as in the existing scripts.",
        "", "## All sanity checks", "", "| check | source | report says | recomputed | |", "|---|---|---|---|---|"]
  L += [f"| {c['label']} | {c['source']} | {fmtv(c['expected'], c['fmt'])} | {fmtv(c['got'], c['fmt'])} | {'ok' if c['ok'] else 'DISAGREES'} |" for c in C]
  (OUT / "NOTES.md").write_text("\n".join(L) + "\n")
  print(f"NOTES.md: {len(C) - len(bad)}/{len(C)} checks agree" + ("" if not bad else f"; DISAGREE: {[c['label'] for c in bad]}"))


def write_captions(S: dict) -> None:
  d, c = S["dose"], S["contrast"]
  A, B = S["shock_A"], S["shock_B"]
  cp = S["cp"]
  m = S["mech"]
  ct_a, ct_b = A["contrast"], B["contrast"]
  g = S.get("g1")
  g1n = {a: len(g[a]) for a in g} if g else {}
  g1w = {a: sum(v["track"] >= WALK for v in g[a].values()) for a in g} if g else {}
  g1txt = ""
  if g:
    k = S["g1_contrast"]
    g1txt = (f"G1, gamma 0.99, offset +1, fall penalty 5.7, 2500 iterations, corrected protocol; mean tracking (`track_lin_frac`) per eval point with a 95% bootstrap CI over seeds; "
             f"n per arm = {g1n['none']} / {g1n['value']} / {g1n['bias']} (uncentered / centered / bias-init) from DONE runs only, walks {g1w['none']}/{g1n['none']}, {g1w['value']}/{g1n['value']}, {g1w['bias']}/{g1n['bias']}"
             f"{'' if all(v == 20 for v in g1n.values()) else '; the claim set is still running, so this is a pilot until all arms have 20 seeds'}. "
             f"Paired centered - uncentered {k['centered - uncentered']['mean']:+.3f} [{k['centered - uncentered']['boot'][0]:+.3f}, {k['centered - uncentered']['boot'][1]:+.3f}] (n={k['centered - uncentered']['n']} pairs).")
  txt = f"""# Captions

**Fig 1a (CartPole representations).** Linear DQN, no centering, episodic CartPole-v1; greedy evaluation return (mean of 10 episodes) at 15 checkpoints over 1,000 episodes, mean over n = 20 seeds (3000-3019) with a 95% bootstrap CI over seeds; dashed line = solved (475). Tile coding and SSP use the new (H2) search; discrete is the old search pipeline (RUN_004 replicate block, same seeds). Final means (last-5-checkpoint): tile {cp['tile_h2']['mean']:.1f}, SSP {cp['ssp_h2']['mean']:.1f}, discrete {cp['old_discrete_none_linear']['mean']:.1f}.

**Fig 1b (old vs new SSP search).** Same axes, n = 20 seeds (3000-3019) per line, 95% bootstrap CI over seeds. Old SSP (length scale searched 0.1-1, chosen {cp['old_ssp_none_linear']['length_scale']:.3f}) vs new SSP (0.01-1, chosen {cp['ssp_h2']['length_scale']:.3f}, with per-dimension input normalisation); final means {cp['old_ssp_none_linear']['mean']:.1f} -> {cp['ssp_h2']['mean']:.1f} (paired difference {S['cp_contrast']['ssp_h2 (new) - ssp old']['mean']:+.1f}, bootstrap 95% CI [{S['cp_contrast']['ssp_h2 (new) - ssp old']['ci'][0]:+.1f}, {S['cp_contrast']['ssp_h2 (new) - ssp old']['ci'][1]:+.1f}]); the same seeds were not used for tuning.

**Fig 2 (Go1 main effect).** Go1, gamma 0.999, 2000 iterations, corrected protocol (time-outs kept; falls reset in place and hidden from PPO); tracking (`track_lin_frac`) mean over n = 20 seeds (3000-3019) with a 95% bootstrap CI over seeds. Left: no offset (E12 + round-5 top-up); right: offset +1, where bias-init sets the critic's output bias to c/(1-gamma). Walks (last-5-eval mean >= 0.5) at +1: uncentered {d[('none', 1.0)]['walks']}/20, centered {d[('value', 1.0)]['walks']}/20, bias-init {d[('bias', 1.0)]['walks']}/20.

**Fig 3 (offset dose-response).** Go1, gamma 0.999, corrected protocol, n = 20 seeds (3000-3019) per point; left: fraction of seeds that walk with Wilson 95% intervals, right: mean final tracking (last-5-eval mean) with bootstrap 95% CIs over seeds. Uncentered walks {', '.join(f"{d[('none', o)]['walks']}/20" for o in OFFSETS)} and centered {', '.join(f"{d[('value', o)]['walks']}/20" for o in OFFSETS)} at c = 0, 0.1, 0.3, 1; bias-init {d[('bias', 0.1)]['walks']}/20 at c = 0.1 and {d[('bias', 1.0)]['walks']}/20 at c = 1. Paired centered - uncentered dtrack at c = 0.1: {c[('value-none', 0.1)]['mean']:+.3f} [{c[('value-none', 0.1)]['boot'][0]:+.3f}, {c[('value-none', 0.1)]['boot'][1]:+.3f}] (bootstrap).

**Fig 4 (mechanism).** Go1, offset +1, gamma 0.999, corrected protocol; ev_state (the state-explained share of the advantage, computed as in `mechanism_e11.py`) averaged over iterations 100-299 per seed, bars = mean over seeds with 95% bootstrap CIs, dots = seeds, labels = walking rate (last-5-eval tracking >= 0.5). n = {m[('none', 'ev_state')]['n']} uncentered (E11), {m[('clip', 'ev_state')]['n']} clip-off (pilot), {m[('bias', 'ev_state')]['n']} bias-init, {m[('value', 'ev_state')]['n']} centered (E11); right panel: const_frac over the same window.

**Fig 5 (shock test).** Go1, offset-0 mothers trained to iteration 1200, then branched into a shocked continuation (solid) and an unshocked control (dashed); pooled seeds 3000-3019, tracking mean with a 95% bootstrap CI over seeds, walking-mothers analysis (pre-registered filter: mean of the mother's last 3 evals before 1200 >= 0.8): uncentered n = {B['n_mothers']['none']['kept']}, centered n = {B['n_mothers']['value']['kept']}. Arm contrast (centered - uncentered, shocked minus control, t 95% CI): +1.0 offset {ct_b['off+1.0']['mean']:+.3f} [{ct_b['off+1.0']['ci'][0]:+.3f}, {ct_b['off+1.0']['ci'][1]:+.3f}] on the walking mothers and {ct_a['off+1.0']['mean']:+.3f} [{ct_a['off+1.0']['ci'][0]:+.3f}, {ct_a['off+1.0']['ci'][1]:+.3f}] over all 20 seeds; action-rate x5 {ct_b['actx5']['mean']:+.3f} [{ct_b['actx5']['ci'][0]:+.3f}, {ct_b['actx5']['ci'][1]:+.3f}] (walking) and {ct_a['actx5']['mean']:+.3f} [{ct_a['actx5']['ci'][0]:+.3f}, {ct_a['actx5']['ci'][1]:+.3f}] (all 20).

**Fig 6 (G1, second robot).** {g1txt if g1txt else 'G1 cache missing; run with --pull-g1.'}
"""
  (OUT / "captions.md").write_text(txt)
  print("wrote", OUT / "captions.md")


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--pull-g1", action="store_true", help="refresh the G1 claim runs from R2 (read-only) before plotting")
  ap.add_argument("--only", default="all", help="all | fig1a fig1b fig2 fig3 fig4 fig5 fig6 tables (tables always need every figure's statistics)")
  a = ap.parse_args()
  apply_style()
  OUT.mkdir(parents=True, exist_ok=True)
  if a.pull_g1:
    pull_g1()
  ensure_cartpole()
  S: dict = {"cp": {}}
  want = lambda n: a.only in ("all", n)
  # the figures also fill the statistics the tables need, so for 'tables'/'all' every figure function runs
  run_all = a.only in ("all", "tables")
  if run_all or a.only == "fig1a":
    fig1a(S)
  if run_all or a.only == "fig1b":
    fig1b(S)
  if run_all or a.only == "fig2":
    fig2(S)
  if run_all or a.only == "fig3":
    fig3(S)
  if run_all or a.only == "fig4":
    fig4(S)
  if run_all or a.only == "fig5":
    fig5(S)
  if run_all or a.only == "fig6":
    fig6(S)
  if run_all:
    rows, md = build_tables(S)
    (OUT / "summary_table.md").write_text("\n".join(md) + "\n")
    write_csv(rows)
    print("wrote", OUT / "summary_table.md", "+ .csv")
    write_captions(S)
    write_notes(S, checks(S))


if __name__ == "__main__":
  main()
