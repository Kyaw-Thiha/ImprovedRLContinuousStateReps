#!/usr/bin/env python3
"""RUN_004 (eval_v3) figures.

Reads the eval_v3 artifacts (policy pass, robustness, deployment, DQN-settings)
from the local mirror and writes one PNG plus a sidecar CSV per figure.

Style deliberately mirrors `experiments/plotting/ablation_plotting.py`:
  * centering -> color   (none blue, simple gold, value red)
  * representation -> line style (discrete solid, SSP dashed, tile coding dash-dot)
  * learning curves -> mean line + faint band (5th-95th percentile by default)
  * distributions -> pale box + thick median + per-seed dots + "% solved" labels

Usage
-----
    .venv/bin/python scripts/plot_eval_v3.py --local-root outputs/eval_v3 \
        --out-dir outputs/eval_v3/_figures            # all figures that have data
    .venv/bin/python scripts/plot_eval_v3.py --figures learning_curves,probe ...
    .venv/bin/python scripts/plot_eval_v3.py --list   # list figure names

Every figure is wrapped: a missing artifact or a plotting error is reported and the
remaining figures still run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

import eval_v3_layout as layout  # noqa: E402

SOLVED = 475.0
CEILING = 500.0  # CartPole-v1 caps an episode at 500 steps
LAST_K = 5
N_BOOT = 5000

# ---------------------------------------------------------------------------------------------
# style (mirrors experiments/plotting/ablation_plotting.py)
# ---------------------------------------------------------------------------------------------
MODE_COLOR = {"none": "#597fa8", "simple": "#ad8d44", "value": "#ab5d5d"}
MODE_BAND = {"none": "#8aaed1", "simple": "#e0c27a", "value": "#d78888"}
MODE_LABEL = {"none": "No Centering", "simple": "Simple Centering", "value": "Value Centering"}
REP_LINESTYLE = {"discrete": "-", "ssp": "--", "tile_coding": "-."}
REP_LABEL = {"discrete": "Discrete", "ssp": "SSP", "tile_coding": "TileCoding"}
ARCH_LABEL = {"linear": "DQN-linear", "deep": "DQN-deep", "a2c": "A2C"}
ARCHS = ("linear", "deep", "a2c")

# Selectable per-seed metric. `score` is the RUN_003/004 headline (mean of the last 5 greedy checkpoints x 10
# episodes); `final` is the 100-episode greedy eval of the FINAL saved policy (the same policy the robustness
# suite and deployment use); `online` is the paper-era metric (mean return of the last 100 training episodes,
# exploration on). They are NOT interchangeable: for the A2C SSP cells `score` reads ~25 points below `final`
# because the cell dips mid-run and recovers (see RUN_004 3.6 / 3.7).
METRICS = {
    "score": ("mean of the last 5 greedy checkpoints (10 episodes each)", ""),
    "final": ("greedy eval of the final policy (100 episodes)", "_finalpolicy"),
    "online": ("online return, last 100 training episodes (paper metric)", "_online"),
}
REPS = ("discrete", "ssp", "tile_coding")
CENTERINGS = ("none", "simple", "value")
# Family order for robustness plots (roughly increasing severity / interest).
FAMILY_ORDER = ["baseline", "pole_length", "pole_mass", "cart_mass", "gravity", "force",
                "obs_noise", "action_delay", "random_action", "wide_init"]

# RUN_003 section 4.1 (final eval, 20 seeds, greedy last-5 mean; filter mean in parens).
# Transcribed for the replicate / winner's-curse figures; override with --run003-json.
RUN003 = {
    "tile_coding/none/linear": (498.6, 3.6, 499.6), "tile_coding/value/deep": (498.0, 2.8, 499.7),
    "tile_coding/simple/linear": (496.3, 7.1, 498.6), "tile_coding/value/linear": (495.4, 6.6, 494.9),
    "tile_coding/none/deep": (489.7, 20.1, 493.9), "ssp/none/deep": (486.4, 21.4, 496.2),
    "tile_coding/simple/deep": (479.5, 18.4, 461.7), "ssp/value/deep": (474.6, 26.7, 491.7),
    "ssp/none/linear": (469.0, 30.7, 478.1), "ssp/value/linear": (468.7, 34.6, 492.5),
    "ssp/simple/deep": (363.6, 67.3, 388.4), "ssp/simple/linear": (311.5, 101.6, 317.0),
    "discrete/value/linear": (265.8, 62.2, 299.4), "discrete/none/linear": (260.4, 68.0, 299.5),
    "discrete/simple/deep": (259.4, 57.0, 294.6), "discrete/simple/linear": (255.1, 50.5, 263.6),
    "discrete/none/deep": (232.9, 53.8, 280.5), "discrete/value/deep": (220.8, 45.1, 273.9),
}


def apply_style():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight",
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9})


# ---------------------------------------------------------------------------------------------
# cell / study helpers
# ---------------------------------------------------------------------------------------------
def parse_cell(cell):
    rep, cen, arch = cell.split("/")
    return rep, cen, arch


def all_cells():
    return list(layout.ALL_CELLS) + [c for c in layout.A2C_CELLS if c not in layout.ALL_CELLS]


def study_of(cell):
    return layout.study_name(cell)


def cell_dir(local_root, cell, variant=None):
    """Local mirror directory of a cell, via layout.local_path (keys are never hand-written)."""
    return os.path.dirname(layout.local_path(layout.train_summary_key(study_of(cell), variant), local_root))


def sort_cells(cells):
    rep_ix = {r: i for i, r in enumerate(REPS)}
    cen_ix = {c: i for i, c in enumerate(CENTERINGS)}
    arch_ix = {"linear": 0, "deep": 1, "a2c": 2}
    return sorted(cells, key=lambda c: (rep_ix.get(parse_cell(c)[0], 9),
                                        cen_ix.get(parse_cell(c)[1], 9),
                                        arch_ix.get(parse_cell(c)[2], 9)))


def discover_cells(local_root):
    """Cells with a train_summary.json in the local mirror."""
    out = []
    for cell in all_cells():
        if os.path.exists(os.path.join(cell_dir(local_root, cell), "train_summary.json")):
            out.append(cell)
    return sort_cells(out)


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def load_cell(local_root, cell, variant=None):
    d = cell_dir(local_root, cell, variant)
    return {
        "cell": cell, "dir": d,
        "runs": load_jsonl(os.path.join(d, "train_runs.jsonl")),
        "summary": load_json(os.path.join(d, "train_summary.json")),
        "rob": load_jsonl(os.path.join(d, "robustness_runs.jsonl")),
        "deploy": load_json(os.path.join(d, "deployment.json")),
        "manifest": load_json(os.path.join(d, "manifest.json")),
    }


def variant_base(local_root, cell):
    """<local_root>/dqn_settings/<study> for the DQN-settings variants of a cell."""
    key = layout.train_summary_key(study_of(cell), "x")
    return os.path.dirname(os.path.dirname(layout.local_path(key, local_root)))


def variant_names(local_root, cell):
    base = variant_base(local_root, cell)
    if not os.path.isdir(base):
        return []
    return sorted(n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n)))


def load_variant(local_root, cell, variant):
    d = cell_dir(local_root, cell, variant)
    return {"runs": load_jsonl(os.path.join(d, "train_runs.jsonl")),
            "summary": load_json(os.path.join(d, "train_summary.json"))}


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def bootstrap_ci(xs, n_boot=N_BOOT, alpha=0.05, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    if xs.size == 0:
        return (float("nan"), float("nan"))
    if xs.size == 1:
        return (float(xs[0]), float(xs[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, xs.size, (n_boot, xs.size))
    means = xs[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def ratio_ci(num, den, n_boot=N_BOOT, alpha=0.05, seed=0):
    """95% bootstrap CI of mean(num) / mean(den), seed-resampled independently (the convention used by
    scripts/analyze_robustness.py::bootstrap_diff with a_den/b_den). Returns (lo, hi)."""
    num = np.asarray([x for x in num if x is not None and np.isfinite(x)], dtype=float)
    den = np.asarray([x for x in den if x is not None and np.isfinite(x)], dtype=float)
    if num.size == 0 or den.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = num[rng.integers(0, num.size, (n_boot, num.size))].mean(axis=1)
    b = den[rng.integers(0, den.size, (n_boot, den.size))].mean(axis=1)
    r = np.where(b != 0, a / b, np.nan)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(r, 100 * alpha / 2)), float(np.percentile(r, 100 * (1 - alpha / 2))))


def band(ys, mode="p5_95"):
    """ys: (n_seeds, n_x) -> (mean, lo, hi) with nans ignored per column."""
    ys = np.asarray(ys, dtype=float)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(ys, axis=0)
        n = np.sum(np.isfinite(ys), axis=0)
        sd = np.nanstd(ys, axis=0)
    if mode == "p5_95":
        lo = np.nanpercentile(ys, 5, axis=0)
        hi = np.nanpercentile(ys, 95, axis=0)
    elif mode == "sem":
        se = sd / np.sqrt(np.maximum(n, 1))
        lo, hi = mean - se, mean + se
    else:  # std
        lo, hi = mean - sd, mean + sd
    return mean, lo, hi


def eval_freq_of(summary, runs):
    p = (summary or {}).get("params", {})
    trials = int(p.get("trials", 1000) or 1000)
    n_ck = max((len(r.get("eval_history", [])) for r in runs), default=0)
    if n_ck <= 0:
        return trials // 15
    return max(1, trials // n_ck)


def stack_history(runs, key="eval_history"):
    hists = [r.get(key) or [] for r in runs]
    n = min((len(h) for h in hists if h), default=0)
    return np.array([h[:n] for h in hists if len(h) >= n], dtype=float) if n else np.zeros((0, 0))


def stability_stats(runs, solved=SOLVED):
    ever = stay = 0
    first_reach, share_after, drops = [], [], []
    n = 0
    for r in runs:
        h = np.asarray(r.get("eval_history") or [], dtype=float)
        if h.size == 0:
            continue
        n += 1
        if h.max() >= solved:
            ever += 1
            i0 = int(np.argmax(h >= solved))
            first_reach.append(i0)
            post = h[i0:]
            share_after.append(float((post >= solved).mean()))
            if bool((post >= solved).all()):
                stay += 1
            peak = np.maximum.accumulate(h)
            drops.append(float(np.max(peak - h)))
        else:
            drops.append(float(np.max(np.maximum.accumulate(h) - h)))
    return {"n": n, "ever": ever, "stay": stay, "first_reach": first_reach,
            "share_after": share_after, "drops": drops}


def score_of(run):
    v = run.get("score")
    return float(v) if v is not None and np.isfinite(v) else None


def pass_scores(runs, last_k=LAST_K):
    out = []
    for r in runs:
        s = score_of(r)
        if s is not None:
            continue
        h = np.asarray(r.get("eval_history") or [], dtype=float)
        if h.size:
            out.append(float(h[-last_k:].mean()))
    return out


def cell_scores(runs):
    return cell_metric(runs, "score")


def cell_metric(runs, metric="score"):
    """Per-seed values of the selected metric (see METRICS); missing values are dropped."""
    xs = []
    for r in runs:
        if metric == "final":
            v = r.get("final_eval_reward")
        elif metric == "online":
            v = r.get("terminal_reward")
        else:
            v = score_of(r)
            if v is None and r.get("eval_history"):
                v = float(np.mean(r["eval_history"][-LAST_K:]))
        if v is not None and np.isfinite(v):
            xs.append(float(v))
    return xs


def cond_level(cond):
    """Numeric level of a perturbation condition (baseline -> 1.0)."""
    if cond == "baseline":
        return 1.0
    tail = cond.split("_")[-1].lstrip("x")
    try:
        return float(tail)
    except ValueError:
        return None


def family_scores(rob_rows, cell):
    """{family: score} = mean over the family's conditions of the seed means, and retained vs baseline."""
    by_fam = {}
    for r in rob_rows:
        by_fam.setdefault(r.get("family", "baseline"), []).append(float(r["mean"]))
    fam = {k: float(np.mean(v)) for k, v in by_fam.items()}
    base = fam.get("baseline", float("nan"))
    ret = {k: (v / base if base and np.isfinite(base) and base != 0 else float("nan")) for k, v in fam.items()}
    return fam, ret, base


# ---------------------------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------------------------
def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def save_fig(fig, out_dir, name, csv_header=None, csv_rows=None):
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(png)
    plt.close(fig)
    if csv_header is not None:
        save_csv(os.path.join(out_dir, f"{name}.csv"), csv_header, csv_rows or [])
    print(f"[plot] {png}", flush=True)


def legend_by_rep_mode(cells, loc="best", ncol=2, ax=None, **kw):
    """One legend entry per (representation, centering) pair: several cells share a rep/centering (the
    architectures), so the raw cell list would repeat every label once per architecture."""
    seen, handles = set(), []
    for cell in cells:
        rep, cen, _ = parse_cell(cell)
        label = f"{REP_LABEL[rep]} / {MODE_LABEL[cen]}"
        if label in seen:
            continue
        seen.add(label)
        handles.append(plt.Line2D([], [], color=MODE_COLOR[cen], ls=REP_LINESTYLE[rep],
                                  lw=2, label=label))
    (plt.legend if ax is None else ax.legend)(handles=handles, loc=loc, ncol=ncol, **kw)


# ---------------------------------------------------------------------------------------------
# figures: policy pass
# ---------------------------------------------------------------------------------------------
def fig_learning_curves(data, out_dir, band_mode="p5_95", **kw):
    cells = [c for c in data if data[c]["runs"]]
    if not cells:
        raise FileNotFoundError("no train_runs.jsonl")
    groups = [("linear", [c for c in cells if parse_cell(c)[2] == "linear"]),
              ("deep", [c for c in cells if parse_cell(c)[2] == "deep"]),
              ("a2c", [c for c in cells if parse_cell(c)[2] == "a2c"])]
    groups = [(n, cs) for n, cs in groups if cs]
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 5.6), squeeze=False,
                             sharey=True, constrained_layout=True)
    rows = []
    for ax, (gname, gcells) in zip(axes[0], groups):
        for cell in gcells:
            d = data[cell]
            ys = stack_history(d["runs"])
            if ys.size == 0:
                continue
            ef = eval_freq_of(d["summary"], d["runs"])
            x = (np.arange(ys.shape[1]) + 1) * ef
            mean, lo, hi = band(ys, band_mode)
            rep, cen, _ = parse_cell(cell)
            ax.plot(x, mean, color=MODE_COLOR[cen], ls=REP_LINESTYLE[rep], lw=1.8)
            ax.fill_between(x, lo, hi, color=MODE_BAND[cen], alpha=0.10, lw=0)
            for xi, m, l, h in zip(x, mean, lo, hi):
                rows.append([cell, int(xi), float(m), float(l), float(h)])
        ax.axhline(SOLVED, color="#222222", lw=1.2, ls="--", alpha=0.8)
        ax.set_title(f"{ARCH_LABEL.get(gname, gname)}")
        ax.set_xlabel("Episode")
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"Greedy return ({LAST_K}-checkpoint mean)")
    fig.suptitle("Policy pass: learning curves"
                 + (" (5th-95th pct band)" if band_mode == "p5_95" else f" ({band_mode} band)"), fontsize=13)
    legend_by_rep_mode(cells, loc="lower right", ncol=3, fontsize=7.5, ax=axes[0][0])
    save_fig(fig, out_dir, "learning_curves", ["cell", "episode", "mean", "lo", "hi"], rows)


def fig_spaghetti(data, out_dir, cells=None, **kw):
    cells = cells or [c for c in data if data[c]["runs"]]
    rep_ix = {r: i for i, r in enumerate(REPS)}
    cen_ix = {c: i for i, c in enumerate(CENTERINGS)}
    cells = sorted(cells, key=lambda c: (rep_ix.get(parse_cell(c)[0], 9), cen_ix.get(parse_cell(c)[1], 9)))[:6]
    nrow, ncol = int(np.ceil(len(cells) / 3)), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    rows = []
    for ax, cell in zip(axes.ravel(), cells):
        d = data[cell]
        ys = stack_history(d["runs"])
        if ys.size == 0:
            continue
        ef = eval_freq_of(d["summary"], d["runs"])
        x = (np.arange(ys.shape[1]) + 1) * ef
        rep, cen, _ = parse_cell(cell)
        ax.plot(x, ys.T, color=MODE_COLOR[cen], lw=0.9, alpha=0.55)
        ax.plot(x, np.nanmean(ys, axis=0), color=MODE_COLOR[cen], lw=2.4)
        ax.axhline(SOLVED, color="#222222", lw=1.0, ls="--", alpha=0.7)
        ax.set_title(f"{REP_LABEL[rep]} / {MODE_LABEL[cen]} / {ARCH_LABEL.get(parse_cell(cell)[2])}", fontsize=10)
        ax.set_ylim(0, 505)
        for si, seed in enumerate([r.get("seed") for r in d["runs"]]):
            for xi, v in zip(x, ys[si]):
                rows.append([cell, seed, int(xi), float(v)])
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    fig.suptitle("Policy pass: per-seed greedy trajectories", fontsize=13)
    save_fig(fig, out_dir, "spaghetti", ["cell", "seed", "episode", "score"], rows)


def fig_final_score_boxes(data, out_dir, arch=None, metric="score", **kw):
    cells = [c for c in data if data[c]["runs"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no cells for this arch")
    cells = sort_cells(cells)
    xs, rows = [], []
    fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(cells) + 3.5), 5.6), constrained_layout=True)
    for i, cell in enumerate(cells):
        sc = cell_metric(data[cell]["runs"], metric)
        rep, cen, _ = parse_cell(cell)
        bp = ax.boxplot([sc], positions=[i], widths=0.55, patch_artist=True, showfliers=True,
                        medianprops=dict(color="black", lw=2.2),
                        boxprops=dict(facecolor=MODE_BAND[cen], alpha=0.35, edgecolor=MODE_COLOR[cen]),
                        whiskerprops=dict(color=MODE_COLOR[cen]), capprops=dict(color=MODE_COLOR[cen]),
                        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="#444444", ms=5))
        rng = np.random.default_rng(0)
        ax.scatter(np.full(len(sc), i) + rng.uniform(-0.12, 0.12, len(sc)), sc,
                   color=MODE_COLOR[cen], alpha=0.75, s=22, zorder=3)
        # mean with its 95% bootstrap CI (the box shows the seed distribution; this shows the uncertainty
        # on the mean the cell is quoted with)
        mean = float(np.mean(sc)) if sc else float("nan")
        ci_lo, ci_hi = bootstrap_ci(sc)
        if np.isfinite(mean):
            ax.errorbar(i, mean, yerr=[[mean - ci_lo], [ci_hi - mean]], fmt="D", color="black", ms=4.5,
                        lw=1.4, capsize=3.5, zorder=5,
                        label="mean with 95% bootstrap CI" if i == 0 else None)
        solved = float(np.mean([s >= SOLVED for s in sc])) if sc else float("nan")
        ax.text(i, 512, f"{100 * solved:.0f}%", ha="center", fontsize=8, color="#333333")
        xs.append(f"{REP_LABEL[rep]}\n{MODE_LABEL[cen]}")
        rows.append([cell, len(sc), mean, float(np.std(sc)), solved, ci_lo, ci_hi, metric])
    ax.axhline(SOLVED, color="#222222", lw=1.2, ls="--", alpha=0.8)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 525)
    mlabel, msuffix = METRICS[metric]
    ax.set_ylabel(f"Greedy return ({mlabel})")
    ax.set_title(f"Policy pass by cell ({ARCH_LABEL.get(arch, 'all')}) — {mlabel}; "
                 f"% solved at {SOLVED:.0f}; black diamond = mean with 95% bootstrap CI")
    if cells:
        ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, out_dir, f"final_score_boxes_{arch or 'all'}{msuffix}",
             ["cell", "n", "mean", "std", "solved_share", "ci_lo", "ci_hi", "metric"], rows)


def fig_replicate(data, out_dir, run003=None, **kw):
    run003 = run003 or RUN003
    cells = [c for c in data if c in run003 and data[c]["runs"]]
    if not cells:
        raise FileNotFoundError("no overlap with RUN_003 cells")
    fig, ax = plt.subplots(figsize=(6.6, 6.2), constrained_layout=True)
    rows, lo_all, hi_all = [], [], []
    for cell in sort_cells(cells):
        sc = cell_scores(data[cell]["runs"])
        m, s = float(np.mean(sc)), float(np.std(sc))
        lo, hi = bootstrap_ci(sc)
        m3, s3, _f = run003[cell]
        rep, cen, arch = parse_cell(cell)
        marker = {"linear": "o", "deep": "^", "a2c": "s"}[arch]
        ax.errorbar(float(m3), float(m), xerr=float(s3 / np.sqrt(max(len(sc), 1))),
                    yerr=[[float(m - lo)], [float(hi - m)]],
                    fmt=marker, ms=7, color=MODE_COLOR[cen], ecolor=MODE_COLOR[cen], alpha=0.85,
                    markeredgecolor="white", markeredgewidth=0.6, ls="none")
        rows.append([cell, m3, s3, m, s, lo, hi, m - m3])
        lo_all.append(min(m3, m)); hi_all.append(max(m3, m))
    lo, hi = min(lo_all) - 15, max(hi_all) + 15
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.2, ls=":")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("RUN_003 final eval mean (seeds 2000-2019)")
    ax.set_ylabel(f"Policy pass mean (seeds 3000-3019)")
    ax.set_title("Replicate check: policy pass vs RUN_003")
    handles = [plt.Line2D([], [], marker=m, ls="none", color="#555555", label=n)
               for n, m in (("linear", "o"), ("deep", "^"), ("A2C", "s"))]
    ax.legend(handles=handles, loc="upper left")
    save_fig(fig, out_dir, "replicate_vs_run003",
             ["cell", "run003_mean", "run003_std", "pass_mean", "pass_std", "pass_lo", "pass_hi", "diff"], rows)


def fig_winners_curse(data, out_dir, run003=None, **kw):
    run003 = run003 or RUN003
    cells = [c for c in data if c in run003 and data[c]["runs"]]
    if not cells:
        raise FileNotFoundError("no overlap with RUN_003 cells")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)
    for ax, (arch, title) in zip(axes, (("linear", "linear"), ("deep", "deep"))):
        cs = sort_cells([c for c in cells if parse_cell(c)[2] == arch])
        fam, fin, pas = [], [], []
        for cell in cs:
            m3, _s3, filt = run003[cell]
            rep, cen, _ = parse_cell(cell)
            fam.append(filt); fin.append(m3); pas.append(float(np.mean(cell_scores(data[cell]["runs"]))))
            rows.append([cell, filt, m3, pas[-1], filt - m3, pas[-1] - m3])
        x = np.arange(len(cs))
        ax.plot(x, fam, "o-", color="#888888", ms=6, lw=1.2, label="Filter mean (RUN_003, 10 seeds)")
        ax.plot(x, fin, "s-", color="#333333", ms=6, lw=1.2, label="Final eval (RUN_003, 20 seeds)")
        ax.plot(x, pas, "D--", color="#c0392b", ms=5, lw=1.2, label="Policy pass (seeds 3000+)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]]}\n{MODE_LABEL[parse_cell(c)[1]]}" for c in cs],
                           rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"Winner's curse — {title}")
        ax.set_ylim(180, 515)
    axes[0].set_ylabel("Greedy return (last-5 mean)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Winner's curse: filter mean vs RUN_003 final eval vs this pass", fontsize=13)
    save_fig(fig, out_dir, "winners_curse",
             ["cell", "filter_mean", "run003_mean", "pass_mean", "curse", "pass_minus_run003"], rows)


def fig_stability(data, out_dir, arch=None, **kw):
    cells = [c for c in data if data[c]["runs"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no cells")
    cells = sort_cells(cells)
    rows, stats = [], {c: stability_stats(data[c]["runs"]) for c in cells}
    fig, axes = plt.subplots(1, 2, figsize=(max(12.0, 0.52 * len(cells) + 5), 5.0), constrained_layout=True)
    x = np.arange(len(cells))
    for ax, key, title, ylab in ((axes[0], "drops", "Learn-then-collapse size", "Mean largest drop from own peak"),
                                 (axes[1], "stay", "Seeds staying solved", f"Share of seeds all-ckpt >= {SOLVED:.0f}")):
        for i, cell in enumerate(cells):
            rep, cen, arch_ = parse_cell(cell)
            if key == "drops":
                v = float(np.mean(stats[cell]["drops"])) if stats[cell]["drops"] else float("nan")
                ax.bar(i, v, color=MODE_BAND[cen], edgecolor=MODE_COLOR[cen], alpha=0.8)
            else:
                n = max(stats[cell]["n"], 1)
                v = stats[cell]["stay"] / n
                ax.bar(i, v, color=MODE_BAND[cen], edgecolor=MODE_COLOR[cen], alpha=0.8)
                ax.text(i, v + 0.02, f"{stats[cell]['stay']}/{n}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]][:4]}\n{MODE_LABEL[parse_cell(c)[1]][:3]}\n{ARCH_LABEL[parse_cell(c)[2]]}"
                            for c in cells], fontsize=6.5)
        ax.set_title(title); ax.set_ylabel(ylab)
    for cell in cells:
        s = stats[cell]
        rows.append([cell, s["n"], s["ever"], s["stay"],
                     float(np.mean(s["drops"])) if s["drops"] else float("nan"),
                     float(np.mean(s["share_after"])) if s["share_after"] else float("nan")])
    save_fig(fig, out_dir, f"stability_{arch or 'all'}",
             ["cell", "n_seeds", "ever_solved", "stays_solved", "mean_largest_drop", "mean_share_ckpt_solved"], rows)


def fig_sample_efficiency(data, out_dir, arch=None, **kw):
    cells = [c for c in data if data[c]["runs"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no cells")
    cells = sort_cells(cells)
    rows = []
    fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(cells) + 3.5), 5.4), constrained_layout=True)
    for i, cell in enumerate(cells):
        st = stability_stats(data[cell]["runs"])
        ef = eval_freq_of(data[cell]["summary"], data[cell]["runs"])
        ep = sorted((k + 1) * ef for k in st["first_reach"])
        rep, cen, _ = parse_cell(cell)
        if ep:
            q1, med, q3 = np.percentile(ep, [25, 50, 75])
            ax.bar(i, med, color=MODE_BAND[cen], edgecolor=MODE_COLOR[cen], alpha=0.8, width=0.6)
            ax.plot([i, i], [q1, q3], color="#333333", lw=2.0)
            ax.scatter(np.full(len(ep), i), ep, color=MODE_COLOR[cen], s=14, alpha=0.7, zorder=3)
        else:
            ax.bar(i, 0, color="#dddddd", width=0.6)
        n = max(st["n"], 1)
        ax.text(i, -60, f"{st['ever']}/{n}", ha="center", fontsize=7.5, color="#333333")
        rows.append([cell, st["n"], st["ever"], float(np.median(ep)) if ep else float("nan"),
                     float(np.percentile(ep, 25)) if ep else float("nan"),
                     float(np.percentile(ep, 75)) if ep else float("nan")])
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]]}\n{MODE_LABEL[parse_cell(c)[1]]}\n{ARCH_LABEL[parse_cell(c)[2]]}"
                        for c in cells], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"Episode of first checkpoint >= {SOLVED:.0f}")
    ax.set_title("Sample efficiency: time to first solve (bar = median, line = IQR, dots = seeds)")
    save_fig(fig, out_dir, f"sample_efficiency_{arch or 'all'}",
             ["cell", "n_seeds", "ever_solved", "median_episode", "q25_episode", "q75_episode"], rows)


def fig_probe(data, out_dir, arch=None, **kw):
    cells = [c for c in data if data[c]["runs"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no cells")
    cells = sort_cells(cells)
    is_a2c = any(parse_cell(c)[2] == "a2c" for c in cells)
    keys = ["mean_max_q", "mean_action_gap", "avg_reward"] + (["mean_value"] if is_a2c else [])
    titles = {"mean_max_q": "Probe mean max-Q (preference for A2C)",
              "mean_action_gap": "Probe mean action gap", "avg_reward": "Centering R-bar (avg_reward)",
              "mean_value": "Probe mean V"}
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 4.8), squeeze=False, constrained_layout=True)
    rows = []
    for ax, key in zip(axes[0], keys):
        plotted = False
        for cell in cells:
            d = data[cell]
            hists = [r.get("probe_history") or [] for r in d["runs"]]
            n = min((len(h) for h in hists if h), default=0)
            if n == 0:
                continue
            ys = np.array([[ph.get(key, np.nan) for ph in h[:n]] for h in hists], dtype=float)
            xs = np.array([ph.get("episode") for ph in hists[0][:n]], dtype=float)
            rep, cen, _ = parse_cell(cell)
            ax.plot(xs, np.nanmean(ys, axis=0), color=MODE_COLOR[cen], ls=REP_LINESTYLE[rep], lw=1.7)
            plotted = True
            for xi, v in zip(xs, np.nanmean(ys, axis=0)):
                rows.append([cell, key, float(xi), float(v)])
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("Episode")
        if not plotted:
            ax.axis("off")
    fig.suptitle(f"Probe diagnostics ({ARCH_LABEL.get(arch or '', 'all')})", fontsize=13)
    legend_by_rep_mode(cells, loc="best", ncol=1)
    save_fig(fig, out_dir, f"probe_{arch or 'all'}", ["cell", "metric", "episode", "value"], rows)


def fig_cost_scatter(data, out_dir, **kw):
    cells = [c for c in data if data[c]["runs"] and data[c]["summary"]]
    if not cells:
        raise FileNotFoundError("no summaries")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    specs = [("n_params", "Parameters (log)", True), ("dimensionality", "Feature dimension", False),
             ("wall_min", "Wall minutes (20 seeds, contended)", False)]
    rows = []
    for ax, (key, xlabel, logx) in zip(axes, specs):
        for cell in sort_cells(cells):
            d = data[cell]
            cost = (d["summary"].get("cost") or {})
            v = cost.get(key)
            if v is None and key == "wall_min":
                v = d["summary"].get("wall_min")
            sc = cell_scores(d["runs"])
            if v is None or not sc:
                continue
            rep, cen, arch = parse_cell(cell)
            marker = {"linear": "o", "deep": "^", "a2c": "s"}[arch]
            ax.scatter(v, float(np.mean(sc)), marker=marker, s=46, color=MODE_COLOR[cen],
                       edgecolor="white", lw=0.6, alpha=0.9)
            rows.append([cell, key, float(v), float(np.mean(sc))])
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Policy pass mean score")
        ax.axhline(SOLVED, color="#222222", lw=1.0, ls="--", alpha=0.7)
    handles = [plt.Line2D([], [], marker=m, ls="none", color="#555555", label=n)
               for n, m in (("linear", "o"), ("deep", "^"), ("A2C", "s"))]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8)
    fig.suptitle("Cost vs score (descriptive; winner configs come from non-overlapping search ranges)", fontsize=12)
    save_fig(fig, out_dir, "cost_scatter", ["cell", "metric", "value", "mean_score"], rows)


# ---------------------------------------------------------------------------------------------
# figures: robustness
# ---------------------------------------------------------------------------------------------
def fig_robustness_curves(data, out_dir, arch=None, band_mode="std", **kw):
    cells = [c for c in data if data[c]["rob"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no robustness_runs.jsonl")
    cells = sort_cells(cells)
    fams = [f for f in FAMILY_ORDER if any(f in {r.get("family") for r in data[c]["rob"]} for c in cells)
            and f != "baseline"]
    fig, axes = plt.subplots(3, max(1, int(np.ceil(len(fams) / 3))), figsize=(4.3 * max(1, int(np.ceil(len(fams) / 3))), 11.0),
                             squeeze=False, constrained_layout=True)
    rows = []
    for ax, fam in zip(axes.ravel(), fams):
        for cell in cells:
            rob = [r for r in data[cell]["rob"] if r.get("family") == fam]
            if not rob:
                continue
            levels = sorted({cond_level(r["condition"]) for r in rob if cond_level(r["condition"]) is not None})
            if not levels:
                continue
            xs, means, los, his = [], [], [], []
            for lv in levels:
                vals = [float(r["mean"]) for r in rob if cond_level(r["condition"]) == lv]
                m, lo, hi = band(np.array([vals]), band_mode)
                xs.append(lv); means.append(float(m[0])); los.append(float(lo[0])); his.append(float(hi[0]))
                rows.append([cell, fam, lv, float(m[0]), float(lo[0]), float(hi[0])])
            rep, cen, _ = parse_cell(cell)
            ax.plot(xs, means, color=MODE_COLOR[cen], ls=REP_LINESTYLE[rep], lw=1.7, marker="o", ms=4)
            ax.fill_between(xs, los, his, color=MODE_BAND[cen], alpha=0.10, lw=0)
        ax.set_title(fam, fontsize=10)
        ax.set_xlabel("perturbation factor")
        ax.set_ylabel("Greedy return")
        ax.set_xscale("log")
    for ax in axes.ravel()[len(fams):]:
        ax.axis("off")
    fig.suptitle(f"Robustness: degradation by family ({ARCH_LABEL.get(arch or '', 'all')})", fontsize=13)
    legend_by_rep_mode(cells, loc="best", ncol=1)
    save_fig(fig, out_dir, f"robustness_curves_{arch or 'all'}",
             ["cell", "family", "level", "mean", "lo", "hi"], rows)


def fig_robustness_heatmap(data, out_dir, arch=None, **kw):
    cells = [c for c in data if data[c]["rob"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no robustness_runs.jsonl")
    cells = sort_cells(cells)
    conds = ["baseline"] + [c for f in FAMILY_ORDER if f != "baseline"
                            for c in layout_conditions_in_order(f, data, cells)]
    conds = [c for c in conds if any(any(r["condition"] == c for r in data[cell]["rob"]) for cell in cells)]
    M = np.full((len(cells), len(conds)), np.nan)
    TIE = np.zeros((len(cells), len(conds)), dtype=bool)   # 95% CI of the ratio includes 1.0
    Lo = np.full((len(cells), len(conds)), np.nan)
    Hi = np.full((len(cells), len(conds)), np.nan)
    for i, cell in enumerate(cells):
        base_vals = [r["mean"] for r in data[cell]["rob"] if r["condition"] == "baseline"]
        base = float(np.mean(base_vals)) if base_vals else 0.0
        for j, cond in enumerate(conds):
            vals = [float(r["mean"]) for r in data[cell]["rob"] if r["condition"] == cond]
            if vals:
                M[i, j] = float(np.mean(vals)) / base if base else np.nan
            if vals and base and cond != "baseline":
                lo, hi = ratio_ci(vals, base_vals, seed=abs(hash((cell, cond))) % (2 ** 31))
                Lo[i, j], Hi[i, j] = lo, hi
                TIE[i, j] = np.isfinite(lo) and np.isfinite(hi) and lo <= 1.0 <= hi
    fig, ax = plt.subplots(figsize=(max(9.0, 0.42 * len(conds) + 4), max(4.0, 0.30 * len(cells) + 2.5)),
                           constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ti, tj = np.where(TIE)
    if len(ti):
        ax.scatter(tj, ti, s=16, facecolors="none", edgecolors="#222222", linewidths=0.9, zorder=3)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"{REP_LABEL[parse_cell(c)[0]]}/{MODE_LABEL[parse_cell(c)[1]][:3]}\n"
                        f"{ARCH_LABEL[parse_cell(c)[2]]}" for c in cells], fontsize=7)
    fig.colorbar(im, ax=ax, label="retained (family mean / baseline)")
    ax.set_title(f"Robustness retention ({ARCH_LABEL.get(arch or '', 'all')})\n"
                 f"baseline = this cell's own greedy baseline on the FINAL saved policies (not the "
                 f"last-5-checkpoint pass metric);\n"
                 f"open circle = 95% bootstrap CI of the ratio includes 1.0 (indistinguishable from that "
                 f"cell's own baseline)")
    save_fig(fig, out_dir, f"robustness_heatmap_{arch or 'all'}",
             ["cell"] + conds, [[c] + ["" if not np.isfinite(M[i, j]) else round(float(M[i, j]), 4)
                                       for j in range(len(conds))] for i, c in enumerate(cells)])
    save_csv(os.path.join(out_dir, f"robustness_heatmap_{arch or 'all'}_ci.csv"),
             ["cell", "condition", "retained", "ci_lo", "ci_hi", "ci_includes_1"],
             [[cells[i], conds[j], round(float(M[i, j]), 4), round(float(Lo[i, j]), 4),
               round(float(Hi[i, j]), 4), bool(TIE[i, j])]
              for i in range(len(cells)) for j in range(len(conds)) if np.isfinite(M[i, j])])


def layout_conditions_in_order(fam, data, cells):
    seen = {}
    for cell in cells:
        for r in data[cell]["rob"]:
            if r.get("family") == fam:
                seen[r["condition"]] = cond_level(r["condition"])
    return [c for c, _ in sorted(seen.items(), key=lambda kv: (kv[1] is None, kv[1]))]


def fig_robustness_worst(data, out_dir, arch=None, **kw):
    cells = [c for c in data if data[c]["rob"] and (arch is None or parse_cell(c)[2] == arch)]
    if not cells:
        raise FileNotFoundError("no robustness_runs.jsonl")
    cells = sort_cells(cells)
    rows = []
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(cells) + 3.5), 5.2), constrained_layout=True)
    for i, cell in enumerate(cells):
        per_seed = {}
        for r in data[cell]["rob"]:
            fam = r.get("family")
            if fam == "baseline":
                continue
            per_seed.setdefault(r["seed"], {}).setdefault(fam, []).append(float(r["mean"]))
        worst = []
        for seed, fams in per_seed.items():
            fam_means = {f: float(np.mean(v)) for f, v in fams.items()}
            if not fam_means:
                continue
            wf = min(fam_means, key=fam_means.get)
            worst.append(fam_means[wf])
            rows.append([cell, seed, wf, fam_means[wf]])
        rep, cen, _ = parse_cell(cell)
        ax.scatter(np.full(len(worst), i) + np.random.default_rng(1).uniform(-0.13, 0.13, len(worst)), worst,
                   color=MODE_COLOR[cen], s=20, alpha=0.75)
        if worst:
            ax.plot([i - 0.28, i + 0.28], [np.median(worst)] * 2, color="#333333", lw=2.0)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]]}\n{MODE_LABEL[parse_cell(c)[1]][:3]}\n{ARCH_LABEL[parse_cell(c)[2]]}"
                        for c in cells], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Worst-family mean return")
    ax.set_title(f"Worst-condition performance per seed ({ARCH_LABEL.get(arch or '', 'all')})")
    save_fig(fig, out_dir, f"robustness_worst_{arch or 'all'}", ["cell", "seed", "worst_family", "worst_mean"], rows)


def fig_robustness_forest(data, out_dir, analysis_path=None, **kw):
    a = load_json(analysis_path)
    if not a:
        raise FileNotFoundError(f"no analysis json at {analysis_path}")
    comps = a.get("comparisons") or []
    if not comps:
        raise ValueError("analysis json has no comparisons (are both cells of a primary pair present?)")
    keys = set(comps[0].keys())
    def pick(*names):
        for n in names:
            if n in keys:
                return n
        return None
    k_pair = pick("pair", "cells", "label")
    k_fam = pick("family", "fam")
    k_metric = pick("metric")
    k_diff = pick("diff", "mean_diff", "delta")
    k_lo = pick("lo", "ci_lo", "diff_lo")
    k_hi = pick("hi", "ci_hi", "diff_hi")
    k_ph = pick("p_holm", "holm_p", "p_adjusted")
    k_pr = pick("p_raw", "p", "p_value")
    missing = [n for n, v in (("pair", k_pair), ("metric", k_metric), ("diff", k_diff),
                              ("lo", k_lo), ("hi", k_hi)) if v is None]
    if missing:
        raise ValueError(f"unexpected comparison schema; missing {missing}; available={sorted(keys)}")
    metrics = sorted({c[k_metric] for c in comps})
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), max(4.0, 0.42 * len(comps) / len(metrics) + 2)),
                             squeeze=False, constrained_layout=True)
    rows = []
    for ax, metric in zip(axes[0], metrics):
        cs = [c for c in comps if c[k_metric] == metric]
        for i, c in enumerate(cs):
            d, lo, hi = float(c[k_diff]), float(c[k_lo]), float(c[k_hi])
            holm = float(c[k_ph]) if k_ph and c.get(k_ph) is not None else float("nan")
            raw = float(c[k_pr]) if k_pr and c.get(k_pr) is not None else float("nan")
            row = (c[k_fam] if k_fam else "")
            sig = bool(c.get("significant")) if "significant" in keys else (np.isfinite(holm) and holm < 0.05)
            color = "#c0392b" if sig else "#666666"
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", ms=5, color=color,
                        ecolor=color, lw=1.4, alpha=0.9, markeredgecolor="white", markeredgewidth=0.5)
            rows.append([row, metric, d, lo, hi, raw, holm, int(sig), c.get("label", "")])
        ax.axvline(0.0, color="#222222", lw=1.0, ls="--", alpha=0.8)
        ax.set_yticks(range(len(cs)))
        ax.set_yticklabels([f"{c[k_fam] if k_fam else i} | {c.get('label', '')}" for i, c in enumerate(cs)], fontsize=7)
        ax.set_title(f"tile_coding - ssp: {metric}")
        ax.set_xlabel("difference in score (95% CI; red = significant after Holm)")
    fig.suptitle("Primary robustness comparisons (Holm-adjusted within each metric)", fontsize=12)
    save_fig(fig, out_dir, "robustness_forest",
             ["family", "metric", "diff", "lo", "hi", "p_raw", "p_holm", "significant", "label"], rows)


# ---------------------------------------------------------------------------------------------
# figures: deployment
# ---------------------------------------------------------------------------------------------
def _size_total(dep):
    try:
        return float(dep["size"]["total"]["fp32"])
    except Exception:  # noqa: BLE001
        return None


def fig_deployment_size(data, out_dir, **kw):
    have = [c for c in data if data[c]["deploy"] and data[c]["runs"]]
    if not have:
        raise FileNotFoundError("no deployment.json")
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    rows = []
    for cell in sort_cells(have):
        dep, sc = data[cell]["deploy"], cell_scores(data[cell]["runs"])
        b = _size_total(dep)
        if b is None or not sc:
            continue
        rep, cen, arch = parse_cell(cell)
        marker = {"linear": "o", "deep": "^", "a2c": "s"}[arch]
        ax.scatter(b, float(np.mean(sc)), marker=marker, s=52, color=MODE_COLOR[cen],
                   edgecolor="white", lw=0.6, alpha=0.9)
        rows.append([cell, b, float(np.mean(sc)), dep.get("size", {}).get("n_params")])
    ax.set_xscale("log")
    ax.set_xlabel("Model bytes, fp32 (network + encoder, as stored; log)")
    ax.set_ylabel("Policy pass mean score")
    ax.axhline(SOLVED, color="#222222", lw=1.0, ls="--", alpha=0.7)
    handles = [plt.Line2D([], [], marker=m, ls="none", color="#555555", label=n)
               for n, m in (("linear", "o"), ("deep", "^"), ("A2C", "s"))]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Deployment size vs score (NOT size-matched: search ranges did not overlap)")
    save_fig(fig, out_dir, "deployment_size", ["cell", "fp32_bytes", "mean_score", "n_params"], rows)


def fig_deployment_quant(data, out_dir, **kw):
    have = [c for c in data if data[c]["deploy"] and data[c]["deploy"].get("quantization")]
    if not have:
        raise FileNotFoundError("no quantization section")
    have = sort_cells(have)
    variants = ["fp32", "fp16_weights", "int8_weights", "fp16_encoder"]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * len(have) + 3.5), 5.2), constrained_layout=True)
    w = 0.2
    rows = []
    for k, var in enumerate(variants):
        xs, ys = [], []
        for i, cell in enumerate(have):
            q = (data[cell]["deploy"]["quantization"].get("variants") or {}).get(var)
            if not q:
                continue
            xs.append(i + (k - 1.5) * w); ys.append(float(q.get("retained", np.nan)))
            rows.append([cell, var, q.get("mean"), q.get("std"), q.get("retained")])
        ax.bar(xs, ys, width=w, label=var, alpha=0.85)
    ax.axhline(1.0, color="#222222", lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]][:4]}/{MODE_LABEL[parse_cell(c)[1]][:3]}/{ARCH_LABEL[parse_cell(c)[2]]}"
                        for c in have], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Retained return vs fp32")
    ax.legend(fontsize=8)
    ax.set_title("Simulated quantization retention (paired with fp32)")
    save_fig(fig, out_dir, "deployment_quant", ["cell", "variant", "mean", "std", "retained"], rows)


def fig_deployment_latency(data, out_dir, **kw):
    have = [c for c in data if data[c]["deploy"] and data[c]["deploy"].get("latency")]
    if not have:
        raise FileNotFoundError("no latency section")
    have = sort_cells(have)
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 0.58 * len(have) + 6), 5.0), constrained_layout=True)
    rows = []
    parts = [("encode", "encode"), ("q_from_phi", "Q"), ("act", "act")]
    bottoms = np.zeros(len(have))
    for key, label in parts:
        vals = []
        for cell in have:
            t = (data[cell]["deploy"]["latency"].get("timings") or {}).get(key) or {}
            vals.append(float(t.get("median_us", np.nan)))
            rows.append([cell, key, t.get("median_us"), t.get("p95_us")])
        vals = np.array(vals)
        axes[0].bar(range(len(have)), vals, bottom=bottoms, label=label, alpha=0.85)
        bottoms = bottoms + np.nan_to_num(vals)
    axes[0].set_ylabel("Median latency (us)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Per-call latency (one quiet pod, 1 thread)")
    for cell in have:
        oc = (data[cell]["deploy"]["latency"].get("operation_counts") or {})
        axes[1].bar(cell, oc.get("total_ops", np.nan), color=MODE_COLOR[parse_cell(cell)[1]], alpha=0.8)
    axes[1].set_ylabel("Analytic ops per action (machine-independent)")
    axes[1].set_title("Analytic operation counts")
    for ax in axes:
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels([f"{REP_LABEL[parse_cell(c)[0]][:4]}/{MODE_LABEL[parse_cell(c)[1]][:3]}/{ARCH_LABEL[parse_cell(c)[2]]}"
                            for c in have], rotation=30, ha="right", fontsize=7)
    cpu = (data[have[0]]["deploy"].get("pod") or {}).get("cpu_model")
    fig.suptitle(f"Deployment latency (pod CPU: {cpu}; wall times are implementation-specific)", fontsize=12)
    save_fig(fig, out_dir, "deployment_latency",
             ["cell", "part", "median_us", "p95_us"], rows)


# ---------------------------------------------------------------------------------------------
# figures: DQN settings and cross-algorithm
# ---------------------------------------------------------------------------------------------
def fig_dqn_settings(data, out_dir, local_root, **kw):
    rows, comps = [], []
    for cell in [c for c in sort_cells(data) if parse_cell(c)[2] in ("linear", "deep")]:
        base = data[cell]["runs"]
        if not base:
            continue
        base_by_seed = {r["seed"]: cell_scores([r])[0] for r in base if cell_scores([r])}
        for var in variant_names(local_root, cell):
            v = load_variant(local_root, cell, var)
            if not v["runs"]:
                continue
            diffs, scen, vsc = [], [], []
            for r in v["runs"]:
                s = cell_scores([r])
                if not s or r["seed"] not in base_by_seed:
                    continue
                diffs.append(s[0] - base_by_seed[r["seed"]])
                scen.append(base_by_seed[r["seed"]]); vsc.append(s[0])
            if not diffs:
                continue
            lo, hi = bootstrap_ci(diffs)
            comps.append((cell, var, float(np.mean(diffs)), lo, hi, len(diffs),
                          float(np.mean([d < -10 for d in diffs]))))
            rows.append([cell, var, float(np.mean(diffs)), lo, hi, float(np.mean(vsc)), float(np.mean(scen))])
    if not comps:
        raise FileNotFoundError("no dqn_settings variants found")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    for i, (cell, var, d, lo, hi, n, collapse) in enumerate(comps):
        color = MODE_COLOR[parse_cell(cell)[1]]
        axes[0].errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", ms=5, color=color, ecolor=color, lw=1.4)
        axes[1].scatter(i, collapse, color=color, s=40)
    axes[0].axvline(0.0, color="#222222", lw=1.0, ls="--", alpha=0.8)
    axes[0].set_yticks(range(len(comps)))
    axes[0].set_yticklabels([f"{REP_LABEL[parse_cell(c)[0]][:4]}/{ARCH_LABEL[parse_cell(c)[2]]}/{v}" for c, v, *_ in comps],
                            fontsize=7)
    axes[0].set_xlabel("Paired difference vs baseline, same seeds (95% CI)")
    axes[0].set_title("DQN-settings variants: paired difference")
    axes[1].set_yticks(range(len(comps)))
    axes[1].set_yticklabels([])
    axes[1].set_ylabel("Share of paired seeds collapsing by >10")
    axes[1].set_title("Collapse rate")
    fig.suptitle("DQN-settings test (lr/eps NOT retuned; 'no effect' is weak evidence)", fontsize=12)
    save_fig(fig, out_dir, "dqn_settings", ["cell", "variant", "mean_diff", "lo", "hi", "variant_mean", "baseline_mean"], rows)


def fig_a2c_vs_dqn(data, out_dir, metric="score", err_mode="ci", **kw):
    have = [c for c in data if data[c]["runs"]]
    if not have:
        raise FileNotFoundError("no runs")
    rows = []
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    groups = []
    for rep in REPS:
        for cen in CENTERINGS:
            groups.append((rep, cen))
    w = 0.26
    for k, arch in enumerate(ARCHS):
        xs, ys, errs = [], [], []
        for i, (rep, cen) in enumerate(groups):
            cell = f"{rep}/{cen}/{arch}"
            sc = cell_metric(data[cell]["runs"], metric) if cell in data and data[cell]["runs"] else []
            if not sc:
                continue
            mean = float(np.mean(sc))
            lo, hi = bootstrap_ci(sc)
            if err_mode == "std":
                lo, hi = mean - float(np.std(sc)), mean + float(np.std(sc))
            # CartPole-v1 caps an episode at 500, so no whisker may be drawn past the ceiling.
            lo, hi = max(lo, 0.0), min(hi, float(CEILING))
            xs.append(i + (k - 1) * w); ys.append(mean)
            errs.append([mean - lo, hi - mean])
            rows.append([cell, len(sc), mean, float(np.std(sc)), lo, hi,
                         int(sum(1 for s in sc if s >= CEILING)), metric])
        ax.bar(xs, ys, yerr=np.asarray(errs).T if errs else None, width=w,
               label=ARCH_LABEL[arch], alpha=0.85, capsize=2)
    ax.axhline(SOLVED, color="#222222", lw=1.2, ls="--", alpha=0.8)
    ax.axhline(CEILING, color="#777777", lw=1.0, ls=":", alpha=0.9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{REP_LABEL[r]}\n{MODE_LABEL[c][:3]}" for r, c in groups], rotation=30, ha="right", fontsize=7.5)
    mlabel, msuffix = METRICS[metric]
    ax.set_ylim(0, CEILING + 45)
    ax.set_ylabel("Policy pass mean return")
    ax.legend(fontsize=9)
    ax.set_title(f"{ARCH_LABEL['a2c']} vs {ARCH_LABEL['linear']} vs {ARCH_LABEL['deep']}, same protocol\n"
                 f"metric: {mlabel}; bar = mean over seeds, error bar = "
                 f"{'95% bootstrap CI of the mean' if err_mode == 'ci' else '1 SD over seeds'}\n"
                 f"dotted line = CartPole-v1 ceiling {CEILING:.0f} (seeds at the ceiling are censored)")
    save_fig(fig, out_dir, f"a2c_vs_dqn{msuffix}",
             ["cell", "n_seeds", "mean", "std", "ci_lo", "ci_hi", "n_at_ceiling", "metric"], rows)


# ---------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------
FIGURES = {
    "learning_curves": fig_learning_curves,
    "spaghetti": fig_spaghetti,
    "final_score_boxes_linear": lambda d, o, **k: fig_final_score_boxes(d, o, arch="linear", **k),
    "final_score_boxes_deep": lambda d, o, **k: fig_final_score_boxes(d, o, arch="deep", **k),
    "final_score_boxes_a2c": lambda d, o, **k: fig_final_score_boxes(d, o, arch="a2c", **k),
    "replicate_vs_run003": fig_replicate,
    "winners_curse": fig_winners_curse,
    "stability_linear": lambda d, o, **k: fig_stability(d, o, arch="linear", **k),
    "stability_deep": lambda d, o, **k: fig_stability(d, o, arch="deep", **k),
    "stability_a2c": lambda d, o, **k: fig_stability(d, o, arch="a2c", **k),
    "sample_efficiency_linear": lambda d, o, **k: fig_sample_efficiency(d, o, arch="linear", **k),
    "sample_efficiency_deep": lambda d, o, **k: fig_sample_efficiency(d, o, arch="deep", **k),
    "sample_efficiency_a2c": lambda d, o, **k: fig_sample_efficiency(d, o, arch="a2c", **k),
    "probe_linear": lambda d, o, **k: fig_probe(d, o, arch="linear", **k),
    "probe_deep": lambda d, o, **k: fig_probe(d, o, arch="deep", **k),
    "probe_a2c": lambda d, o, **k: fig_probe(d, o, arch="a2c", **k),
    "cost_scatter": fig_cost_scatter,
    "robustness_curves_linear": lambda d, o, **k: fig_robustness_curves(d, o, arch="linear", **k),
    "robustness_curves_deep": lambda d, o, **k: fig_robustness_curves(d, o, arch="deep", **k),
    "robustness_curves_a2c": lambda d, o, **k: fig_robustness_curves(d, o, arch="a2c", **k),
    "robustness_heatmap_linear": lambda d, o, **k: fig_robustness_heatmap(d, o, arch="linear", **k),
    "robustness_heatmap_deep": lambda d, o, **k: fig_robustness_heatmap(d, o, arch="deep", **k),
    "robustness_heatmap_a2c": lambda d, o, **k: fig_robustness_heatmap(d, o, arch="a2c", **k),
    "robustness_worst_linear": lambda d, o, **k: fig_robustness_worst(d, o, arch="linear", **k),
    "robustness_worst_deep": lambda d, o, **k: fig_robustness_worst(d, o, arch="deep", **k),
    "robustness_worst_a2c": lambda d, o, **k: fig_robustness_worst(d, o, arch="a2c", **k),
    "robustness_forest": fig_robustness_forest,
    "deployment_size": fig_deployment_size,
    "deployment_quant": fig_deployment_quant,
    "deployment_latency": fig_deployment_latency,
    "dqn_settings": fig_dqn_settings,
    "a2c_vs_dqn": fig_a2c_vs_dqn,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local-root", default="outputs/eval_v3")
    ap.add_argument("--out-dir", default=None, help="default: <local-root>/_figures")
    ap.add_argument("--figures", default="all", help="comma list or 'all'")
    ap.add_argument("--band", default="p5_95", choices=["p5_95", "sem", "std"])
    ap.add_argument("--metric", default="score", choices=sorted(METRICS),
                    help="per-seed metric for a2c_vs_dqn / final_score_boxes_*: score (last-5 ckpt mean, "
                         "default), final (final policy greedy), online (paper metric)")
    ap.add_argument("--err-mode", default="ci", choices=["ci", "std"],
                    help="error bars in a2c_vs_dqn: ci = 95%% bootstrap CI of the mean (default), std = 1 SD")
    ap.add_argument("--run003-json", default=None, help="override the transcribed RUN_003 numbers")
    ap.add_argument("--analysis-json", default=None,
                    help="default: <local-root>/_analysis/robustness_summary.json (for the forest plot)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for name in FIGURES:
            print(name)
        return 0
    out_dir = args.out_dir or os.path.join(args.local_root, "_figures")
    os.makedirs(out_dir, exist_ok=True)
    apply_style()
    data = {cell: load_cell(args.local_root, cell) for cell in discover_cells(args.local_root)}
    print(f"[plot] {len(data)} cells with train_summary.json under {args.local_root}", flush=True)
    run003 = RUN003
    if args.run003_json:
        raw = load_json(args.run003_json) or {}
        run003 = {k: tuple(v) for k, v in raw.items()}
    analysis = args.analysis_json or os.path.join(args.local_root, "_analysis", "robustness_summary.json")
    names = list(FIGURES) if args.figures == "all" else [n.strip() for n in args.figures.split(",") if n.strip()]
    ok, failed = [], []
    for name in names:
        fn = FIGURES.get(name)
        if fn is None:
            failed.append((name, "unknown figure")); continue
        try:
            fn(data, out_dir, local_root=args.local_root, band_mode=args.band, run003=run003,
               analysis_path=analysis, metric=args.metric, err_mode=args.err_mode)
            ok.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            if os.environ.get("PLOT_TRACEBACK"):
                traceback.print_exc()
    print(f"\n[plot] ok: {len(ok)}  failed: {len(failed)}")
    for name, err in failed:
        print(f"  SKIP {name}: {err}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
