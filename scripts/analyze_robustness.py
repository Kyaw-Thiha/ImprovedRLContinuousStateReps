"""
Statistics for the eval_v3 robustness suite (reads robustness_runs.jsonl of each cell; needs no policies).

Unit of analysis = the policy SEED: episodes are averaged within a seed first, then every bootstrap
resamples seeds (fixed rng seeds, so reruns give identical numbers). Quantities:

  score(seed, family)   = mean over the family's conditions of that seed's mean return
  retained(cell, family) = mean_seed(score) / mean_seed(baseline)   (ratio of means; CI by seed bootstrap)

Pre-declared comparison (label `primary`): tile_coding minus ssp, centering in {none, value}, both archs, for
every family; unpaired bootstrap of the difference (seeds are independent draws, nothing is paired across
cells), Holm-adjusted across all primary comparisons of one metric (score and retained are corrected as two
separate sets). Everything else (centering=simple, discrete vs the others) is `exploratory` and gets raw
p-values only. A sensitivity pass repeats everything on seeds whose in-harness baseline mean is >= 475
(i.e. seeds that actually solved the task, so retention is not dominated by a failed baseline).

Usage:
    python scripts/analyze_robustness.py --local-root outputs/eval_v3 --out-dir outputs/eval_v3/_analysis
    python scripts/analyze_robustness.py --local-root DIR --pull --cells ssp/none/deep,tile_coding/none/deep --out-dir DIR2
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import eval_v3_layout as layout  # noqa: E402

# Fallback copy of rl.perturbations.FAMILIES so the analysis runs without importing the simulator.
_FAMILIES_FALLBACK = {
    "baseline": ["baseline"],
    "pole_length": ["pole_length_x0.5", "pole_length_x1.5", "pole_length_x2", "pole_length_x4"],
    "pole_mass": ["pole_mass_x0.5", "pole_mass_x2"],
    "cart_mass": ["cart_mass_x0.5", "cart_mass_x2"],
    "gravity": ["gravity_x0.8", "gravity_x1.2"],
    "force": ["force_x0.7", "force_x1.3"],
    "obs_noise": ["obs_noise_0.01", "obs_noise_0.05", "obs_noise_0.1"],
    "action_delay": ["action_delay_1", "action_delay_2"],
    "random_action": ["random_action_0.05"],
    "wide_init": ["wide_init_0.1", "wide_init_0.2"],
}


def default_families() -> dict:
    try:
        from rl.perturbations import FAMILIES
        return {k: list(v) for k, v in FAMILIES.items()}
    except Exception:  # noqa: BLE001
        return {k: list(v) for k, v in _FAMILIES_FALLBACK.items()}


SOLVED = 475.0


# ---------------------------------------------------------------------------------------------
# Pure statistics
# ---------------------------------------------------------------------------------------------
def _rng(seed) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _label_seed(label: str, base: int = 12345) -> int:
    """Deterministic per-quantity rng seed (zlib, never builtin hash which is salted per process)."""
    return (zlib.crc32(label.encode()) + base) % 2 ** 31


def _boot_stat(num: np.ndarray, den, n_boot: int, rng) -> np.ndarray:
    """Bootstrap distribution over seeds of mean(num) or, when `den` is given, mean(num)/mean(den)."""
    n = len(num)
    idx = rng.integers(0, n, size=(n_boot, n))
    m = num[idx].mean(axis=1)
    if den is None:
        return m
    d = den[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m / d


def _point(num: np.ndarray, den) -> float:
    if den is None:
        return float(np.mean(num))
    d = float(np.mean(den))
    return float(np.mean(num)) / d if d != 0 else float("nan")


def bootstrap_ci_mean(x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0, den=None) -> dict:
    """Percentile bootstrap CI of the mean over seeds (or of the ratio of means when `den` is given).
    Returns {"mean","lo","hi","n"}."""
    x = np.asarray(x, dtype=np.float64)
    d = None if den is None else np.asarray(den, dtype=np.float64)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    est = _point(x, d)
    if len(x) == 1:
        return {"mean": est, "lo": est, "hi": est, "n": 1}
    dist = _boot_stat(x, d, n_boot, _rng(seed))
    lo, hi = np.nanpercentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": est, "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def bootstrap_diff(a, b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0,
                   a_den=None, b_den=None) -> dict:
    """Unpaired bootstrap of stat(a) - stat(b); each group is resampled independently. stat is the mean, or
    the ratio of means when a_den / b_den are given. p is the two-sided bootstrap p-value
    2*min(P*(diff<=0), P*(diff>=0)) with a +1 correction so it is never exactly 0.
    Returns {"diff","lo","hi","p","n_a","n_b"}."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ad = None if a_den is None else np.asarray(a_den, dtype=np.float64)
    bd = None if b_den is None else np.asarray(b_den, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        nan = float("nan")
        return {"diff": nan, "lo": nan, "hi": nan, "p": nan, "n_a": len(a), "n_b": len(b)}
    diff = _point(a, ad) - _point(b, bd)
    rng = _rng(seed)
    dist = _boot_stat(a, ad, n_boot, rng) - _boot_stat(b, bd, n_boot, rng)
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        nan = float("nan")
        return {"diff": diff, "lo": nan, "hi": nan, "p": nan, "n_a": len(a), "n_b": len(b)}
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_le = (np.sum(dist <= 0) + 1) / (len(dist) + 1)
    p_ge = (np.sum(dist >= 0) + 1) / (len(dist) + 1)
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, 2 * min(p_le, p_ge))),
            "n_a": int(len(a)), "n_b": int(len(b))}


def holm(pvals, alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns (adjusted p-values in input order, reject flags at `alpha`).
    adj_(k) = max over j<=k of min(1, (m-j+1) * p_(j)) so the adjusted values are monotone."""
    p = np.asarray(list(pvals), dtype=np.float64)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return [float(v) for v in adj], [bool(v <= alpha) for v in adj]


# ---------------------------------------------------------------------------------------------
# Records -> per-seed tables
# ---------------------------------------------------------------------------------------------
def dedupe_records(records) -> list:
    """Drop error lines; the last record of a (seed_idx, condition) wins (matches parallel_jobs)."""
    out = {}
    for r in records:
        if "error" in r or r.get("kind", "rob") != "rob":
            continue
        out[(int(r["seed_idx"]), r["condition"])] = r
    return list(out.values())


def seed_condition_means(records) -> dict:
    """{seed_idx: {condition: mean return of that seed's episodes}}."""
    out = {}
    for r in dedupe_records(records):
        m = r.get("mean")
        if m is None:
            m = float(np.mean(r["returns"]))
        out.setdefault(int(r["seed_idx"]), {})[r["condition"]] = float(m)
    return out


def family_scores(records, families: dict = None) -> dict:
    """{seed_idx: {family: score}}: mean over the family's conditions of the seed's mean return. A family is
    only reported for a seed when ALL its conditions are present (so scores are comparable across seeds)."""
    families = families or default_families()
    out = {}
    for seed, conds in seed_condition_means(records).items():
        row = {}
        for fam, names in families.items():
            if all(c in conds for c in names):
                row[fam] = float(np.mean([conds[c] for c in names]))
        out[seed] = row
    return out


def filter_solved(fs: dict, threshold: float = SOLVED) -> dict:
    """Sensitivity subset: seeds whose in-harness baseline mean is >= threshold (seeds without baseline drop)."""
    return {s: row for s, row in fs.items() if row.get("baseline", -np.inf) >= threshold}


def _arrays(fs: dict, family: str, retained: bool):
    """(numerator, denominator|None) arrays over the seeds that have the family (and the baseline)."""
    seeds = sorted(s for s, row in fs.items() if family in row and (not retained or "baseline" in row))
    num = np.array([fs[s][family] for s in seeds], dtype=np.float64)
    den = np.array([fs[s]["baseline"] for s in seeds], dtype=np.float64) if retained else None
    return num, den


def summarize_cell(records, families: dict = None, n_boot: int = 5000, seed: int = 0, label: str = "") -> dict:
    families = families or default_families()
    scm = seed_condition_means(records)
    fs = family_scores(records, families)
    conditions = {}
    all_conds = [c for names in families.values() for c in names]
    for c in all_conds:
        vals = [row[c] for row in scm.values() if c in row]
        if vals:
            conditions[c] = bootstrap_ci_mean(vals, n_boot, seed=_label_seed(f"{label}|{c}", seed))
    fam_out = {}
    for fam in families:
        num, den = _arrays(fs, fam, retained=False)
        if len(num) == 0:
            continue
        entry = {"score": bootstrap_ci_mean(num, n_boot, seed=_label_seed(f"{label}|{fam}|score", seed))}
        if fam != "baseline":
            rn, rd = _arrays(fs, fam, retained=True)
            if len(rn):
                entry["retained"] = bootstrap_ci_mean(rn, n_boot, seed=_label_seed(f"{label}|{fam}|ret", seed),
                                                      den=rd)
        fam_out[fam] = entry
    base = [row["baseline"] for row in fs.values() if "baseline" in row]
    non_base = {c: v for c, v in conditions.items() if c != "baseline"}
    worst = None
    if non_base:
        wc = min(non_base, key=lambda c: non_base[c]["mean"])
        worst = {"condition": wc, **non_base[wc]}
    return {"n_seeds": len(scm),
            "baseline_solved_share": float(np.mean([b >= SOLVED for b in base])) if base else None,
            "baseline_mean": float(np.mean(base)) if base else None,
            "conditions": conditions, "families": fam_out, "worst_condition": worst}


def compare_cells(fs_a: dict, fs_b: dict, cell_a: str, cell_b: str, families: dict = None, n_boot: int = 5000,
                  seed: int = 0, label_prefix: str = "") -> list:
    """Comparison rows `a - b` per family and metric (score; retained except for the baseline family)."""
    families = families or default_families()
    rows = []
    for fam in families:
        for metric in ("score", "retained"):
            if metric == "retained" and fam == "baseline":
                continue
            an, ad = _arrays(fs_a, fam, retained=metric == "retained")
            bn, bd = _arrays(fs_b, fam, retained=metric == "retained")
            if len(an) == 0 or len(bn) == 0:
                continue
            res = bootstrap_diff(an, bn, n_boot, seed=_label_seed(f"{label_prefix}|{cell_a}|{cell_b}|{fam}|{metric}", seed),
                                 a_den=ad, b_den=bd)
            rows.append({"cell_a": cell_a, "cell_b": cell_b, "family": fam, "metric": metric,
                         "mean_a": _point(an, ad), "mean_b": _point(bn, bd), **res})
    return rows


def comparison_pairs(cells) -> list:
    """[(cell_a, cell_b, label)]: primary = tile_coding minus ssp for centering none/value (both archs);
    exploratory = the same pair at centering=simple and discrete vs ssp / tile_coding everywhere."""
    cells = set(cells)
    pairs = []
    for cen in layout.CENTERINGS:
        for arch in layout.ARCHS + (layout.A2C_ARCH,):
            tile, ssp, disc = (f"{r}/{cen}/{arch}" for r in ("tile_coding", "ssp", "discrete"))
            if tile in cells and ssp in cells:
                lab = "primary" if cen in layout.PRIMARY_CENTERINGS else "exploratory"
                pairs.append((tile, ssp, lab))
            if disc in cells and ssp in cells:
                pairs.append((disc, ssp, "exploratory"))
            if disc in cells and tile in cells:
                pairs.append((disc, tile, "exploratory"))
    return pairs


def run_comparisons(fs_by_cell: dict, families: dict = None, n_boot: int = 5000, seed: int = 0,
                    tag: str = "all", alpha: float = 0.05) -> list:
    """All comparison rows with labels; Holm over the primary rows of each metric (score / retained)."""
    rows = []
    for a, b, lab in comparison_pairs(fs_by_cell):
        for r in compare_cells(fs_by_cell[a], fs_by_cell[b], a, b, families, n_boot, seed, label_prefix=tag):
            r["label"] = lab
            r["p_holm"] = None
            r["reject"] = None
            rows.append(r)
    for metric in ("score", "retained"):
        prim = [r for r in rows if r["label"] == "primary" and r["metric"] == metric and r["p"] == r["p"]]
        adj, rej = holm([r["p"] for r in prim], alpha)
        for r, p_adj, rj in zip(prim, adj, rej):
            r["p_holm"], r["reject"] = p_adj, rj
    return rows


# ---------------------------------------------------------------------------------------------
# IO / report
# ---------------------------------------------------------------------------------------------
def load_cell(cell: str, local_root: str, pull: bool = False):
    from parallel_jobs import pull_file, read_jsonl
    study = layout.study_name(cell)
    key = layout.robustness_key(study)
    path = layout.local_path(key, local_root)
    if pull:
        print(f"[analysis] pull {key}: {pull_file(key, path)}", flush=True)
    if not os.path.exists(path):
        return None
    return read_jsonl(path)


def _fmt(ci: dict, digits: int = 1) -> str:
    return f"{ci['mean']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def _fmt_ret(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]"


def _cmp_line(r: dict) -> str:
    dg = 3 if r["metric"] == "retained" else 1
    holm_txt = "" if r["p_holm"] is None else f", p_holm={r['p_holm']:.4f}{' SIGNIFICANT' if r['reject'] else ''}"
    return (f"- {r['cell_a']} - {r['cell_b']} | {r['family']} | {r['metric']}: "
            f"{r['mean_a']:.{dg}f} vs {r['mean_b']:.{dg}f}, diff {r['diff']:+.{dg}f} "
            f"[{r['lo']:+.{dg}f}, {r['hi']:+.{dg}f}], p={r['p']:.4f}{holm_txt} (n={r['n_a']}/{r['n_b']})")


def render_markdown(summary: dict) -> str:
    L = ["# Robustness suite: tables", "",
         f"Bootstrap over policy seeds (n_boot={summary['n_boot']}), 95% percentile CIs. score = mean over a family's "
         "conditions of the seed's mean return; retained = family score / the same seeds' baseline (ratio of means).", ""]
    L += ["## Per cell: baseline and worst condition", ""]
    for cell, s in summary["cells"].items():
        w = s["worst_condition"]
        worst = f"{w['condition']} {w['mean']:.1f}" if w else "n/a"
        share = "n/a" if s["baseline_solved_share"] is None else f"{100 * s['baseline_solved_share']:.0f}%"
        bm = "n/a" if s["baseline_mean"] is None else f"{s['baseline_mean']:.1f}"
        L.append(f"- {cell}: n_seeds={s['n_seeds']}, baseline mean {bm}, "
                 f"seeds with baseline >= {SOLVED:.0f}: {share}, worst condition: {worst}")
    L += ["", "## Per cell: family score and retained", ""]
    for cell, s in summary["cells"].items():
        L += [f"Cell {cell}", "", "| family | score | retained |", "|---|---|---|"]
        for fam, e in s["families"].items():
            ret = _fmt_ret(e["retained"]) if "retained" in e else "-"
            L.append(f"| {fam} | {_fmt(e['score'])} | {ret} |")
        L.append("")
    for title, key in (("Primary comparisons (pre-declared: tile_coding minus ssp, centering none/value)",
                        "comparisons"),
                       ("Sensitivity: same comparisons on seeds with in-harness baseline >= 475",
                        "comparisons_sensitivity")):
        rows = summary[key]
        L += [f"## {title}", "", "Holm is applied across the primary rows of each metric (families x cells)."
              if key == "comparisons" else "Holm recomputed within this restricted set.", ""]
        prim = [r for r in rows if r["label"] == "primary"]
        for metric in ("score", "retained"):
            L += [f"Metric: {metric}", ""]
            L += [_cmp_line(r) for r in prim if r["metric"] == metric] or ["(no data)"]
            L.append("")
        n_sig = sum(1 for r in prim if r["reject"])
        L += [f"Significant after Holm: {n_sig} of {len(prim)} primary rows.", ""]
    L += ["## Exploratory comparisons (raw p, no correction; centering=simple tile vs ssp and discrete pairs)", ""]
    expl = [r for r in summary["comparisons"] if r["label"] == "exploratory"]
    L += [_cmp_line(r) for r in expl] or ["(none)"]
    L.append("")
    return "\n".join(L)


def analyze(cells_records: dict, out_dir: str, n_boot: int = 5000, families: dict = None, seed: int = 0) -> dict:
    families = families or default_families()
    fs_by_cell = {c: family_scores(r, families) for c, r in cells_records.items()}
    sens_by_cell = {c: filter_solved(fs) for c, fs in fs_by_cell.items()}
    summary = {"n_boot": n_boot, "solved_threshold": SOLVED, "families": families,
               "cells": {c: summarize_cell(r, families, n_boot, seed, label=c) for c, r in cells_records.items()},
               "comparisons": run_comparisons(fs_by_cell, families, n_boot, seed, tag="all"),
               "comparisons_sensitivity": run_comparisons(sens_by_cell, families, n_boot, seed, tag="solved")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "robustness_tables.md"), "w") as f:
        f.write(render_markdown(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--pull", action="store_true", help="download robustness_runs.jsonl from R2 first")
    ap.add_argument("--cells", default=None, help="comma separated; default all DQN cells with a local file")
    ap.add_argument("--a2c", action="store_true", help="default to the 9 A2C cells instead of the 18 DQN cells")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)
    cells = [c.strip() for c in args.cells.split(",")] if args.cells else list(
        layout.A2C_CELLS if args.a2c else layout.ALL_CELLS)
    data = {}
    for c in cells:
        recs = load_cell(c, args.local_root, pull=args.pull)
        if recs is None:
            print(f"[analysis] {c}: no robustness_runs.jsonl, skipped", flush=True)
            continue
        data[c] = recs
        print(f"[analysis] {c}: {len(recs)} records", flush=True)
    if not data:
        raise SystemExit("no robustness data found")
    analyze(data, args.out_dir, args.n_boot)
    print(f"[analysis] wrote {args.out_dir}/robustness_tables.md and robustness_summary.json", flush=True)


if __name__ == "__main__":
    main()
