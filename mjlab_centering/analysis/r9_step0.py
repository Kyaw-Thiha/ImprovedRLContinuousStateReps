"""REPORT_009 Step 0 (post hoc, free): does "walks iff t_level < t_commit" hold in the existing logs?

    python analysis/r9_step0.py [--fig] [--json results/r9_step0.json]

Per run (iter.jsonl, one row per iteration):
  t_level  - first iteration where the critic's mean value `v_mean` reaches 90% of the analytic target c/(1-gamma)
             (uncentered, bias-init, value-normalisation, clip-off, return-std arms); for the centred arms (value
             centering, simple centering) the first iteration where `const_frac` < 0.1. `const_frac` < 0.1 is also
             computed for every arm as a sensitivity ("lvl=cf").
  t_commit - first iteration where `action_std` < 0.8 x its iteration-0 value.
  walks    - mean of the last five evals of `track_lin_frac` >= 0.5 (the study's rule, REPORT_003+).
Rule under test: walks <=> t_level < t_commit (strict; inf < inf is False, so a run that never reaches the level
and never commits is predicted not to walk).

Three variants of the two times (all computed; the spec's literal one is variant A):
  A  literal   : first iteration at which the raw per-iteration value passes.
  B  smoothed  : first iteration at which the trailing-10-iteration mean passes.
  C  settled   : t_level = 1 + last iteration (of the run) at which the trailing-10 mean has NOT passed (the level
                 criterion has held from then on); t_commit = variant B.
Nothing here gates any run; this is a retrospective reading, labelled post hoc in the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT.parent / "docs" / "mjlab_centering" / "figures" / "report9"
INF = float("inf")
WALK = 0.5

# name, kind, dirs, pattern, gamma, c, seeds
GO1 = list(range(3000, 3020))
CELLS = []
for c, cs in ((1.0, "1"), (0.1, "0.1")):
  if c == 1.0:
    none_p = ("merged_r2", "w2main_go1_raw_none_g0.999_o1_s{}")
    val_p = ("merged_r2", "w2main_go1_raw_value_g0.999_o1_s{}")
    bias_p = ("merged_r4", "r5bias_go1_g0.999_o1_none_s{}")
  else:
    none_p = ("merged_r4", "r4sweep_go1_g0.999_o0.1_none_s{}")
    val_p = ("merged_r4", "r4sweep_go1_g0.999_o0.1_value_s{}")
    bias_p = ("merged_r4", "r5bias01_go1_g0.999_o0.1_none_s{}")
  CELLS += [
    (f"Go1 uncentered c={cs}", "level", *none_p, 0.999, c, GO1),
    (f"Go1 value-centred c={cs}", "centred", *val_p, 0.999, c, GO1),
    (f"Go1 bias-init c={cs}", "level", *bias_p, 0.999, c, GO1),
    (f"Go1 value-norm c={cs}", "level", "merged_r4", f"r7vnsc_go1_vn_g0.999_o{cs}_s{{}}", 0.999, c, GO1),
    (f"Go1 simple-centred c={cs}", "centred", "merged_r4", f"r7vnsc_go1_sc_g0.999_o{cs}_s{{}}", 0.999, c, GO1),
    (f"Go1 return-std c={cs}", "level", "merged_r4", f"r7rs_go1_rs_g0.999_o{cs}_s{{}}", 0.999, c, GO1),
  ]
CELLS.append(("Go1 clip-off c=1", "level", "merged_r4", "clip1_go1_raw_none_clipoff_g0.999_o1_s{}", 0.999, 1.0, GO1))
CELLS += [
  ("G1 uncentered c=1", "level", "r9_g1_iter", "r6g1_g1_g0.99_o1_none_s{}", 0.99, 1.0, GO1),
  ("G1 value-centred c=1", "centred", "r9_g1_iter", "r6g1_g1_g0.99_o1_value_s{}", 0.99, 1.0, GO1),
  ("G1 bias-init c=1", "level", "r9_g1_iter", "r6g1_g1_g0.99_o1_bias_none_s{}", 0.99, 1.0, GO1),
]
# eval files for the two G1 (and every other) cell live next to iter.jsonl or in the supervisor/merged caches
EVAL_DIRS = ["merged_r4", "merged_r2", "supervisor_cache/g1"]


def jl(p: Path) -> list[dict]:
  """Rows of one JSONL, merged to one row per iteration.

  A re-claimed job resumes from its rolling checkpoint and APPENDS from that iteration, so the file can contain
  0..N, then start_it..1999 (non-monotone), and with the 2026-11-09 claim bug two pods could interleave. Taking the last
  `iter == 0` start is not enough: for each iteration the row from the latest attempt that reached it is the one that
  describes the policy that continued, so later rows overwrite earlier ones and the result is sorted by iteration.
  """
  rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
  if not rows:
    return rows
  by: dict[int, dict] = {}
  for r in rows:
    if "iter" in r:
      by[int(r["iter"])] = r
    else:
      by.setdefault(-1, r)
  return [by[k] for k in sorted(by)]


def find(d: str, jid: str, name: str) -> Path | None:
  for base in [d] + EVAL_DIRS:
    p = RES / base / jid / name
    if p.exists() and p.stat().st_size > 0:
      return p
  return None


def trail(x: np.ndarray, w: int = 10) -> np.ndarray:
  cs = np.cumsum(np.insert(x, 0, 0.0))
  out = np.empty_like(x, dtype=float)
  for i in range(len(x)):
    lo = max(0, i - w + 1)
    out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
  return out


def first(mask: np.ndarray) -> float:
  i = np.flatnonzero(mask)
  return float(i[0]) if len(i) else INF


def settled(ok: np.ndarray) -> float:
  """1 + last index at which the criterion does not hold (0 if it always holds, inf if it never holds at the end)."""
  bad = np.flatnonzero(~ok)
  if len(bad) == 0:
    return 0.0
  if bad[-1] == len(ok) - 1:
    return INF
  return float(bad[-1] + 1)


def first_eval(rows: list[dict], level: float) -> float | None:
  """Iteration at which the eval tracking curve first reaches `level` (linear interpolation between evals)."""
  prev_it = prev_v = None
  for r in sorted(rows, key=lambda r: r["iter"]):
    v, it = r.get("track_lin_frac"), r["iter"] + 1
    if v is None:
      continue
    if v >= level:
      if prev_v is None or prev_v >= level:
        return float(it)
      f = (level - prev_v) / (v - prev_v) if v > prev_v else 1.0
      return float(prev_it + f * (it - prev_it))
    prev_it, prev_v = it, v
  return None


def times(rows: list[dict], kind: str, gamma: float, c: float) -> dict:
  v = np.array([r["v_mean"] for r in rows], float)
  cf = np.array([r["const_frac"] for r in rows], float)
  sd = np.array([r["action_std"] for r in rows], float)
  tgt = c / (1 - gamma)
  sd0 = sd[0]
  lv_raw = (cf < 0.1) if kind == "centred" else (v >= 0.9 * tgt)
  lv_sm = (trail(cf) < 0.1) if kind == "centred" else (trail(v) >= 0.9 * tgt)
  cf_raw, cf_sm = cf < 0.1, trail(cf) < 0.1
  cm_raw, cm_sm = sd < 0.8 * sd0, trail(sd) < 0.8 * sd0
  return {
    "A": {"t_level": first(lv_raw), "t_commit": first(cm_raw), "t_level_cf": first(cf_raw)},
    "B": {"t_level": first(lv_sm), "t_commit": first(cm_sm), "t_level_cf": first(cf_sm)},
    "C": {"t_level": settled(lv_sm), "t_commit": first(cm_sm), "t_level_cf": settled(cf_sm)},
    "sd0": float(sd0), "tgt": tgt,
  }


def outcome(jid: str, d: str) -> tuple[bool | None, float | None, float | None]:
  p = find(d, jid, "eval.jsonl")
  if p is None:
    return None, None, None
  rows = jl(p)
  if len(rows) < 5:
    return None, None, None
  tr = float(np.mean([r["track_lin_frac"] for r in rows[-5:]]))
  t50 = next((r["iter"] + 1 for r in rows if r["track_lin_frac"] >= WALK), None)
  return tr >= WALK, tr, t50


def collect() -> list[dict]:
  out = []
  for name, kind, d, pat, gamma, c, seeds in CELLS:
    for s in seeds:
      jid = pat.format(s)
      ip = find(d, jid, "iter.jsonl")
      if ip is None:
        continue
      walks, tr, t50 = outcome(jid, d)
      if walks is None:
        continue
      rows = jl(ip)
      t = times(rows, kind, gamma, c)
      out.append({"cell": name, "kind": kind, "job": jid, "seed": s, "gamma": gamma, "c": c, "n_rows": len(rows),
                  "walks": walks, "track": tr, "t50": t50, **{k: t[k] for k in ("A", "B", "C", "sd0", "tgt")}})
  return out


def predict(r: dict, var: str, lvl: str = "level") -> bool:
  t = r[var]
  tl = t["t_level_cf"] if lvl == "cf" else t["t_level"]
  return tl < t["t_commit"]


def f(x: float) -> str:
  return "inf" if x == INF else f"{x:.0f}"


def tables(runs: list[dict]) -> str:
  cells = list(dict.fromkeys(r["cell"] for r in runs))
  lines = []
  for var, title in (("A", "A literal (first raw iteration)"), ("B", "B smoothed (trailing-10 mean)"),
                     ("C", "C settled (level held from t onward; commit as B)")):
    lines += [f"**Variant {title}**", "",
              "| cell | n | walks k/n | rule accuracy | pred. walk & walks | pred. walk & not | pred. no-walk & walks "
              "| pred. no-walk & not | median t_level (walkers / non) | median t_commit (walkers / non) | acc. (lvl=cf) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = tot_cf = n_all = 0
    for cn in cells:
      rs = [r for r in runs if r["cell"] == cn]
      pr = [predict(r, var) for r in rs]
      pc = [predict(r, var, "cf") for r in rs]
      w = [r["walks"] for r in rs]
      acc = sum(p == y for p, y in zip(pr, w))
      acc_cf = sum(p == y for p, y in zip(pc, w))
      tot += acc; tot_cf += acc_cf; n_all += len(rs)

      def med(key: str, sel: bool) -> str:
        xs = [r[var][key] for r in rs if r["walks"] == sel]
        return f(float(np.median(xs))) if xs else "-"
      lines.append(
        f"| {cn} | {len(rs)} | {sum(w)}/{len(rs)} | **{acc}/{len(rs)}** | "
        f"{sum(p and y for p, y in zip(pr, w))} | {sum(p and not y for p, y in zip(pr, w))} | "
        f"{sum((not p) and y for p, y in zip(pr, w))} | {sum((not p) and not y for p, y in zip(pr, w))} | "
        f"{med('t_level', True)} / {med('t_level', False)} | {med('t_commit', True)} / {med('t_commit', False)} | "
        f"{acc_cf}/{len(rs)} |")
    lines += [f"| **all cells** | {n_all} | | **{tot}/{n_all} ({tot / n_all:.0%})** | | | | | | | "
              f"{tot_cf}/{n_all} ({tot_cf / n_all:.0%}) |", ""]
  return "\n".join(lines)


def majority_baseline(runs: list[dict]) -> str:
  """The per-cell majority-class baseline: predicting each cell's own commoner outcome, which knows the cell but not the
  seed. A rule that does not beat this has not explained anything about the mechanism."""
  cells: dict[str, list[bool]] = {}
  for r in runs:
    cells.setdefault(r["cell"], []).append(r["walks"])
  hit = sum(max(sum(v), len(v) - sum(v)) for v in cells.values())
  return f"per-cell majority baseline: **{hit}/{len(runs)} ({hit / len(runs):.0%})**"


def within_cell_auc(runs: list[dict], var: str) -> str:
  """Does the race margin separate walkers from failures inside the cells that contain both?

  AUC of (t_commit - t_level) against the outcome, computed only where a cell has both outcomes (else undefined). 0.5 is
  no information; below 0.5 means the rule orders the seeds the wrong way.
  """
  def auc(xs, ys):
    pos = [x for x, y in zip(xs, ys) if y]
    neg = [x for x, y in zip(xs, ys) if not y]
    if not pos or not neg:
      return None
    c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return c / (len(pos) * len(neg))
  lines = [f"| cell | n | walks | AUC(margin, {var}) |", "|---|---|---|---|"]
  for cn in dict.fromkeys(r["cell"] for r in runs):
    rs = [r for r in runs if r["cell"] == cn]
    if len({r["walks"] for r in rs}) < 2:
      continue
    m = [r[var]["t_commit"] - r[var]["t_level"] for r in rs]
    a = auc(m, [r["walks"] for r in rs])
    lines.append(f"| {cn} | {len(rs)} | {sum(r['walks'] for r in rs)} | **{a:.2f}** |")
  return "\n".join(lines)


def figure(runs: list[dict], path: Path) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  cap = 2800.0
  fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
  rng = np.random.default_rng(0)
  mk = {"uncentered": "o", "value-centred": "s", "bias-init": "^", "value-norm": "D", "simple-centred": "P",
        "return-std": "X", "clip-off": "v"}
  for ax, var in zip(axes, "ABC"):
    for r in runs:
      key = next(k for k in mk if k in r["cell"].lower().replace("value-centred", "value-centred"))
      t = r[var]
      x = min(t["t_commit"], cap) + 1.0
      y = min(t["t_level"], cap) + 1.0
      jit = np.exp(rng.normal(0, 0.04, 2))
      ax.scatter(x * jit[0], y * jit[1], marker=mk[key], s=34 if r["gamma"] > 0.995 else 60,
                 facecolors="#2ca02c" if r["walks"] else "#d62728",
                 edgecolors="k" if r["gamma"] < 0.995 else "none", linewidths=0.8, alpha=0.6)
    lo, hi = 0.8, cap * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.axhline(cap + 1, color="grey", lw=0.5, ls=":"); ax.axvline(cap + 1, color="grey", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("t_commit (iteration, +1; right edge = never)"); ax.set_ylabel("t_level (iteration, +1; top = never)")
    ax.set_title({"A": "A literal", "B": "B smoothed", "C": "C settled level"}[var] + " (below diagonal: rule predicts walk)")
    ax.text(0.97, 0.03, "t_level < t_commit\n(predict walk)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
  from matplotlib.lines import Line2D
  h = [Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="none", label="walks"),
       Line2D([], [], marker="o", ls="", mfc="#d62728", mec="none", label="does not walk")]
  h += [Line2D([], [], marker=m, ls="", mfc="grey", mec="none", label=k) for k, m in mk.items()]
  h += [Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="G1 (black edge)")]
  fig.legend(handles=h, loc="lower center", ncol=10, fontsize=8)
  fig.suptitle("REPORT_009 Step 0 (post hoc): t_level vs t_commit, every existing run, coloured by outcome")
  fig.tight_layout(rect=(0, 0.06, 1, 0.95))
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=130)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--fig", action="store_true")
  ap.add_argument("--json", default=str(RES / "r9_step0.json"))
  a = ap.parse_args()
  runs = collect()
  Path(a.json).write_text(json.dumps(runs, default=lambda x: "inf" if x == INF else x))
  print(f"{len(runs)} runs")
  print(tables(runs))
  print(majority_baseline(runs), "\n")
  print("**Within-cell separation** (only cells with both outcomes):\n")
  print(within_cell_auc(runs, "A"))
  print()
  if a.fig:
    figure(runs, OUT / "r9_01_step0_race.png")
