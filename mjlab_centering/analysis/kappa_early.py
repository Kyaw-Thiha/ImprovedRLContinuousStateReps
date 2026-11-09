"""EXPLORATORY, POST HOC (Task 1): can kappa = L / S be estimated without a centered run?  Cached logs only, no runs, no spend.

    cd mjlab_centering && .venv/bin/python analysis/kappa_early.py

Pre-registered in docs/mjlab_centering/02-decisions.md d153 (written before any kappa-hat was computed; hash in
docs/analysis/kappa_early/prereg_d153.sha256.json). Three candidates, no others, all taken from the UNCENTERED arm:

  S_ret   std of the batch returns, per iteration, averaged over iterations 0..50.  `ret_std` is not a logged key; the std of the same
          tensor is recovered exactly from two logged keys:  ev = 1 - err_mse/(var_ret + 1e-8)  =>  var_ret = err_mse / (1 - ev).
  S_rew   std of the per-step TASK reward across envs, iterations 0..50, x 1/sqrt(1-gamma^2).  NOT AVAILABLE: the logged `rew_std` is
          updated only under reward_scale == "std" and is its initial 1.0 in every row here (audited below); candidate dropped.
  S_v100  the uncentered arm's logged `v_std`, averaged over iterations 50..100.

kappa_hat = L / S with L exactly as in kappa_analysis.py. Cells, seeds, walking rule, Firth regression, AUC and the single-threshold
test are the ones of kappa_analysis.py (imported, not copied). Outputs: docs/analysis/kappa_early/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import kappa_analysis as ka  # noqa: E402

OUT = ka.REPO / "docs" / "analysis" / "kappa_early"
W_RET = (0, 50)
W_V = (50, 100)

CANDS = {  # key: (label, column prefix)
  "S_ret": "S_ret (return std, uncentered, it 0-50)",
  "S_rew": "S_rew (per-step reward std x 1/sqrt(1-g^2), it 0-50)",
  "S_v100": "S_v100 (uncentered v_std, it 50-100)",
}
AVAILABLE = ["S_ret", "S_v100"]


# ------------------------------------------------------------------------------------------ field audit
def audit() -> dict:
  """Facts about the logs that decide availability (computed over every iter row of every cell used)."""
  n_rows = n_rew1 = n_rew_other = n_ev_bad = n_var_bad = 0
  keys_ret = set()
  rs = set()
  for cell in ka.go1_cells() + ka.g1_cells():
    for arm in ("unc", "cen"):
      for jid in (cell[arm] or {}).values():
        rows = ka.iter_rows(jid, cell["robot"])["rows"]
        for it, r in rows.items():
          if it > 100:
            continue
          n_rows += 1
          if r.get("rew_std") == 1.0:
            n_rew1 += 1
          else:
            n_rew_other += 1
          keys_ret |= {k for k in r if "ret" in k}
          if not (r["ev"] < 1.0):
            n_ev_bad += 1
          elif not (r["err_mse"] / (1.0 - r["ev"]) > 0):
            n_var_bad += 1
  for p in list((ka.RES / "merged_r4").glob("*/config.json"))[:400]:
    try:
      rs.add(json.loads(p.read_text()).get("reward_scale"))
    except Exception:
      pass
  for p in list(ka.CACHE_SUP.glob("*/config.json")):
    rs.add(json.loads(p.read_text()).get("reward_scale"))
  return dict(rows_it0_100=n_rows, rew_std_equals_1=n_rew1, rew_std_other=n_rew_other, ret_like_keys=sorted(keys_ret),
              reward_scale_values_seen=sorted(map(str, rs)), rows_ev_ge_1=n_ev_bad, rows_nonpositive_var_ret=n_var_bad)


# ------------------------------------------------------------------------------------------ per-cell S
def ret_std_row(r: dict) -> float | None:
  if "ev" not in r or "err_mse" not in r or not (r["ev"] < 1.0):
    return None
  v = r["err_mse"] / (1.0 - r["ev"])
  return math.sqrt(v) if v > 0 else None


def cell_S(cell: dict, base: dict) -> dict:
  """Per-seed S_ret and S_v100 from the uncentered arm, over the seeds that the kappa report used for this cell."""
  s_ret, s_v = {}, {}
  for s in base["seeds_walk"]:
    rows = ka.iter_rows(cell["unc"][s], cell["robot"])["rows"]
    rr = {i: {"ret_std": ret_std_row(r)} for i, r in rows.items()}
    for i in list(rr):
      if rr[i]["ret_std"] is None:
        del rr[i]
    a = ka.window_mean(rr, "ret_std", *W_RET)
    b = ka.window_mean(rows, "v_std", *W_V)
    if a is not None:
      s_ret[s] = a
    if b is not None:
      s_v[s] = b
  return dict(S_ret_seeds=s_ret, S_v100_seeds=s_v)


def build() -> list[dict]:
  cells = [ka.compute_cell(c) for c in ka.go1_cells() + ka.g1_cells()]
  reg = {c["key"]: c for c in ka.go1_cells() + ka.g1_cells()}
  for c in cells:
    c.update(cell_S(reg[c["key"]], c))
    for name, key in (("S_ret", "S_ret_seeds"), ("S_v100", "S_v100_seeds")):
      v = list(c[key].values())
      c[name] = float(np.mean(v)) if v else float("nan")
      c[name + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
      c[name + "_n"] = len(v)
      c["kappa_" + name] = c["L"] / c[name] if v else float("nan")
  order = {"claim": 0, "pilot": 1}
  cells.sort(key=lambda c: (order[c["kind"]], c["robot"] != "go1", c["gamma"] != 0.999, c["c"]))
  return cells


# ------------------------------------------------------------------------------------------ regression
def fit_x(claim: list[dict], col: str) -> dict:
  rows = [(c, s, w) for c in claim for s, w in c["seeds_walk"].items()]
  y = np.array([w for _, _, w in rows], float)
  robot = np.array([1.0 if c["robot"] == "g1" else 0.0 for c, _, _ in rows])
  x = np.log10(np.array([c[col] for c, _, _ in rows]))
  xc = x - x.mean()
  one = np.ones_like(xc)
  fa = ka.firth_logit(np.column_stack([one, xc]), y)
  fb = ka.firth_logit(np.column_stack([one, xc, robot]), y)
  pts = [(float(np.log10(c[col])), c["rate"]) for c in claim]
  srt = sorted(pts)
  return dict(alone=fa, with_robot=fb, auc=ka.auc(x, y, True), d_aic_gain=fa["aic"] - fb["aic"], single_threshold=ka.single_threshold_separable(pts),
              rank_order=[(c["key"], c["rate"], float(c[col])) for c in sorted(claim, key=lambda c: c[col])],
              strictly_monotone=all(a[1] >= b[1] for a, b in zip(srt, srt[1:])),
              x_mean=float(x.mean()),
              pred_g1=float(ka.sigmoid(fa["beta"][0] + fa["beta"][1] * (math.log10([c for c in claim if c["robot"] == "g1"][0][col]) - x.mean()))),
              n_seeds=len(y), n_walk=int(y.sum()))


def verdict(f: dict) -> dict:
  a = bool(f["single_threshold"])
  b = bool(f["auc"] >= 0.90)
  c = bool(f["d_aic_gain"] < 10)
  return dict(a=a, b=b, c=c, passes=a and b and c)


# ------------------------------------------------------------------------------------------ tables
def f(x, nd=3):
  return ka.fnum(x, nd)


def cell_table(cells: list[dict]) -> str:
  hdr = ("| cell | n | L | S_cen (kappa.md) | kappa (S_cen mid) | S_ret ± SD | kappa-hat ret | S_v100 ± SD | kappa-hat v100 | S_rew | walks k/n [Wilson 95%] | mean tracking (unc) |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
  lines = []
  for c in cells:
    dag = " †" if c["n_centered"] == 0 else ""
    lines.append(f"| {c['key']} | {c['n']} | {f(c['L'])} | {f(c['S_mid'])}{dag} | {f(c['kappa_mid'])} | {f(c['S_ret'])} ± {f(c['S_ret_sd'], 2)} | {f(c['kappa_S_ret'])} | "
                 f"{f(c['S_v100'])} ± {f(c['S_v100_sd'], 2)} | {f(c['kappa_S_v100'])} | not logged | {c['walks']}/{c['n']} [{c['wilson'][0]:.2f}, {c['wilson'][1]:.2f}] | {c['track_mean']:.3f} |")
  return hdr + "\n".join(lines)


def fit_table(fits: dict) -> str:
  hdr = ("| x | model | slope per decade | SE | robot (G1) coef | log-lik | AIC | seed-level AUC | AIC gain from robot | one threshold, 5 cells? | strictly monotone rates? |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|\n")
  lines = []
  for name, ft in fits.items():
    a, b = ft["alone"], ft["with_robot"]
    lines.append(f"| {name} | x alone | {a['beta'][1]:+.2f} | {a['se'][1]:.2f} | – | {a['ll']:.1f} | {a['aic']:.1f} | {ft['auc']:.3f} | | {'yes' if ft['single_threshold'] else 'no'} | {'yes' if ft['strictly_monotone'] else 'no'} |")
    lines.append(f"| | x + robot | {b['beta'][1]:+.2f} | {b['se'][1]:.2f} | {b['beta'][2]:+.2f} (SE {b['se'][2]:.2f}) | {b['ll']:.1f} | {b['aic']:.1f} | | **{ft['d_aic_gain']:.1f}** | | |")
  return hdr + "\n".join(lines)


# ------------------------------------------------------------------------------------------ figure
def figure(cells: list[dict], fits: dict, verd: dict) -> None:
  import matplotlib.pyplot as plt
  import make_supervisor_figs as SF

  SF.apply_style()
  FIG = {"go1": dict(marker="o", color=SF.ARM["none"]["color"]), "g1": dict(marker="s", color=SF.ARM["value"]["color"])}
  OUT.mkdir(parents=True, exist_ok=True)
  panels = [("kappa_S_ret", "S_ret", "κ̂ = L / S_ret   (uncentered return std, it 0-50)"),
            ("kappa_S_v100", "S_v100", "κ̂ = L / S_v100   (uncentered v_std, it 50-100)"),
            ("kappa_mid", "κ (S_cen)", "κ = L / S   (centered-arm v_std, it 100-300; kappa.md)")]
  fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)

  def short(c):
    return f"{c['robot'].capitalize()} γ{c['gamma']:g} c={c['c']:g}" + (" pilot" if c["kind"] == "pilot" else "")

  for ax, (col, nm, xl) in zip(list(axes), panels):
    texts = []
    for c in cells:
      st = FIG[c["robot"]]
      hollow = c["kind"] == "pilot"
      x, y = c[col], c["rate"]
      ax.errorbar(x, y, yerr=[[y - c["wilson"][0]], [c["wilson"][1] - y]], fmt="none", ecolor=st["color"], alpha=0.45, lw=1.2, capsize=2.5)
      ax.plot(x, y, marker=st["marker"], ms=9, ls="none", mfc="white" if hollow else st["color"], mec=st["color"], mew=1.8, zorder=3)
      texts.append((x, y, short(c) + f"  {c['walks']}/{c['n']}", st["color"]))
    placed = []
    for x, y, t, colr in sorted(texts, key=lambda a: (np.log10(a[0]), a[1])):
      dy = 0
      while any(abs(np.log10(x) - px) < 0.55 and abs((y + dy * 0.055) - py) < 0.05 for px, py in placed):
        dy += 1
      placed.append((np.log10(x), y + dy * 0.055))
      ax.annotate(t, (x, y), xytext=(7, 7 + 10 * dy), textcoords="offset points", fontsize=7.0, color=colr, ha="left", zorder=2.5,
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.set_xscale("log")
    ax.set_xlabel(xl, fontsize=8.5)
    ax.set_ylim(-0.08, 1.14)
    ax.grid(alpha=0.25, which="both")
    key = {"kappa_S_ret": "S_ret", "kappa_S_v100": "S_v100", "kappa_mid": "centered κ (S mid)"}[col]
    ft = fits[key]
    tag = ""
    if key in verd:
      v = verd[key]
      tag = f"   {'PASS' if v['passes'] else 'FAIL'} (a {'ok' if v['a'] else 'no'}, b {'ok' if v['b'] else 'no'}, c {'ok' if v['c'] else 'no'})"
    ax.set_title(f"{nm}: AUC {ft['auc']:.2f}, ΔAIC robot {ft['d_aic_gain']:.1f}{tag}", fontsize=9)
  axes[0].set_ylabel("uncentered walking rate (k/n, Wilson 95%)")
  hs = [plt.Line2D([], [], marker="o", ls="none", mfc=FIG["go1"]["color"], mec=FIG["go1"]["color"], ms=8, label="Go1 (claim, n=20)"),
        plt.Line2D([], [], marker="s", ls="none", mfc=FIG["g1"]["color"], mec=FIG["g1"]["color"], ms=8, label="G1 (claim)"),
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=8, mew=1.8, label="hollow = pilot (n=3), not in the fit")]
  fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=8.5, frameon=False)
  fig.text(0.5, 0.005, "S_rew (per-step reward std x 1/sqrt(1-g^2)) is not shown: the logged `rew_std` is 1.0 in every row (reward_scale = none); candidate dropped, none substituted.", ha="center", fontsize=7.6, color="#555555")
  fig.suptitle("EXPLORATORY, POST HOC: walking rate vs κ̂ with S estimated from the UNCENTERED arm (right: centered-arm κ from kappa.md for reference)", fontsize=10.2)
  fig.tight_layout(rect=(0, 0.10, 1, 0.95))
  for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_kappa_early.{ext}", dpi=300 if ext == "png" else None)
  plt.close(fig)
  print("wrote", OUT / "fig_kappa_early.png")


def diagnostics(cells: list[dict]) -> dict:
  """NOT a candidate and not used in any verdict or fit. Explains the result: the same two windows measured on the CENTERED arm of each cell,
  and the spread of the uncentered per-seed values. (No kappa-hat is formed from the centered values: that would be a fourth estimator.)"""
  reg = {c["key"]: c for c in ka.go1_cells() + ka.g1_cells()}
  out = {}
  for c in cells:
    cell = reg[c["key"]]
    d = {}
    for nm, key in (("S_ret", "S_ret_seeds"), ("S_v100", "S_v100_seeds")):
      v = np.array(list(c[key].values()))
      d[nm + "_unc_median"], d[nm + "_unc_min"], d[nm + "_unc_max"] = float(np.median(v)), float(v.min()), float(v.max())
    if cell["cen"]:
      sr, sv = [], []
      for s_, jid in cell["cen"].items():
        rows = ka.iter_rows(jid, cell["robot"])["rows"]
        rr = {i: {"x": ret_std_row(r)} for i, r in rows.items() if ret_std_row(r) is not None}
        a, b = ka.window_mean(rr, "x", *W_RET), ka.window_mean(rows, "v_std", *W_V)
        if a is not None:
          sr.append(a)
        if b is not None:
          sv.append(b)
      d.update(S_ret_cen=float(np.mean(sr)), S_v100_cen=float(np.mean(sv)), n_cen=len(sr))
    out[c["key"]] = d
  return out


def diag_table(dg: dict) -> str:
  hdr = ("| cell | S_ret unc (median [min, max] over seeds) | S_ret, same window, centered arm | S_v100 unc (median [min, max]) | S_v100, same window, centered arm |\n|---|---|---|---|---|\n")
  ls = []
  for k, d in dg.items():
    cr = f"{d['S_ret_cen']:.2f}" if "S_ret_cen" in d else "n/a"
    cv = f"{d['S_v100_cen']:.2f}" if "S_v100_cen" in d else "n/a"
    ls.append(f"| {k} | {d['S_ret_unc_median']:.2f} [{d['S_ret_unc_min']:.2f}, {d['S_ret_unc_max']:.2f}] | {cr} | {d['S_v100_unc_median']:.2f} [{d['S_v100_unc_min']:.2f}, {d['S_v100_unc_max']:.2f}] | {cv} |")
  return hdr + "\n".join(ls)


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  aud = audit()
  print("AUDIT", json.dumps(aud))
  cells = build()
  claim = [c for c in cells if c["kind"] == "claim"]
  fits = {"L alone": fit_x(claim, "L"), "centered κ (S mid)": fit_x(claim, "kappa_mid"), "S_ret": fit_x(claim, "kappa_S_ret"), "S_v100": fit_x(claim, "kappa_S_v100")}
  verd = {k: verdict(fits[k]) for k in AVAILABLE}
  # reference rows keep their own keys in `fits` for the figure
  fits["centered κ (S mid)"]["_note"] = "reference: kappa.md primary"
  dg = diagnostics(cells)
  res = dict(diagnostics=dg, audit=aud, cells=cells, fits=fits, verdict=verd, candidates=CANDS, available=AVAILABLE, dropped={"S_rew": "rew_std is 1.0 in every row; reward_scale=none; no per-step reward spread logged"})
  (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
  md = ["## Cell table (generated)\n", cell_table(cells), "\n† S_cen taken from the uncentered arm (no centered counterpart).\n",
        "## Regressions on the 5 claim cells (generated)\n", fit_table(fits), "\n",
        "## Diagnostic only: spread of the uncentered S, and the same windows on the centered arm (generated; not a candidate)\n", diag_table(dg), "\n",
        "## Verdicts (generated)\n", "\n".join(f"- {k}: (a) {v['a']}  (b) {v['b']}  (c) {v['c']}  => {'PASS' if v['passes'] else 'FAIL'}" for k, v in verd.items()), "\n"]
  (OUT / "tables.md").write_text("\n".join(md))
  print("\n".join(md))
  figure(cells, fits, verd)


if __name__ == "__main__":
  main()
