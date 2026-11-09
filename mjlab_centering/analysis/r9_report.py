"""REPORT_009 pilot readout: the race arms against the existing uncentered/value cells, paired by seed.

    python analysis/r9_report.py --campaign r9pilot-xxxxxxxx [--pull] [--json results/r9_claim.json] [--fig]

Conventions are REPORT_003-007's and are not re-derived here: metric = last-five-eval mean of `track_lin_frac`;
"walks" = that mean >= 0.5; paired contrasts are per-seed differences with a two-sided 95 % t-interval; fewer than
20 complete pairs is a pilot. The rescue rule is d303's: walks on >= 7 of 10 AND the paired dtrack CI against the
uncentered cell excludes 0.

The race statistics reuse `analysis/r9_step0.py` so the pilot and the retrospective table are computed by the same
code: t_level = first iteration at 90 % of c/(1-gamma) (`v_mean`; centred arms: const_frac < 0.1), t_commit = first
iteration with `action_std` < 0.8 x its initial value, plus the `const_frac` level and the settled variants.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_step0 as S  # noqa: E402
import report4 as R4  # noqa: E402
from report4 import mean_last5, paired, seed_of  # noqa: E402
from mjlab_centering import r2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "merged_r9"
OUT = ROOT.parent / "docs" / "mjlab_centering" / "figures" / "report9"
WALK = 0.5
ARMS = ("a1_05", "a1_005", "clipoff", "a2_off", "a2_on", "a3")
LABEL = {
  "a1_05": "A1 fast bias, lr 0.5 (clip off)",
  "a1_005": "A1 fast bias, lr 0.05 (clip off)",
  "clipoff": "control: clip off only",
  "a2_off": "A2 critic warm-up, clip OFF",
  "a2_on": "A2 critic warm-up, clip ON",
  "a3": "A3 actor lr x0.1, 500 iterations",
  "none": "no centering (reference)",
  "value": "value centering (reference)",
}
PALETTE = {"a1_05": "#1f77b4", "a1_005": "#aec7e8", "clipoff": "#8c564b", "a2_off": "#2ca02c",
           "a2_on": "#98df8a", "a3": "#9467bd", "none": "#c0392b", "value": "#ff7f0e"}


def pull(campaign: str, jobs: list[str]) -> None:
  done = r2.done_set(campaign)
  for jid in jobs:
    if jid not in done:
      continue
    d = MERGED / jid
    if (d / "eval.jsonl").exists() and (d / "iter.jsonl").exists():
      continue
    d.mkdir(parents=True, exist_ok=True)
    for name in ("eval.jsonl", "iter.jsonl", "final.json", "config.json"):
      with tempfile.TemporaryDirectory() as td:
        p = Path(td) / name
        if r2.download_file(f"{campaign}/runs/{jid}/{name}", str(p)):
          (d / name).write_bytes(p.read_bytes())


def _rows(jid: str, name: str) -> list[dict]:
  """Merged rows (one per iteration / eval) from this round's mirror or the older ones, via the same loader as Step 0."""
  for base in (MERGED, ROOT / "results" / "merged_r4", ROOT / "results" / "merged_r2"):
    p = base / jid / name
    if p.exists():
      rows = S.jl(p)
      if rows:
        return rows
  return []


def evals(jid: str) -> list[dict]:
  return _rows(jid, "eval.jsonl")


def iters(jid: str) -> list[dict]:
  return _rows(jid, "iter.jsonl")


def reference(arm: str, seed: int) -> str:
  return f"w2main_go1_raw_{arm}_g0.999_o1_s{seed}"


def clipoff_job(seed: int) -> str:
  """Seeds 3000-3002 predate this round (`clip1_*`, the round-4 pilot); 3003-3009 are new (`r9a_*`)."""
  if seed <= 3002:
    return f"clip1_go1_raw_none_clipoff_g0.999_o1_s{seed}"
  return f"r9a_go1_clipoff_g0.999_o1_s{seed}"


def cell(jid: str) -> dict:
  ev = evals(jid)
  if not ev:
    return {"job_id": jid, "ok": False}
  tr = mean_last5(ev, "track_lin_frac")
  j = mean_last5(ev, "J_task")
  rows = iters(jid)
  t = S.times(rows, "level", 0.999, 1.0) if rows else None
  cf = [r.get("const_frac") for r in rows[100:600] if r.get("const_frac") is not None]
  bias90 = next((r["iter"] + 1 for r in rows if r.get("critic_bias") is not None
                 and r["critic_bias"] >= 900.0), None)
  warm_end = next((int(r["warmup_done_at"]) for r in rows if r.get("warmup_done_at", -1) >= 0), None)
  return {"job_id": jid, "ok": True, "track": tr, "J": j, "walks": tr >= WALK,
          "falls": mean_last5(ev, "fall_rate_per_1k"), "n_evals": len(ev),
          "t50": S.first_eval(ev, 0.5), "t90rel": S.first_eval(ev, 0.9 * tr) if tr >= WALK else None,
          "cf_hi": float(np.mean([c > 0.5 for c in cf])) if cf else None, "bias90": bias90,
          "warm_end": warm_end, "t": t}


def med_int(xs) -> str:
  """Median as an integer label; 'inf' when the median itself is unbounded (every run never reached the time)."""
  if not xs:
    return "—"
  m = float(np.median(xs))
  return "inf" if not np.isfinite(m) else f"{int(m)}"


def fmt(x, nd=3):
  return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pv(p: dict) -> str:
  return f"{p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  ap.add_argument("--pull", action="store_true")
  ap.add_argument("--json", default=str(ROOT / "results" / "r9_pilot.json"))
  ap.add_argument("--fig", action="store_true")
  a = ap.parse_args()
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  ids = [j["job_id"] for j in man["jobs"]]
  if a.pull:
    pull(a.campaign, ids)
  done = r2.done_set(a.campaign)

  seeds = sorted({seed_of(i) for i in ids})
  cells: dict[str, dict[int, dict]] = {}
  for s in seeds:
    cells.setdefault("none", {})[s] = cell(reference("none", s))
    cells.setdefault("value", {})[s] = cell(reference("value", s))
    for arm in ARMS:
      jid = clipoff_job(s) if arm == "clipoff" else f"r9a_go1_{arm}_g0.999_o1_s{s}"
      c = cell(jid)
      if c.get("ok"):
        cells.setdefault(arm, {})[s] = c
  for s in seeds:  # value/none references exist for every seed; drop the placeholder-only entries
    for k in ("none", "value"):
      if not cells[k].get(s, {}).get("ok"):
        cells[k].pop(s, None)

  print(f"campaign {a.campaign}: {len(done)}/{len(ids)} runs complete\n")
  head = ("| arm | n | walks k/n | tracking | J_task | dtrack vs uncentered (95% CI) | dtrack vs value (95% CI) | "
          "dJ vs uncentered (95% CI) | t50 | t_level | t_commit | cf>0.5 (it100-600) | rescue (d303) |")
  print(head)
  print("|" + "---|" * 13)
  out = {}
  for arm in ARMS:
    d = cells.get(arm, {})
    if not d:
      print(f"| {LABEL[arm]} | 0 | — | — | — | — | — | — | — | — | — | — | — |")
      continue
    tr = {s: c["track"] for s, c in d.items()}
    j = {s: c["J"] for s, c in d.items()}
    ref_tr = {s: cells["none"][s]["track"] for s in d if s in cells["none"]}
    ref_j = {s: cells["none"][s]["J"] for s in d if s in cells["none"]}
    v_tr = {s: cells["value"][s]["track"] for s in d if s in cells["value"]}
    p_n = paired(tr, ref_tr)
    p_v = paired(tr, v_tr)
    p_jn = paired(j, ref_j)
    walks = sum(c["walks"] for c in d.values())
    tl = [c["t"]["A"]["t_level"] for c in d.values() if c["t"]]
    tc = [c["t"]["A"]["t_commit"] for c in d.values() if c["t"]]
    t50 = [c["t50"] for c in d.values() if c["t50"]]
    rescue = (walks >= 7 and p_n["ci"][0] > 0) if p_n["n"] >= 2 else False
    verdict = ("**rescues**" if rescue else "no") if len(d) >= 7 else f"n={len(d)} (pilot)"
    print(f"| {LABEL[arm]} | {len(d)} | {walks}/{len(d)} | {np.mean(list(tr.values())):.3f} | "
          f"{np.mean(list(j.values())):.4f} | {pv(p_n)} | {pv(p_v)} | {pv(p_jn)} | "
          f"{med_int(t50)} | {med_int(tl)} | {med_int(tc)} | "
          f"{np.mean([c['cf_hi'] for c in d.values() if c['cf_hi'] is not None]):.2f} | {verdict} |")
    out[arm] = {"n": len(d), "walks": walks, "track": float(np.mean(list(tr.values()))),
                "J": float(np.mean(list(j.values()))), "vs_none": p_n, "vs_value": p_v, "dJ_vs_none": p_jn,
                "failed_seeds": [s for s, c in d.items() if not c["walks"]],
                "t_level": tl, "t_commit": tc, "t50": t50, "rescue": bool(rescue)}
  for k in ("none", "value"):
    d = cells[k]
    tr = [c["track"] for c in d.values()]
    jj = [c["J"] for c in d.values()]
    print(f"| {LABEL[k]} | {len(d)} | {sum(c['walks'] for c in d.values())}/{len(d)} | "
          f"{np.mean(tr):.3f} | {np.mean(jj):.4f} | — | — | — | — | — | — | — | — |")
  print("\n**Mechanism readout** (median over completed seeds; `bias90` = first iteration the critic's output bias "
        "reaches 0.9 x c/(1-gamma); `warm-up ended` = the A2 rule's own trigger):\n")
  print("| arm | n | bias90 | warm-up ended | t_level | t_commit | walks |")
  print("|---|---|---|---|---|---|---|")
  for arm in ARMS:
    d = cells.get(arm, {})
    if not d:
      continue
    b = [c["bias90"] for c in d.values() if c.get("bias90")]
    w = [c["warm_end"] for c in d.values() if c.get("warm_end") is not None]
    tl = [c["t"]["A"]["t_level"] for c in d.values() if c.get("t")]
    tc = [c["t"]["A"]["t_commit"] for c in d.values() if c.get("t")]
    med = med_int
    print(f"| {LABEL[arm]} | {len(d)} | {med(b)} | {med(w)} | {med(tl)} | {med(tc)} | "
          f"{sum(c['walks'] for c in d.values())}/{len(d)} |")
  early_levels(cells)
  Path(a.json).write_text(json.dumps(out, indent=1))
  print(f"\nwrote {a.json}")
  if a.fig:
    figure(cells)
    print(f"wrote {OUT / 'r9_02_pilot_tracking.png'}")
    figure_race(cells)
    print(f"wrote {OUT / 'r9_03_race_all.png'}")


def early_levels(cells: dict[str, dict[int, dict]]) -> None:
  """The mechanism table: how fast the critic's mean value rises, and how much of its error is state-constant.

  `v_mean` at a few early iterations (median over the arm's completed seeds) beside `const_frac` at iteration 100, next
  to the two reference cells. This is what separates the arms that walk from the arms that do not, and it is not the
  speed of the mean level.
  """
  groups = {"no centering (reference)": {}, "bias-init (reference)": {}}
  for s_ in cells.get("none", {}):
    groups["no centering (reference)"][s_] = f"w2main_go1_raw_none_g0.999_o1_s{s_}"
    groups["bias-init (reference)"][s_] = f"r5bias_go1_g0.999_o1_none_s{s_}"
  for arm in ARMS:
    groups[LABEL[arm]] = {s_: c["job_id"] for s_, c in cells.get(arm, {}).items()}
  print("\n**Early level and its state-constant share** (median over completed seeds):\n")
  print("| group | n | v_mean @25 | @50 | @100 | @200 | @400 | const_frac @100 | v_mean/ret_mean @100 |")
  print("|---|---|---|---|---|---|---|---|---|")
  for g, jobs in groups.items():
    if not jobs:
      continue
    def med(key, it):
      vals = []
      for jid in jobs.values():
        d = {r["iter"]: r for r in iters(jid)}
        if it in d and key in d[it]:
          vals.append(d[it][key])
      return float(np.mean(vals)) if vals else float("nan")
    ratio = med("v_mean", 100) / med("ret_mean", 100) if med("ret_mean", 100) else float("nan")
    print(f"| {g} | {len(jobs)} | " + " | ".join(f"{med('v_mean', it):.0f}" for it in (25, 50, 100, 200, 400))
          + f" | {med('const_frac', 100):.2f} | {ratio:.2f} |")


def figure(cells: dict[str, dict[int, dict]]) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(9, 5.2))
  for arm, d in cells.items():
    if not d:
      continue
    curves = []
    for s, c in d.items():
      ev = evals(c["job_id"])
      if ev:
        curves.append([r["track_lin_frac"] for r in ev])
    if not curves:
      continue
    n = max(len(c) for c in curves)
    m = np.array([c + [np.nan] * (n - len(c)) for c in curves])
    it = np.array([r["iter"] + 1 for r in evals(next(iter(d.values()))["job_id"])])[:n]
    ax.plot(it, np.nanmean(m, axis=0), color=PALETTE[arm], lw=2.2 if arm not in ("none", "value") else 1.4,
            ls="--" if arm in ("none", "value") else "-", label=f"{LABEL[arm]} (n={len(d)})")
  ax.axhline(WALK, color="grey", lw=0.8, ls=":")
  ax.set_xlabel("iteration")
  ax.set_ylabel("tracking (eval, mean of last five not applied)")
  ax.set_title("REPORT_009 pilot: eval tracking per arm (mean over completed seeds)")
  ax.legend(fontsize=8)
  fig.tight_layout()
  OUT.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUT / "r9_02_pilot_tracking.png", dpi=130)


def figure_race(cells: dict[str, dict[int, dict]]) -> None:
  """The race figure the deliverable asks for: every existing cell plus every pilot arm, t_level vs t_commit."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  runs = S.collect()
  cap, rng = 2800.0, np.random.default_rng(1)
  fig, ax = plt.subplots(figsize=(8.2, 6.4))
  for r in runs:  # existing cells: grey, outcome by colour
    t = r["A"]
    ax.scatter((min(t["t_commit"], cap) + 1) * np.exp(rng.normal(0, .03)),
               (min(t["t_level"], cap) + 1) * np.exp(rng.normal(0, .03)),
               s=22, c="#2ca02c" if r["walks"] else "#d62728", alpha=.35, edgecolors="none")
  for arm, d in cells.items():
    for s_, c in d.items():
      if not c.get("t"):
        continue
      t = c["t"]["A"]
      ax.scatter(min(t["t_commit"], cap) + 1, min(t["t_level"], cap) + 1, s=90,
                 marker={"a1_05": "o", "a1_005": "o", "clipoff": "v", "a2_off": "^", "a2_on": "^", "a3": "P",
                         "none": "X", "value": "*"}[arm],
                 facecolors=PALETTE[arm], edgecolors="k" if c["walks"] else "none", linewidths=.7, alpha=.9)
  lo, hi = .8, cap * 1.5
  ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
  ax.axhline(cap + 1, color="grey", lw=.5, ls=":"); ax.axvline(cap + 1, color="grey", lw=.5, ls=":")
  ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
  ax.set_xlabel("t_commit (iteration, +1; right edge = never)")
  ax.set_ylabel("t_level (iteration, +1; top edge = never)")
  ax.set_title("REPORT_009: t_level vs t_commit, all existing cells (pale) and pilot arms (bold)")
  h = [Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="none", label="walks"),
       Line2D([], [], marker="o", ls="", mfc="#d62728", mec="none", label="does not walk")]
  h += [Line2D([], [], marker=m, ls="", mfc=PALETTE[k], mec="none", label=LABEL[k])
        for k, m in (("a1_05", "o"), ("a1_005", "o"), ("clipoff", "v"), ("a2_off", "^"), ("a2_on", "^"), ("a3", "P"))]
  ax.legend(handles=h, fontsize=8, loc="lower right")
  fig.tight_layout()
  fig.savefig(OUT / "r9_03_race_all.png", dpi=130)


if __name__ == "__main__":
  main()
