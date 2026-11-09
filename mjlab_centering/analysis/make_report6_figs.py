"""Figures for REPORT_006, computed from the run files (R2), never retyped.

    python analysis/make_report6_figs.py [--out ../docs/mjlab_centering/figures/report6]

1. `r6_01_g1_probe.png`  - G1 probe: tracking curves, centered vs bias-init, with the 2000 check and the 2075 verdict.
2. `r6_02_g1_claim.png`  - G1 claim set: per-seed final tracking for the three arms.
3. `r6_03_h2_arms.png`   - H2: per-seed final-eval scores for the three arms, with the paired difference interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

PROBE_CAMP = "r6a-ad66ecf2"
CLAIM_CAMP = "r6g1queue-0ede5636"
SEEDS = (1000, 1001, 1002)
CLAIM_SEEDS = tuple(range(3000, 3020))


def _get(campaign: str, job: str, name: str) -> str | None:
  from mjlab_centering import r2
  try:
    return r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{campaign}/runs/{job}/{name}"))["Body"].read().decode()
  except Exception:  # noqa: BLE001
    return None


def _rows(campaign: str, job: str) -> list[dict]:
  body = _get(campaign, job, "eval.jsonl")
  if not body:
    return []
  rows = [json.loads(l) for l in body.splitlines() if l.strip()]
  first = min(r["iter"] for r in rows)
  starts = [i for i, r in enumerate(rows) if r["iter"] == first]
  return sorted(rows[starts[-1]:], key=lambda r: r["iter"])


def fig_probe(out: Path) -> bool:
  arms = {"centered": f"r6probe_g1_g0.99_o1_value_s{{}}", "bias-init": f"r6probe_g1_g0.99_o1_bias_none_s{{}}"}
  data = {k: [_rows(PROBE_CAMP, p.format(s)) for s in SEEDS] for k, p in arms.items()}
  if not any(any(x) for x in data.values()):
    print("probe: no rows")
    return False
  fig, ax = plt.subplots(figsize=(7.2, 4.2))
  colors = {"centered": "#1f77b4", "bias-init": "#d62728"}
  for name, runs in data.items():
    for i, rows in enumerate(runs):
      if not rows:
        continue
      it = [r["iter"] + 1 for r in rows]
      y = [r["track_lin_frac"] for r in rows]
      ax.plot(it, y, color=colors[name], alpha=0.35, lw=1)
    full = [[r["track_lin_frac"] for r in rows] for rows in runs if rows]
    if full:
      n = min(len(v) for v in full)
      m = np.mean([v[:n] for v in full], axis=0)
      its = [r["iter"] + 1 for r in runs[[i for i, v in enumerate(runs) if v][0]]][:n]
      ax.plot(its, m, color=colors[name], lw=2.4, label=name)
  ax.axvline(2000, color="grey", ls=":", lw=1.2)
  ax.axvline(2075, color="black", ls="--", lw=1.1)
  ax.text(2010, 0.70, "2075\n(flat)", fontsize=8, va="top", ha="left")
  ax.text(1990, 0.70, "2000 ", fontsize=8, va="top", ha="right", color="#1f77b4")
  ax.set_xlabel("iteration")
  ax.set_ylabel("track_lin_frac")
  ax.set_title("G1 probe: centered climbs to 2000, bias-init plateaus earlier")
  ax.legend(loc="lower right")
  fig.tight_layout()
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r6_01_g1_probe.png", dpi=140)
  plt.close(fig)
  return True


def fig_claim(out: Path) -> bool:
  patterns = {"none": "r6g1_g1_g0.99_o1_none_s{}", "value": "r6g1_g1_g0.99_o1_value_s{}",
              "bias": "r6g1_g1_g0.99_o1_bias_none_s{}"}
  vals: dict[str, list[float]] = {}
  for name, p in patterns.items():
    v = []
    for s in CLAIM_SEEDS:
      rows = _rows(CLAIM_CAMP, p.format(s))
      if len(rows) >= 5:
        v.append(float(np.mean([r["track_lin_frac"] for r in rows[-5:]])))
    if v:
      vals[name] = v
  if len(vals) < 2:
    print("claim: not enough finished seeds")
    return False
  fig, ax = plt.subplots(figsize=(6.4, 3.8))
  for i, (name, v) in enumerate(vals.items()):
    ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=18, alpha=0.8,
               color={"none": "#7f7f7f", "value": "#1f77b4", "bias": "#d62728"}[name])
    ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="black", lw=2)
    ax.text(i, 0.45, f"{sum(x >= 0.5 for x in v)}/{len(v)} walk", ha="center", fontsize=8)
  ax.axhline(0.5, color="grey", ls="--", lw=1)
  ax.set_xticks(range(len(vals)), list(vals))
  ax.set_ylabel("last-5 tracking")
  ax.set_title("G1 claim set: one line of critic-bias init matches centering")
  fig.tight_layout()
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r6_02_g1_claim.png", dpi=140)
  plt.close(fig)
  return True


def fig_h2(out: Path) -> bool:
  import h2_report
  sums = {a: h2_report.load(a) for a in h2_report.ARMS}
  if not all(sums.values()):
    print("h2: missing summaries")
    return False
  rep = h2_report.read(sums)
  fig, ax = plt.subplots(figsize=(6.6, 4.0))
  order = ["ssp_h2", "ssp_h2raw", "tile_h2"]
  for i, a in enumerate(order):
    v = sums[a]["final_eval"]["per_seed_scores"]
    ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=16, alpha=0.8)
    ax.hlines(np.mean(v), i - 0.25, i + 0.25, color="black", lw=2)
  ax.set_xticks(range(3), ["SSP norm", "SSP raw", "tile"])
  ax.set_ylabel("final-eval score (20 seeds)")
  d = rep["vs_tile"]
  ax.set_title(f"H2: SSP norm - tile = {d['mean']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}], margin {d['margin']:.1f}")
  fig.tight_layout()
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r6_03_h2_arms.png", dpi=140)
  plt.close(fig)
  return True


def fig_stage_a(out: Path) -> bool:
  import ssp_readout
  rep = ssp_readout.width_report(ROOT / "results" / "merged_r2")
  cells = rep.get("cells", {})
  order = [("raw", "raw"), ("ssp2x2@0.03", "2x2\n0.03"), ("ssp2x2@0.1", "2x2\n0.1"), ("ssp2x2@0.3", "2x2\n0.3"),
           ("ssp2x2@1", "2x2\n1.0"), ("ssp4x4@0.03", "4x4\n0.03"), ("ssp4x4@0.1", "4x4\n0.1"),
           ("ssp4x4@0.3", "4x4\n0.3"), ("ssp4x4@1", "4x4\n1.0"), ("rff2x2", "RFF\n2x2"), ("rff4x4", "RFF\n4x4")]
  have = [(k, lab) for k, lab in order if k in cells and cells[k]["n"] == cells[k]["of"]]
  if not have:
    print("stage A: no complete cells")
    return False
  labels = [lab for _, lab in have]
  means = [cells[k]["mean"] for k, _ in have]
  walks = [cells[k]["walks"] for k, _ in have]
  fig, ax = plt.subplots(figsize=(8.6, 4.2))
  colors = ["#444444" if k == "raw" else "#8c8c8c" if k.startswith("rff") else "#1f77b4" for k, _ in have]
  bars = ax.bar(range(len(have)), means, color=colors)
  for i, (b, w) in enumerate(zip(bars, walks)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{w}/3", ha="center", fontsize=8)
  ax.set_xticks(range(len(have)), labels, fontsize=8)
  ax.axhline(0.886, color="#444444", ls=":", lw=1.2)
  ax.text(len(have) - 0.6, 0.90, "raw = 0.886", fontsize=8, ha="right", color="#444444")
  ax.set_ylim(0, 1.0)
  ax.set_ylabel("mean tracking (3 search seeds)")
  ax.set_title("Stage A: width x length scale, with equal-width random Fourier controls")
  fig.tight_layout()
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r6_04_stage_a.png", dpi=140)
  plt.close(fig)
  return True


def fig_pilot(out: Path) -> bool:
  """Per-seed paired scatter: SSP (2x2 @ ls 1.0) against the raw cell on the same seed, both offsets, both arms."""
  import ssp_readout
  rep = ssp_readout.pilot_report(2, 1.0, ROOT / "results" / "merged_r2")
  if not rep["offsets"]["0.1"]["none"]["n_pairs"]:
    print("pilot: no pairs")
    return False
  fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
  for ax, off in zip(axes, ["0.1", "1.0"]):
    for arm, color, marker in (("none", "#1f77b4", "o"), ("value", "#d62728", "s")):
      rec = rep["offsets"][off][arm]
      if not rec.get("n_pairs"):
        continue
      # per-seed pairs straight from the mirror
      xs, ys = [], []
      for s_i, s in enumerate(ssp_readout.PILOT_SEEDS):
        a = ssp_readout.track(ssp_readout.pilot_ids(2, 1.0, float(off), arm, s)[0], ROOT / "results" / "merged_r2")
        b = ssp_readout.track(ssp_readout.pilot_ids(2, 1.0, float(off), arm, s)[1], ROOT / "results" / "merged_r2")
        if a is not None and b is not None:
          xs.append(b); ys.append(a)
      ax.scatter(xs, ys, s=42, color=color, marker=marker, alpha=0.85, label=f"{arm} arm")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("raw tracking")
    ax.set_title(f"offset {off}")
    ax.legend(fontsize=8, loc="upper left")
  axes[0].set_ylabel("SSP tracking (2x2 @ ls 1.0)")
  fig.suptitle("Stage B pilot, per seed: points above the diagonal favour SSP", fontsize=9)
  fig.tight_layout()
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "r6_05_pilot_pairs.png", dpi=140)
  plt.close(fig)
  return True


def main(argv: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=str(ROOT.parent / "docs" / "mjlab_centering" / "figures" / "report6"))
  a = ap.parse_args()
  out = Path(a.out)
  for fn in (fig_probe, fig_claim, fig_h2, fig_stage_a, fig_pilot):
    print(fn.__name__, "ok" if fn(out) else "skipped")


if __name__ == "__main__":
  main(sys.argv[1:])
