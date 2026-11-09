"""Figures for REPORT_003 (the corrected-protocol study), generated straight from the R2 mirrors.

    python analysis/make_report3_figs.py --out ../docs/mjlab_centering/figures/report3

Every number in the figures is recomputed from `eval.jsonl` (last-5-eval means, newest attempt only). Nothing is
copied from a table in a document. Downloads are cached in /home/<user>/.cache so a rerun is quick.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mjlab_centering import r2

CACHE = Path.home() / ".cache" / "report3"
WALK = 0.5
C_NONE, C_VAL = "#c0392b", "#1f77b4"


def rows(campaign: str, jid: str, name: str) -> list[dict]:
  f = CACHE / campaign / f"{jid}.{name}.jsonl"
  if not f.exists():
    f.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
      p = Path(td) / "x.jsonl"
      if not r2.download_file(f"{campaign}/runs/{jid}/{name}.jsonl", str(p)):
        return []
      f.write_text(p.read_text())
  out = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
  starts = [i for i, r in enumerate(out) if r.get("iter") == 0]
  return out[starts[-1]:] if len(starts) > 1 else out


def last5(ev: list[dict], key: str) -> float:
  v = [r[key] for r in ev[-5:] if key in r]
  return float(np.mean(v)) if v else float("nan")


def load(campaign: str) -> list[dict]:
  man = r2.get_json(f"{campaign}/manifest.json") or {"jobs": []}
  done = r2.done_set(campaign)
  out = []
  for j in man["jobs"]:
    if j["job_id"] not in done:
      continue
    ev = rows(campaign, j["job_id"], "eval")
    if not ev:
      continue
    out.append({**j, "ev": ev, "track": last5(ev, "track_lin_frac"), "J": last5(ev, "J_task")})
  return out


def strip(ax, groups: dict[str, np.ndarray], colors: list[str], ylabel: str, hline=None, title=""):
  for i, (name, v) in enumerate(groups.items()):
    x = np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v))
    ax.scatter(x, v, s=28, color=colors[i], alpha=0.8, zorder=3)
    ax.hlines(np.mean(v), i - 0.3, i + 0.3, color="k", lw=2, zorder=4)
  ax.set_xticks(range(len(groups)))
  ax.set_xticklabels(list(groups))
  ax.set_ylabel(ylabel)
  ax.set_title(title, fontsize=10)
  if hline is not None:
    ax.axhline(hline, ls="--", color="grey", lw=1)
  ax.grid(alpha=0.25)


def fig_main(night: list[dict], out: Path) -> None:
  w = [r for r in night if r["block"] == "w2main"]
  n = {r["seed"]: r for r in w if r["center"] == "none"}
  v = {r["seed"]: r for r in w if r["center"] == "value"}
  fig, ax = plt.subplots(1, 3, figsize=(13, 4))
  strip(ax[0], {"no centering": np.array([r["track"] for r in n.values()]),
                "value centering": np.array([r["track"] for r in v.values()])}, [C_NONE, C_VAL],
        "tracking (last-5 evals)", WALK, "20 seeds: linear-velocity tracking\n(dashed = 0.5 'walks' line)")
  strip(ax[1], {"no centering": np.array([r["J"] for r in n.values()]),
                "value centering": np.array([r["J"] for r in v.values()])}, [C_NONE, C_VAL],
        "J_task (per-step task reward)", 0.077, "20 seeds: task reward\n(dashed = standing policy 0.077)")
  sh = sorted(set(n) & set(v))
  d = np.array([v[s]["J"] - n[s]["J"] for s in sh])
  ax[2].bar(range(len(d)), d, color=C_VAL)
  ax[2].axhline(d.mean(), color="k", lw=1.5)
  ax[2].set_xlabel("seed (3000-3019)")
  ax[2].set_ylabel("paired dJ = centered - uncentered")
  se = d.std(ddof=1) / np.sqrt(len(d))
  ax[2].set_title(f"paired gap: mean {d.mean():+.4f}, se {se:.4f}, n={len(d)}", fontsize=10)
  ax[2].grid(alpha=0.25)
  fig.suptitle("Go1, gamma .999, +1 reward offset, 2000 iterations (corrected protocol)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "r3_01_main_effect.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def mean_curve(rs: list[dict], key="track_lin_frac"):
  its = sorted({e["iter"] for r in rs for e in r["ev"]})
  mat = np.full((len(rs), len(its)), np.nan)
  for i, r in enumerate(rs):
    m = {e["iter"]: e.get(key) for e in r["ev"]}
    for k, it in enumerate(its):
      mat[i, k] = m.get(it, np.nan)
  return np.array(its), np.nanmean(mat, 0), np.nanmin(mat, 0), np.nanmax(mat, 0)


def fig_curves(night: list[dict], out: Path) -> None:
  w = [r for r in night if r["block"] == "w2main"]
  fig, ax = plt.subplots(1, 2, figsize=(12, 4))
  for c, col, lab in (("none", C_NONE, "no centering"), ("value", C_VAL, "value centering")):
    rs = [r for r in w if r["center"] == c]
    its, m, lo, hi = mean_curve(rs)
    ax[0].plot(its, m, color=col, label=f"{lab} (n={len(rs)})")
    ax[0].fill_between(its, lo, hi, color=col, alpha=0.15)
  ax[0].axhline(0.267, ls=":", color="grey")
  ax[0].text(1500, 0.28, "standing", color="grey", fontsize=8)
  ax[0].set_xlabel("PPO iteration")
  ax[0].set_ylabel("tracking")
  ax[0].set_title("Go1 offset +1, gamma .999: mean and min-max band over 20 seeds")
  ax[0].legend()
  ax[0].grid(alpha=0.25)
  o = [r for r in night if r["block"] == "w2off0"]
  for c, col, lab in (("none", C_NONE, "no centering"), ("value", C_VAL, "value centering")):
    rs = [r for r in o if r["center"] == c]
    its, m, lo, hi = mean_curve(rs)
    ax[1].plot(its, m, color=col, label=f"{lab} (n={len(rs)})")
    ax[1].fill_between(its, lo, hi, color=col, alpha=0.15)
  ax[1].set_xlabel("PPO iteration")
  ax[1].set_title("Go1 offset 0, gamma .999: both arms walk (10 seeds)")
  ax[1].legend()
  ax[1].grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out / "r3_02_curves_offset1_vs_offset0.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_offset_grid(night: list[dict], out: Path) -> None:
  cells = [("offset +1, gamma .999\n(20 seeds)", "w2main"), ("offset 0, gamma .999\n(10 seeds)", "w2off0"),
           ("offset 0, gamma .99\n(3 seeds)", "w2off0g99")]
  fig, ax = plt.subplots(figsize=(8, 4.2))
  width = 0.35
  for i, (lab, blk) in enumerate(cells):
    for k, (c, col) in enumerate((("none", C_NONE), ("value", C_VAL))):
      v = np.array([r["track"] for r in night if r["block"] == blk and r["center"] == c])
      x = i + (k - 0.5) * width
      ax.bar(x, v.mean(), width * 0.9, color=col, alpha=0.8, label=("no centering" if c == "none" else "value centering") if i == 0 else None)
      ax.scatter(np.full(len(v), x) + np.random.default_rng(1).uniform(-0.06, 0.06, len(v)), v, s=10, color="k", zorder=3)
  ax.set_xticks(range(len(cells)))
  ax.set_xticklabels([c[0] for c in cells])
  ax.set_ylabel("tracking (last-5 evals)")
  ax.axhline(0.5, ls="--", color="grey")
  ax.set_title("Where centering matters: only when the +1 offset is present (Go1)")
  ax.legend()
  ax.grid(alpha=0.25, axis="y")
  fig.tight_layout()
  fig.savefig(out / "r3_03_offset_necessity.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_g1(night: list[dict], out: Path) -> None:
  w = [r for r in night if r["robot"] == "g1" and r["job_id"].startswith(("w2g1c57_", "w2g1c57b_"))]
  n = {r["seed"]: r for r in w if r["center"] == "none"}
  v = {r["seed"]: r for r in w if r["center"] == "value"}
  fig, ax = plt.subplots(1, 3, figsize=(13, 4))
  strip(ax[0], {"no centering": np.array([r["track"] for r in n.values()]),
                "value centering": np.array([r["track"] for r in v.values()])}, [C_NONE, C_VAL],
        "tracking (last-5 evals)", WALK, "G1, 6 seeds, gamma .99, offset +1")
  for c, col, lab in (("none", C_NONE, "no centering"), ("value", C_VAL, "value centering")):
    rs = [r for r in w if r["center"] == c]
    its, m, lo, hi = mean_curve(rs)
    ax[1].plot(its, m, color=col, label=f"{lab} (n={len(rs)})")
    ax[1].fill_between(its, lo, hi, color=col, alpha=0.15)
  ax[1].axhline(0.267, ls=":", color="grey")
  ax[1].set_xlabel("PPO iteration")
  ax[1].set_title("G1 learning curves")
  ax[1].legend()
  ax[1].grid(alpha=0.25)
  hi = [r for r in night if r["robot"] == "g1" and r["gamma"] == 0.999 and r["offset"] == 0.0 and r["job_id"].startswith("w2g1c999")]
  strip(ax[2], {"no centering": np.array([r["track"] for r in hi if r["center"] == "none"]),
                "value centering": np.array([r["track"] for r in hi if r["center"] == "value"])}, [C_NONE, C_VAL],
        "tracking (last-5 evals)", WALK, "G1, gamma .999, offset 0 (3 seeds):\nunstable, no claim")
  fig.tight_layout()
  fig.savefig(out / "r3_04_g1_second_robot.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_shock(night: list[dict], out: Path) -> None:
  sh = [r for r in night if r["block"] == "w2shock"]
  fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
  for k, (kind, title) in enumerate((("off+0.1", "+0.1 per-step reward offset"), ("actx5", "action-rate penalty weight x5"))):
    rs = [r for r in sh if f"_{kind}_" in r["job_id"]]
    for r in rs:
      its = [e["iter"] for e in r["ev"]]
      ax[k].plot(its, [e["track_lin_frac"] for e in r["ev"]], alpha=0.5, lw=1)
    its, m, lo, hi = mean_curve(rs)
    ax[k].plot(its, m, color="k", lw=2, label=f"mean of {len(rs)} seeds")
    ax[k].axvline(1200, color="red", ls="--")
    ax[k].text(1210, 0.5, "shock", color="red")
    ax[k].set_title(title)
    ax[k].set_xlabel("PPO iteration")
    ax[k].grid(alpha=0.25)
    ax[k].legend(loc="lower right")
  ax[0].set_ylabel("tracking")
  fig.suptitle("Shock test: a walking, centered Go1 hit at iteration 1200 (800 iterations to recover)", y=1.02)
  fig.tight_layout()
  fig.savefig(out / "r3_05_shock.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def fig_bug(out: Path) -> None:
  """RUN_M04/M05: the fall-reset bug. Buggy protocol vs corrected, G1 seeds 1000-1002, gamma .99, offset 0."""
  fig, ax = plt.subplots(1, 2, figsize=(12, 4))
  groups = [("pilot1-9f3a1c", ["pilot_g1_raw_none_o0_s1000", "pilot_g1_raw_none_o0_s1001", "pilot_g1_raw_none_o0_s1002"],
             C_NONE, "fall did NOT reset the robot (decision-77 bug), gamma .999"),
            ("fix1-b7c40a", ["fix_g1_raw_none_o0_s1000", "fix_g1_raw_none_o0_s1001", "fix_g1_raw_none_o0_s1002"],
             "#2ca02c", "fall resets the robot, hidden from PPO (fixed), gamma .99")]
  for camp, jids, col, lab in groups:
    for k, jid in enumerate(jids):
      ev = rows(camp, jid, "eval")
      it = rows(camp, jid, "iter")
      if not ev:
        continue
      ax[0].plot([e["iter"] for e in ev], [e["track_lin_frac"] for e in ev], color=col, alpha=0.6, label=lab if k == 0 else None)
      if it:
        ax[1].plot([r["iter"] for r in it], [r["fall_rate_per_1k"] for r in it], color=col, alpha=0.6, label=lab if k == 0 else None)
  ax[0].axhline(0.267, ls=":", color="grey")
  ax[0].set_xlabel("PPO iteration")
  ax[0].set_ylabel("tracking")
  ax[0].set_title("G1 tracking: the bug made the humanoid unlearnable")
  ax[0].legend(fontsize=8)
  ax[1].set_xlabel("PPO iteration")
  ax[1].set_ylabel("fall-flag steps per 1k")
  ax[1].set_title("Fall rate: ~960/1k = robot lying on the floor")
  ax[1].legend(fontsize=8)
  for a in ax:
    a.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(out / "r3_06_fall_reset_bug.png", dpi=140, bbox_inches="tight")
  plt.close(fig)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  night = load("night1-3e71b4")
  print(f"loaded {len(night)} completed night1 runs")
  fig_main(night, out)
  fig_curves(night, out)
  fig_offset_grid(night, out)
  fig_g1(night, out)
  fig_shock(night, out)
  fig_bug(out)
  print("figures written to", out)


if __name__ == "__main__":
  main()
