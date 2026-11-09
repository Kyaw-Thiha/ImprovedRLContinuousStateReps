"""Critic-window mechanism analysis on corrected-protocol runs, straight from the per-iteration diagnostics.

    # E11 (Go1, +1 offset, gamma .999, 20 seeds x {none, value})
    python analysis/mechanism_e11.py --runs results/merged_r2 --out ../docs/mjlab_centering/figures/report4

    # the clip-off pilot (one arm), same statistics (decision 102)
    python analysis/mechanism_e11.py --runs results/merged_r2 --out ../docs/mjlab_centering/figures/report4 \
        --set none=clip1_go1_raw_none_clipoff_g0.999_o1_s* --name clip1

Uses only fields `CenteredPPO.compute_returns` already logs (no new compute):

  err        = ret - V   (rsl-rl: returns = GAE advantages + values, so err IS the batch's GAE advantage)
  err_mse    = mean(err^2)              const_part = err_mean^2          state_part = err_mse - err_mean^2
  const_frac = err_mean^2 / err_mse     (share of critic error that is a constant offset)
  ev         = 1 - err_mse / var(ret)   ev_state   = 1 - state_part / var(ret), with var(ret) = err_mse / (1 - ev)
  V/target   = v_mean / (r_seen_rate / (1 - gamma))   (uncentered arms only: a centered critic predicts a differential
                                                       value, so the ratio is not defined for it)

Caveat built into the definitions: `err` is a bootstrapped lambda-return error, not a Monte-Carlo error against the true
value, and two arms that have taken off run different policies; the early window (iters 100-300) is the like-for-like one.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

G = 0.999
WINDOWS = [(100, 300), (600, 1000), (1750, 2000)]
COLORS = {"none": "#c0392b", "value": "#1f77b4", "clipoff": "#e67e22", "rff": "#7f7f7f"}


def load(path: str) -> list[dict]:
  r = [json.loads(l) for l in open(path) if l.strip()]
  starts = [i for i, x in enumerate(r) if x["iter"] == 0]
  return r[starts[-1]:] if len(starts) > 1 else r  # newest attempt only


def series(rows: list[dict]) -> dict[str, np.ndarray]:
  g = lambda k: np.array([x[k] for x in rows], dtype=float)
  mse, em, ev = g("err_mse"), g("err_mean"), g("ev")
  state = mse - em**2
  varret = mse / np.maximum(1 - ev, 1e-9)
  r = dict(it=g("iter"), mse=mse, const=em**2, state=state, const_frac=g("const_frac"), ev=ev, varret=varret,
           ev_state=1 - state / varret, v=g("v_mean"), tgt=g("r_seen_rate") / (1 - G),
           ratio=g("v_mean") / (g("r_seen_rate") / (1 - G)), std=g("action_std") if "action_std" in rows[0] else np.full(len(rows), np.nan))
  return r


def roll(x: np.ndarray, k: int = 25) -> np.ndarray:
  if len(x) < k:
    return x
  return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", default="results/merged_r2")
  ap.add_argument("--set", action="append", default=None, help="name=glob (repeatable)")
  ap.add_argument("--name", default="e11", help="output file stem")
  ap.add_argument("--out", default="../docs/mjlab_centering/figures/report4")
  ap.add_argument("--title", default=None)
  a = ap.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  specs = a.set or ["none=w2main_go1_raw_none_g0.999_o1_s*", "value=w2main_go1_raw_value_g0.999_o1_s*"]

  S: dict[str, dict[int, dict]] = {}
  allure: dict[str, list[float]] = {}
  for spec in specs:
    name, pat = spec.split("=", 1)
    S[name], allure[name] = {}, []
    for d in sorted(glob.glob(f"{a.runs}/{pat}")):
      seed = int(d.rsplit("_s", 1)[1])
      S[name][seed] = series(load(d + "/iter.jsonl"))
      evp = Path(d) / "eval.jsonl"
      if evp.exists():
        ev = [json.loads(l) for l in open(evp) if l.strip()]
        allure[name].append(float(np.mean([e["track_lin_frac"] for e in ev[-5:]])))
  for k, v in S.items():
    print(f"{k}: {len(v)} seeds, walk mean {np.mean(allure[k]):.3f}" if allure[k] else f"{k}: {len(v)} seeds")

  def wmean(arm: str, key: str, lo: int, hi: int) -> np.ndarray:
    return np.array([s[key][(s["it"] >= lo) & (s["it"] < hi)].mean() for s in S[arm].values()])

  res: dict = {"sets": {k: len(v) for k, v in S.items()}, "windows": {}}
  lines = ["| window | arm | V/target | const_frac | ev | state part | var(ret) | ev_state | action std |",
           "|---|---|---|---|---|---|---|---|---|"]
  for lo, hi in WINDOWS:
    for arm in S:
      m = {k: wmean(arm, k, lo, hi) for k in ("ratio", "const_frac", "ev", "state", "varret", "ev_state", "std")}
      n = len(m["state"])
      res["windows"][f"{lo}-{hi}_{arm}"] = {k: [float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))] if n > 1 else [float(v.mean()), 0.0] for k, v in m.items()}
      r = "n/a" if arm.lower().startswith(("value", "rff")) or "value" in arm else f"{m['ratio'].mean():.3f}"
      lines.append(f"| {lo}-{hi} | {arm} | {r} | {m['const_frac'].mean():.3f} | {m['ev'].mean():.3f} | "
                   f"{m['state'].mean():.3f} (se {(m['state'].std(ddof=1) / np.sqrt(n)) if n > 1 else 0:.3f}) | {m['varret'].mean():.2f} | "
                   f"{m['ev_state'].mean():.3f} | {m['std'].mean():.2f} |")
  names = list(S)
  if len(names) >= 2 and all(len(S[n]) == len(S[names[0]]) for n in names):
    x, y = names[0], names[1]
    lr = np.log(wmean(x, "state", 1750, 2000) / wmean(y, "state", 1750, 2000))
    se = float(lr.std(ddof=1) / np.sqrt(len(lr)))
    res["late_state_log_ratio"] = {"arms": [x, y], "mean": float(lr.mean()), "se": se,
                                   "ci95_ratio": [float(np.exp(lr.mean() - 2.093 * se)), float(np.exp(lr.mean() + 2.093 * se))]}
  (out / f"mechanism_{a.name}.json").write_text(json.dumps(res, indent=1))
  (out / f"mechanism_{a.name}_table.md").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))

  # ---------------------------------------------------------------- figure
  fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.2))

  def band(a_, key, arm):
    M = np.stack([roll(s[key]) for s in S[arm].values()])
    it = next(iter(S[arm].values()))["it"]
    c = COLORS.get(arm, None)
    a_.plot(it, np.median(M, 0), color=c, lw=1.8, label=arm)
    if len(M) > 1:
      a_.fill_between(it, np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=c, alpha=0.18)

  for key, a_, ttl, yl in [
    ("ratio", ax[0, 0], "(a) uncentered critic: mean value vs analytic target", "V / (r_seen / (1-gamma))"),
    ("const_frac", ax[0, 1], "(b) share of critic error that is a constant offset", "const_frac"),
    ("state", ax[1, 0], "(c) state-dependent critic error (= variance of the GAE advantage)", "state part (reward$^2$)"),
    ("ev", ax[1, 1], "(d) critic explained variance (solid: logged ev, dashed: constant removed)", "explained variance"),
  ]:
    for arm in S:
      if key == "ratio" and (arm.lower().startswith(("value", "rff")) or "value" in arm):
        continue
      band(a_, key, arm)
      if key == "ev":
        M = np.stack([roll(s["ev_state"]) for s in S[arm].values()])
        a_.plot(next(iter(S[arm].values()))["it"], np.median(M, 0), color=COLORS.get(arm), lw=1.2, ls="--")
    if key == "ratio":
      a_.axhline(1.0, color="grey", ls="--", lw=1)
      a_.axhline(0.98, color="grey", ls=":", lw=1)
      a_.set_ylim(-0.05, 1.12)
    if key == "ev":
      a_.set_ylim(-1.5, 1.02)
    if key == "state":
      a_.set_yscale("log")
    a_.set_title(ttl, fontsize=9.5)
    a_.set_xlabel("iteration")
    a_.set_ylabel(yl)
    a_.grid(alpha=0.25)
    a_.legend(fontsize=8, loc="best")
  sub = a.title or (f"late (1750-1999) state-dependent error ratio " +
                    f"{np.exp(res['late_state_log_ratio']['mean']):.1f}x (95% CI "
                    f"{res['late_state_log_ratio']['ci95_ratio'][0]:.1f}-{res['late_state_log_ratio']['ci95_ratio'][1]:.1f})"
                    if "late_state_log_ratio" in res else "")
  fig.suptitle(f"Mechanism, {' + '.join(f'{k} (n={len(v)})' for k, v in S.items())}: median, 10-90% band, 25-iteration smoothing\n{sub}",
               fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(out / f"r4_mech_{a.name}.png", dpi=140)
  print("wrote", out / f"r4_mech_{a.name}.png")


if __name__ == "__main__":
  main()
