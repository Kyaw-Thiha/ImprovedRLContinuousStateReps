"""REPORT_009 round 9b, free retrospective: how large is the truncated-GAE leak next to the real advantage signal?

    python analysis/r9_leak_retro.py [--fig] [--json results/r9_leak_retro.json]

Theory (docs/mjlab_centering/02-decisions.md, d1100+): a state-constant critic error e adds -(1-gamma)*e to every TD error.
GAE sums over the rollout only up to its end (T = 24), so the advantage at position t carries
    bias_t = delta_bar * w_t,   w_t = sum_{k < T-t} (gamma*lambda)^k,   delta_bar = mean TD error (logged as `delta_mean`),
which depends on t and is therefore NOT removed by per-batch normalisation. Its spread across positions is
    leak_std = |delta_bar| * std_t(w_t),
to be compared with the spread of the raw advantage, sqrt(err_mse - err_mean^2) (`err = returns - values` is the raw
GAE advantage, so `err_mse` and `err_mean` are exactly its second moment and mean).

    rho = leak_std / std(raw advantage)     (per iteration; the run's score is the median over iterations 100-600)

rho ~ 1 means the position-dependent bias alone could account for all of the advantage variance; rho << 1 means it is
negligible. This is a **post hoc** readout of existing logs (it gates nothing). Cells and outcome rule are Step 0's.

**Circularity, stated up front.** Value centering's update is `r_bar += eta * mean(delta)`, i.e. it drives `delta_mean` to
zero *by construction*; simple centering drives the raw reward's running mean instead and leaves `delta_mean` free. So the
centred cells' rho is mechanically small and is NOT evidence for or against the theory; the informative cells are the
uncentered, bias-init, value-norm, return-std and clip-off arms. `CIRCULAR_CELLS` marks the centred ones and the verdict
table excludes them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_step0 as S  # noqa: E402

ROOT = S.ROOT
T_ROLL = 24
LAM = 0.95
WINDOWS = {"take-off 100-600": (100, 600), "early 50-300": (50, 300)}
# `center="value"` sets r_bar += eta*mean(delta), so delta_mean is zero by construction in these cells
CIRCULAR_CELLS = frozenset({"Go1 value-centred c=1", "Go1 value-centred c=0.1",
                            "G1 value-centred c=1"})


def gae_weights(gamma: float, lam: float = LAM, T: int = T_ROLL) -> np.ndarray:
  """w_t = sum_{k=0}^{T-t-1} (gamma*lam)^k for t = 0..T-1 (position 0 = first step of the rollout)."""
  g = gamma * lam
  return np.array([sum(g ** k for k in range(T - t)) for t in range(T)])


def leak_ratio(delta_mean: np.ndarray, err_mean: np.ndarray, err_mse: np.ndarray, gamma: float,
               lam: float = LAM, T: int = T_ROLL) -> np.ndarray:
  """Per-iteration rho = |delta_bar| * std_t(w) / std(raw advantage)."""
  sw = float(np.std(gae_weights(gamma, lam, T)))
  adv_std = np.sqrt(np.maximum(err_mse - err_mean ** 2, 1e-12))
  return np.abs(delta_mean) * sw / adv_std


def run_rho(rows: list[dict], gamma: float, lo: int, hi: int) -> float:
  d = {int(r["iter"]): r for r in rows}
  its = [i for i in range(lo, hi) if i in d]
  if not its:
    return float("nan")
  dm = np.array([d[i]["delta_mean"] for i in its])
  em = np.array([d[i]["err_mean"] for i in its])
  ms = np.array([d[i]["err_mse"] for i in its])
  return float(np.median(leak_ratio(dm, em, ms, gamma)))


def auc(xs, ys):
  pos = [x for x, y in zip(xs, ys) if y]
  neg = [x for x, y in zip(xs, ys) if not y]
  if not pos or not neg:
    return None
  return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


PILOT_CELLS = {  # round-9a arms (n = 4, descriptive): does rho order them the way the outcomes do?
  "r9a pilot A1 lr 0.5": ("merged_r9", "r9a_go1_a1_05_g0.999_o1_s{}"),
  "r9a pilot A1 lr 0.05": ("merged_r9", "r9a_go1_a1_005_g0.999_o1_s{}"),
  "r9a pilot clip-off": ("merged_r9", "r9a_go1_clipoff_g0.999_o1_s{}"),
  "round-4 clip-off": ("merged_r4", "clip1_go1_raw_none_clipoff_g0.999_o1_s{}"),
  "r9a pilot A2 clip off": ("merged_r9", "r9a_go1_a2_off_g0.999_o1_s{}"),
  "r9a pilot A2 clip on": ("merged_r9", "r9a_go1_a2_on_g0.999_o1_s{}"),
  "r9a pilot A3": ("merged_r9", "r9a_go1_a3_g0.999_o1_s{}"),
}
PILOT_SEEDS = tuple(range(3000, 3010))


def collect() -> list[dict]:
  out = []
  for name, kind, d, pat, gamma, c, seeds in S.CELLS:
    for s in seeds:
      jid = pat.format(s)
      ip = S.find(d, jid, "iter.jsonl")
      if ip is None:
        continue
      walks, tr, _ = S.outcome(jid, d)
      if walks is None:
        continue
      rows = S.jl(ip)
      rec = {"cell": name, "job": jid, "seed": s, "gamma": gamma, "walks": bool(walks), "track": tr,
             "circular": name in CIRCULAR_CELLS}
      for wn, (lo, hi) in WINDOWS.items():
        rec[wn] = run_rho(rows, gamma, lo, hi)
      out.append(rec)
  # round-9a pilot arms (also post hoc; n = 4 each)
  for name, (d, pat) in PILOT_CELLS.items():
    for s in PILOT_SEEDS:
      jid = pat.format(s)
      ip = S.find(d, jid, "iter.jsonl")
      if ip is None:
        continue
      walks, tr, _ = S.outcome(jid, d)
      if walks is None:
        continue
      rows = S.jl(ip)
      rec = {"cell": name, "job": jid, "seed": s, "gamma": 0.999, "walks": bool(walks), "track": tr, "circular": False}
      for wn, (lo, hi) in WINDOWS.items():
        rec[wn] = run_rho(rows, 0.999, lo, hi)
      out.append(rec)
  return out


def table(runs: list[dict], window: str) -> str:
  lines = [f"**Leak ratio rho, window {window}** (median over the cell's runs; walkers / non-walkers). "
           "`circular` = the cell's `delta_mean` is zeroed by its own update rule, so its rho is not evidence.", "",
           "| cell | n | walks | rho median (all) | rho walkers | rho failures | within-cell AUC(rho -> fails) | circular |",
           "|---|---|---|---|---|---|---|---|"]
  for cn in dict.fromkeys(r["cell"] for r in runs):
    rs = [r for r in runs if r["cell"] == cn]
    v = [r[window] for r in rs]
    w = [r[window] for r in rs if r["walks"]]
    f = [r[window] for r in rs if not r["walks"]]
    a = auc([r[window] for r in rs], [not r["walks"] for r in rs]) if w and f else None
    fm = lambda x: "—" if not x else f"{np.median(x):.2f}"  # noqa: E731
    lines.append(f"| {cn} | {len(rs)} | {sum(r['walks'] for r in rs)} | {np.median(v):.2f} | {fm(w)} | {fm(f)} | "
                 f"{'—' if a is None else f'{a:.2f}'} | {'yes' if cn in CIRCULAR_CELLS else ''} |")
  return "\n".join(lines)


def pooled_auc(runs: list[dict], window: str, exclude_circular: bool = True) -> float:
  """Across runs: does a larger rho predict failure? Descriptive and cell-confounded, but the circular cells are dropped
  so the number is not manufactured by value centering's own update rule."""
  rs = [r for r in runs if not (exclude_circular and r.get("circular"))]
  return auc([r[window] for r in rs], [not r["walks"] for r in rs])


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default=str(ROOT / "results" / "r9_leak_retro.json"))
  a = ap.parse_args()
  runs = collect()
  Path(a.json).write_text(json.dumps(runs, default=float))
  print(f"{len(runs)} runs; w_t (gamma .999, lambda .95): first {gae_weights(0.999)[0]:.2f}, last {gae_weights(0.999)[-1]:.2f}, "
        f"std {np.std(gae_weights(0.999)):.2f}\n")
  for wn in WINDOWS:
    print(table(runs, wn))
    print(f"\npooled AUC(rho -> failure), {wn}: {pooled_auc(runs, wn):.2f} "
          f"(centred cells' delta_mean is zero by construction and is excluded:\n"
          f"  with them included it would be {pooled_auc(runs, wn, exclude_circular=False):.2f})\n")
