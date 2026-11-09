"""Summarise linear-slice results: per (features, gamma, offset, center) mean +- sd over seeds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def load(path):
  return [json.loads(l) for l in open(path) if l.strip()]


def table(rows, keys, fields):
  g = defaultdict(list)
  for r in rows:
    g[tuple(r[k] for k in keys)].append(r)
  out = []
  for k in sorted(g):
    cell = g[k]
    out.append((k, len(cell), {f: (float(np.mean([c[f] for c in cell])), float(np.std([c[f] for c in cell], ddof=1)) if len(cell) > 1 else 0.0) for f in fields}))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--td", default=None)
  ap.add_argument("--ac", default=None)
  a = ap.parse_args()
  if a.td:
    print("TD(0), fixed policy: critic error vs exact truth (stationary-weighted)")
    print("feat gamma offset center | n | mse | state_part (decision-relevant) | const_part | const_frac | floor")
    for k, n, f in table(load(a.td), ("feat", "gamma", "offset", "center"),
                         ("mse", "state_part", "const_part", "const_frac", "floor_mse")):
      print(*k, "|", n, "|", *(f"{f[x][0]:.4g}±{f[x][1]:.1g}" for x in ("mse", "state_part", "const_part", "const_frac", "floor_mse")))
  if a.ac:
    rows = load(a.ac)
    print("\nlinear actor-critic: average task reward (last 3 windows)")
    print("feat gamma offset center r_bar_init | n | reward_last (task return, offset excluded)")
    for k, n, f in table(rows, ("feat", "gamma", "offset", "center", "r_bar_init"), ("reward_last",)):
      print(*k, "|", n, "|", f"{f['reward_last'][0]:.4g}±{f['reward_last'][1]:.2g}")


if __name__ == "__main__":
  main()
