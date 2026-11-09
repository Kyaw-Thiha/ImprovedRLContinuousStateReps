#!/usr/bin/env python3
"""Validate the A2C port against RECORDED legacy value series (the strongest available check).

`data/reward_centering/<Rep>/<centering>/ACTrial#<id>/{states,rewards,values}.csv` hold the raw
trace of real Nengo A2C runs (wide CSVs: columns are `trial<i>`, rows are environment steps).

The state value V(s) evolves independently of the chosen actions (row 0 of W is the only row V
depends on, and its update does not involve the action), so feeding the recorded states/rewards
through `LinearActorCritic` reproduces the recorded `values.csv` series exactly if the port's
critic + centering arithmetic matches the legacy rule. That makes this an action-free, exact test
of every centering mode -- which is precisely what the online-metric gate cannot isolate.

Writes nothing; prints a per-mode max-abs-diff table.

    .venv/bin/python scripts/validate_a2c_replay.py --cells PlaceSSP/simple,PlaceSSP/value,PlaceSSP/none
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

META = os.path.join(REPO, "data", "processed", "reward-centering-metadata-summary.csv")
DATA = os.path.join(REPO, "data", "reward_centering")


def build_rep(row):
    from rl.representations.ssp_legacy import LegacySSPRep
    rep_name = row["rep_"]
    if rep_name in ("PlaceSSP", "HexSSP"):
        return LegacySSPRep(4, length_scale=float(row["length_scale"]), n_rotates=int(row["n_rotates"]))
    raise SystemExit(f"replay not wired for rep {rep_name!r} (SSP only for now)")


def replay(states, rewards, rep, row, n_cols, n_steps):
    """(pred, finite) per column; the trace is truncated at the first NaN (episode end padding).

    Actions are not needed: row 0 of W (the value) never depends on which action was taken, so the
    recorded states/rewards alone determine the recorded value series.
    """
    from rl.networks.acLinear import LinearActorCritic
    mode = row["reward_center_mode"]
    lr = float(row["lr"])
    out = []
    for col in n_cols:
        s = states[:n_steps, col * 4: col * 4 + 4]
        r = rewards[:n_steps, col]
        good = np.isfinite(s).all(axis=1) & np.isfinite(r)
        if good.sum() < 2:
            out.append(None); continue
        end = int(np.argmin(good)) if not good.all() else len(good)
        s, r = s[:end], r[:end]
        phis = [rep.map(x) for x in s]
        ac = LinearActorCritic(rep.size_out, 2, lr=lr, act_dis=float(row["act_dis"]),
                               state_dis=float(row["state_dis"]), centering=mode,
                               beta=float(row.get("reward_center_beta") or 0.001),
                               eta=float(row.get("reward_center_eta") or 1.0),
                               avg_reward_init=float(row.get("reward_center_init") or 0.0))
        ac.reset(phis[0])
        pred = []
        for t in range(1, len(phis)):
            ac.step(phis[t], 0, float(r[t]))
            pred.append(ac.value(phis[t - 1]))   # legacy returns the updated value of the PREVIOUS state
        out.append((np.asarray(pred), end))
    return out


def best_alignment(pred, recorded, min_len=5):
    """Align the length-n-1 prediction series against the recorded series; return (max_abs_diff, shift).

    Recorded episodes are padded with NaN after termination, so only the finite prefix is compared.
    """
    best = (np.inf, None, 0)
    for shift in (0, 1):
        p = np.asarray(pred, dtype=float)
        q_full = np.asarray(recorded, dtype=float)[shift:shift + len(p)]
        if len(q_full) != len(p):
            continue
        finite = np.isfinite(p) & np.isfinite(q_full)
        if finite.sum() < min_len:
            continue
        d = float(np.max(np.abs(p[finite] - q_full[finite])))
        if d < best[0]:
            best = (d, shift, int(finite.sum()))
    return best


def parse_txt_params(path):
    """Parse the `key = value` .txt written beside each ACTrial directory."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'").strip('"')
    return out


def row_from_txt(p):
    def f(key, default=None):
        v = p.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    mode = p.get("reward_center_mode", "none")
    return {
        "rep_": p.get("rep_"), "reward_center_mode": mode, "lr": f("lr"), "act_dis": f("act_dis", 0.9),
        "state_dis": f("state_dis", 0.99), "length_scale": f("length_scale", 1.0), "n_rotates": f("n_rotates", 5),
        "reward_center_beta": f("reward_center_beta", 0.001), "reward_center_eta": f("reward_center_eta", 1.0),
        "reward_center_init": f("reward_center_init", 0.0), "state_neurons": p.get("state_neurons"),
        "dimensionality": f("dimensionality"),
    }


def replay_dirs(dirs, tol, steps=500, cols=3):
    ok = True
    for d in dirs:
        txt = d.rstrip("/") + ".txt"
        if not os.path.exists(txt):
            print(f"{d}: no params .txt (looked for {txt})"); ok = False; continue
        p = parse_txt_params(txt)
        if str(p.get("state_neurons")).lower() not in ("none", "nil", "null"):
            print(f"{d}: SKIP state_neurons={p.get('state_neurons')} (not the grid config)"); continue
        row = row_from_txt(p)
        rep = build_rep(row)
        states = pd.read_csv(os.path.join(d, "states.csv"), index_col=0).to_numpy()
        rewards = pd.read_csv(os.path.join(d, "rewards.csv"), index_col=0).to_numpy()
        values = pd.read_csv(os.path.join(d, "values.csv"), index_col=0).to_numpy()
        n_avail = min(states.shape[1] // 4, rewards.shape[1], values.shape[1])
        use = list(range(min(cols, n_avail)))
        preds = replay(states, rewards, rep, row, use, steps)
        worst = 0.0
        for k, col in enumerate(use):
            if preds[k] is None:
                print(f"  {os.path.basename(d)} col{col}: skipped (no finite episode)"); continue
            pred, end = preds[k]
            diff, shift, nf = best_alignment(pred, values[:end, col], min_len=5)
            worst = max(worst, diff)
            print(f"  {os.path.basename(d)} col{col}: max|port - legacy| = {diff:.3e}  (shift {shift}, {nf} steps)")
        good = worst <= tol
        ok &= good
        print(f"{os.path.basename(d)}: mode={row['reward_center_mode']} dim={rep.size_out} "
              f"(metadata {row['dimensionality']}) lr={row['lr']} worst={worst:.3e} -> "
              f"{'MATCH' if good else 'MISMATCH'}\n")
    print("REPLAY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", default="PlaceSSP/simple,PlaceSSP/value,PlaceSSP/none")
    ap.add_argument("--trial-dirs", default=None,
                    help="comma list of ACTrial# directories (grid runs; params read from the sibling .txt)")
    ap.add_argument("--cols", type=int, default=3, help="how many recorded runs (columns) to replay per cell")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    if args.trial_dirs:
        return replay_dirs([d for d in args.trial_dirs.split(",") if d], args.tol, args.steps, args.cols)

    df = pd.read_csv(META)
    overall_ok = True
    for cell in args.cells.split(","):
        rep_name, cen = cell.split("/")
        sub = df[(df["rep_"] == rep_name) & (df["reward_center_mode"] == cen)]
        if sub.empty:
            print(f"{cell}: no metadata rows"); overall_ok = False; continue
        # one ACTrial directory = 40 runs of the same params; pick the first directory.
        trial_id = sub["trial_ID"].iloc[0]
        rows = df[df["trial_ID"] == trial_id].sort_values("seed").reset_index(drop=True)
        d = os.path.join(DATA, rep_name, cen, trial_id)
        states = pd.read_csv(os.path.join(d, "states.csv"), index_col=0).to_numpy()
        rewards = pd.read_csv(os.path.join(d, "rewards.csv"), index_col=0).to_numpy()
        values = pd.read_csv(os.path.join(d, "values.csv"), index_col=0).to_numpy()
        row = rows.iloc[0]
        rep = build_rep(row)
        cols = list(range(min(args.cols, rows.shape[0], values.shape[1])))
        preds = replay(states, rewards, rep, row, cols, args.steps)
        diffs = []
        for k, col in enumerate(cols):
            if preds[k] is None:
                print(f"  {cell} column{col}: skipped (no finite episode)")
                continue
            pred, end = preds[k]
            diff, shift, nf = best_alignment(pred, values[:end, col])
            diffs.append(diff)
            print(f"  {cell} column{col}: max|port - legacy| = {diff:.3e}  (shift {shift}, {nf} steps compared)")
        worst = max(diffs) if diffs else float("inf")
        ok = worst <= args.tol
        overall_ok &= ok
        print(f"{cell}: dim={rep.size_out} (metadata says {int(row['dimensionality'])}), "
              f"lr={row['lr']}, beta={row['reward_center_beta']}, eta={row['reward_center_eta']}, "
              f"worst={worst:.3e} -> {'MATCH' if ok else 'MISMATCH'}\n")
    print("REPLAY:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
