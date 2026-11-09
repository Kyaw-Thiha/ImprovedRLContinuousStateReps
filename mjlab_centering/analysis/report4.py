"""Readouts for REPORT_004 (Steps 1-4), straight from the R2 mirrors.

    python analysis/report4.py clip  --campaign clip1-r4a1
    python analysis/report4.py sweep --campaigns r4sweep-xxxxxxxx [--topup r4sweep-yyyyyyyy]
    python analysis/report4.py shock --campaign r4shock-xxxxxxxx
    python analysis/report4.py width --campaign r4width-xxxxxxxx

Each command pulls the campaign's runs into results/merged_r4/ (cached), applies the pre-registered rule from
docs/mjlab_centering/02-decisions.md (d102-d105) and prints the table that goes into the report, plus a JSON record.
Conventions: metric = last-5-eval mean of `track_lin_frac`; "walks" = that mean >= 0.5; paired contrasts are per-seed
differences with a 95% t-interval. A cell with 3 seeds is labelled a pilot, never a claim.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from mjlab_centering import r2

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "results" / "merged_r4"
WALK = 0.5


# ---------------------------------------------------------------- data
def pull(campaign: str, jobs: list[str] | None = None) -> None:
  man = r2.get_json(f"{campaign}/manifest.json") or {"jobs": []}
  ids = [j["job_id"] for j in man["jobs"]] if jobs is None else jobs
  done = r2.done_set(campaign)
  for jid in ids:
    if jid not in done:
      continue
    d = MERGED / jid
    if (d / "eval.jsonl").exists():
      continue
    d.mkdir(parents=True, exist_ok=True)
    for name in ("eval.jsonl", "iter.jsonl", "final.json"):
      with tempfile.TemporaryDirectory() as td:
        p = Path(td) / name
        if r2.download_file(f"{campaign}/runs/{jid}/{name}", str(p)):
          (d / name).write_bytes(p.read_bytes())


def newest(path: Path) -> list[dict]:
  rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
  starts = [i for i, r in enumerate(rows) if r.get("iter") == 0]
  return rows[starts[-1]:] if len(starts) > 1 else rows


def evals(jid: str) -> list[dict]:
  p = MERGED / jid / "eval.jsonl"
  return newest(p) if p.exists() else []


def mean_last5(ev: list[dict], key: str = "track_lin_frac") -> float:
  v = [r[key] for r in ev[-5:] if key in r]
  return float(np.mean(v)) if v else float("nan")


def cell(jid: str) -> dict:
  ev = evals(jid)
  if not ev:  # baseline cells (E11/E12) live in the REPORT_003 mirror
    p2 = ROOT / "results" / "merged_r2" / jid / "eval.jsonl"
    ev = newest(p2) if p2.exists() else []
  return {"job_id": jid, "track": mean_last5(ev, "track_lin_frac"), "J": mean_last5(ev, "J_task"),
          "falls": mean_last5(ev, "fall_rate_per_1k"), "n_evals": len(ev),
          "walks": bool(ev) and mean_last5(ev, "track_lin_frac") >= WALK}


def paired(a: dict[int, float], b: dict[int, float]) -> dict:
  seeds = sorted(set(a) & set(b))
  d = np.array([a[s] - b[s] for s in seeds])
  n = len(d)
  if n < 2:
    return {"n": n, "mean": float(d.mean()) if n else float("nan"), "se": float("nan"), "ci": [float("nan")] * 2}
  se = float(d.std(ddof=1) / math.sqrt(n))
  h = float(stats.t.ppf(0.975, n - 1) * se)
  return {"n": n, "mean": float(d.mean()), "se": se, "ci": [float(d.mean() - h), float(d.mean() + h)],
          "positive": int((d > 0).sum())}


def fmt_paired(name: str, p: dict) -> str:
  if p["n"] < 2:
    return f"{name}: n={p['n']} (too few)"
  return (f"{name}: {p['mean']:+.4f} (se {p['se']:.4f}, 95% CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}], "
          f"{p['positive']}/{p['n']} positive)")


def seed_of(jid: str) -> int:
  return int(jid.rsplit("_s", 1)[1].split("_")[0])


# ---------------------------------------------------------------- steps
def cmd_clip(campaign: str) -> dict:
  man = r2.get_json(f"{campaign}/manifest.json") or {"jobs": []}
  ids = [j["job_id"] for j in man["jobs"]]
  pull(campaign, ids)
  cells = [cell(i) for i in ids]
  walked = [c for c in cells if c["walks"]]
  print("| seed | tracking | J_task | walks |")
  print("|---|---|---|---|")
  for c in cells:
    print(f"| {seed_of(c['job_id'])} | {c['track']:.3f} | {c['J']:.4f} | {'yes' if c['walks'] else 'no'} |")
  out = {"campaign": campaign, "cells": cells, "n_walk": len(walked), "n": len(cells),
         "rule": "d100: >=1 of 3 walks -> 20-seed clip-off run; 0 of 3 -> skip, legacy-only evidence"}
  print(f"\nwalking rate {len(walked)}/{len(cells)}; rule outcome: "
        f"{'LAUNCH 20 seeds' if walked else 'SKIP (report centering vs clip-off as legacy-only)'}")
  return out


def offset_of(jid: str) -> float | None:
  m = re.search(r"_o(\d+(?:\.\d+)?)_", jid)
  return float(m.group(1)) if m else None


def _sweep_cells(offset: float, jids: list[str], gamma: float = 0.999) -> dict[str, dict[int, dict]]:
  out: dict[str, dict[int, dict]] = defaultdict(dict)
  for j in jids:
    if offset_of(j) is None or abs(offset_of(j) - offset) > 1e-9 or f"g{gamma:g}" not in j:
      continue
    arm = "value" if "_value_" in j else "none"
    c = cell(j)
    if c["track"] == c["track"]:  # skip cells with no evals yet
      out[arm][seed_of(j)] = c
  return out


def baseline_jids(folder: Path) -> list[str]:
  """Existing corrected-protocol Go1 cells (offset 0 = E12, offset +1 = E11) used as the c = 0 and c = 1 rows."""
  out = []
  for arm in ("none", "value"):
    out += [p.name for p in folder.glob(f"w2main_go1_raw_{arm}_g0.999_o1_s*") if (p / "eval.jsonl").exists()]
    out += [p.name for p in folder.glob(f"w2off0_go1_raw_{arm}_g0.999_o0_s*") if (p / "eval.jsonl").exists()]
  return out


def cmd_sweep(campaigns: list[str], baseline_dir: Path, offsets: list[float], blocks: set[str] | None = None) -> dict:
  """`blocks` filters the manifest jobs before analysis. Needed when a campaign mixes blocks: shock mothers and the
  bias pilot both carry offset 0 or +1 and would otherwise leak into the sweep's c = 0 / c = 1 rows (the REPORT_004
  section 4 defect)."""
  jids: list[str] = []
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or {"jobs": []}
    jobs = [j for j in man["jobs"] if blocks is None or j.get("block") in blocks]
    ids = [j["job_id"] for j in jobs]
    pull(c, ids)
    jids += ids
  jids += [j for j in baseline_jids(baseline_dir) if j not in jids]  # read in place, not pulled
  res = {"campaigns": campaigns, "cells": {}}
  print("| offset c | arm | n | tracking | J_task | walks |")
  print("|---|---|---|---|---|---|")
  rows: dict[float, dict[str, dict[int, dict]]] = {}
  for o in offsets:
    rows[o] = _sweep_cells(o, jids)
    for arm in ("none", "value"):
      cs = rows[o].get(arm, {})
      if not cs:
        continue
      t = np.array([c["track"] for c in cs.values()])
      jj = np.array([c["J"] for c in cs.values()])
      w = sum(c["walks"] for c in cs.values())
      res["cells"][f"{o:g}_{arm}"] = {"n": len(cs), "track": float(t.mean()), "J": float(jj.mean()), "walks": int(w)}
      print(f"| {o:g} | {arm} | {len(cs)} | {t.mean():.3f} | {jj.mean():.4f} | {w}/{len(cs)} |")
  print()
  for o in offsets:
    for arm in ("none", "value"):
      cs = rows[o].get(arm, {})
      if len(cs) < 2:
        continue
      a = {s: c["track"] for s, c in cs.items()}
      b = {s: c["J"] for s, c in cs.items()}
      res[f"within_{o:g}_{arm}"] = {"track_sd": float(np.std(list(a.values()), ddof=1)),
                                    "J_sd": float(np.std(list(b.values()), ddof=1))}
  # arm contrast at each offset
  for o in offsets:
    a, b = rows[o].get("value", {}), rows[o].get("none", {})
    if len(a) < 2 or len(b) < 2:
      continue
    pt = paired({s: c["track"] for s, c in a.items()}, {s: c["track"] for s, c in b.items()})
    pj = paired({s: c["J"] for s, c in a.items()}, {s: c["J"] for s, c in b.items()})
    wn = sum(c["walks"] for c in b.values())
    wv = sum(c["walks"] for c in a.values())
    res[f"contrast_{o:g}"] = {"dtrack": pt, "dJ": pj, "walks_none": wn, "walks_value": wv, "n": len(b)}
    print(f"c={o:g}: value - none, " + fmt_paired("dtrack", pt))
    print(f"         " + fmt_paired("dJ", pj) + f"; walks none {wn}/{len(b)} vs value {wv}/{len(a)}")
    differ = abs(wv - wn) >= 3 or (pt["ci"][0] > 0 or pt["ci"][1] < 0)
    res[f"contrast_{o:g}"]["topup_fires"] = bool(differ)
    print(f"         d103 top-up rule: {'FIRES' if differ else 'does not fire'}")
  return res


def cmd_shock(campaigns: list[str]) -> dict:
  """Accepts several campaigns (Step 3 is sharded by seed across pods)."""
  jobs: list[dict] = []
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or {"jobs": []}
    ids = [j["job_id"] for j in man["jobs"]]
    pull(c, ids)
    jobs += man["jobs"]
  campaign = ",".join(campaigns)
  ids = [j["job_id"] for j in jobs]
  meta = {j["job_id"]: j for j in jobs}
  res: dict = {"campaign": campaign, "branches": {}, "effect": {}, "arm_contrast": {}}
  print("| shock | arm | seed | pre (mother 1149) | dip | min | t_rec | end | r_bar 1200 | r_bar end |")
  print("|---|---|---|---|---|---|---|---|---|---|")
  by: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for jid in ids:
    j = meta[jid]
    if j["block"] != "r4branch":
      continue
    m = meta[j["extra"]["after"]]
    mev = evals(m["job_id"])
    pre = [r["track_lin_frac"] for r in mev if r["iter"] < j["shock_at"]][-3:]
    base = float(np.mean(pre)) if pre else float("nan")
    ip = MERGED / jid / "iter.jsonl"
    rows = newest(ip) if ip.exists() else []
    ev = evals(jid)
    post = [(r["iter"], r["track_lin_frac"]) for r in ev if r["iter"] >= j["shock_at"]]
    traj = [v for _, v in post]
    mn = min(traj) if traj else float("nan")
    end = mean_last5(ev) if ev else float("nan")
    dip = mn - base
    t_rec = float("nan")
    thr = 0.95 * base
    for it, v in post:
      if v >= thr:
        t_rec = it - j["shock_at"]
        break
    rb0 = next((r["r_bar"] for r in rows if r["iter"] >= j["shock_at"]), float("nan"))
    rb1 = rows[-1]["r_bar"] if rows else float("nan")
    kind = "ctrl" if j["shock_kind"] == "none" else ("off+0.1" if j["shock_kind"] == "offset" and (j["extra"].get("shock_delta") == 0.1) else
                                                     ("off+1.0" if j["shock_kind"] == "offset" else "actx5"))
    rec = {"seed": j["seed"], "arm": j["center"], "kind": kind, "base": base, "min": mn, "dip": dip,
           "end": end, "t_rec": t_rec, "r_bar_shock": rb0, "r_bar_end": rb1, "n_post_evals": len(post)}
    by[(kind, j["center"])].append(rec)
    res["branches"][jid] = rec
    print(f"| {kind} | {j['center']} | {j['seed']} | {base:.3f} | {dip:+.3f} | {mn:.3f} | {t_rec if t_rec == t_rec else float('nan'):.0f} | "
          f"{end:.3f} | {rb0:.3f} | {rb1:.3f} |")
  print()
  ctrl_final = {(r["arm"], r["seed"]): r for r in by.get(("ctrl", "none"), []) + by.get(("ctrl", "value"), [])}
  for (kind, arm), rs in sorted(by.items()):
    if kind == "ctrl":
      continue
    # shock effect = shocked branch minus the unshocked control of the same arm and seed
    dz = {r["seed"]: r["end"] - ctrl_final[(arm, r["seed"])]["end"] for r in rs if (arm, r["seed"]) in ctrl_final}
    dd = {r["seed"]: r["dip"] - ctrl_final[(arm, r["seed"])]["dip"] for r in rs if (arm, r["seed"]) in ctrl_final}
    pz, pd = paired(dz, {s: 0.0 for s in dz}), paired(dd, {s: 0.0 for s in dd})
    rec_n = sum(1 for r in rs if r["t_rec"] == r["t_rec"])
    res["effect"][f"{kind}_{arm}"] = {"dz_vs_ctrl": pz, "ddip_vs_ctrl": pd, "recovered": rec_n, "n": len(rs)}
    print(f"{kind} / {arm}: " + fmt_paired("final - control", pz) + f", recovered {rec_n}/{len(rs)}")
  # arm contrast within each shock type
  for kind in sorted({k for k, _ in by if k != "ctrl"}):
    n_ = {r["seed"]: r for r in by.get((kind, "none"), [])}
    v_ = {r["seed"]: r for r in by.get((kind, "value"), [])}
    if len(n_) < 2 or len(v_) < 2:
      continue
    p = paired({s: r["end"] - ctrl_final[("value", s)]["end"] for s, r in v_.items() if ("value", s) in ctrl_final},
               {s: r["end"] - ctrl_final[("none", s)]["end"] for s, r in n_.items() if ("none", s) in ctrl_final})
    res.setdefault("arm_contrast", {})[kind] = p
    print(f"{kind}: value vs none, shocked-minus-control final " + fmt_paired("difference", p))
  return res


def cmd_width(campaigns: list[str]) -> dict:
  ids: list[str] = []
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or {"jobs": []}
    sub = [j["job_id"] for j in man["jobs"]]
    pull(c, sub)
    ids += sub
  campaign = ",".join(campaigns)
  arms: dict[str, dict[int, dict]] = defaultdict(dict)
  for jid in ids:
    tag = jid.split("go1_", 1)[1].rsplit("_s", 1)[0]
    arms[tag][seed_of(jid)] = cell(jid)
  res = {"campaign": campaign, "arms": {}}
  print("| arm | n | tracking | J_task | walks | action std (end) |")
  print("|---|---|---|---|---|---|")
  for tag, cs in arms.items():
    t = np.array([c["track"] for c in cs.values()])
    w = sum(c["walks"] for c in cs.values())
    res["arms"][tag] = {"n": len(cs), "track": float(t.mean()), "track_per_seed": t.round(4).tolist(), "walks": int(w),
                        "J": float(np.mean([c["J"] for c in cs.values()]))}
    print(f"| {tag} | {len(cs)} | {t.mean():.3f} | {res['arms'][tag]['J']:.4f} | {w}/{len(cs)} | |")
  print()
  raw = res["arms"].get("raw")
  best, best_v = None, -1.0
  for tag, a in res["arms"].items():
    if tag == "raw" or tag.startswith("rff"):
      continue
    if a["track"] > best_v:
      best, best_v = tag, a["track"]
  if best:
    res["best"] = {"arm": best, "track": res["arms"][best]["track"], "gap_to_raw": res["arms"]["raw"]["track"] - best_v if raw else None,
                   "closes_gap": bool(raw and raw["track"] - best_v <= 0.03)}
    n = best.split("x")[0][-1]
    rff = f"rff_low{n}x{n}"
    if rff in res["arms"]:
      res["best"]["beats_rff"] = bool(best_v - res["arms"][rff]["track"] >= 0.05 and res["arms"][best]["walks"] >= res["arms"][rff]["walks"])
      res["best"]["rff"] = rff
    print(f"best SSP: {best} (tracking {best_v:.3f}); gap to raw {res['best']['gap_to_raw']:+.3f}; "
          f"closes gap: {res['best']['closes_gap']}")
    if "beats_rff" in res["best"]:
      print(f"vs matched RFF ({rff}, {res['arms'][rff]['track']:.3f}): beats = {res['best']['beats_rff']}")
    if best.startswith("ssp_low4x4") and best_v - res["arms"].get("ssp_low2x2", {"track": -9})["track"] >= 0:
      print("d105 clause (iv): 4x4 wins the stage -> the 8x8 3-seed pilot is unlocked")
  return res


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=["clip", "sweep", "shock", "width"])
  ap.add_argument("--campaign", default=None)
  ap.add_argument("--campaigns", default=None)
  ap.add_argument("--offsets", default="0,0.1,0.3,1")
  args = ap.parse_args()
  if args.cmd == "clip":
    out = cmd_clip(args.campaign)
  elif args.cmd == "sweep":
    camps = (args.campaigns or args.campaign).split(",")
    out = cmd_sweep(camps, ROOT / "results" / "merged_r2", [float(x) for x in args.offsets.split(",")])
  elif args.cmd == "shock":
    out = cmd_shock((args.campaigns or args.campaign).split(","))
  else:
    out = cmd_width((args.campaigns or args.campaign).split(","))
  Path(ROOT / "results" / f"report4_{args.cmd}.json").write_text(json.dumps(out, indent=1, default=float))
  print(f"\nwrote results/report4_{args.cmd}.json")


if __name__ == "__main__":
  main()
