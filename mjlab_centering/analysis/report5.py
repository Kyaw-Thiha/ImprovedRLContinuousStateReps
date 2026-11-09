"""Readouts for REPORT_005 (round 5, d113-d121), straight from the R2 mirrors, reusing the report4 helpers.

    python analysis/report5.py bias   --campaigns r5bias-xxxx[,r5biasext-yyyy]
    python analysis/report5.py sweep  --campaigns <r4 sweep campaigns>,<r5 sweep campaigns>
    python analysis/report5.py off0   --campaigns <r5 off0 campaigns>
    python analysis/report5.py shock  --campaigns <r5 branch campaigns>
    python analysis/report5.py width  --campaigns <r4width raw campaign>,<r5 width campaigns>
    python analysis/report5.py sspc   --campaigns <r5 sspconfirm campaigns> [--arm-prefix ssp_low4x4]
    python analysis/report5.py g1     --campaigns <r5 g1 campaigns>

Conventions are the report4 ones: metric = last-5-eval mean of `track_lin_frac`; walks = >= 0.5; paired contrasts are
per-seed differences with a 95% t-interval. Fewer than 20 complete pairs is labelled a pilot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report4 as R4  # noqa: E402
from report4 import MERGED, ROOT, cell, evals, fmt_paired, mean_last5, newest, paired, pull, seed_of  # noqa: E402
from mjlab_centering import r2  # noqa: E402

MOTHER_CAMPAIGN = "r4topupshock-049efe29"  # d116
WALK_MOTHER = 0.8


def _jobs(campaigns: list[str]) -> list[dict]:
  jobs: list[dict] = []
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or {"jobs": []}
    pull(c, [j["job_id"] for j in man["jobs"]])
    jobs += man["jobs"]
  return jobs


# ------------------------------------------------------------------------------------------------ step 0
def cmd_bias(campaigns: list[str]) -> dict:
  jobs = [j for j in _jobs(campaigns) if j.get("block") == "r5bias"]  # a combined campaign also holds every other block
  cells = [cell(j["job_id"]) for j in jobs]
  walked = [c for c in cells if c["walks"]]
  print("| seed | tracking | J_task | walks |\n|---|---|---|---|")
  for c in sorted(cells, key=lambda c: seed_of(c["job_id"])):
    print(f"| {seed_of(c['job_id'])} | {c['track']:.3f} | {c['J']:.4f} | {'yes' if c['walks'] else 'no'} |")
  n = len(cells)
  w = len(walked)
  base = [cell(f"w2main_go1_raw_none_g0.999_o1_s{seed_of(c['job_id'])}") for c in cells]
  verdict = ("early value-level error is a sufficient cause (revise d108)" if n >= 3 and w >= 2 and n == 3 else
             "level initialisation is not enough (d108 stands)" if w == 0 else
             "exactly 1 of 3: extend to seeds 3003-3009 (d114)" if n == 3 and w == 1 else f"{w}/{n} walk (extension cell)")
  print(f"\nwalking rate {w}/{n}; E11 uncentered on the same seeds: {sum(b['walks'] for b in base)}/{len(base)} walk "
        f"(mean tracking {np.nanmean([b['track'] for b in base]):.3f}); d114 reading: {verdict}")
  return {"campaigns": campaigns, "cells": cells, "walks": w, "n": n, "verdict": verdict,
          "e11_same_seeds": [b["track"] for b in base]}


# ------------------------------------------------------------------------------------------------ shock
def _branch_record(j: dict, mother_id: str) -> dict:
  mev = evals(mother_id)
  pre = [r["track_lin_frac"] for r in mev if r["iter"] < j["shock_at"]][-3:]
  base = float(np.mean(pre)) if pre else float("nan")
  ev = evals(j["job_id"])
  post = [(r["iter"], r["track_lin_frac"]) for r in ev if r["iter"] >= j["shock_at"]]
  traj = [v for _, v in post]
  mn = min(traj) if traj else float("nan")
  end = mean_last5(ev) if ev else float("nan")
  t_rec = float("nan")
  for it, v in post:
    if v >= 0.95 * base:
      t_rec = it - j["shock_at"]
      break
  ip = MERGED / j["job_id"] / "iter.jsonl"
  rows = newest(ip) if ip.exists() else []
  rb0 = next((r["r_bar"] for r in rows if r["iter"] >= j["shock_at"]), float("nan"))
  rb1 = rows[-1]["r_bar"] if rows else float("nan")
  ex = j.get("extra") or {}
  kind = "ctrl" if j["shock_kind"] == "none" else ("off+0.1" if j["shock_kind"] == "offset" and ex.get("shock_delta") == 0.1
                                                  else ("off+1.0" if j["shock_kind"] == "offset" else "actx5"))
  return {"seed": j["seed"], "arm": j["center"], "kind": kind, "base": base, "min": mn, "dip": mn - base, "end": end,
          "t_rec": t_rec, "r_bar_shock": rb0, "r_bar_end": rb1, "n_post_evals": len(post)}


def _jobs_with_campaign(campaigns: list[str]) -> list[tuple[dict, str]]:
  out: list[tuple[dict, str]] = []
  for c in campaigns:
    man = r2.get_json(f"{c}/manifest.json") or {"jobs": []}
    pull(c, [j["job_id"] for j in man["jobs"]])
    out += [(j, c) for j in man["jobs"]]
  return out


def cmd_shock(campaigns: list[str]) -> dict:
  """Branch records for one or more rounds. A branch's mother is read from `extra.after_campaign` when present (round 5,
  d116) and from the branch's own campaign otherwise (round 4), so the same code handles both and can be validated
  against REPORT_004's numbers."""
  pairs = _jobs_with_campaign(campaigns)
  src: dict[str, set[str]] = {}
  for j, c in pairs:
    if j["block"] == "r4branch":
      src.setdefault(j["extra"].get("after_campaign") or c, set()).add(j["extra"]["after"])
  for camp, mothers in src.items():
    pull(camp, sorted(mothers))
  out: dict = {"campaign": ",".join(campaigns), "mother_campaigns": {k: sorted(v)[:3] for k, v in src.items()},
               "branches": {}}
  for j, c in pairs:
    if j["block"] == "r4branch":
      out["branches"][j["job_id"]] = _branch_record(j, j["extra"]["after"])
  assert len(out["branches"]) == 4 * sum(len(v) for v in src.values()), "unexpected branch count"
  return out


def print_shock_tables(round5_json: str = "results/report5_shock.json", round4_json: str = "results/report4_shock.json") -> str:
  import shock_tables as ST
  r4, r5 = ST.load([str(ROOT / round4_json)]), ST.load([str(ROOT / round5_json)])
  txt = []
  for name, rows in (("ROUND 4 alone (seeds 3000-3009)", r4), ("ROUND 5 alone (seeds 3010-3019)", r5),
                     ("POOLED (seeds 3000-3019)", r4 + r5)):
    txt.append(f"# {name}")
    txt.append(ST.fmt(ST.tables(rows, False), "(A) all seeds, as pre-registered"))
    txt.append("")
    txt.append(ST.fmt(ST.tables(rows, True), f"(B) walking mothers only (>= {ST.WALK_MOTHER})"))
    txt.append("")
  return "\n".join(txt)


# ------------------------------------------------------------------------------------------------ sspc / g1
def cmd_pairs(campaigns: list[str], ref_pattern: str | None, tag: str) -> dict:
  """Cells of a confirmation campaign by (arm, seed), paired against reference cells (`ref_pattern` has {arm}, {seed})."""
  jobs = _jobs(campaigns)
  cells: dict[str, dict[int, dict]] = {"none": {}, "value": {}}
  for j in jobs:
    c = cell(j["job_id"])
    if c["track"] == c["track"]:
      cells[j["center"]][j["seed"]] = c
  res: dict = {"campaigns": campaigns, "cells": {}, "per_seed": {}}
  print("| arm | n | tracking | J_task | walks |\n|---|---|---|---|---|")
  for arm, cs in cells.items():
    if not cs:
      continue
    t = np.array([c["track"] for c in cs.values()])
    jj = np.array([c["J"] for c in cs.values()])
    w = sum(c["walks"] for c in cs.values())
    res["cells"][arm] = {"n": len(cs), "track": float(t.mean()), "J": float(jj.mean()), "walks": int(w)}
    res["per_seed"][arm] = {s: c["track"] for s, c in cs.items()}
    print(f"| {tag}/{arm} | {len(cs)} | {t.mean():.3f} | {jj.mean():.4f} | {w}/{len(cs)} |")
  a, b = cells["value"], cells["none"]
  if len(a) > 1 and len(b) > 1:
    pt = paired({s: c["track"] for s, c in a.items()}, {s: c["track"] for s, c in b.items()})
    pj = paired({s: c["J"] for s, c in a.items()}, {s: c["J"] for s, c in b.items()})
    res["centering_effect"] = {"dtrack": pt, "dJ": pj}
    print("centering effect (value - none): " + fmt_paired("dtrack", pt) + "; " + fmt_paired("dJ", pj))
  if ref_pattern:
    for arm, cs in cells.items():
      ref = {s: cell(ref_pattern.format(arm=arm, seed=s)) for s in cs}
      ref = {s: c for s, c in ref.items() if c["track"] == c["track"]}
      if len(ref) > 1:
        pt = paired({s: cs[s]["track"] for s in ref}, {s: c["track"] for s, c in ref.items()})
        pj = paired({s: cs[s]["J"] for s in ref}, {s: c["J"] for s, c in ref.items()})
        res.setdefault("vs_ref", {})[arm] = {"dtrack": pt, "dJ": pj, "ref_walks": sum(c["walks"] for c in ref.values()),
                                             "n": len(ref), "std_track": float(np.mean([c["track"] for c in ref.values()])),
                                             "ref_per_seed": {s: c["track"] for s, c in ref.items()}}
        print(f"{tag} minus raw ({arm}): " + fmt_paired("dtrack", pt) + "; " + fmt_paired("dJ", pj))
  return res


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=["bias", "sweep", "off0", "shock", "width", "sspc", "g1"])
  ap.add_argument("--campaigns", required=True)
  ap.add_argument("--offsets", default="0,0.1,0.3,1")
  ap.add_argument("--tag", default="ssp")
  a = ap.parse_args()
  camps = a.campaigns.split(",")
  if a.cmd == "bias":
    out = cmd_bias(camps)
  elif a.cmd in ("sweep", "off0"):
    offs = [float(x) for x in a.offsets.split(",")] if a.cmd == "sweep" else [0.0]
    blocks = {"r4sweep"} if a.cmd == "sweep" else {"r4off0"}
    out = R4.cmd_sweep(camps, ROOT / "results" / "merged_r2", offs, blocks=blocks)
  elif a.cmd == "shock":
    out = cmd_shock(camps)
    Path(ROOT / "results" / "report5_shock.json").write_text(json.dumps(out, indent=1, default=float))
    print(print_shock_tables())
  elif a.cmd == "width":
    out = R4.cmd_width(camps)
  elif a.cmd == "sspc":
    out = cmd_pairs(camps, "w2main_go1_raw_{arm}_g0.999_o1_s{seed}", a.tag)
  else:
    out = cmd_pairs(camps, None, "g1")
  if a.cmd != "shock":
    Path(ROOT / "results" / f"report5_{a.cmd}.json").write_text(json.dumps(out, indent=1, default=float))
  print(f"\nwrote results/report5_{a.cmd}.json")


if __name__ == "__main__":
  main()
