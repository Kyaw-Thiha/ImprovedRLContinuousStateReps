"""REPORT_009 round 9b readout: per-position advantage normalisation and lambda = 0, against the uncentered cell.

    python analysis/r9_leak_report.py --campaign r9aleak-xxxxxxxx [--pull] [--json results/r9_leak.json] [--fig]
    python analysis/r9_leak_report.py --local results/r9smoke_pos_norm0.95 ...   # smoke dirs, for validating the maths

Pre-registered reading (d1102/d1103):
  rescue  - walks on >= 7 of 10 AND the paired dtrack CI against the uncentered cell excludes 0 (unchanged from d303);
  lambda  - the lambda = 0 arm is read ONLY under the validity gate "the centred lambda = 0 control walks >= 7 of 10".

Mechanism diagnostics, from the per-iteration `adv_pos_mean` vector every 9b run logs:
  * the per-position profile of the raw advantage, mean over a window (default iterations 100-300);
  * `r2_pos` = share of the advantage's variance explained by the rollout position (median over the window);
  * a least-squares fit of the profile on the theoretical weights w_t: slope b, its R^2, and
    e_fit = -b / (1 - gamma), to be compared with the independent estimate e_indep = v_mean - r_seen_rate/(1-gamma)
    (a crude steady-state reading of the critic's level error; reported with that caveat, not as truth).

Nothing here changes a run; the script only reads artefacts (and `--pull` copies them from R2).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r9_step0 as S  # noqa: E402
from report4 import paired, seed_of  # noqa: E402
from mjlab_centering import r2  # noqa: E402

MERGED = ROOT / "results" / "merged_r9b"
WALK = 0.5
DIAG_SEEDS = (3000, 3001, 3002)
ARMS = ("posnorm", "poscenter", "lam0_none", "lam0_value")
LABEL = {
  "posnorm": "per-position normalisation (mean+std per position)",
  "poscenter": "per-position centring, global scale",
  "lam0_none": "lambda = 0, uncentered",
  "lam0_value": "lambda = 0, value-centred (validity gate)",
  "none": "no centering (reference)",
  "value": "value centering (reference)",
}


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


def rows(jid: str, name: str) -> list[dict]:
  for base in (MERGED, ROOT / "results" / "merged_r9", ROOT / "results" / "merged_r4", ROOT / "results" / "merged_r2"):
    p = base / jid / name
    if p.exists():
      out = S.jl(p)
      if out:
        return out
  return []


def cell(jid: str) -> dict:
  ev = rows(jid, "eval.jsonl")
  if len(ev) < 5:
    return {"job_id": jid, "ok": False}
  tr = float(np.mean([r["track_lin_frac"] for r in ev[-5:]]))
  j = float(np.mean([r["J_task"] for r in ev[-5:]]))
  it = rows(jid, "iter.jsonl")
  prof = position_profile(it)
  return {"job_id": jid, "ok": True, "track": tr, "J": j, "walks": tr >= WALK,
          "t50": S.first_eval(ev, 0.5), "prof": prof}


GAMMA_9B = 0.999  # every round-9b cell is Go1 at gamma .999; the lambda is read from each run's own rows


def _lam_of(it: list[dict]) -> float:
  """The run's own GAE lambda (logged every iteration since d1101). The fit MUST use it: at lambda = 0 the weights are
  flat, and fitting against the lambda = 0.95 ramp would report a slope for an arm whose prediction is 'no slope'."""
  for r in it:
    v = r.get("lam")
    if isinstance(v, (int, float)) and np.isfinite(v):
      return float(v)
  return 0.95


def position_profile(it: list[dict], lo: int = 100, hi: int = 300) -> dict | None:
  """Mean per-position advantage profile over a window, plus the position R^2 and the fit on w_t (the run's own lambda)."""
  sel = [r for r in it if lo <= r.get("iter", -1) <= hi and r.get("adv_pos_mean")]
  if not sel:
    return None
  prof = np.mean([r["adv_pos_mean"] for r in sel], axis=0)
  r2pos = float(np.median([r.get("adv_pos_r2", np.nan) for r in sel]))
  lam = _lam_of(it)
  w = S_w(GAMMA_9B, lam, len(prof))
  b, a = np.polyfit(w, prof, 1)
  fit = a + b * w
  ss = float(np.sum((prof - prof.mean()) ** 2) or 1e-12)
  r2fit = float(1 - np.sum((prof - fit) ** 2) / ss)
  e_ind = float(np.median([r.get("v_mean", np.nan) - r.get("r_seen_rate", np.nan) / (1 - GAMMA_9B) for r in sel]))
  return {"prof": [float(x) for x in prof], "r2_pos": r2pos, "w": [float(x) for x in w], "lam": lam,
          "slope": float(b), "r2_fit": r2fit, "e_fit": float(-b / (1 - GAMMA_9B)), "e_indep": e_ind,
          "n_iters": len(sel)}


def S_w(gamma: float, lam: float, T: int) -> np.ndarray:
  return np.array([sum((gamma * lam) ** k for k in range(T - t)) for t in range(T)])


def pv(p: dict) -> str:
  return f"{p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default=None)
  ap.add_argument("--pull", action="store_true")
  ap.add_argument("--local", nargs="*", default=None, help="smoke result dirs (validates the maths, not a result)")
  ap.add_argument("--json", default=str(ROOT / "results" / "r9_leak.json"))
  a = ap.parse_args()

  if a.local:
    print("| dir | adv profile (first, last) | r2_pos | fit R^2 | slope | e_fit | e_indep |")
    print("|---|---|---|---|---|---|---|")
    for name in a.local:
      d = Path(name)
      it = [json.loads(l) for l in (d / "iter.jsonl").read_text().splitlines() if l.strip()]
      p = position_profile(it, lo=0, hi=10_000)
      if not p:
        print(f"| {d.name} | no adv_pos_mean | | | | | |")
        continue
      print(f"| {d.name} | {p['prof'][0]:.2f} -> {p['prof'][-1]:.2f} | {p['r2_pos']:+.3f} | {p['r2_fit']:.3f} | "
            f"{p['slope']:+.4f} | {p['e_fit']:.0f} | {p['e_indep']:.0f} |")
    return

  assert a.campaign, "--campaign or --local"
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  ids = [j["job_id"] for j in man["jobs"]]
  if a.pull:
    pull(a.campaign, ids)
  done = r2.done_set(a.campaign)
  seeds = sorted({seed_of(i) for i in ids})

  cells: dict[str, dict[int, dict]] = {}
  for s in seeds:
    cells.setdefault("none", {})[s] = cell(f"w2main_go1_raw_none_g0.999_o1_s{s}")
    cells.setdefault("value", {})[s] = cell(f"w2main_go1_raw_value_g0.999_o1_s{s}")
    for arm in ARMS:
      jid = f"r9leak_go1_{arm}_g0.999_o1_s{s}"
      c = cell(jid)
      if c.get("ok"):
        cells.setdefault(arm, {})[s] = c
  for k in ("none", "value"):
    for s in [s for s, c in cells[k].items() if not c.get("ok")]:
      cells[k].pop(s)

  print(f"campaign {a.campaign}: {len(done)}/{len(ids)} runs complete\n")
  print("| arm | n | walks k/n | tracking | J_task | dtrack vs uncentered (95% CI) | dtrack vs value (95% CI) | t50 | verdict (d1102/d1103) |")
  print("|---|---|---|---|---|---|---|---|---|")
  out = {}
  for arm in ARMS:
    d = cells.get(arm, {})
    if not d:
      print(f"| {LABEL[arm]} | 0 | — | — | — | — | — | — | not run |")
      continue
    tr = {s: c["track"] for s, c in d.items()}
    j = {s: c["J"] for s, c in d.items()}
    p_n = paired(tr, {s: cells["none"][s]["track"] for s in d if s in cells["none"]})
    p_v = paired(tr, {s: cells["value"][s]["track"] for s in d if s in cells["value"]})
    walks = sum(c["walks"] for c in d.values())
    t50 = [c["t50"] for c in d.values() if c["t50"]]
    rescue = walks >= 7 and p_n["n"] >= 2 and p_n["ci"][0] > 0
    if len(d) < 10:
      verdict = f"n={len(d)} (pilot)"
    elif arm == "lam0_value":
      verdict = "validity gate: " + ("PASS" if walks >= 7 else "FAIL (lambda arm uninformative)")
    else:
      verdict = "**rescues**" if rescue else "does not rescue"
    print(f"| {LABEL[arm]} | {len(d)} | {walks}/{len(d)} | {np.mean(list(tr.values())):.3f} | "
          f"{np.mean(list(j.values())):.4f} | {pv(p_n)} | {pv(p_v)} | "
          f"{int(np.median(t50)) if t50 else '—'} | {verdict} |")
    out[arm] = {"n": len(d), "walks": walks, "track": float(np.mean(list(tr.values()))),
                "vs_none": p_n, "vs_value": p_v, "rescue": bool(rescue),
                "failed_seeds": [s for s, c in d.items() if not c["walks"]],
                "lam": sorted({round(c["prof"]["lam"], 6) for c in d.values() if c.get("prof")})}

  print("\n**Per-position advantage profile** (mean over iterations 100-300; `e_fit` from the slope on w_t, "
        "`e_indep` = v_mean - r_seen_rate/(1-gamma)):\n")
  print("| arm | n | profile first -> last | r2_pos | fit R^2 | slope | e_fit | e_indep |")
  print("|---|---|---|---|---|---|---|---|")
  for arm in ARMS:
    d = cells.get(arm, {})
    profs = [c["prof"] for c in d.values() if c.get("prof")]
    if not profs:
      continue
    prof = np.mean([p["prof"] for p in profs], axis=0)
    lams = sorted({round(p["lam"], 6) for p in profs})
    b = float(np.mean([p["slope"] for p in profs]))
    print(f"| {LABEL[arm]} | {len(profs)} | {prof[0]:.2f} -> {prof[-1]:.2f} | "
          f"{np.median([p['r2_pos'] for p in profs]):+.3f} | {np.median([p['r2_fit'] for p in profs]):.3f} | "
          f"{b:+.4f} | {np.median([p['e_fit'] for p in profs]):.0f} | "
          f"{np.median([p['e_indep'] for p in profs]):.0f} |")
  out["profiles"] = {}
  for arm, d in cells.items():
    ps = [c["prof"] for c in d.values() if c.get("prof")]
    if ps:
      out["profiles"][arm] = {"prof": np.mean([p["prof"] for p in ps], axis=0).tolist(),
                              "w": np.mean([p["w"] for p in ps], axis=0).tolist(),
                              "lam": sorted({round(p["lam"], 6) for p in ps}),
                              "r2_fit": [p["r2_fit"] for p in ps], "e_fit": [p["e_fit"] for p in ps]}
  # ---------------------------------------------------------------- the diagnostic arms and the pre-registered reading
  diag = {}
  for kind in ("none", "value"):
    profs = [position_profile(rows(f"r9diag_go1_diag_{kind}_g0.999_o1_s{s}", "iter.jsonl")) for s in DIAG_SEEDS]
    profs = [p for p in profs if p]
    if not profs:
      continue
    diag[kind] = {"n": len(profs),
                  "r2_fit": float(np.median([p["r2_fit"] for p in profs])),
                  "r2_pos": float(np.median([p["r2_pos"] for p in profs])),
                  "slope": float(np.median([p["slope"] for p in profs])),
                  "e_fit": float(np.median([p["e_fit"] for p in profs])),
                  "e_indep": float(np.median([p["e_indep"] for p in profs])),
                  "prof": np.mean([p["prof"] for p in profs], axis=0).tolist()}
  if diag:
    print("\n**Diagnostic arms** (600-iteration runs, profile over iterations 100-300):\n")
    print("| diag arm | n | profile first -> last | r2_pos | fit R^2 | e_fit | e_indep | ratio |e_fit|/|e_indep| |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, d in diag.items():
      ratio = abs(d["e_fit"]) / abs(d["e_indep"]) if d["e_indep"] else float("nan")
      print(f"| diag_{kind} | {d['n']} | {d['prof'][0]:.2f} -> {d['prof'][-1]:.2f} | {d['r2_pos']:+.3f} | "
            f"{d['r2_fit']:.3f} | {d['e_fit']:.0f} | {d['e_indep']:.0f} | {ratio:.2f} |")
    out["diag"] = {k: {kk: vv for kk, vv in v.items()} for k, v in diag.items()}

  # d1103's reading, applied mechanically. Missing pieces are reported as "not yet decidable", never assumed.
  def _walks(arm):
    return out.get(arm, {}).get("walks", 0), out.get(arm, {}).get("n", 0)

  wn, nn = _walks("posnorm")
  wc, nc = _walks("poscenter")
  wv, nv = _walks("lam0_value")
  gate = nv >= 10 and wv >= 7
  alarms = (nn >= 10 and wn >= 7) or (nc >= 10 and wc >= 7)
  fit_none = diag.get("none", {}).get("r2_fit")
  slope_none = abs(diag.get("none", {}).get("slope", 0.0))
  slope_val = abs(diag.get("value", {}).get("slope", 0.0))
  diag_match = (fit_none is not None and fit_none >= 0.8 and slope_val > 0
                and slope_none >= 5 * slope_val)
  print("\n**Pre-registered reading (d1103):**")
  print(f"- lambda arm validity gate: {'PASS' if gate else 'FAIL' if nv >= 10 else 'not yet decidable'}"
        f" (lam0_value walks {wv}/{nv}); lambda arm rescues: "
        f"{'yes' if gate and nv >= 10 and out.get('lam0_none', {}).get('walks', 0) >= 7 else 'no' if gate and nv >= 10 else 'uninformative without the gate'}")
  print(f"- a position arm rescues: {'yes' if alarms else 'no' if (nn >= 10 and nc >= 10) else 'not yet decidable'}"
        f" (posnorm {wn}/{nn}, poscenter {wc}/{nc})")
  print(f"- diagnostic matches the theory: {'yes' if diag_match else 'no' if diag else 'not yet decidable'}"
        f" (uncentered fit R^2 {fit_none if fit_none is None else round(fit_none, 3)}, "
        f"|slope| ratio {slope_none / slope_val if slope_val else float('nan'):.1f})")
  if alarms and diag_match:
    print("- **reading (i): the leak is supported as a sufficient cause.**")
  elif alarms and diag and not diag_match:
    print("- **reading (ii): a rescue without the theory's position profile — no mechanism claim.**")
  elif (nn >= 10 and nc >= 10) and not alarms:
    print("- **reading (iii): neither position arm rescues — the leak is not a sufficient cause here; fall back to the "
          "descriptive 'constant share tracks failure'.**")
  else:
    print("- not yet decidable at this n.")

  Path(a.json).write_text(json.dumps(out, indent=1, default=float))
  print(f"\nwrote {a.json}")


if __name__ == "__main__":
  main()
