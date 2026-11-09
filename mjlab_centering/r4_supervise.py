"""Unattended supervisor for the REPORT_004 round (Steps 1-4 plus the pre-registered conditionals).

    python r4_supervise.py --cap 20

It owns the sequence so nothing depends on a session staying alive:

  * each step is one or more campaigns (Step 3 is sharded by seed, because its branches need `podrun`);
  * Steps 2 and 4 use the claim-based worker with several pods (a claimed job is never run twice, and one GPU-bound
    process per pod is 2.4x cheaper in wall clock than six processes on one pod at the same $/iteration);
  * when a step drains it runs the readout, applies the pre-registered rule (d100, d103, d104, d105, d106) and
    enqueues what the rule allows, refusing any step whose projected cost would cross the cap;
  * a dead orchestrator is restarted, a step with no progress and no pod for 90 min is retired, and the loop ends on
    a global deadline (9 h) whatever happens;
  * at the end it deletes every pod of ours, confirms none remain, writes results/r4_final.json and exits.

Spend is the larger of the RunPod billing total for today and the sum of this round's orchestrator ledgers (d101).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
STATE = RESULTS / "r4_supervisor_state.json"
PY = str(ROOT / ".venv" / "bin" / "python")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from mjlab_centering import pods, r2  # noqa: E402

USD_PER_KIT = 0.095  # measured: secure 4090, GPU-bound at ~3.2 iterations/s => $0.078/1000 it, plus eval + slack


def load_state() -> dict:
  return json.loads(STATE.read_text()) if STATE.exists() else {"steps": {}}


def save(st: dict) -> None:
  STATE.write_text(json.dumps(st, indent=1, default=float))


def log(msg: str) -> None:
  print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spend() -> float:
  try:
    out = subprocess.run([PY, str(ROOT / "analysis" / "spend.py")], capture_output=True, text=True, timeout=180)
    bill = float(json.loads(out.stdout)["total"])
  except Exception as e:  # noqa: BLE001
    log(f"billing read failed ({e!r})")
    bill = 0.0
  est = 0.0
  for f in RESULTS.glob("orch_*.json"):
    if not (f.stem.startswith("orch_r4") or f.stem.startswith("orch_clip1-r4")):
      continue
    try:
      est += float(json.loads(f.read_text()).get("spend", 0.0))
    except Exception:
      continue
  # The ledger over-counts by ~35%: it charges every pod until it is *closed out* (plus the absent grace), while the
  # invoice attributes only real container hours (checked per pod: 3 pods billed $0.78 against a $1.32 ledger). The
  # billing API is authoritative (d101), so it wins; the shrunk ledger is the fallback and the lag guard.
  return max(bill, 0.75 * est)


def campaign_pending(camp: str) -> tuple[int, int]:
  man = r2.get_json(f"{camp}/manifest.json") or {"jobs": []}
  done = r2.done_set(camp)
  jobs = [j["job_id"] for j in man["jobs"]]
  return len([j for j in jobs if j not in done]), len(jobs)


def our_pods() -> list[dict]:
  try:
    return [p for p in pods.list_pods() if str(p.get("name", "")).startswith("mjc-")]
  except Exception as e:  # noqa: BLE001
    log(f"pod list failed: {e!r}")
    return []


def orch_pid(camp: str) -> int | None:
  out = subprocess.run(["pgrep", "-f", f"orchestrator.py --campaign {camp}"], capture_output=True, text=True)
  pids = [int(x) for x in out.stdout.split()]
  return pids[0] if pids else None


def start_orch(camp: str, manifest: Path, mode: str, units: int, hard: float) -> None:
  if orch_pid(camp):
    return
  if mode == "podrun":
    mode_args = ["--launch", "1", "--podrun-procs", str(units)]
  else:  # claim-based worker: several one-job pods, which is 2.4x faster per dollar than 6 procs on one pod
    mode_args = ["--launch", str(units), "--max-pods-created", str(units + 2)]
  cmd = [PY, "orchestrator.py", "--campaign", camp, "--manifest", str(manifest), *mode_args,
         "--poll", "20", "--community-rounds", "6", "--absent-min", "6", "--min-cuda", "13.0",
         "--soft-cap", str(max(hard * 0.85, 0.5)), "--hard-cap", str(hard)]
  with open(RESULTS / f"orch_{camp}.log", "a") as f:
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
  log(f"started orchestrator {camp} ({mode} x{units}, hard cap {hard:.2f})")


def build(kind: str, *args: str) -> tuple[str, Path]:
  out = subprocess.run([PY, "jobs_r4.py", kind, *args], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
  if out.returncode != 0:
    raise RuntimeError(f"jobs_r4 {kind} failed: {out.stderr[-500:]}")
  info = json.loads(out.stdout.strip().splitlines()[-1])
  return info["campaign"], Path(info["manifest"])


def readout(cmd: str, *args: str) -> dict:
  out = subprocess.run([PY, str(ROOT / "analysis" / "report4.py"), cmd, *args], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=2400)
  if out.returncode != 0:
    log(f"readout {cmd} failed: {out.stderr[-800:]}")
    return {}
  for line in out.stdout.strip().splitlines()[-3:]:
    log(f"  {line}")
  rec = RESULTS / f"report4_{cmd}.json"
  return json.loads(rec.read_text()) if rec.exists() else {}


# fixed campaigns from before the supervisor owned them
SWEEP = ["r4sweepshard-3ea9c1cd", "r4sweepshard-20afb621", "r4sweepshard-d53fbcbb"]
# d110: the two A40 shards were replaced by these two 4090 campaigns holding only their unfinished cells
SWEEP = ["r4sweepshard-d53fbcbb", "r4sweepfin-fd982038", "r4sweepfin-5e13e8c4"]
SWEEP_ALL = ["r4sweepshard-3ea9c1cd", "r4sweepshard-20afb621"] + SWEEP
SHOCK_SHARDS = ["r4shockshard-9b6c18fc", "r4shockshard-ff8dccf1", "r4shockshard-f4928819"]
WIDTH = ["r4widthshard-072e7ac0", "r4widthshard-dbd482b3", "r4widthshard-45310f2e"]
CLIP = ["clip1-r4a1"]

STEPS = {
  "clip1": {"campaigns": CLIP, "mode": "podrun", "units": 3, "iters": 6000,
            "cmd": ("clip", "--campaign", "clip1-r4a1")},
  # one pod per shard, 6 procs each: measured $0.066/1000 it on an A40, against $0.149 for one job per pod
  "sweep": {"campaigns": SWEEP, "mode": "podrun", "units": 6, "iters": 80000,
            "cmd": ("sweep", "--campaigns", ",".join(SWEEP_ALL), "--offsets", "0,0.1,0.3,1")},
  "shock": {"campaigns": SHOCK_SHARDS, "mode": "podrun", "units": 6, "iters": 56000,
            "cmd": ("shock", "--campaign", ",".join(SHOCK_SHARDS))},
  "width": {"campaigns": WIDTH, "mode": "podrun", "units": 6, "iters": 30000,
            "cmd": ("width", "--campaign", ",".join(WIDTH))},
}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cap", type=float, default=20.0)
  ap.add_argument("--deadline-h", type=float, default=9.0)
  a = ap.parse_args()
  st = load_state()
  log(f"supervisor start; cap ${a.cap:.2f}; spent ${spend():.2f}")

  def enqueue(name: str, kind: str, args: tuple[str, ...], units: int, iters: int, note: str = "",
              mode: str = "podrun") -> None:
    if st["steps"].get(name, {}).get("campaigns"):
      return
    proj = iters / 1000.0 * USD_PER_KIT
    now = spend()
    if now + proj > a.cap:
      log(f"SKIP {name}: ${now:.2f} spent + ${proj:.2f} projected > cap ${a.cap:.2f}")
      st["steps"][name] = {"skipped": "budget", "projected": proj, "at_spend": now, "note": note}
      save(st)
      return
    camp, manifest = build(kind, *args)
    st["steps"][name] = {"campaigns": [camp], "mode": mode, "units": units, "iters": iters,
                         "manifest": str(manifest), "projected": proj}
    save(st)
    log(f"ENQUEUE {name} -> {camp} ({iters} iterations, projected ${proj:.2f}) {note}")
    start_orch(camp, manifest, mode, units, hard=min(a.cap, now + proj + 0.8))

  for name, spec in STEPS.items():
    st.setdefault("steps", {}).setdefault(name, {})["campaigns"] = spec["campaigns"]
    st["steps"][name].setdefault("mode", spec["mode"])
    st["steps"][name].setdefault("units", spec["units"])
    st["steps"][name].setdefault("iters", spec["iters"])
  save(st)

  t_start = time.time()
  stall: dict[str, tuple[int, float]] = {}
  # honour retirements recorded in a previous supervisor run (the budget guard writes them to state)
  retired: set[str] = {k for k, v in st["steps"].items() if v.get("retired") or v.get("finalized")}

  def kill_step(name: str, why: str) -> None:
    spec = STEPS.get(name) or {}
    camps = (st["steps"].get(name) or {}).get("campaigns") or spec.get("campaigns") or []
    for camp in camps:
      subprocess.run(["pkill", "-f", f"orchestrator.py --campaign {camp}"], capture_output=True)
    for p in our_pods():
      if any(c in str(p.get("name", "")) for c in camps):
        try:
          pods.delete_pod(p["id"])
          log(f"budget guard: deleted pod {p['id']} ({p['name']})")
        except Exception as e:  # noqa: BLE001
          log(f"budget guard: delete {p['id']} failed: {e!r}")
    st["steps"].setdefault(name, {})["retired"] = why
    retired.add(name)
    save(st)
    log(f"BUDGET GUARD: retired {name} ({why}) at ${spend():.2f}")

  while True:
    now_spend = spend()
    # d101: hard cap. Width goes first (cheapest to re-run and least load-bearing), then sweep shards are stopped
    # before shock, which carries the round's novel result.
    if now_spend > 0.90 * a.cap and "width" not in retired and not st["steps"].get("width", {}).get("finalized"):
      kill_step("width", f"budget: ${now_spend:.2f} of ${a.cap:.2f}")
    if now_spend > 0.99 * a.cap and "sweep" not in retired and not st["steps"].get("sweep", {}).get("finalized"):
      kill_step("sweep", f"budget: ${now_spend:.2f} of ${a.cap:.2f}")
    if now_spend > 0.999 * a.cap:
      for nm in ("shock", "clip1"):
        if nm not in retired and not st["steps"].get(nm, {}).get("finalized"):
          kill_step(nm, f"budget: ${now_spend:.2f} of ${a.cap:.2f}")
    any_live = bool(our_pods())
    all_done = True
    for name, spec in STEPS.items():
      if name in retired:
        continue
      stp = st["steps"].setdefault(name, {})
      camps = stp.get("campaigns") or spec["campaigns"]
      def manifest_for(c: str) -> Path:
        p = ROOT / "manifests" / f"{c}.json"
        return p if p.exists() else Path(stp.get("manifest", ""))
      pend_tot = 0
      unshipped = False
      for camp in camps:
        pending, total = campaign_pending(camp)
        pend_tot += pending
        if total == 0:
          unshipped = True
          all_done = False
          if not orch_pid(camp):
            start_orch(camp, manifest_for(camp), spec["mode"], spec["units"],
                       hard=min(a.cap, now_spend + spec["iters"] / 1000.0 * USD_PER_KIT + 0.8))
        elif pending > 0 and not orch_pid(camp):
          start_orch(camp, manifest_for(camp), spec["mode"], spec["units"],
                     hard=min(a.cap, now_spend + spec["iters"] / 1000.0 * USD_PER_KIT + 0.8))
      if unshipped:  # a campaign that has not shipped yet is not drained
        all_done = False
        continue
      if pend_tot > 0:
        all_done = False
        seen = stall.get(name, (pend_tot, time.time()))
        if seen[0] != pend_tot:
          seen = (pend_tot, time.time())
        stall[name] = seen
        if time.time() - seen[1] > 5400 and not any_live:
          log(f"{name}: {pend_tot} jobs left, no progress and no pod for 90 min; retiring the step")
          stp["retired_unfinished"] = pend_tot
          retired.add(name)
      elif not stp.get("finalized"):
        rec = readout(*spec["cmd"])
        stp["finalized"] = True
        stp["result"] = rec
        save(st)
        log(f"DONE {name}")
        if name == "clip1":
          if rec.get("n_walk", 0) >= 1:
            enqueue("clip20", "clip20", (), 3, 17 * 2000, "d100 fired")
          else:
            st["steps"]["clip20"] = {"skipped": "d100: 0 of 3 walked"}
        if name == "sweep":
          for key, v in (rec or {}).items():
            if key.startswith("contrast_") and isinstance(v, dict) and v.get("topup_fires"):
              enqueue(f"topup_sweep_{key.split('_', 1)[1]}", "topup-sweep", (key.split("_", 1)[1],), 3, 40000,
                      f"d103 fired at c={key.split('_', 1)[1]}")
        if name == "shock":
          eff = (rec or {}).get("effect") or {}
          for kind in sorted({k.rsplit("_", 1)[0] for k in eff}):
            differ = False
            for arm in ("none", "value"):
              pz = (eff.get(f"{kind}_{arm}") or {}).get("dz_vs_ctrl") or {}
              ci = pz.get("ci") or [0, 0]
              if pz.get("n", 0) >= 3 and (ci[0] > 0 or ci[1] < 0):
                differ = True
            rn = (eff.get(f"{kind}_none") or {}).get("recovered")
            rv = (eff.get(f"{kind}_value") or {}).get("recovered")
            nn = (eff.get(f"{kind}_none") or {}).get("n", 0) or 0
            if rn is not None and rv is not None and nn and abs(rn - rv) >= max(2, round(0.3 * nn)):
              differ = True
            if differ:
              enqueue(f"topup_shock_{kind}", "topup-shock", ("3010-3019", kind), 4, 20 * 1200 + 20 * 400,
                      f"d104 fired for {kind}")
            else:
              st["steps"][f"topup_shock_{kind}"] = {"skipped": f"d104: {kind} arms do not differ"}
              save(st)
        if name == "width":
          best = (rec or {}).get("best") or {}
          if best.get("arm"):
            n_ = int(best["arm"].split("x")[0][-1])
            arm_rec = rec["arms"][best["arm"]]
            if arm_rec["walks"] / max(arm_rec["n"], 1) >= 2 / 3:
              enqueue("ssp_confirm", "ssp-confirm", ("ssp_low", str(n_), str(n_)), 3, 40 * 2000,
                      f"d106: {best['arm']} walks {arm_rec['walks']}/{arm_rec['n']}")
            else:
              log(f"d106: {best['arm']} does not walk reliably; no confirmation")
            if best.get("beats_rff") or (best.get("gap_to_raw") is not None and abs(best["gap_to_raw"]) < 0.03):
              enqueue("rff_confirm", "rff-confirm", (str(n_), str(n_)), 3, 20 * 2000, "d106: SSP ties/beats raw")
            if best["arm"].startswith("ssp_low4x4"):
              enqueue("width8", "width8", (), 2, 6 * 2000, "d105(iv): 4x4 won")
    # extra campaigns (top-ups, confirmations) keep the loop alive
    extra = 0
    for name, stp in list(st["steps"].items()):
      if name in STEPS or name in retired:
        continue
      camps = stp.get("campaigns") or ([stp["campaign"]] if stp.get("campaign") else [])
      for camp in camps:
        p, _ = campaign_pending(camp)
        extra += p
        if p > 0 and not orch_pid(camp):
          start_orch(camp, Path(stp.get("manifest", "")), stp.get("mode", "podrun"),
                     stp.get("units", 3), hard=min(a.cap, now_spend + 1.5))
    if all_done and extra == 0:
      log("every step drained")
      break
    if time.time() - t_start > a.deadline_h * 3600:
      log("global deadline reached")
      break
    time.sleep(60)

  # consolidated readouts
  try:
    log("consolidated readouts (top-ups included)")
    sweep_extra = [v["campaigns"][0] for k, v in st["steps"].items() if k.startswith("topup_sweep") and v.get("campaigns")]
    readout("sweep", "--campaigns", ",".join(SWEEP_ALL + sweep_extra), "--offsets", "0,0.1,0.3,1")
    shock_extra = [v["campaigns"][0] for k, v in st["steps"].items() if k.startswith("topup_shock") and v.get("campaigns")]
    readout("shock", "--campaign", ",".join(SHOCK_SHARDS + shock_extra))
    readout("width", "--campaign", ",".join(WIDTH))
  except Exception as e:  # noqa: BLE001
    log(f"consolidated readout failed: {e!r}")

  log(f"live pods before close-out: {[(p['id'], p['name']) for p in our_pods()]}")
  deadline = time.time() + 900
  while our_pods() and time.time() < deadline:
    for p in our_pods():
      try:
        pods.delete_pod(p["id"])
        log(f"deleted pod {p['id']} ({p['name']})")
      except Exception as e:  # noqa: BLE001
        log(f"delete {p['id']} failed: {e!r}")
    time.sleep(15)
  final = {"spent": spend(), "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "live_pods": our_pods(),
           "steps": {k: {kk: vv for kk, vv in v.items() if kk != "result"} for k, v in st["steps"].items()}}
  (RESULTS / "r4_final.json").write_text(json.dumps(final, indent=1, default=float))
  log(f"FINAL spent ${final['spent']:.2f}; live pods {len(final['live_pods'])}")


if __name__ == "__main__":
  main()
