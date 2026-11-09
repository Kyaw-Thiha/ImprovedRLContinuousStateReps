"""The unattended gate: when the pilot finishes, check the harness, read the bill, and publish Stage 2 only if the
pre-registered rule allows it (d155, d166).

    python analysis/r7_keeper.py --pilot r7pilot-63bd51f3 --queue manifests/r7vnsc-a565805a.json [--dry-run]

This exists because a captured community pod is the scarcest resource in the project: with a pod kept warm (d165) the
next stage should start the moment the gate passes, without waiting for a human. The checks are mechanical and were
fixed before the pilot finished:

  1. harness: all pilot jobs `DONE`; every run has its full eval series and reached its final iteration; every run's last
     iteration row is finite and carries the arm's own diagnostic key (SC `r_bar`, VN `vn_mean`/`vn_std`, RS `rs_ret_std`).
     A run that is missing, truncated, or missing its diagnostic means the arm is not wired as intended -> do NOT publish.
  2. budget: rate = measured pilot cost / 24 (dollars per 1000 iterations) from `r7_spend.py`'s per-pod attribution;
     project Stage 2 at 160,000 iterations; publish only if `spent + 1.3 x projected <= 18` (d155).
  3. once published, exit: the warm pod picks the queue up on its next 60 s poll.

Nothing else is launched, and no stage is published on a failed check. `--dry-run` prints the decision only.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import r7_gate as GATE  # noqa: E402
import r7_publish_queue as PUB  # noqa: E402
from mjlab_centering import r2  # noqa: E402

DIAG = {"sc": ("r_bar", "r_task_rate"), "vn": ("vn_mean", "vn_std", "r_task_rate"),
        "vn_noclamp": ("vn_mean", "vn_std"), "rs": ("rs_ret_std", "r_task_rate")}
MIN_EVALS = 39          # 2000 iterations at eval_every 50 -> 40 evals; one short is tolerated, two is not
FINAL_ITER = 1990


def arm_of(jid: str) -> str:
  for a in ("vn_noclamp", "vn", "sc", "rs"):
    if f"_go1_{a}_" in jid:
      return a
  return "?"


def _rows(campaign: str, jid: str, name: str) -> list[dict]:
  try:
    body = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{campaign}/runs/{jid}/{name}"))["Body"].read()
  except Exception:
    return []
  return [json.loads(l) for l in body.decode().splitlines() if l.strip()]


def harness_checks(campaign: str, manifest: dict) -> list[str]:
  """Returns a list of failures; empty means the pilot is complete and every arm is wired."""
  fails: list[str] = []
  done = r2.done_set(campaign)
  for j in manifest["jobs"]:
    jid = j["job_id"]
    if jid not in done:
      fails.append(f"{jid}: not DONE")
      continue
    ev = _rows(campaign, jid, "eval.jsonl")
    it = _rows(campaign, jid, "iter.jsonl")
    if len(ev) < MIN_EVALS:
      fails.append(f"{jid}: {len(ev)} evals (< {MIN_EVALS})")
    if not it or int(it[-1].get("iter", -1)) < FINAL_ITER:
      fails.append(f"{jid}: final iteration {it[-1].get('iter') if it else None} (< {FINAL_ITER})")
      continue
    arm = arm_of(jid)
    for k in DIAG.get(arm, ()):
      v = it[-1].get(k)
      if v is None or (isinstance(v, float) and not math.isfinite(v)):
        fails.append(f"{jid}: diagnostic {k}={v}")
  return fails


def spend_now() -> dict:
  out = subprocess.run([sys.executable, str(HERE / "r7_spend.py"), "--json"], capture_output=True, text=True)
  if out.returncode != 0:
    raise RuntimeError(out.stderr[-400:])
  return json.loads(out.stdout)


def decide(campaign: str, queue_manifest: Path, pilot_iters: int, stage_iters: int, cap: float = 18.0,
           sunk: float = 0.0) -> dict:
  """`sunk` is spend that cannot recur (here: the pods lost before any job ran, d160/d162). It belongs in `spent`, not in
  the marginal rate — otherwise the same money is counted twice and a stage is blocked for a cost that is already paid."""
  man = r2.get_json(f"{campaign}/manifest.json") or {"jobs": []}
  fails = harness_checks(campaign, man)
  sp = spend_now()
  spent = float(sp["round7_total"])
  if spent <= 0:
    fails.append("measured spend is 0: the billing read is not yet populated, so the gate cannot be evaluated")
  rate = GATE.rate_per_1000it(max(spent - sunk, 1e-9), pilot_iters) if spent > 0 else float("nan")
  g = GATE.gate(spent, rate * stage_iters / 1000.0, cap=cap) if spent > 0 else {"pass": False, "projected": float("nan"),
                                                                               "gate_need": float("nan"), "cap": cap}
  return {"campaign": campaign, "fails": fails, "spent": spent, "sunk_not_counted_in_rate": sunk,
          "marginal_cost_of_pilot": round(spent - sunk, 4), "rate_per_1000it": rate,
          "stage_iters": stage_iters, "gate": g, "publish": (not fails) and bool(g["pass"])}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pilot", required=True)
  ap.add_argument("--queue", required=True, help="manifest to publish as the next stage")
  ap.add_argument("--pilot-iters", type=int, default=24000)
  ap.add_argument("--stage-iters", type=int, default=160000)
  ap.add_argument("--cap", type=float, default=18.0)
  ap.add_argument("--sunk", type=float, default=0.87,
                  help="measured spend that cannot recur (pods lost before any job ran: d160/d162, $0.87); excluded from "
                       "the marginal rate so it is not counted twice")
  ap.add_argument("--wait-min", type=float, default=240.0, help="stop waiting for the pilot after this many minutes")
  ap.add_argument("--poll", type=float, default=300.0)
  ap.add_argument("--dry-run", action="store_true")
  a = ap.parse_args()
  deadline = time.time() + a.wait_min * 60.0
  while True:
    try:
      d = decide(a.pilot, Path(a.queue), a.pilot_iters, a.stage_iters, a.cap, a.sunk)
    except Exception as e:  # noqa: BLE001 - a transient R2/API failure must not end the keeper
      print(f"[keeper] transient error ({type(e).__name__}: {e}); retrying in {a.poll:g}s", flush=True)
      if time.time() > deadline:
        print("[keeper] deadline reached during a transient failure; exiting without publishing", flush=True)
        return
      time.sleep(a.poll)
      continue
    print(json.dumps({**d, "at": time.strftime("%H:%M:%SZ", time.gmtime())}, indent=1), flush=True)
    if d["publish"]:
      if a.dry_run:
        print("[keeper] dry run: would publish")
        return
      p = PUB.publish(a.pilot, Path(a.queue), note="stage 2, unattended gate (d166)")
      print(f"[keeper] published {len(p['jobs'])} jobs to {a.pilot}/queue.json", flush=True)
      return
    if d["fails"] and all("not DONE" in f for f in d["fails"]) and time.time() < deadline:
      time.sleep(a.poll)  # the pilot is still running; keep waiting
      continue
    print("[keeper] not publishing: " + ("; ".join(d["fails"]) or "the budget gate did not pass"), flush=True)
    return


if __name__ == "__main__":
  main()
