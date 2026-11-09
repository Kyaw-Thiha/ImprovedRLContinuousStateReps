"""One-line progress report for the round-7 claim set (and whichever pods are carrying it).

    python analysis/r7_progress.py                     # pods, claims, DONE counts per offset/arm, spend
    python analysis/r7_progress.py --stage2-campaign r7vnsc-a565805a
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]
from mjlab_centering import pods, r2  # noqa: E402


def arm_of(jid: str) -> str:
  for a in ("vn_noclamp", "vn", "sc", "rs"):
    if f"_go1_{a}_" in jid:
      return a
  return "?"


def offset_of(jid: str) -> str:
  try:
    return jid.split("_g0.999_o")[1].split("_")[0]
  except IndexError:
    return "?"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--launch-campaign", default="r7pilot-63bd51f3")
  ap.add_argument("--stage2-campaign", default="r7vnsc-a565805a")
  a = ap.parse_args()
  ps = [p for p in pods.list_pods() if str(p.get("name", "")).startswith("mjc-r7")]
  print(f"pods: {len(ps)}")
  for p in ps:
    print(f"  {p['id']} {p.get('name')} {p.get('costPerHr')}/h {p.get('desiredStatus')}/{p.get('runtimeStatus')}")
  done = r2.done_set(a.stage2_campaign)
  man = r2.get_json(f"{a.stage2_campaign}/manifest.json") or {"jobs": []}
  claims = r2.list_keys(f"{a.stage2_campaign}/claims/")
  by = Counter((offset_of(j), arm_of(j)) for j in done)
  tot = Counter((offset_of(j["job_id"]), arm_of(j["job_id"])) for j in man["jobs"])
  print(f"stage 2: {len(done)}/{len(man['jobs'])} DONE, {len(claims)} claimed now")
  # ETA from the fleet: each pod runs 6 jobs at once and a job takes about 76 min (0.44 iterations/s for 2000)
  running = [p for p in ps if str(p.get("runtimeStatus", "")).lower() == "running"]
  left = len(man["jobs"]) - len(done)
  if running and left:
    import datetime as _dt2
    waves = left / (6.0 * len(running))
    eta = _dt2.datetime.now(_dt2.timezone.utc) + _dt2.timedelta(minutes=76 * waves)
    print(f"  fleet {len(running)} pod(s) x 6 jobs, {left} to go -> {waves:.1f} waves, ETA {eta.strftime('%H:%M')}Z")
  # ETA from the completion timestamps of the finished runs (jobs/hour since the first completion)
  if done:
    import datetime as _dt
    ts = []
    for o in r2.client().list_objects_v2(Bucket=r2.bucket(), Prefix=r2.key(f"{a.stage2_campaign}/runs/")).get("Contents", []):
      if o["Key"].endswith("/DONE"):
        ts.append(o["LastModified"])
    if len(ts) >= 2:
      t0, t1 = min(ts), max(ts)
      hours = max((t1 - t0).total_seconds() / 3600.0, 1e-6)
      rate = (len(ts) - 1) / hours
      left = len(man["jobs"]) - len(done)
      eta = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=left / rate if rate > 0 else float("inf"))
      print(f"  throughput {rate:.1f} runs/h; {left} to go, ETA {eta.strftime('%H:%M')}Z")
  for k in sorted(tot):
    print(f"  c={k[0]:<4} {k[1]:<5} {by.get(k, 0):2d}/{tot[k]:2d}")
  try:
    import r7_spend as S
    rows = S._billing_pods("2026-11-06T00:00:00Z")
    own = {k: v for k, v in S.round7_pods().items() if v["campaign"].split("-")[0] in ("r7pilot", "r7vnsc", "r7rs")}
    total = sum(float(r["amount"]) for r in rows if r.get("podId") in own)
    print(f"spend (own pods, measured): ${total:.2f} of $18")
  except Exception as e:  # noqa: BLE001
    print(f"spend read failed: {type(e).__name__}")


if __name__ == "__main__":
  main()
