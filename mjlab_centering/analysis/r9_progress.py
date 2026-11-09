"""One-line progress for the round-9a pilot.

    python analysis/r9_progress.py [--campaign r9pilot-3ae3cc84]

Pods are filtered on this round's own prefix (`mjc-r9`), never the whole account: another campaign may be running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis")]
from mjlab_centering import pods, r2  # noqa: E402
import spend  # noqa: E402


def arm_of(jid: str) -> str:
  for a in ("a1_005", "a1_05", "clipoff", "a2_off", "a2_on", "a3"):
    if f"_go1_{a}_" in jid:
      return a
  return "?"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  a = ap.parse_args()
  ps = [p for p in pods.list_pods() if str(p.get("name", "")).startswith("mjc-r9")]
  print(f"pods: {len(ps)}")
  for p in ps:
    print(f"  {p['id']} {p.get('name')} {p.get('costPerHr')}/h {p.get('desiredStatus')}/{p.get('runtimeStatus')}")
  done = r2.done_set(a.campaign)
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  claims = r2.list_keys(f"{a.campaign}/claims/")
  by = Counter(arm_of(j) for j in done)
  tot = Counter(arm_of(j["job_id"]) for j in man["jobs"])
  print(f"{len(done)}/{len(man['jobs'])} DONE, {len(claims)} claimed now")
  for k in sorted(tot):
    print(f"  {k:8s} {by.get(k, 0):2d}/{tot[k]}")
  st = ROOT / "results" / f"orch_{a.campaign}.json"
  if st.exists():
    print(f"orchestrator ledger: ${json.loads(st.read_text()).get('spend', 0):.2f}")
  # the freshest iteration rows of any run in flight (a progress signal that does not need the pod list)
  best = []
  for jid in [j["job_id"] for j in man["jobs"] if j["job_id"] not in done][:6]:
    try:
      body = r2.client().get_object(Bucket=r2.bucket(), Key=r2.key(f"{a.campaign}/runs/{jid}/iter.jsonl"))["Body"].read()
      n = body.count(b"\n")
      best.append(f"{jid} {n}/2000")
    except Exception:  # noqa: BLE001
      continue
  for b in best:
    print("  in flight:", b)


if __name__ == "__main__":
  main()
