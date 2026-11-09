"""Wait for the first mirrored training rows of a campaign, then print them (smoke check for a launch).

    python analysis/wait_for_rows.py --campaign g1diag-7c41e2 --rows 25 --timeout 7200

Exits as soon as every job in the manifest has >= `rows` rows on R2 (or a job is DONE/FAILED, or the timeout passes,
or no live pod remains and nothing moved for 15 minutes). Prints the offset/centering fields the launch was supposed
to set, which is the check that a continuing-offset run really ran at the offset it claims.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from mjlab_centering import pods, r2


def rows(rel: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "x.jsonl"
    if not r2.download_file(rel, str(p)):
      return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--rows", type=int, default=25)
  ap.add_argument("--timeout", type=float, default=7200.0)
  ap.add_argument("--pod-prefix", default=None)
  ap.add_argument("--any-k", type=int, default=0, help="exit once K jobs have rows (a smoke check) instead of all")
  a = ap.parse_args()
  prefix = a.pod_prefix or f"mjc-{a.campaign}"
  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  jids = [j["job_id"] for j in man["jobs"]]
  t0 = time.time()
  while True:
    got, busy = {}, 0
    for jid in jids:
      if r2.exists(f"{a.campaign}/runs/{jid}/DONE"):
        st = "DONE"
      elif r2.exists(f"{a.campaign}/runs/{jid}/FAILED"):
        st = "FAILED"
      else:
        st = "running"
      r = rows(f"{a.campaign}/runs/{jid}/iter.jsonl")
      got[jid] = (st, r)
      if len(r) < a.rows and st == "running":
        busy += 1
    if a.any_k:
      if sum(1 for st, r in got.values() if len(r) >= a.rows) >= a.any_k:
        break
    elif busy == 0:
      break
    live = [p for p in pods.list_pods() if str(p.get("name", "")).startswith(prefix)
            and p.get("runtimeStatus") == "running"]
    print(f"[wait] {time.strftime('%H:%M:%S')} +{int(time.time() - t0)}s live_pods={len(live)} "
          + " ".join(f"{k}:{v[0]}/{len(v[1])}" for k, v in got.items()), flush=True)
    if not live and all(len(v[1]) == 0 for v in got.values()):
      time.sleep(900)
      live = [p for p in pods.list_pods() if str(p.get("name", "")).startswith(prefix)
              and p.get("runtimeStatus") == "running"]
      if not live:
        print("[wait] no live pod and no rows: giving up", flush=True)
        break
    if time.time() - t0 > a.timeout:
      print("[wait] timeout", flush=True)
      break
    time.sleep(60)
  print("\n=== state ===", flush=True)
  for jid, (st, r) in got.items():
    print(f"\n== {jid}  status={st}  rows={len(r)}")
    if r:
      k = r[-1]
      print("   " + json.dumps({x: k.get(x) for x in
                                ("iter", "offset_now", "r_bar", "r_bar_gap", "action_std", "fall_rate_per_1k",
                                 "resets", "fps", "r_task_rate")}, default=float))


if __name__ == "__main__":
  main()
