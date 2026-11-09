"""Delete a campaign's pods once its queue has drained (or after a deadline).

    python analysis/cleanup_campaign.py --campaign pilot1-9f3a1c [--deadline-min 45]

Workers are supposed to self-delete when drained, but that has failed before (16 min of idle billing on 2026-11-01),
so this is the belt-and-braces cleanup. It only ever deletes pods whose name starts with `mjc-<campaign>`.
"""

from __future__ import annotations

import argparse
import time

from mjlab_centering import pods, r2


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--deadline-min", type=float, default=60.0)
  ap.add_argument("--poll", type=float, default=60.0)
  a = ap.parse_args()
  prefix = f"mjc-{a.campaign}"
  t0 = time.time()
  while True:
    man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
    pending = [j["job_id"] for j in man["jobs"] if j["job_id"] not in r2.done_set(a.campaign)]
    live = [p for p in pods.list_pods() if str(p.get("name", "")).startswith(prefix)
            and p.get("runtimeStatus") == "running"]
    print(f"[cleanup] {time.strftime('%H:%M:%S')} pending={len(pending)} live={len(live)}", flush=True)
    if not live:
      print("[cleanup] nothing left to delete", flush=True)
      return
    if not pending or time.time() - t0 > a.deadline_min * 60:
      for p in live:
        try:
          pods.delete_pod(p["id"])
          print(f"[cleanup] deleted {p['name']} ({p['id']})", flush=True)
        except Exception as e:  # keep going: report, do not crash
          print(f"[cleanup] delete failed for {p.get('id')}: {e!r}", flush=True)
      return
    time.sleep(a.poll)


if __name__ == "__main__":
  main()
