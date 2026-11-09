"""Delete round-7 pods whose container never started (stalled image pull).

    python analysis/r7_reap_stalled.py            # report + delete pods with uptime 0 for more than --minutes
    python analysis/r7_reap_stalled.py --dry-run

A pod whose container has not started still bills its GPU hour, has no in-container watchdog (the watchdog lives in the
bootstrap, which never runs), and cannot be caught by the orchestrator: it is *present* in the pod list, so it never looks
"absent". Two were paid for like this on 2026-11-06 (d175). Run every 10 minutes by r7reaper.timer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mjlab_centering import pods, r2  # noqa: E402

PREFIX = "mjc-r7"


def _get(pod_id: str) -> dict:
  env = {"PATH": "/usr/bin:/bin:" + str(Path.home() / ".local/bin"), "HOME": str(Path.home()),
         "RUNPOD_API_KEY": pods._key()}
  out = subprocess.run(["runpodctl", "pod", "get", pod_id], capture_output=True, text=True, env=env, timeout=90)
  try:
    return json.loads(out.stdout)
  except Exception:
    return {}


def _created_ts(d: dict) -> float:
  from datetime import datetime
  try:
    return datetime.strptime(d["createdAt"][:19], "%Y-%m-%d %H:%M:%S").replace(
      tzinfo=__import__("datetime").timezone.utc).timestamp()
  except Exception:
    return time.time()


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--minutes", type=float, default=25.0)
  ap.add_argument("--silent-minutes", type=float, default=30.0,
                  help="delete a pod that has pushed no boot log after this long (stuck in its first install)")
  ap.add_argument("--dry-run", action="store_true")
  a = ap.parse_args()
  now = time.time()
  out = []
  for p in pods.list_pods():
    name = str(p.get("name", ""))
    if not name.startswith(PREFIX):
      continue
    d = _get(p["id"])
    up, created = d.get("uptimeSeconds"), _created_ts(d)
    age_min = (now - created) / 60.0
    rec = {"id": p["id"], "name": name, "uptime_s": up, "age_min": round(age_min, 1)}
    # a container that is up but has never pushed a boot log is stuck in its first installs: the bootstrap pushes one
    # within about a minute of starting, so silence for half an hour means nothing has happened.
    silent = False
    if age_min > a.silent_minutes:
      # the campaign a pod belongs to is encoded in its name: mjc-<campaign>-<n>. Looking only under the pilot's campaign
      # made every Stage-2 pod look silent and the reaper deleted three of them mid-run (d185).
      parts = name.split("-")
      camp = "-".join(parts[1:3]) if len(parts) >= 3 else None
      seen = False
      for c in ([camp] if camp else []) + ["r7pilot-63bd51f3", "r7vnsc-a565805a"]:
        try:
          r2.client().head_object(Bucket=r2.bucket(), Key=r2.key(f"{c}/logs/boot_{p['id']}.log"))
          seen = True
          break
        except Exception:
          continue
      silent = not seen
    rec["no_boot_log"] = silent
    if (up == 0 and age_min > a.minutes) or silent:
      rec["action"] = "would delete" if a.dry_run else "deleted"
      if not a.dry_run:
        try:
          pods.delete_pod(p["id"])
          for camp in ("r7pilot-63bd51f3",):
            try:
              r2.release_claims_of(camp, p["id"])
            except Exception:
              pass
        except Exception as e:  # noqa: BLE001
          rec["action"] = f"delete failed: {e}"
    out.append(rec)
  # claims held by pods that no longer exist block their jobs for the whole 4 h TTL; release them here (d204)
  live = {x["id"] for x in pods.list_pods()}
  freed = 0
  for camp in ("r7vnsc-a565805a", "r7pilot-63bd51f3"):
    pref = r2.key(f"{camp}/claims/")
    for o in r2.client().list_objects_v2(Bucket=r2.bucket(), Prefix=pref).get("Contents", []):
      try:
        body = json.loads(r2.client().get_object(Bucket=r2.bucket(), Key=o["Key"])["Body"].read())
      except Exception:
        continue
      owner = body.get("owner") or body.get("pod")
      if owner and owner not in live:
        if not a.dry_run:
          r2.client().delete_object(Bucket=r2.bucket(), Key=o["Key"])
        freed += 1
  print(json.dumps({"checked": len(out), "pods": out, "stale_claims_released": freed}, indent=1))


if __name__ == "__main__":
  main()
