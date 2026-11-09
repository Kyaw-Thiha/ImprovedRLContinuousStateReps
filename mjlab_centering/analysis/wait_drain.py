"""Exit when a campaign has no real pending jobs (or after the timeout), printing what is left.

    python analysis/wait_drain.py --campaign night1-3e71b4 --timeout 3300

Kept inside the repo rather than /tmp: the workstation cleared /tmp mid-run once already (decision 91 notes the same
class of loss for the orchestrator log).
"""

from __future__ import annotations

import argparse
import time

from mjlab_centering import r2


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--timeout", type=float, default=3300)
  ap.add_argument("--poll", type=float, default=180)
  a = ap.parse_args()
  t0 = time.time()
  while True:
    man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
    done = r2.done_set(a.campaign)
    pending = [j["job_id"] for j in man["jobs"] if j["job_id"] not in done]
    if not pending:
      print("DRAINED")
      return
    if time.time() - t0 > a.timeout:
      print(f"timeout; still pending ({len(pending)}): {pending[:6]}")
      return
    print(f"[drain] {time.strftime('%H:%M:%S')} pending={len(pending)} first={pending[0]}", flush=True)
    time.sleep(a.poll)


if __name__ == "__main__":
  main()
