"""Close the round-9a fleet when the queue drains (or when the cap is reached).

    python analysis/r9_closer.py [--campaign r9pilot-3ae3cc84] [--cap 15]

A warm pod polls `<campaign>/TERMINATE` every 60 s and deletes itself when it finds it; without this marker a drained
pod idles for KEEP_MAX_H (3 h) at $0.22/h. This waits for the queue to be empty (no pending job, no live claim, no
running pod with work) and then publishes the marker, so an unattended round does not spend its last dollars on idle
GPUs. It also publishes the marker if the measured spend reaches the cap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mjlab_centering import pods, r2  # noqa: E402


def spend_of(campaign: str) -> float:
  st = ROOT / "results" / f"orch_{campaign}.json"
  return float(json.loads(st.read_text()).get("spend", 0.0)) if st.exists() else 0.0


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  ap.add_argument("--cap", type=float, default=15.0)
  a = ap.parse_args()
  man = json.loads((ROOT / "manifests" / f"{a.campaign}.json").read_text())
  while True:
    done = r2.done_set(a.campaign)
    claims = r2.list_keys(f"{a.campaign}/claims/")
    live = [p for p in pods.list_pods() if str(p.get("name", "")).startswith(f"mjc-{a.campaign}")]
    sp = spend_of(a.campaign)
    print(f"[closer] {time.strftime('%H:%M:%S')} done={len(done)}/{len(man['jobs'])} claims={len(claims)} "
          f"pods={len(live)} spend=${sp:.2f}", flush=True)
    if len(done) >= len(man["jobs"]) or sp >= a.cap:
      if live and sp < a.cap:
        # let the last runs finish and their artefacts reach R2 before the pods delete themselves
        if claims:
          time.sleep(120)
          continue
      r2.put_json(f"{a.campaign}/TERMINATE", {"when": time.time(), "done": len(done), "spend": sp})
      print(f"[closer] TERMINATE published ({len(done)}/{len(man['jobs'])} done, ${sp:.2f})", flush=True)
      if live and sp >= a.cap:
        for p in live:
          try:
            pods.delete_pod(p["id"])
            print(f"[closer] deleted {p['id']} (cap reached)", flush=True)
          except Exception as e:  # noqa: BLE001
            print(f"[closer] delete failed: {e!r}", flush=True)
      return
    time.sleep(300)


if __name__ == "__main__":
  main()
