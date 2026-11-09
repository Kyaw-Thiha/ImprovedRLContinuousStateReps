"""Publish the next queue to a warm pod, or tell it to shut down (d165).

A captured community pod no longer deletes itself when its queue drains: it polls `<campaign>/queue.json` and runs
whatever manifest appears there, on the same warm installation (torch, mjlab, warp kernels already paid for). This script
is the only way to feed it, so the next stage starts when — and only when — the gate says it may.

    python analysis/r7_publish_queue.py --campaign r7pilot-63bd51f3 --manifest manifests/r7vnsc-a565805a.json
    python analysis/r7_publish_queue.py --campaign r7pilot-63bd51f3 --status
    python analysis/r7_publish_queue.py --campaign r7pilot-63bd51f3 --terminate     # pod deletes itself

The pod also self-deletes after KEEP_MAX_H hours (default 3), so a forgotten warm pod cannot spend indefinitely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mjlab_centering import r2  # noqa: E402


def payload(manifest: dict, note: str = "") -> dict:
  """The queue object: the manifest plus when and why it was published."""
  jobs = manifest.get("jobs") or []
  assert jobs, "refusing to publish a queue with no jobs"
  return {"campaign": manifest.get("campaign"), "published": time.time(), "note": note, "jobs": jobs}


def publish(campaign: str, manifest_path: Path, note: str = "") -> dict:
  man = json.loads(Path(manifest_path).read_text())
  p = payload(man, note)
  r2.put_json(f"{campaign}/queue.json", p)
  # also publish it as the *target campaign's* manifest: the runs are written under that campaign (podrun --r2 auto), and
  # the readouts discover job ids from <campaign>/manifest.json (no orchestrator ever shipped that campaign)
  target = man.get("campaign") or campaign
  r2.put_json(f"{target}/manifest.json", {"campaign": target, "jobs": man["jobs"], "published": p["published"],
                                          "note": f"published by r7_publish_queue from {campaign}; {note}"})
  return p


def status(campaign: str) -> str:
  q = r2.get_json(f"{campaign}/queue.json")
  if q is None:
    return "no queue published"
  return f"queue published {time.strftime('%H:%M:%SZ', time.gmtime(q['published']))}: {len(q['jobs'])} jobs ({q.get('note','')})"


def terminate(campaign: str) -> None:
  r2.put_json(f"{campaign}/TERMINATE", {"at": time.time()})


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--manifest", default=None)
  ap.add_argument("--note", default="")
  ap.add_argument("--status", action="store_true")
  ap.add_argument("--terminate", action="store_true")
  a = ap.parse_args()
  if a.status:
    print(status(a.campaign))
    return
  if a.terminate:
    terminate(a.campaign)
    print("TERMINATE marker written; the pod deletes itself on its next poll (<=60 s)")
    return
  if not a.manifest:
    raise SystemExit("give --manifest, --status or --terminate")
  p = publish(a.campaign, Path(a.manifest), a.note)
  print(json.dumps({"campaign": a.campaign, "jobs": len(p["jobs"]), "note": p["note"]}, indent=1))


if __name__ == "__main__":
  main()
