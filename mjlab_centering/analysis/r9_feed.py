"""Keep the round-9a warm pods fed.

The pod bootstrap runs `podrun.py`, which makes ONE pass over the manifest it downloaded at boot: a job another pod has
already claimed at the moment it looks is skipped and never retried, so a pod that loses those races ends its pass early
and idles (measured 2026-11-09: 3 of 4 pods idle at $0.22/h each with 40 jobs pending). Warm pods poll
`<campaign>/queue.json`; this feeder republishes the *pending* list every few minutes so an idle pod immediately has
work. Claims are atomic, so republishing cannot make two pods run one job.

    python analysis/r9_feed.py [--campaign r9pilot-3ae3cc84] [--every 240] [--hours 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mjlab_centering import r2  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", default="r9pilot-3ae3cc84")
  ap.add_argument("--every", type=float, default=240.0)
  ap.add_argument("--hours", type=float, default=8.0)
  a = ap.parse_args()
  man = json.loads((ROOT / "manifests" / f"{a.campaign}.json").read_text())
  end = time.time() + a.hours * 3600
  while time.time() < end:
    done = r2.done_set(a.campaign)
    claimed = {k.split("/")[-1].replace(".json", "") for k in r2.list_keys(f"{a.campaign}/claims/")}
    pend = [j for j in man["jobs"] if j["job_id"] not in done and j["job_id"] not in claimed]
    if not pend:
      print(f"[feed] {time.strftime('%H:%M:%S')} queue empty ({len(done)} done); stopping", flush=True)
      return
    r2.put_json(f"{a.campaign}/queue.json", {"campaign": a.campaign, "jobs": pend})
    print(f"[feed] {time.strftime('%H:%M:%S')} {len(done)}/{len(man['jobs'])} done, "
          f"{len(claimed)} claimed, {len(pend)} queued", flush=True)
    time.sleep(a.every)


if __name__ == "__main__":
  main()
