"""Campaign status from R2 alone (pod logs are unreliable; DONE markers and JSONL are the source of truth).

    python analysis/status.py --campaign b1-4d73b6d1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from mjlab_centering import r2


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--campaign", required=True)
  ap.add_argument("--tail", type=int, default=12)
  a = ap.parse_args()

  man = r2.get_json(f"{a.campaign}/manifest.json") or {"jobs": []}
  jobs = man["jobs"]
  done = r2.done_set(a.campaign)
  pending = [j for j in jobs if j["job_id"] not in done]
  print(f"campaign {a.campaign}: {len(done)}/{len(jobs)} done, {len(pending)} pending")

  rows = []
  for k in r2.list_keys(f"{a.campaign}/runs/"):
    if not k.endswith("/iter.jsonl"):
      continue
    job = k.split("/runs/")[1].rsplit("/iter.jsonl", 1)[0]
    try:
      body = r2.client().get_object(Bucket=r2.bucket(), Key=k)["Body"].read().decode()
      lines = [ln for ln in body.splitlines() if ln.strip()]
      if not lines:
        continue
      last = json.loads(lines[-1])
      rows.append({
        "job": job,
        "iters": len(lines),
        "fps": last.get("fps", 0),
        "J_task": last.get("r_task_rate", 0),
        "falls": last.get("fall_rate_per_1k", 0),
        "done": job in done,
      })
    except Exception as e:
      print(f"  [warn] {job}: {e!r}")
  rows.sort(key=lambda r: (r["done"], r["job"]))
  for r in rows[: a.tail]:
    print(f"  {'DONE ' if r['done'] else 'run  '} {r['job'][:52]:52} iters={r['iters']:4d} "
          f"fps={r['fps']:7.0f} J={r['J_task']:+.4f} falls/1k={r['falls']:.2f}")
  if rows:
    fps = [r["fps"] for r in rows if r["fps"]]
    print(f"  median fps across running jobs: {sorted(fps)[len(fps)//2]:.0f}" if fps else "")

  claims = [k for k in r2.list_keys(f"{a.campaign}/claims/")]
  print(f"  live claims: {len(claims)}")
  for k in claims[:5]:
    c = r2.get_json(k.replace(f"{r2.PREFIX}/", "", 1)) or {}
    ts = c.get("ts", 0)
    age = (datetime.now(timezone.utc).timestamp() - ts) / 60 if ts else -1
    print(f"    {k.rsplit('/', 1)[-1]} pod={c.get('pod')} age={age:.0f}min")


if __name__ == "__main__":
  main()
