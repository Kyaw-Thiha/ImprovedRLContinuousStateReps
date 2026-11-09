"""Authoritative spend from the RunPod billing API (not the orchestrator's estimate, which priced 4090s at $0.34/h).

    python analysis/spend.py                 # today so far, plus the running campaign totals
    python analysis/spend.py --since 2026-11-03T10:00:00Z
    python analysis/spend.py --mark step1    # record a checkpoint of the cumulative total
    python analysis/spend.py --since-mark step1   # spend since that checkpoint (the step's cost)

Cumulative totals are kept in results/spend_marks.json. Hour buckets, so a delta has +-1 hour of resolution; the
step functions in this project run for at least half an hour, which is the granularity this is meant for.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKS = ROOT / "results" / "spend_marks.json"
ENV = {**os.environ, "RUNPOD_API_KEY": "", "PATH": "/usr/bin:/bin:/home/kevin/.local/bin", "HOME": "/home/kevin"}


def _key() -> str:
  for line in (ROOT.parent / ".env").read_text().splitlines():
    if line.startswith("RUNPOD_API_KEY="):
      return line.split("=", 1)[1].strip()
  raise RuntimeError("RUNPOD_API_KEY not in .env")


def _billing(sub: str, since: str) -> list[dict]:
  r = subprocess.run(["runpodctl", "billing", sub, "--start-time", since, "--bucket-size", "hour"],
                     capture_output=True, text=True, env={**ENV, "RUNPOD_API_KEY": _key()}, timeout=120)
  if r.returncode != 0:
    raise RuntimeError(r.stderr.strip()[:400])
  out = json.loads(r.stdout or "[]")
  return out if isinstance(out, list) else []


def total(since: str) -> dict:
  rows = _billing("pods", since) + _billing("network-volume", since)
  by_gpu: dict[str, float] = {}
  for x in rows:
    by_gpu[x.get("gpuId") or "network-volume"] = by_gpu.get(x.get("gpuId") or "network-volume", 0.0) + float(x["amount"])
  return {"since": since, "total": round(sum(by_gpu.values()), 4), "by_gpu": {k: round(v, 4) for k, v in by_gpu.items()}}


def marks() -> dict:
  return json.loads(MARKS.read_text()) if MARKS.exists() else {}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--since", default=None)
  ap.add_argument("--mark", default=None)
  ap.add_argument("--since-mark", default=None)
  a = ap.parse_args()
  m = marks()
  if a.mark:
    since = m.get("_since") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
    t = total(since)
    m[a.mark] = {"at": time.time(), "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"), "total": t["total"]}
    m["_since"] = since
    MARKS.write_text(json.dumps(m, indent=1))
    print(json.dumps({a.mark: m[a.mark], "by_gpu": t["by_gpu"]}, indent=1))
    return
  if a.since_mark:
    base = m.get(a.since_mark)
    if not base:
      raise SystemExit(f"no mark {a.since_mark!r}; have {sorted(k for k in m if not k.startswith('_'))}")
    now = total(m["_since"])["total"]
    print(json.dumps({"mark": a.since_mark, "at_total": base["total"], "now_total": now,
                      "delta": round(now - base["total"], 4)}, indent=1))
    return
  since = a.since or datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
  print(json.dumps(total(since), indent=1))


if __name__ == "__main__":
  main()
