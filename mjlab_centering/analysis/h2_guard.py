"""Cost guard for the H2 CPU pods (d132). Run detached; it never launches anything, it only deletes.

    python analysis/h2_guard.py --mark-name h2_start --cap 3.0 --max-hours 12

Every `--poll` seconds it
  * lists pods whose name contains `h2` and deletes any older than `--max-hours` (a stuck search must not bill forever),
  * reads the billing API and deletes every H2 pod if the CPU spend since the mark exceeds `--cap` (non-NVIDIA rows, so
    GPU work running at the same time is not counted against the H2 cap),
  * exits when no H2 pod is left.
Billing has one-hour buckets, so the cap is checked with that resolution; the wall-clock rule is the tighter guard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
from mjlab_centering import pods  # noqa: E402
import spend  # noqa: E402


def cpu_spend_since(since_iso: str) -> float:
  rows = spend._billing("pods", since_iso)
  return round(sum(float(x["amount"]) for x in rows if "NVIDIA" not in str(x.get("gpuId") or "")), 4)


def h2_pods() -> list[dict]:
  return [p for p in pods.list_pods() if "h2" in str(p.get("name", "")).lower()]


STATE = ROOT / "results" / "h2_guard_state.json"


def _accumulate(ps: list[dict]) -> tuple[float, dict]:
  """Lifetime dollars per pod, integrated over polls (a pod's uptime resets when its container restarts)."""
  st = json.loads(STATE.read_text()) if STATE.exists() else {}
  for p in ps:
    pid = str(p.get("id"))
    up = float((p.get("runtime") or {}).get("uptimeInSeconds") or p.get("uptimeSeconds") or 0.0)
    rate = float(p.get("costPerHr") or 0.0)
    rec = st.setdefault(pid, {"rate": rate, "last_uptime": 0.0, "usd": 0.0})
    d = up - rec["last_uptime"]
    if d < 0:  # container restarted: count what we had, then continue from the new uptime
      d = 0.0
    rec["usd"] += d * rate / 3600.0
    rec["last_uptime"] = up
    rec["rate"] = rate
  STATE.parent.mkdir(parents=True, exist_ok=True)
  STATE.write_text(json.dumps(st, indent=1))
  return round(sum(v["usd"] for v in st.values()), 4), st


def h2_estimate(ps: list[dict]) -> float:
  """Dollars spent by these pods so far, from each pod's own rate x uptime.

  The billing API labels CPU pods with the account's GPU name (observed 2026-11-04: three `cpu3c` pods reported as
  "NVIDIA GeForce RTX 5090"), so a gpuId filter cannot separate CPU from GPU spend. Rate x uptime can, and it is what
  the cap needs.
  """
  return round(sum(float(p.get("costPerHr") or 0.0) * float((p.get("runtime") or {}).get("uptimeInSeconds")
                or p.get("uptimeSeconds") or 0.0) / 3600.0 for p in ps), 4)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mark-name", default="h2_start")
  ap.add_argument("--cap", type=float, default=3.0)
  ap.add_argument("--max-hours", type=float, default=12.0)
  ap.add_argument("--poll", type=int, default=600)
  a = ap.parse_args()
  m = spend.marks().get(a.mark_name)
  if not m:
    sys.exit(f"no mark {a.mark_name!r}: run `python analysis/spend.py --mark {a.mark_name}` right before launching")
  since = m["iso"].replace("+00:00", "Z")
  seen_any = False
  while True:
    ps = h2_pods()
    seen_any |= bool(ps)
    now = time.time()
    try:
      s = cpu_spend_since(since)
    except Exception as e:  # billing hiccup: keep the wall-clock rule alive
      s = float("nan")
      print(f"[guard] billing read failed: {e}", flush=True)
    est, _st = _accumulate(ps)
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "h2_pods": len(ps), "h2_estimate_usd": est,
                      "billing_since_mark_all_pods": s}), flush=True)
    over_cap = est > a.cap
    for p in ps:
      ts = pods.pod_created_ts(p)
      old = ts is not None and (now - ts) / 3600.0 > a.max_hours
      if over_cap or old:
        why = f"spend {s} > cap {a.cap}" if over_cap else f"older than {a.max_hours} h"
        print(f"[guard] deleting {p.get('id')} ({p.get('name')}): {why}", flush=True)
        try:
          pods.delete_pod(p["id"])
        except Exception as e:
          print(f"[guard] delete failed: {e}", flush=True)
    if seen_any and not ps:
      print("[guard] no H2 pods left; exiting", flush=True)
      return
    if not seen_any and now - m["at"] > 3 * 3600:
      print("[guard] no H2 pod appeared within 3 h of the mark; exiting", flush=True)
      return
    time.sleep(a.poll)


if __name__ == "__main__":
  main()
