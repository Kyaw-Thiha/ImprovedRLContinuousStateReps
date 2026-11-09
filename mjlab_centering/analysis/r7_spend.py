"""Round-7 spend, attributed to this round's own pods (d155).

The account-level billing delta cannot be used as the round's spend: the SSP width campaign is running on the same
account and its pods spend money at the same time. The billing API does expose `podId` (--grouping podId), and every pod
this round launches is recorded in `results/orch_r7<stage>-<id>.json`, so the round's spend is the sum of the billing
rows of those pods. Read after every stage, as pre-registered.

    python analysis/r7_spend.py                    # this round's spend so far (all r7 campaigns)
    python analysis/r7_spend.py --since 2026-11-06T12:00:00Z
    python analysis/r7_spend.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spend  # noqa: E402


def _billing_pods(since: str) -> list[dict]:
  r = subprocess.run(["runpodctl", "billing", "pods", "--start-time", since, "--bucket-size", "hour",
                      "--grouping", "podId"],
                     capture_output=True, text=True, env={**spend.ENV, "RUNPOD_API_KEY": spend._key()}, timeout=120)
  if r.returncode != 0:
    raise RuntimeError(r.stderr.strip()[:400])
  out = json.loads(r.stdout or "[]")
  return out if isinstance(out, list) else []


def round7_pods() -> dict[str, dict]:
  """pod id -> {campaign, price, gpu} for every pod a round-7 orchestrator launched."""
  pods: dict[str, dict] = {}
  for p in sorted((ROOT / "results").glob("orch_r7*.json")):
    st = json.loads(p.read_text())
    camp = p.stem[len("orch_"):]
    for pid, rec in (st.get("pods") or {}).items():
      pods[pid] = {"campaign": camp, "price": rec.get("price"), "gpu": rec.get("gpu")}
  return pods


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--since", default="2026-11-06T00:00:00Z")
  ap.add_argument("--json", action="store_true")
  a = ap.parse_args()
  pods = round7_pods()
  rows = _billing_pods(a.since)
  by_pod: dict[str, float] = {}
  by_camp: dict[str, float] = {}
  for r in rows:
    pid = r.get("podId")
    if pid not in pods:
      continue
    by_pod[pid] = by_pod.get(pid, 0.0) + float(r["amount"])
    by_camp[pods[pid]["campaign"]] = by_camp.get(pods[pid]["campaign"], 0.0) + float(r["amount"])
  # this brief's campaigns are r7pilot/r7vnsc/r7rs; anything else (the d151/d152 video runs) is reported but not charged
  ours = {k: v for k, v in by_camp.items() if k.split("-")[0] in ("r7pilot", "r7vnsc", "r7rs")}
  other = {k: v for k, v in by_camp.items() if k not in ours}
  out = {"since": a.since, "round7_total": round(sum(ours.values()), 4),
         "by_campaign": {k: round(v, 4) for k, v in sorted(ours.items())},
         "other_r7_campaigns_not_charged": {k: round(v, 4) for k, v in sorted(other.items())},
         "by_pod": {k: round(v, 4) for k, v in sorted(by_pod.items(), key=lambda x: -x[1])},
         "account_total_since": spend.total(a.since)["total"],
         "note": "account_total_since also contains the running SSP campaign and every other pod on the account"}
  if a.json:
    print(json.dumps(out, indent=1))
    return
  print(f"this brief's stages (own pods only), since {a.since}: **${out['round7_total']:.2f}**")
  if other:
    print(f"  other r7-named campaigns, not charged to this cap: {out['other_r7_campaigns_not_charged']}")
  for camp, v in out["by_campaign"].items():
    print(f"  {camp}: ${v:.2f}")
  print(f"  account total over the same window (all campaigns): ${out['account_total_since']:.2f}")
  for pid, v in list(out["by_pod"].items())[:12]:
    print(f"    {pid} {pods[pid]['gpu']} ${pods[pid]['price']}/h  ${v:.2f}")


if __name__ == "__main__":
  main()
