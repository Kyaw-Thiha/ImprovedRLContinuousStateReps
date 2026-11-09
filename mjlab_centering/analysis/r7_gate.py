"""The round-7 budget gate, as arithmetic rather than memory (d155).

    python analysis/r7_gate.py --pilot-cost 1.9 --pilot-iters 24000 \\
        --done-runs 12 --spent 2.1 \\
        --stage "vnsc:160000" --stage "rs:80000"

Prints, for each stage, the projection from the *measured* dollars per 1000 iterations, and the pre-registered gate:
a stage launches only if `spent + 1.3 x projected <= 18`. Also prints the cut order (Stage 3 first, then nothing else:
Stage 2 is the claim). Pure arithmetic on numbers the caller has read from `analysis/r7_spend.py` and the pilot's
finished runs — it never touches R2 or a pod.
"""

from __future__ import annotations

import argparse
import json

CAP = 18.0
CONTINGENCY = 1.3


def rate_per_1000it(pilot_cost: float, pilot_iters: int) -> float:
  if pilot_iters <= 0:
    raise SystemExit("--pilot-iters must be > 0")
  return pilot_cost / (pilot_iters / 1000.0)


def gate(spent: float, projected: float, cap: float = CAP, contingency: float = CONTINGENCY) -> dict:
  need = spent + contingency * projected
  return {"projected": round(projected, 3), "gate_need": round(need, 3), "cap": cap, "pass": need <= cap}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--spent", type=float, required=True, help="this brief's measured spend so far (r7_spend.py)")
  ap.add_argument("--pilot-cost", type=float, default=None, help="measured cost of the pilot (billing)")
  ap.add_argument("--pilot-iters", type=int, default=24000)
  ap.add_argument("--diagnostic-cost", type=float, default=0.0,
                  help="cost of any part of the pilot that is not charged to the arms (e.g. lost pods); informational")
  ap.add_argument("--stage", action="append", default=[], help="name:iterations, e.g. vnsc:160000")
  ap.add_argument("--json", action="store_true")
  a = ap.parse_args()
  out = {"spent": a.spent, "cap": CAP, "stages": []}
  if a.pilot_cost is not None:
    out["usd_per_1000it"] = round(rate_per_1000it(a.pilot_cost, a.pilot_iters), 4)
  if "usd_per_1000it" not in out:
    raise SystemExit("give --pilot-cost (and --pilot-iters) so the projection is measured, not guessed")
  cum = a.spent
  for spec in a.stage:
    name, iters = spec.split(":")
    proj = out["usd_per_1000it"] * int(iters) / 1000.0
    g = gate(cum, proj)
    out["stages"].append({"stage": name, "iters": int(iters), **g})
    if g["pass"]:
      cum += proj
  out["spent_if_all_stages_run"] = round(cum, 3)
  out["cut_order"] = "Stage 3 (RS) is cut first and is the only cuttable stage; Stage 2 is the claim"
  if a.json:
    print(json.dumps(out, indent=1))
    return
  print(f"measured ${out['usd_per_1000it']:.4f} per 1000 iterations; spent so far ${a.spent:.2f} of ${CAP:.2f}")
  for s in out["stages"]:
    v = "PASS" if s["pass"] else "BLOCKED"
    print(f"  {s['stage']:6s} {s['iters']:>7d} it  projected ${s['projected']:.2f}  "
          f"gate ${s['gate_need']:.2f} <= ${s['cap']:.2f}  {v}")
  print(f"cumulative if every passing stage runs: ${out['spent_if_all_stages_run']:.2f}")
  print(out["cut_order"])


if __name__ == "__main__":
  main()
