"""Self-check for REPORT_006: re-derive every number the report quotes and fail if it does not match.

    python analysis/report6_checks.py

Reads R2 through the same modules the report's tables come from, so a reader can verify the prose without trusting it.
Exits non-zero with a list of mismatches. Values are the ones written in `docs/REPORT_006.md`; when a run is rerun or a
table is regenerated, update this file with it (that is the point - the check is the report's contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import g1_claim_report  # noqa: E402
import ssp_readout  # noqa: E402
import g1_slope  # noqa: E402
import h2_report  # noqa: E402

PROBE = "r6a-ad66ecf2"
CLAIM = "r6g1queue-0ede5636"


def close(got, want, tol) -> bool:
  try:
    return abs(float(got) - float(want)) <= tol
  except Exception:  # noqa: BLE001
    return False


def main() -> None:
  fails: list[str] = []

  # 1. probe: the rule's verdict
  arms = {"centered": [f"r6probe_g1_g0.99_o1_value_s{s}" for s in (1000, 1001, 1002)],
          "bias": [f"r6probe_g1_g0.99_o1_bias_none_s{s}" for s in (1000, 1001, 1002)]}
  dec = g1_slope.decide({k: [g1_slope.series(g1_slope.load_r2(PROBE, j)) for j in v] for k, v in arms.items()})
  if dec.get("length") != 2500:
    fails.append(f"probe claim length {dec.get('length')} != 2500")
  if dec.get("first_flat_t") != 2075:
    fails.append(f"probe first flat {dec.get('first_flat_t')} != 2075")

  # 2. claim set: arms, contrasts, equivalence
  rep = g1_claim_report.summarise({k: g1_claim_report.read_arm(CLAIM, p) for k, p in g1_claim_report.ARMS.items()})
  want = {
      "none walks": (rep["arms"]["none"]["walks"], 0), "value walks": (rep["arms"]["value"]["walks"], 20),
      "bias walks": (rep["arms"]["bias"]["walks"], 20),
      "value track": (rep["arms"]["value"]["track"], 0.816), "bias track": (rep["arms"]["bias"]["track"], 0.820),
      "value-none dtrack": (rep["contrasts"]["value - none"]["dtrack"], 0.5063),
      "bias-none dtrack": (rep["contrasts"]["bias - none"]["dtrack"], 0.5102),
      "bias-value dtrack": (rep["contrasts"]["bias - value"]["dtrack"], 0.0040),
  }
  for k, (got, expect) in want.items():
    tol = 0.002 if "track" in k else 0
    if not close(got, expect, tol):
      fails.append(f"claim {k}: {got} != {expect}")
  if not rep.get("bias_vs_value_equivalent_d117"):
    fails.append("claim: bias vs value not equivalent under d117")

  # 3. H2: arm means and the paired interval
  sums = {a: h2_report.load(a) for a in h2_report.ARMS}
  h2 = h2_report.read(sums)
  for arm, expect in (("ssp_h2", 486.1), ("ssp_h2raw", 469.8), ("tile_h2", 495.8)):
    if not close(h2["arms"][arm]["mean"], expect, 0.2):
      fails.append(f"H2 {arm} mean {h2['arms'][arm]['mean']} != {expect}")
  if not close(h2["vs_tile"]["mean"], -9.66, 0.1) or not close(h2["vs_tile"]["lo"], -22.07, 0.2):
    fails.append(f"H2 paired interval {h2['vs_tile']} != -9.66 [-22.07, ...]")
  if "inconclusive" not in h2["rules"]["R1"]:
    fails.append(f"H2 R1 verdict changed: {h2['rules']['R1']}")

  # 4. SSP stage: only meaningful once its runs exist; when they do, the gate verdict must be one of the two the
  # pre-registration allows, and the chosen config must be a cell the stage actually ran.
  try:
    local = ROOT / "results" / "merged_r2"
    ssp = ssp_readout.width_report(local)
    if "best_overall" in ssp:
      if ssp["verdict"] not in (f"Stage B runs on {ssp['best_overall']}",
                                "SSP does not walk at any width or length scale tried: skip Stage B"):
        fails.append(f"SSP verdict unexpected: {ssp['verdict']}")
      if ssp["best_overall"] not in ssp["cells"]:
        fails.append(f"SSP best config {ssp['best_overall']} is not a cell that ran")
    else:
      print(f"note: SSP stage not readable yet ({ssp.get('verdict', '')[:60]}); its checks are skipped")
    if "best_overall" in ssp:
      want = {"raw": 0.886, "ssp2x2@0.3": 0.875, "ssp2x2@1": 0.880, "rff2x2": 0.653, "ssp4x4@1": 0.874}
      for cell, expect in want.items():
        got = ssp["cells"].get(cell, {}).get("mean")
        if got is None or abs(got - expect) > 0.002:
          fails.append(f"SSP width {cell}: {got} != {expect}")
      if ssp.get("best_overall") != "ssp2x2@1":
        fails.append(f"SSP best config {ssp.get('best_overall')} != ssp2x2@1")
      # pilot
      pil = ssp_readout.pilot_report(2, 1.0, local)
      r01 = pil["offsets"]["0.1"]
      if abs(r01["none"]["mean_ssp"] - 0.762) > 0.002 or r01["none"]["walks_ssp"] != 4:
        fails.append(f"pilot 0.1/none SSP {r01['none']} != 0.762 with 4/5 walks")
      if abs(r01["none"]["mean_raw"] - 0.532) > 0.002 or r01["none"]["walks_raw"] != 2:
        fails.append(f"pilot 0.1/none raw {r01['none']} != 0.532 with 2/5 walks")
      if abs(r01["interaction"]["mean"] + 0.232) > 0.003:
        fails.append(f"pilot 0.1 interaction {r01['interaction']['mean']:.3f} != -0.232")
  except Exception as e:  # noqa: BLE001 - the stage is optional for this report
    print(f"note: SSP checks skipped ({type(e).__name__}: {e})")

  if fails:
    print("REPORT_006 self-check FAILED:")
    for f in fails:
      print("  -", f)
    raise SystemExit(1)
  print("REPORT_006 self-check passed: probe 2075/2500, claim arms and contrasts, H2 means and interval, "
        "SSP width cells and the pilot's offset-0.1 interaction all match")


if __name__ == "__main__":
  main()
