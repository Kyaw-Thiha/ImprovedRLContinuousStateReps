"""Checks for the rewritten decision rules (handoff 09 step H, docs/11 section 4.5).

    OMP_NUM_THREADS=1 ../../.venv/bin/python test_analysis.py

No real sweep data is used: each scenario is a synthetic set of hold-out rows, so the *rule* is tested rather than
the numbers it happens to be applied to.  The four scenarios are the ones the rule must get right:

  1. equal convergence, 3x faster                      -> thesis_pass_candidate
  2. a win only at the hand-wired rung 0               -> structural_prior_only
  3. equal everywhere                                  -> algebra_buys_nothing
  4. a learned rung far below the baseline             -> structure_not_discoverable
  5. a learned win matched by the fwp control          -> structural_prior_only (fragile)
"""

import numpy as np

import analysis as AN

SEEDS = 5
JITTER = 0.0005          # a tiny seed-to-seed spread, so the pooled se is small but non-zero


def rows_for(variant, spec, seeds=SEEDS):
    """Synthetic hold-out rows.  ``spec``: arm -> dict(id, extrap_T, t50, t90, state, rung, pool)."""
    rows = []
    for arm, d in spec.items():
        for s in range(seeds):
            j = JITTER * ((s % 3) - 1)
            summary = {
                "conv_id": d["id"] + j,
                "conv_train": d["id"] + j,
                "conv_extrap_T": d.get("extrap_T", d["id"]) + j,
                "t50_id": d.get("t50"), "t90_id": d.get("t90"),
                "aulc_id": d["id"] + j,
                "last5_id": d["id"] + j,
            }
            cfg = {"rung": d["rung"]} if d.get("rung") else {}
            rows.append({"variant": variant, "arm": arm, "phase": "final", "seed": 1000 + s, "cfg": cfg,
                         "budget": 256, "steps": 1500, "pool": d.get("pool"),
                         "info": {"state_floats": d.get("state", 256), "params": 1000}, "summary": summary})
    return rows


def run(variant, spec, gate=None):
    rows = rows_for(variant, spec)
    agg, sizes, budgets, rungs, _ = AN.aggregate(rows)
    lowdata = AN.aggregate_lowdata(rows)
    v, why, _ = AN.verdict_for_variant(variant, agg, sizes, budgets, rungs, gate or {}, lowdata, {})
    return v, why


def test_equal_convergence_but_faster_is_a_candidate():
    v, why = run("index", {
        "ssp": {"id": 0.400, "t50": 400, "t90": 800},
        "gru": {"id": 0.998, "t50": 200, "t90": 400},
        "ssp_index_learned": {"id": 0.999, "t50": 100, "t90": 200, "rung": 1},
    })
    print(f"  scenario 1 -> {v}: {why}")
    assert v == "thesis_pass_candidate", v


def test_hand_wired_win_only_is_structural_prior():
    v, why = run("pointer", {
        "ssp": {"id": 0.900, "extrap_T": 0.800, "t50": 100, "t90": 200},
        "gru": {"id": 0.300, "extrap_T": 0.150, "t50": 200, "t90": 400},
    })
    print(f"  scenario 2 -> {v}: {why}")
    assert v == "structural_prior_only", v


def test_equal_everywhere_buys_nothing():
    v, why = run("delay", {
        "ssp": {"id": 0.999, "t50": 200, "t90": 400},
        "gru": {"id": 0.998, "t50": 200, "t90": 400},
        "lru": {"id": 0.998, "t50": 200, "t90": 400},
    })
    print(f"  scenario 3 -> {v}: {why}")
    assert v == "algebra_buys_nothing", v


def test_learned_rung_below_baseline_is_not_discoverable():
    v, why = run("index", {
        "ssp": {"id": 0.999, "t50": 100, "t90": 200},
        "gru": {"id": 0.998, "t50": 200, "t90": 400},
        "ssp_index_learned": {"id": 0.300, "t50": 900, "t90": 1800, "rung": 1},
    })
    print(f"  scenario 4 -> {v}: {why}")
    assert v == "structure_not_discoverable", v


def test_fwp_control_matching_makes_a_win_fragile():
    v, why = run("index", {
        "gru": {"id": 0.900, "t50": 200, "t90": 400},
        "fwp": {"id": 0.999, "t50": 200, "t90": 400},
        "ssp_index_learned": {"id": 0.999, "t50": 100, "t90": 200, "rung": 1},
    })
    print(f"  scenario 5 -> {v}: {why}")
    assert v == "structural_prior_only", v


def test_gate_failed_baseline_is_not_used():
    """tf_full at 0.37 must not be the bar: with only a failing baseline, the verdict is inconclusive."""
    spec = {"tf_full": {"id": 0.370, "state": 2048, "t50": 200, "t90": 400},
            "ssp": {"id": 0.999, "t50": 100, "t90": 200}}
    v, why = run("index", spec, gate={"tf_full": False})
    print(f"  scenario 6 -> {v}: {why}")
    assert v == "inconclusive", v


def test_win_over_a_weak_baseline_on_one_split_is_not_a_win():
    """The r0fix delay data: the LRU collapses on held-out episodes but the GRU does not.  An arm that beats only the
    (id-best) LRU there has not beaten the best usable baseline on that split, so it must not count as a win."""
    v, why = run("delay", {
        "lru": {"id": 1.000, "extrap_T": 0.10, "t50": 200, "t90": 400},
        "gru": {"id": 0.998, "extrap_T": 0.95, "t50": 200, "t90": 400},
        "ssp": {"id": 0.990, "extrap_T": 0.50, "t50": 200, "t90": 400},
    })
    print(f"  scenario 7 -> {v}: {why}")
    assert v == "algebra_buys_nothing", v


def test_unreached_threshold_is_not_a_speed_claim():
    """A time averaged over only the seeds that reached the threshold is survivorship-biased; require all seeds."""
    rows = rows_for("index", {"ssp": {"id": 0.999, "t50": 100, "t90": 200}, "gru": {"id": 0.998, "t50": 400, "t90": 800}})
    for r in rows:
        if r["arm"] == "ssp" and r["seed"] >= 1002:       # 2 of 5 seeds never reach 90%
            r["summary"]["t90_id"] = None
    agg, sizes, budgets, rungs, _ = AN.aggregate(rows)
    ev = AN.evaluate_candidate("ssp", agg, sizes, budgets["ssp"], {}, {}, "conv")
    print(f"  speed wins with 3/5 seeds reaching t90: {ev['speed_win']}")
    assert ev["speed_win"] is None or "t90" not in ev["speed_win"], ev["speed_win"]


def test_identity_init_is_not_a_learned_rung():
    """An identity-initialised learned arm is rung 0 in disguise.  If it counted as a learned rung, a capability check
    would turn into a 'thesis pass' (this happened on the first learn1 analysis)."""
    rows = rows_for("index", {
        "ssp": {"id": 0.998, "t50": 100, "t90": 200},
        "gru": {"id": 0.999, "t50": 400, "t90": 800},
        "ssp_index_learned_r1_ident": {"id": 0.998, "t50": 100, "t90": 200, "rung": 1},
        "ssp_index_learned_r1_plain": {"id": 0.400, "t50": 900, "t90": 1800, "rung": 1},
    })
    for r in rows:
        if r["arm"].endswith("_ident"):
            r["cfg"] = {"rung": 1, "init": "identity"}
    agg, sizes, budgets, rungs, _ = AN.aggregate(rows)
    v, why, _ = AN.verdict_for_variant("index", agg, sizes, budgets, rungs, {}, {}, {})
    print(f"  rungs={rungs}\n  scenario 8 -> {v}: {why}")
    assert rungs["ssp_index_learned_r1_ident"] == 0
    assert v == "structure_not_discoverable", v


def main():
    checks = [("equal_convergence_faster", test_equal_convergence_but_faster_is_a_candidate),
              ("hand_wired_only", test_hand_wired_win_only_is_structural_prior),
              ("equal_everywhere", test_equal_everywhere_buys_nothing),
              ("learned_below_baseline", test_learned_rung_below_baseline_is_not_discoverable),
              ("fwp_matches", test_fwp_control_matching_makes_a_win_fragile),
              ("gate_failed_baseline_excluded", test_gate_failed_baseline_is_not_used),
              ("weak_baseline_win_is_not_a_win", test_win_over_a_weak_baseline_on_one_split_is_not_a_win),
              ("unreached_threshold_not_speed", test_unreached_threshold_is_not_a_speed_claim),
              ("identity_init_not_learned", test_identity_init_is_not_a_learned_rung)]
    failed = []
    for name, fn in checks:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"FAIL {name}: {e}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
