"""Decision rules from docs/11 sections 4.3-4.6, applied mechanically to sweep JSONL (handoff 09 step H).

    OMP_NUM_THREADS=1 ../../.venv/bin/python analysis.py --tag r0fix [--extra results/sweep_r0fix_pool.jsonl ...]

What changed from the RUN_007 version (which produced the withdrawn verdicts, RUN_007 bugs 7.13-7.14):

* convergence, speed and area are read from three *separately labelled* quantities -- ``conv_*`` (mean of the last 3
  checkpoints), ``t50_*``/``t90_*`` (tokens to 50%/90% of ceiling) and ``aulc_*`` -- never from ``last5_*``, which is
  a short learning-curve area that blends speed with convergence.  If a file predates the fix, analysis says so and
  falls back to ``last5_*`` with a warning rather than silently quoting it as accuracy.
* a baseline that failed its trainability gate (``results/trainability.json``) is reported but **never** used as the
  bar an arm has to beat.
* fragility is checked across rungs *and* splits: an advantage that appears only at the hand-wired rung 0, or only
  in-distribution, or that is matched by the ``fwp`` control or the ``pos_table`` ablation, is not a thesis pass.
* ties go to "buys nothing".

Verdicts:

  * ``thesis_pass_candidate`` -- a **learned** rung is within 5 points absolute of the best matched-state baseline at
    convergence *and* is at least 2 pooled se + 5 points better, or 2x faster, on one of: low-data, a fairly encoded
    extrapolation split, or a robustness split.
  * ``structural_prior_only``  -- only the hand-wired rung 0 separates that way (this is what RUN_007's index and
    pointer results are).
  * ``structure_not_discoverable`` -- a learned rung exists but none is within tolerance of rung 0, even aided.
  * ``algebra_buys_nothing``   -- the arm is within tolerance but nothing separates it (docs/10's kill condition).
  * ``inconclusive``           -- not enough information (no hold-out rows, no comparable baseline, ...).
"""

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

OUT = pathlib.Path(__file__).with_name("results")

TOL = 0.05            # convergence tolerance, points absolute (docs/11 section 4.5)
SPEED_RATIO = 0.5     # "at most half the tokens" to reach 50%/90% of ceiling
K_SE = 2.0            # pooled standard errors for a separation
MARGIN = 0.05         # and a meaningful absolute margin
TOL_STATE = 0.05      # matched-state rule: within 5% in carried floats

VERDICTS = ("thesis_pass_candidate", "structural_prior_only", "structure_not_discoverable",
            "algebra_buys_nothing", "inconclusive")
METRICS = ("conv", "t50", "t90", "aulc", "last5")


# --------------------------------------------------------------------------------------------------------
# loading / aggregation
# --------------------------------------------------------------------------------------------------------
def load(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (float(np.mean(vals)), float(np.std(vals) / max(np.sqrt(len(vals)), 1)), len(vals))


def aggregate(rows, phase="final"):
    """(metric -> arm -> split -> (mean, se, n)) from hold-out rows, plus sizes, budget and rung per arm.

    ``phase`` selects hold-out rows only by default: a search-phase row is a config draw, not a result.
    """
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sizes, budgets, rungs, seen = {}, {}, {}, set()
    for r in rows:
        seen.add(r.get("phase"))
        if phase is not None and r.get("phase") != phase:
            continue
        if r.get("pool"):                   # low-data rows are a different regime: see aggregate_lowdata
            continue
        arm = r["arm"]
        sizes[arm] = r.get("info", {})
        budgets[arm] = r.get("budget")
        cfg = r.get("cfg") or {}
        if cfg.get("init") == "identity":
            rungs[arm] = 0              # seeded with the hand-wired answer: a capability check, never a learned rung
        elif "rung" in cfg:
            rungs[arm] = int(cfg["rung"])
        elif arm.endswith("_learned") or "_learned_" in arm:
            rungs.setdefault(arm, 1)
        for key, val in r["summary"].items():
            for metric in METRICS:
                pre = metric + "_"
                if key.startswith(pre):
                    agg[metric][arm][key[len(pre):]].append(val)
    out = {m: {a: {s: _agg(v) for s, v in splits.items()} for a, splits in arms.items()} for m, arms in agg.items()}
    return out, sizes, budgets, rungs, seen


def aggregate_lowdata(rows):
    """arm -> pool size -> (mean, se, n) of the converged in-distribution score (rows whose pool is not None)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pool = r.get("pool")
        if not pool or r.get("phase") != "final":
            continue
        key = "conv_id" if "conv_id" in r["summary"] else "last5_id"
        agg[r["arm"]][int(pool)].append(r["summary"][key])
    return {a: {n: _agg(v) for n, v in d.items()} for a, d in agg.items()}


def aggregate_budget_curve(rows):
    """arm -> budget -> (mean, se, n) of the converged in-distribution score, hold-out rows, no pool."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("phase") != "final" or r.get("pool"):
            continue
        key = "conv_id" if "conv_id" in r["summary"] else "last5_id"
        agg[r["arm"]][r.get("budget")].append(r["summary"][key])
    return {a: {b: _agg(v) for b, v in d.items()} for a, d in agg.items()}


def gate_status(path=None):
    path = pathlib.Path(path) if path else OUT / "trainability.json"
    if not path.exists():
        return {}, f"(no {path.name}: no trainability gate on record for this tag)"
    data = json.loads(path.read_text())
    return {a: bool(v.get("passed")) for a, v in data.get("arms", {}).items()}, \
        f"(trainability gate from {path.name}, threshold {data.get('threshold')})"


def converged_metric(agg):
    """Use ``conv_*`` when present; otherwise warn once and fall back to ``last5_*`` (pre-fix files)."""
    if agg.get("conv") and any(agg["conv"][a] for a in agg["conv"]):
        return "conv", None
    return "last5", ("this file has no conv_* metric: it predates the measurement fix, so 'conv' below is "
                     "last5_* (a short learning-curve area that blends speed and convergence) -- do not quote it "
                     "as accuracy")


# --------------------------------------------------------------------------------------------------------
# comparisons (pure functions: test_analysis.py drives these with synthetic numbers)
# --------------------------------------------------------------------------------------------------------
def beats(a, b, k=K_SE, margin=MARGIN):
    """True if ``a`` exceeds ``b`` by more than ``max(k pooled se, margin)``.  Ties are not wins."""
    if not a or not b:
        return False
    ma, sa, _ = a
    mb, sb, _ = b
    return (ma - mb) > max(k * float(np.hypot(sa, sb)), margin)


def matched_state_arms(sizes, budget, tol=TOL_STATE):
    if not budget:
        return []
    return [a for a, info in sizes.items()
            if info.get("state_floats") and abs(info["state_floats"] - budget) / budget <= tol]


def generalisation_splits(agg, arm, metric="conv"):
    return sorted(s for s in agg[metric].get(arm, {}) if s.startswith(("extrap", "rob")))


def best_baseline(agg, sizes, budget, exclude, gate, metric="conv"):
    """The best matched-state, gate-passing, non-SSP arm on the in-distribution split."""
    out = {}
    for a in matched_state_arms(sizes, budget):
        if a == exclude or a in ("mlp", "posmlp") or a.startswith("ssp") or a == "fwp":
            continue
        if gate and gate.get(a) is False:
            continue
        val = agg[metric].get(a, {}).get("id")
        if val:
            out[a] = val
    if not out:
        return None
    return max(out, key=lambda a: out[a][0])


def usable_baselines(agg, sizes, budget, exclude, gate, metric="conv"):
    out = []
    for a in matched_state_arms(sizes, budget):
        if a == exclude or a in ("mlp", "posmlp") or a.startswith("ssp") or a == "fwp":
            continue
        if gate and gate.get(a) is False:
            continue
        if agg[metric].get(a, {}).get("id"):
            out.append(a)
    return out


def _reliable_time(agg, name, arm, split="id", metric="conv"):
    """Mean tokens-to-threshold, only if EVERY seed reached it (a mean over the seeds that did is survivorship-biased)."""
    t = agg[name].get(arm, {}).get(split)
    c = agg[metric].get(arm, {}).get(split)
    if t and c and t[2] == c[2]:
        return t[0]
    return None


def evaluate_candidate(arm, agg, sizes, budget, gate, lowdata, metric="conv"):
    """docs/11 section 4.5 comparisons for one arm.

    Each axis is measured against the best *usable* (matched-state, gate-passing) baseline **on that axis**: the
    fastest baseline for speed, the best baseline at each pool size for low-data, the best baseline on each
    extrapolation/robustness split.  Choosing one baseline by its in-distribution score and then crediting wins
    against it on other axes cherry-picks a baseline that happens to be weak there (an LRU collapses on held-out lags
    while a GRU does not).
    """
    base_names = usable_baselines(agg, sizes, budget, arm, gate, metric)
    if not base_names or not agg[metric].get(arm, {}).get("id"):
        return None
    base = max(base_names, key=lambda a: agg[metric][a]["id"][0])
    self_id, base_id = agg[metric][arm]["id"], agg[metric][base]["id"]
    within = self_id[0] >= base_id[0] - TOL

    speed = {}
    for name in ("t50", "t90"):
        ta = _reliable_time(agg, name, arm, "id", metric)
        if ta is None or ta <= 0:
            continue
        tbs = [t for t in (_reliable_time(agg, name, b, "id", metric) for b in base_names) if t]
        fastest = min(tbs) if tbs else None            # no baseline reliably reaches it: any reliable arm wins
        if fastest is None or ta <= SPEED_RATIO * fastest:
            speed[name] = (ta, fastest)
    low = None
    if lowdata and arm in lowdata:
        for n in sorted(lowdata[arm]):
            rivals = [lowdata[b][n] for b in base_names if b in lowdata and n in lowdata[b]]
            if rivals and all(beats(lowdata[arm][n], r) for r in rivals):
                low = (n, lowdata[arm][n][0], max(r[0] for r in rivals))
                break
    gen = None
    for sp in generalisation_splits(agg, arm, metric):
        rivals = [(b, agg[metric][b][sp]) for b in base_names if agg[metric].get(b, {}).get(sp)]
        if rivals and all(beats(agg[metric][arm][sp], r) for _, r in rivals):
            gen = (sp, agg[metric][arm][sp][0], max(r[0] for _, r in rivals))
            break
    return {"baseline": base, "id": self_id[0], "baseline_id": base_id[0], "within_tolerance": bool(within),
            "speed_win": speed or None, "lowdata_win": low, "gen_win": gen}


def fragility(arm, agg, sizes, budget, gate, ablations, variant, metric="conv"):
    """Reasons an apparent win is not a thesis pass: it must not be shared by the controls, and not be
    rung-0-only or in-distribution-only."""
    reasons = []
    for control in ("fwp",):
        if control in agg.get(metric, {}) and agg[metric][control].get("id"):
            if not beats(agg[metric][arm]["id"], agg[metric][control]["id"]):
                reasons.append(f"the {control} (non-SSP associative memory) control matches it in-distribution")
    av = {k[1]: d for k, d in ablations.items() if k[0] == variant}
    for name in ("pos_table", "no_cleanup", "no_decay"):
        if name in av and "id" in av[name].get(metric, {}) and agg[metric].get(arm, {}).get("id"):
            m, s, _ = av[name][metric]["id"]
            ref = agg[metric]["ssp"]["id"] if "ssp" in agg[metric] and "id" in agg[metric]["ssp"] else None
            if ref and (ref[0] - m) <= max(K_SE * float(np.hypot(ref[1], s)), MARGIN):
                reasons.append(f"the {name} ablation matches the full arm (the mechanism is not load-bearing)")
    return reasons


def verdict_for_variant(variant, agg, sizes, budgets, rungs, gate, lowdata, ablations, metric="conv"):
    """The docs/11 section 4.5 rule, returning (verdict, why, per-arm evaluations)."""
    arms = [a for a in agg[metric] if a in sizes]
    if not arms:
        return "inconclusive", "no hold-out rows for this variant", {}
    learned = sorted([a for a in arms if rungs.get(a, 0) >= 1], key=lambda a: -rungs[a])
    hand = "ssp" if "ssp" in arms else None
    evals = {a: evaluate_candidate(a, agg, sizes, budgets.get(a), gate, lowdata, metric) for a in arms}
    if hand is None and not learned:
        return "inconclusive", "no SSP or learned arm measured", evals
    if all(v is None for v in evals.values()):
        return "inconclusive", "no matched-state, gate-passing baseline to compare against", evals

    def passes(ev):
        return bool(ev and ev["within_tolerance"] and (ev["speed_win"] or ev["lowdata_win"] or ev["gen_win"]))

    for arm in learned:
        ev = evals.get(arm)
        if passes(ev):
            frag = fragility(arm, agg, sizes, budgets.get(arm), gate, ablations, variant, metric)
            if frag:
                return "structural_prior_only", f"learned rung {rungs.get(arm)} separates but is fragile: " + "; ".join(frag), evals
            gained = [k for k, v in (("speed", ev["speed_win"]), ("low-data", ev["lowdata_win"]),
                                     ("extrapolation/robustness", ev["gen_win"])) if v]
            return "thesis_pass_candidate", (f"learned rung {rungs[arm]} ({arm}) is within {TOL:.2f} of "
                                             f"{ev['baseline']} at convergence and separates on {', '.join(gained)}"), evals
    if not learned:
        # rung-0-only run: this is the pre-registered stop-rule question (docs/11 section 4.6)
        if hand and passes(evals.get(hand)):
            ev = evals[hand]
            return "structural_prior_only", ("only the hand-wired rung 0 separates from "
                                             f"{ev['baseline']} (a structural prior, not learning the algebra)"), evals
        if hand and evals.get(hand) and evals[hand]["within_tolerance"]:
            ev = evals[hand]
            return "algebra_buys_nothing", (f"rung 0 is within {TOL:.2f} of {ev['baseline']} but not faster, not "
                                            "better in low data and not better on any extrapolation split"), evals
        return "inconclusive", "rung 0 is not within tolerance of any usable baseline", evals
    within_learned = [a for a in learned if evals.get(a) and evals[a]["within_tolerance"]]
    if not within_learned:
        best = max(learned, key=lambda a: rungs.get(a, 0))
        ev = evals.get(best)
        gap = None if not ev else ev["baseline_id"] - ev["id"]
        hand_note = ("rung 0 does separate, so the structure is present but the learner does not discover it"
                     if hand and passes(evals.get(hand)) else "and rung 0 does not separate either")
        return "structure_not_discoverable", (f"no learned rung is within {TOL:.2f} of the best matched-state "
                                              f"baseline (best rung {rungs.get(best)}: "
                                              f"{'n/a' if gap is None else f'{gap:.3f} short'}); {hand_note}"), evals
    return "algebra_buys_nothing", ("a learned rung is within tolerance of the best matched-state baseline but is not "
                                    "faster, not better in low data and not better on any extrapolation split "
                                    "(docs/10 kill condition)"), evals


# --------------------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------------------
def aggregate_ablation(rows):
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        name = r.get("phase", "")
        if not name.startswith("abl_"):
            continue
        key = (r["variant"], name[len("abl_"):])
        for k, val in r["summary"].items():
            for metric in METRICS:
                if k.startswith(metric + "_"):
                    agg[key][metric][k[len(metric) + 1:]].append(val)
    return {k: {m: {s: _agg(v) for s, v in d.items()} for m, d in arms.items()} for k, arms in agg.items()}


def specificity_report(agg, lowdata, abl_v, metric="conv", arm="ssp", control="fwp"):
    """Is the arm's advantage specific to SSP?  Lines comparing it with the non-SSP associative-memory control and
    with the random-position-table ablation, axis by axis.  ``=`` means not separated (ties are not wins)."""
    lines = []

    def cmp(a, b):
        if not a or not b:
            return "n/a"
        if beats(a, b):
            return f"{arm} ahead ({a[0]:.3f} vs {b[0]:.3f})"
        if beats(b, a):
            return f"{control} ahead ({a[0]:.3f} vs {b[0]:.3f})"
        return f"= ({a[0]:.3f} vs {b[0]:.3f})"

    ma, mb = agg.get(metric, {}).get(arm, {}), agg.get(metric, {}).get(control, {})
    if mb:
        for sp in ["id"] + sorted(k for k in ma if k.startswith(("extrap", "rob"))):
            lines.append(f"    vs {control} [{sp}]: {cmp(ma.get(sp), mb.get(sp))}")
        ta, tb = agg.get("t50", {}).get(arm, {}).get("id"), agg.get("t50", {}).get(control, {}).get("id")
        if ta and tb:
            ratio = ta[0] / tb[0] if tb[0] else float("nan")
            lines.append(f"    vs {control} [t50 tokens]: {ta[0]:,.0f} vs {tb[0]:,.0f} (ratio {ratio:.2f})")
        if arm in lowdata and control in lowdata:
            for n in sorted(set(lowdata[arm]) & set(lowdata[control])):
                lines.append(f"    vs {control} [low-data N={n}]: {cmp(lowdata[arm][n], lowdata[control][n])}")
    pt = abl_v.get("pos_table", {}).get(metric, {}).get("id")
    ref = abl_v.get("reference", {}).get(metric, {}).get("id")
    if pt and ref:
        lines.append(f"    pos_table ablation vs ablation reference [id, same fixed config]: {cmp(ref, pt)} "
                     f"(ref {ref[0]:.3f}, pos_table {pt[0]:.3f})".replace(f"{control} ahead", "pos_table ahead"))
    return lines


def _dump(agg):
    """metric -> arm -> split -> tuple  ==>  arm -> metric -> split -> list, for JSON output."""
    out = {}
    for metric, arms in agg.items():
        for arm, splits in arms.items():
            out.setdefault(arm, {})[metric] = {sp: list(v) for sp, v in splits.items() if v is not None}
    return out


def print_table(title, agg, metric, arm_sizes, split_order=None, rungs=None):
    print(f"\n  {title}")
    present = {sp for splits in agg.get(metric, {}).values() for sp in splits}
    keys = [k for k in (split_order or []) if k in present]
    arms = sorted(agg.get(metric, {}), key=lambda a: -(agg[metric][a].get("id") or (0, 0, 0))[0])
    for arm in arms:
        line = f"    {arm:22s} st={arm_sizes.get(arm, {}).get('state_floats', 0):5d} "
        for s in keys:
            val = agg[metric][arm].get(s)
            if not val:
                cell = f" {s}=n/a"
            elif metric in ("t50", "t90"):
                cell = f" {s}={val[0]:,.0f}±{val[1]:,.0f}"
            elif metric == "aulc":
                cell = f" {s}={val[0]:,.0f}"
            else:
                cell = f" {s}={val[0]:+.3f}±{val[1]:.3f}"
            line += cell.rjust(26)
        if rungs:
            line += f"  rung={rungs.get(arm, 0)}"
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--ablation-tag", default="abl")
    p.add_argument("--extra", nargs="*", default=[], help="extra JSONL files (low-data pools, robustness, learned rungs)")
    p.add_argument("--budget", type=int, default=256, help="the budget whose rows form the main verdict table")
    p.add_argument("--gate", default=None, help="trainability gate JSON (default results/trainability.json)")
    args = p.parse_args()

    main_rows = load(OUT / f"sweep_{args.tag}.jsonl")
    extra_rows = [r for f in args.extra for r in load(f)]
    abl_rows = []
    try:
        abl_rows = load(OUT / f"ablation_{args.ablation_tag}.jsonl")
    except SystemExit:
        print(f"(no ablation file for tag {args.ablation_tag}: verdicts cannot flag hand-wiring dependence)")

    all_rows = main_rows + extra_rows
    rows_b = [r for r in all_rows if r.get("budget") in (args.budget, None)]     # budget curve rows are kept apart
    agg, sizes, budgets, rungs, seen = aggregate(rows_b)
    lowdata = aggregate_lowdata(all_rows)
    abl = aggregate_ablation(abl_rows)
    gate, gate_note = gate_status(args.gate)
    metric, warn = converged_metric(agg)
    print(f"tag={args.tag} metric='{metric}' (convergence)  {gate_note}")
    if warn:
        print(f"  WARNING: {warn}")
    if lowdata:
        print(f"  low-data pools present: {sorted({n for d in lowdata.values() for n in d})}")
    if not gate:
        print("  WARNING: no trainability gate on record -- every baseline is treated as usable.")

    out = {"tag": args.tag, "metric": metric, "gate": gate, "variants": {}}
    for variant in sorted({r["variant"] for r in all_rows}):
        vrows = [r for r in all_rows if r["variant"] == variant and r.get("budget") in (args.budget, None)]
        vcurve = aggregate_budget_curve([r for r in all_rows if r["variant"] == variant])
        vagg, vsizes, vbudgets, vrungs, _ = aggregate(vrows)
        vlow = aggregate_lowdata(vrows)
        abl_v = {k: v for k, v in abl.items() if k[0] == variant}
        v, why, evals = verdict_for_variant(variant, vagg, vsizes, vbudgets, vrungs, gate, vlow, abl)
        budget = {b for b in vbudgets.values() if b}
        print(f"\n=== {variant} ===  verdict: {v}\n  why: {why}")
        if len(budget) > 1:
            print(f"  WARNING: mixed budgets in this variant ({sorted(budget)}); matched-state comparison is computed "
                  "per arm budget and may be meaningless")
        order = ["id", "train"] + sorted({s for a in vagg.get(metric, {}) for s in vagg[metric][a]
                                          if s.startswith(("extrap", "rob"))})
        print_table(f"converged ({metric}_*, mean of last 3 checkpoints; ± se over hold-out seeds)",
                    vagg, metric, vsizes, order, vrungs)
        print_table("speed (tokens to 50% / 90% of ceiling)", vagg, "t50", vsizes, order, vrungs)
        print_table("area under the learning curve (per token)", vagg, "aulc", vsizes, order, vrungs)
        if len({b for d in vcurve.values() for b in d}) > 1:
            print("  budget curve (converged id by carried-state budget, floats):")
            for arm in sorted(vcurve):
                print(f"    {arm:22s} " + "  ".join(f"{b}:{v[0]:.3f}" for b, v in sorted(vcurve[arm].items()) if v))
        for arm in sorted(vlow):
            print(f"  low-data {arm:14s}(converged id by pool N): " + ", ".join(f"N={n}:{v[0]:.3f}" for n, v in sorted(vlow[arm].items())))
        for arm, ev in sorted(evals.items()):
            if ev:
                flags = [k for k, val in (("speed", ev["speed_win"]), ("low-data", ev["lowdata_win"]),
                                          ("gen", ev["gen_win"])) if val]
                print(f"  vs {ev['baseline']}: {arm} id={ev['id']:.3f} baseline={ev['baseline_id']:.3f} "
                      f"within_tol={ev['within_tolerance']} wins={flags or 'none'}")
        spec_lines = specificity_report(vagg, vlow, {k[1]: d for k, d in abl_v.items()})
        if spec_lines:
            print("  specificity (is the advantage SSP-specific?):")
            print("\n".join(spec_lines))
        for (_, name), d in sorted(abl_v.items()):
            if "id" in d.get(metric, {}):
                m, se, n = d[metric]["id"]
                ex = d[metric].get("extrap_T")
                ex_s = f" extrap_T={ex[0]:+.3f}±{ex[1]:.3f}" if ex else ""
                print(f"  ablation {name:14s} id={m:+.3f}±{se:.3f}{ex_s} (n={n})")
        out["variants"][variant] = {
            "verdict": v, "why": why,
            "arms": _dump(vagg),
            "evaluations": evals, "lowdata": vlow, "sizes": vsizes, "rungs": vrungs,
            "ablations": {f"{k[0]}/{k[1]}": {m: {sp: list(x) for sp, x in d.items() if x} for m, d in arms.items()}
                         for k, arms in abl_v.items()},
        }
    path = OUT / f"analysis_{args.tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
