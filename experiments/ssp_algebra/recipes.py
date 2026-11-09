"""Training recipes: one rule per problem family (docs/11 section 3.5, 4.7; handoff 09 section 15.4 step 2).

**Why a rule and not a table of constants.** RUN_009 measured that the continuous-address recipe's *form*
(coarse-to-fine kernel width) carries to a second stream length but its *constants* do not: the hand-set T=16 pair
(start width 8, 375-update schedule) gives 0.665 at T=32, while a start width near the address range and a longer
schedule give 0.93-0.94.  Per-problem constants make this a technique, not a recipe, and the search runs plus their
labelled data then belong to the sample cost (docs/11 3.5).  What makes it a recipe is a *rule* that sets every
constant from a task descriptor, here the address range ``R``:

    start kernel width   = a * R
    schedule length      = b * R   updates
    total updates        = c * R   with c = 2b (the schedule is the first half of the run)

``R`` is the span of addresses the memory must resolve: ``T - 2`` for index recall (the task samples ``j`` in
``[0, T-2]``) and the largest training lag for the delay task.

**Status.** The defaults below are *proposals*, not validated constants: docs/11 4.7 requires the rule to be tuned on
two variants (T=16, T=32), frozen, and then to pass on unseen variants (>= 0.95 on >= 9 of 10 seeds, no retuning)
before anything is called a recipe.  Until that run happens this module is the machinery, not the result.
"""

import argparse
import json

# Frozen proposal (to be replaced by the tuned triple, then never re-tuned).  RUN_009: at T=16 a start width of 8
# (a=0.5 at R=14) and a ~375-update schedule work; lr 0.0225 is the top of the working range (0.05 no better,
# 0.1 worse).
DEFAULT_A = 0.5
DEFAULT_B = 24.0
DEFAULT_LR = 0.02
LAM_INIT = 0.99                   # forgetting horizon: frozen constant of the recipe (docs/11 4.7 freezes *every* knob)
TEMP = 0.1                        # cleanup softmax temperature: same
A_GRID = (0.5, 1.0, 1.5)          # tuning grid, handoff 09 section 15.4 step 2
B_GRID = (24.0, 47.0, 94.0)
DEFAULT_SLOPE_EPS = 0.03          # RUN_011's mechanism-derived recipe: the read head's *scale* error (relative), with
                                  # the offset pinned to zero.  Tuned (the search also tried 0.1 relative and 0.5/0.4
                                  # in other parameterisations); see slope_only_cfg and run record RUN_011 section 3.3.


def address_range(variant, ep_len=None, max_lag=None):
    """The address span ``R`` the recipe scales with."""
    if variant == "index":
        if ep_len is None:
            raise ValueError("index needs ep_len (the recipe needs the address range)")
        return max(int(ep_len) - 2, 1)
    if variant == "delay":
        if max_lag is None:
            raise ValueError("delay needs max_lag (the recipe needs the address range)")
        return max(int(max_lag), 1)
    raise ValueError(f"no continuous-address recipe for variant {variant!r}")


def slope_only_cfg(eps=DEFAULT_SLOPE_EPS):
    """The recipe RUN_011 derived from the *mechanism*, for the index and delay tasks: **no address-range constant**.

    A read head initialised to the time ramp with a small *scale* error and **offset exactly zero**, trained with a
    fixed single-scale frequency set (no schedule).  Why it works, measured (RUN_011 3.2): the discoverability of the
    SSP read address is limited by the *offset* of the initial address, not by its size.  A constant offset leaves every
    stream position the same distance from the kernel's informative range (`kappa = sinc(pi d)` is zero at integer
    offsets when `length_scale = 1`), so the gradient has no direction and the read head never moves; a scale error
    leaves the small-index positions nearly correct, which is enough.  Tolerance is therefore narrow in the origin
    coordinate and wide in the scale coordinate: offset 0 passes at 9/10 seeds for scale errors of 0, 3% and 10%, while
    every offset of 1 or more fails 0/10 (RUN_011 3.2's grid).

    Measured (10 seeds, frozen hyperparameters, RUN_011 3.3): index T=16 0.996 (10/10, 25.7k tokens), T=24 0.976
    (10/10), T=32 0.956 (9/10), T=48 0.973 (10/10), delay 1-4 1.000 (10/10), delay 1-8 0.997 (10/10).
    Index T=64 is the one failure (0.952, 6/10 at a 512-float budget) and is suspected capacity-limited (address range
    62 against `D/4 ~ 64`); the controls for that are in `jobs/p5g_t64_ceiling.json` and `jobs/p5g_t64_recipe_1024.json`.

    **Status: a candidate recipe, not yet a validated one.**  `eps` is a *tuned* constant (the search tried 0.4 relative,
    0.5 absolute, 0.1 and 0.03 before this value), the prior is a *strong aid* in the sense of docs/11 3.4, and the
    reliability at T=32 is exactly at the 9/10 bar.
    """
    return {"rung": 1, "init": "affine", "affine_eps": float(eps), "affine_offset_zero": True,
            "length_scale": 1.0, "ls_mode": "fixed", "lr": DEFAULT_LR, "lam_init": LAM_INIT, "temp": TEMP}


def continuous_address_cfg(R, a=DEFAULT_A, b=DEFAULT_B, lr=DEFAULT_LR):
    """The ``--arm-cfg`` dict and step count for the learned continuous-address arm at address range ``R``.

    Returns ``{"arm_cfg": ..., "steps": int, "R": int, "a": float, "b": float}``.  ``steps`` is deliberately *outside*
    ``arm_cfg``: every key of ``arm_cfg`` must be either a harness training key or an arm constructor argument,
    otherwise ``build_learned`` raises on the unknown kwarg (see ``test_recipes.py``).
    """
    R = int(R)
    if R < 1:
        raise ValueError(f"address range must be positive, got {R}")
    width_steps = max(int(round(b * R)), 1)
    arm_cfg = {"rung": 1, "init": "random", "length_scale": 1.0,
               "ls_start": float(a * R), "ls_mode": "sched", "width_steps": width_steps, "lr": float(lr),
               "lam_init": LAM_INIT, "temp": TEMP}
    return {"arm_cfg": arm_cfg, "steps": int(round(2 * b * R)), "R": R, "a": float(a), "b": float(b)}


def job_entry(name, variant, tag, ep_len=16, max_lag=None, a=DEFAULT_A, b=DEFAULT_B, lr=DEFAULT_LR, budget=256,
              configs=6, seeds=10, search_seeds=3, extra="", label=None, rows=None):
    """A ``scripts/ssp_pod_batch.py`` job entry for one recipe condition (fixed by the rule, nothing hand-set)."""
    spec = continuous_address_cfg(address_range(variant, ep_len=ep_len, max_lag=max_lag), a=a, b=b, lr=lr)
    j = {"name": name, "variant": variant, "arms": ["ssp_index_learned"], "tag": tag,
         "arm_cfg": spec["arm_cfg"], "budget": budget, "configs": configs, "seeds": seeds, "steps": spec["steps"],
         "extra": f"--select aulc --search-seeds {search_seeds} {extra}".strip()}
    if label:
        j["label"] = label
    j["rows"] = rows if rows is not None else len(j["arms"]) * (configs * search_seeds + seeds)
    return j


def main():
    p = argparse.ArgumentParser(description="Emit the recipe's --arm-cfg / a pod job entry for a variant.")
    p.add_argument("--variant", default="index", choices=["index", "delay"])
    p.add_argument("--ep-len", type=int, default=16)
    p.add_argument("--max-lag", type=int, default=None)
    p.add_argument("--a", type=float, default=DEFAULT_A)
    p.add_argument("--b", type=float, default=DEFAULT_B)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--job-name", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    spec = continuous_address_cfg(address_range(args.variant, ep_len=args.ep_len, max_lag=args.max_lag),
                                  a=args.a, b=args.b, lr=args.lr)
    if args.job_name:
        print(json.dumps([job_entry(args.job_name, args.variant, args.tag or args.job_name.replace("-", "_"),
                                    ep_len=args.ep_len, max_lag=args.max_lag, a=args.a, b=args.b, lr=args.lr)], indent=1))
        return
    print(f"R={spec['R']} steps={spec['steps']}")
    print("--arm-cfg " + json.dumps(spec["arm_cfg"], separators=(",", ":")))


if __name__ == "__main__":
    main()
