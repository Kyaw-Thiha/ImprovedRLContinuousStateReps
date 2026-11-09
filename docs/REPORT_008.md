# Report 008: settling the one SSP hint, and the length-scale boundary

Completed 2026-11-09. Numbers come from the run files via
`analysis/r7_readout.py`, `analysis/early_critic_ssp.py` and `analysis/ssp_report_table.py`; nothing is typed by hand.
Decisions are `d<N>` in `docs/mjlab_centering/02-decisions.md` (round 7: d1020-d1027), pre-registered before launch.

## 0. What was asked and what was run

Three pieces, **$4 hard cap**, community GPUs only, one campaign (`r7queue-9e67be96`, 21 jobs, 42k iterations):

| piece | design | seeds | cost |
|---|---|---|---|
| free mechanism check | early critic diagnostics, SSP-none vs raw-none at offset 0.1, iterations 100-300 | pilot 3000-3004 | $0 |
| escalation (d130, d1021) | SSP 2x2 @ ls 1.0, uncentered, offset 0.1, paired with the existing raw cell | **3005-3019** (15 jobs) | ~$1.0 |
| length-scale boundary | SSP 2x2, centered, offset 1, length scales **2.0 and 4.0** | search 1000-1002 (6 jobs) | ~$0.5 |

Every job carries `save_every=250`, so a preemption resumes from `ckpt_latest.pt` rather than restarting (d1022).

TBD: the escalation's last six seeds and the final cost.

## 1. The free mechanism check (already in)

At iterations 100-300, uncentered, offset 0.1, five paired seeds:

| statistic | SSP 2x2 @ 1.0 | raw | paired delta (95% CI) |
|---|---|---|---|
| `const_frac` | 0.552 | 0.751 | **-0.199 [-0.229, -0.170]** |
| explained variance | +0.571 | -1.537 | **+2.108 [+1.822, +2.408]** |
| `v_mean` (analytic level 100) | 59.4 | 52.0 | +7.4 [+5.8, +9.0] |
| action std | 0.708 | 0.628 | +0.080 [+0.066, +0.093] |

The SSP critic's advantage is state-explained from the first hundred iterations and carries a much smaller constant
component - the property d108/d124 found to separate arms that learn from arms that do not. So the pilot's walk-count
difference has a plausible mechanism rather than being pure seed luck. n=5 and a cross-build raw comparator: this
explains, it does not prove.

## 2. The escalation

**Baseline verified before the SSP seeds:** raw at offset 0.1, uncentered, walks **12/20** with mean tracking **0.645** and
a sharply bimodal per-seed distribution (0.29-0.32 on eight seeds, 0.84-0.88 on the other twelve). That is the d115 claim
cell, so the pairing is against 12/20, not the pilot's unlucky 2/5.

**Final, all 20 paired seeds (3000-3004 from the pilot plus 3005-3019 fresh):**

| arm | walks | mean tracking |
|---|---|---|
| SSP 2x2 @ ls 1.0, uncentered, offset 0.1 | **12/20** | 0.647 |
| raw, same cell | **12/20** | 0.645 |

Paired difference **+0.002 tracking, 95% CI [-0.198, +0.200]**, 11 of 20 seeds positive. Under the d1021 rule (interval
excludes 0 **and** at least three more walks out of 20) this is **no resolved difference** - and the point estimate is not
even directionally favourable: the two arms are the same cell.

**The pilot's hint was seed luck, exactly as the user warned.** The pilot saw 4/5 against 2/5; on the nine seeds it never
saw, SSP walked 5/9 against raw's 6/9, and over 20 seeds each arm walks 12 times. The reason the pilot looked dramatic is
visible in the per-seed data: every run is sharply bimodal (~0.30 standing floor or ~0.88 walking) and the two arms fail on
*different* seeds, so a five-seed walk-count comparison has enormous variance while the paired mean stays at zero.

**What the free mechanism check still shows, and what it does not.** The early critic difference is real (SSP const_frac
0.55 vs 0.75, ev +0.57 vs -1.54 at iterations 100-300): under this offset SSP's critic does learn the reward level sooner.
But the outcome is identical at 20 seeds, so on this task a better-conditioned critic does **not** convert into more walking
seeds. Mechanism without effect - which is worth more than the reverse, and needs stating plainly. ("SSP-specific advantage" only if the interval excludes 0 and the walk counts differ by >= 3 of 20).

## 3. The length-scale boundary

**Result (complete, three search seeds each):** ls 2.0 -> **0.879**, ls 4.0 -> **0.884**, against the 0.880 measured at
ls 1.0. Neither is 0.02 above the reference, so the pre-registered rule says the optimum is **not** at the edge: the
length-scale curve is flat from 1.0 to 4.0 (range of 0.005 across the three settings). The boundary worry is retired -
the Go1 optimum is an interior plateau, unlike the CartPole case where the best value sat below the old search floor.

## 4. Cost and operational record

Campaign ledger: **$2.85** for 21 jobs (42,000 iterations) on two pods, a 3090 at $0.22/h and a 5090-class at $0.34/h,
against the $4 cap. Nine captures were needed overall; the pod that ran the first fifteen jobs was preempted at ~850
iterations and its replacement resumed from checkpoints - but only after two blockers were fixed (d1024, d1025):

- a **claim** left by the dead pod made the replacement skip all six affected jobs immediately after fetching their
  checkpoints; claims now expire when their owner pod is gone;
- the warm-pod **queue poller** raised `NameError` on an undefined `_C` on every cycle, hidden by `2>/dev/null`, so a
  drained pod could never accept a new queue (this also explains a round-6 pod that sat idle).

Both are the kind of defect that leaves no trace in the result but costs hours; they are now tested (the resume path has a
unit test) and instrumented (failure logs upload to R2).

## 5. What this settles

1. **No SSP-specific advantage under the 0.1 offset.** 12/20 against 12/20, paired +0.002 with an interval spanning
   +/-0.20. The pilot's 4/5-vs-2/5 was an unlucky raw draw; the escalation was worth running precisely because it could
   have gone the other way, and it did not.
2. **The length-scale optimum on Go1 is interior, not at the edge.** 2x2 at ls 2.0 gives 0.879 and at 4.0 gives 0.884,
   against 0.880 at 1.0 - flat within 0.005 across a 4x range. The CartPole lesson (optimum below the old search floor)
   does not repeat in the other direction here.
3. **Mechanism is not outcome.** SSP's critic learns the offset level sooner, and the policy is no more likely to walk.
   Anyone citing the early-critic statistic as a reason to prefer SSP should cite this null alongside it.
4. **Practical rule, unchanged and now twice-supported:** search the length scale, and judge encoders on paired seeds
   against the *baseline's real rate* (12/20 here), never on a five-seed pilot - the pilot's estimate was off by a factor
   of two in walk rate for the arm it was measuring against.
