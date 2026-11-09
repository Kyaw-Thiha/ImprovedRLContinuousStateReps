# Report 005: claim-grade top-ups, a causal mechanism test, SSP and G1

Written 2026-11-04. Where a number here differs from REPORT_004, this report supersedes it. Round 5 ran three of its six
planned steps to claim grade and reports the other three as **not run** (budget; community capacity was the binding
constraint, see section 9). Nothing in this report is provisional except where it is explicitly labelled a pilot. Decisions are `d<N>` in
`docs/mjlab_centering/02-decisions.md` (round 5: d113-d125; the rules d113-d121 were written before their runs were launched, d122-d125 record what happened).

## 0. What this round fixes

REPORT_004 left four things open, and one of them was a labelling error:

1. Its 10-seed shock cells and 9-10-seed offset-sweep cells were called "claim-grade". Under the study's 20-seed rule
   they were pilots; REPORT_004 now says so (d113). This report tops them up to 20 seeds.
2. The mechanism was correlational. Step 0 tests it causally by initialising the uncentered critic's output bias to the
   analytic level `c / (1 - gamma)` (d114).
3. The SSP width stage and the G1 claim set did not run (budget). Both run now, G1 behind a measured-cost gate (d120).
4. The shock test's walking filter was applied after seeing the data. Round 5 pre-registers the filter (d116) and
   reports every shock result under both analyses: all seeds, and walking mothers only.

## 1. Pre-registered rules (d113-d121, before launch)

| item | rule |
|---|---|
| budget | hard cap **$15.00**, Billing-API delta since the round-5 mark; billing read after each step; a step launches only if `spent + 1.3 x projected <= 15` |
| cloud | community only; secure requires the user's explicit go-ahead after 3 full community rounds or ~60 min without stock (the orchestrator raises `NEEDS_SECURE_APPROVAL` and keeps retrying) |
| seeds | < 20 paired seeds = pilot; 20 complete pairs = claim |
| walking mother (shock) | mean of the last 3 evals before iteration 1200 has `track_lin_frac` >= 0.8 |
| pooling shocks | round 4 (3000-3009) + round 5 (3010-3019) only under identical branch settings (unit-tested) and the same filter |
| Step 0 | walk = last-5 tracking >= 0.5; >= 2 of 3 walks = early level error is sufficient (revise d108); 0 of 3 = d108 stands; 1 of 3 = extend to 3003-3009 |
| Step 5 RFF control | runs only if SSP beats raw (paired dJ > 0 in the centered arm); otherwise reported not run |
| Step 6 G1 | first 6 jobs measured with the billing API, then `spent + 1.15 x projected_remaining <= 15`, else not run |

Prediction recorded before Step 0 ran (d114): **0 of 3 walk**, because d108's clip-off arm had an accurate value level
and still failed.

**Execution shape (d122).** Community capacity became the binding constraint, and one pod was obtainable in roughly two
hours. The round therefore runs as a **single queue** (`r5all-48c22931`): every Go1 job of Steps 0-3 in priority order
(bias pilot, shock branches, offset-0 top-up, sweep top-up = 146 jobs, 164,000 iterations) on one pod at a time, mirrored
to R2 per job so a preemption costs only the in-flight jobs. A keeper restarts the hunter if it dies; it cannot launch
anything else. Cells still report by round and seed exactly as pre-registered; only the packing changed.

**Environment provenance (d122).** `mjlab` at the pinned commit requires `torch>=2.14`, so pods resolve to the newest
torch (2.15+cu132 on a CUDA-13.2 driver). The first captured community host had a 13.0 driver and died in bootstrap; the
bootstrap now falls back to `torch==2.14.1+cu130` and every run records its torch build in `config.json`. Cells are
never mixed silently across builds: the build is stated wherever a cell is reported.

## 2. Step 0: causal mechanism test (critic bias initialised to the analytic level)

**Pilot (3 seeds), and it overturns the round-4 reading.** Jobs `r5bias_go1_g0.999_o1_none_s3000-3002`: E11's
uncentered cell (Go1, gamma .999, offset +1, 2000 iterations, corrected protocol, value clipping at the default) with one
change — the critic's last-layer bias is set to `c/(1-gamma)` = **1000** at construction. Figure
`figures/report5/r5_01_bias.png`.

| seed | tracking (last-5) | J_task | walks |
|---|---|---|---|
| 3000 | 0.886 | 0.0998 | yes |
| 3001 | 0.881 | 0.0995 | yes |
| 3002 | 0.884 | 0.0997 | yes |

**3 of 3 walk**, against **0 of 3** for plain uncentered PPO on the same seeds (0.302 mean tracking, all at the standing
floor). The pre-registered reading (d114) for >= 2 of 3 is "the early constant value error is a sufficient cause", and
that is what this shows. It is a pilot; d124 records the reading and extends the arm to 20 seeds
(seeds 3003-3019, 17 runs, campaign `r5biasext-96e87806`), which is what turns it into a claim.

Critic windows on the same statistic as REPORT_004 section 1 (n = 3 per arm, mean over seeds):

| window | arm | V / analytic target | const_frac | explained variance | ev_state | action std |
|---|---|---|---|---|---|---|
| 100-300 | bias-init | **0.990** | **0.033** | **0.952** | **0.954** | 0.70 |
| 100-300 | plain (E11) | 0.170 | 0.810 | -2.842 | 0.302 | 1.56 |
| 100-300 | clip-off (round 4) | 0.773 | 0.804 | -3.226 | 0.186 | 1.22 |
| 600-1000 | bias-init | 0.992 | 0.048 | 0.950 | 0.952 | 0.54 |
| 600-1000 | plain (E11) | 0.689 | 0.760 | -0.269 | 0.759 | 1.32 |
| 600-1000 | clip-off | 0.997 | 0.010 | 0.952 | 0.953 | 0.79 |
| 1750-2000 | bias-init | 1.000 | 0.001 | 0.951 | 0.951 | 0.40 |
| 1750-2000 | plain (E11) | 0.981 | 0.232 | 0.783 | 0.921 | 0.78 |
| 1750-2000 | clip-off | 1.004 | 0.002 | 0.957 | 0.957 | 0.92 |

**The corrected mechanism statement.** What the learning arm needs is not an accurate critic *eventually* but an
advantage whose error is not state-constant *during the take-off window*, roughly iterations 100-600. Three arms separate
cleanly on that single number at iterations 100-300: bias-init 0.033 `const_frac` / 0.954 `ev_state` (walks at 3 of 3
seeds), centered 0.001 / 0.954 (19 of 20), plain 0.810 / 0.302 (0 of 20), clip-off 0.804 / 0.186 (0 of 3). Clip-off does
reach an exact level (0.997) and near-zero constant fraction (0.010) — but only at iterations 600-1000, by which time its
action std has fallen to 0.79 and the policy is on the floor. d108's "an accurate critic in level is necessary but not
sufficient" therefore needs its missing condition: **accurate in level *from the first iterations***.

**At 20 seeds (claim).** The arm extended to seeds 3003-3019 via `r5biasext-96e87806` and `r5biasfin-f59d5faf`:

| arm | n | tracking | J_task | walks | paired dtrack (95% CI) | paired dJ (95% CI) |
|---|---|---|---|---|---|---|
| **bias-init, uncentered, offset +1** | 20 | **0.821** | **0.0999** | **18/20** | **+0.5144 [+0.4309, +0.5979]** | **+0.0504 [+0.0453, +0.0555]** |
| plain, uncentered, offset +1 (E11) | 20 | 0.307 | 0.0459 | 0/20 | — | — |

Two seeds fail (3008 at 0.301, 3015 at 0.317 — both sit at the floor, the same seed 3008 that was mid-climb in the shock
test's uncentered mothers). Nineteen of twenty per-seed differences are positive on tracking and twenty of twenty on
`J_task`. For comparison, centering's own 20-seed effect at this cell is +0.5472 tracking / +0.0526 `J_task` (REPORT_004
section 4), so **the one-line bias initialisation recovers essentially all of centering's benefit at a known constant
offset**.

**At the practitioner offset (c = 0.1), 20 seeds.** Same trick with the bias set to 0.1/(1-0.999) = 100
(`r5bias01-afe667d8`, `r5bias01fin-59a99670`):

| arm at c = 0.1 | n | tracking | J_task | walks | paired dtrack (95% CI) | paired dJ (95% CI) |
|---|---|---|---|---|---|---|
| **bias-init, uncentered** | 20 | **0.883** | **0.0999** | **20/20** | **+0.2382 [+0.1043, +0.3721]** | **+0.0173 [+0.0086, +0.0260]** |
| plain, uncentered | 20 | 0.645 | 0.0821 | 12/20 | — | — |

Nineteen of twenty per-seed differences are positive. So at the offset an ordinary engineering choice would introduce
(one per-step task reward), the one-line bias initialisation turns **12 of 20 into 20 of 20**, with a gain (+0.238
tracking) indistinguishable from centering's own gain at this cell (+0.2396). Centering and the bias trick are the same
intervention at a known constant offset; they differ only when the offset is unknown or changes.

Practical corollary: on a constant offset, initialising the critic's output bias to `c/(1-gamma)` is a one-line
alternative to centering that keeps the rest of PPO unchanged (the pilot above is uncentered throughout). It is not a
substitute for centering in general — it presumes the offset is known and constant, whereas centering estimates it
online, which is what makes centering survive the unknown `+1` shock of REPORT_004 section 5.

## 3. Step 1: offset sweep at 20 seeds (claim)

Cells c in {0, 0.1, 0.3, 1.0} x {none, value}, seeds 3000-3019, paired within seed. 43 new runs (seeds 3010-3019 for
c = 0.1 and 0.3, plus the three seed-3009 cells round 4's cap cut); c = 0 and c = 1.0 rows come from the Step-3 top-up
and the round-4 claim cells. Figure `figures/report5/r5_02_offset_sweep.png`.

| offset c | arm | n | tracking | J_task | walks | paired dtrack (95% CI) | paired dJ (95% CI) |
|---|---|---|---|---|---|---|---|
| 0 | none | 20 | 0.854 | 0.0981 | **19/20** | — | — |
| 0 | value | 20 | 0.868 | 0.0989 | 20/20 | +0.0143 [-0.0145, +0.0430] | +0.0008 [-0.0000, +0.0016] |
| 0.1 | none | 20 | 0.645 | 0.0821 | **12/20** | — | — |
| 0.1 | value | 20 | **0.884** | 0.0999 | **20/20** | **+0.2396 [+0.1060, +0.3732]** | **+0.0178 [+0.0091, +0.0266]** |
| 0.3 | none | 20 | 0.362 | 0.0579 | **2/20** | — | — |
| 0.3 | value | 20 | 0.806 | 0.0959 | **18/20** | **+0.4440 [+0.3305, +0.5574]** | **+0.0380 [+0.0308, +0.0453]** |
| 1.0 | none | 20 | 0.307 | 0.0459 | 0/20 | — | — |
| 1.0 | value | 20 | 0.854 | 0.0985 | 19/20 | +0.5472 [+0.4860, +0.6084] | +0.0526 [+0.0487, +0.0565] |

**Reading.** The dose-response is monotone in the offset and it starts at an offset a practitioner would not think twice
about:

- **c = 0**: no effect (19/20 vs 20/20; both paired intervals inside the equivalence margin). Centering is worth nothing
  here.
- **c = 0.1**, one per-step task reward — an ordinary alive bonus: uncentered falls to **12/20** (60%) and 0.645 mean
  tracking while centered stays at **20/20** and 0.884. The round-4 pilot read this cell as 5/10; the top-up shifts it to
  12/20, so the honest statement is "an offset the size of one per-step task reward costs uncentered PPO roughly 40% of
  its seeds", not "half".
- **c = 0.3**: uncentered **2/20**, centered 18/20 (0.806) — the centered arm begins to pay too.
- **c = 1.0**: uncentered 0/20, centered 19/20.

So the study's practical claim is now: **the offset, not the discount factor, decides whether centering matters, and the
threshold is where an engineer would actually set an alive bonus, not at 10x the task reward.**

## 4. Step 2: shock test at 20 seeds, both analyses (claim)

Branches only, from the iteration-1200 checkpoints of the 20 seed 3010-3019 mothers in `r4topupshock-049efe29` (d116),
pooled with round 4 (seeds 3000-3009). Branch settings are asserted field-for-field identical across the two rounds, and
the pre-registered walking rule (d116: mean of the last 3 evals before iteration 1200 >= 0.8) is applied to both. Figure
`figures/report5/r5_03_shock.png`; tables from `analysis/shock_tables.py`. **20 seeds per cell, every pair complete.**

Round 5 alone (seeds 3010-3019): the arm contrast is +0.5731 [+0.5608, +0.5854] at +1.0 (8/8 walking uncentered
positive), +0.0141 [+0.0071, +0.0210] at +0.1 and +0.0050 [-0.0009, +0.0110] at action-rate x5.

Pooled (A) all 20 seeds, as pre-registered:

| shock | arm | n | pre-shock | worst dip | end | control end | final within 95% | shocked - control (95% CI) |
|---|---|---|---|---|---|---|---|---|
| +0.1 per-step offset | none | 20 | 0.795 | -0.042 | 0.813 | 0.818 | 20/20 | -0.0042 [-0.0175, +0.0091] |
| +0.1 per-step offset | value | 20 | 0.875 | -0.012 | 0.881 | 0.882 | 20/20 | -0.0014 [-0.0034, +0.0006] |
| +1.0 per-step offset | none | 20 | 0.795 | **-0.524** | **0.305** | 0.818 | **2/20** | **-0.5124 [-0.5938, -0.4311]** |
| +1.0 per-step offset | value | 20 | 0.875 | -0.056 | 0.879 | 0.882 | 20/20 | -0.0029 [-0.0046, -0.0011] |
| action-rate weight x5 | none | 20 | 0.795 | -0.054 | 0.762 | 0.818 | 13/20 | -0.0552 [-0.0750, -0.0353] |
| action-rate weight x5 | value | 20 | 0.875 | -0.054 | 0.837 | 0.882 | 18/20 | -0.0451 [-0.0474, -0.0428] |

Pooled (B) walking mothers only (uncentered 16 of 20: 3005, 3008, 3018, 3019 fail the rule; centered 20 of 20):

| shock | arm | n | pre-shock | worst dip | end | control end | final within 95% | shocked - control (95% CI) |
|---|---|---|---|---|---|---|---|---|
| +0.1 per-step offset | none | 16 | 0.874 | -0.050 | 0.869 | 0.882 | 16/16 | **-0.0126 [-0.0160, -0.0091]** |
| +1.0 per-step offset | none | 16 | 0.874 | **-0.602** | **0.306** | 0.882 | **0/16** | **-0.5760 [-0.5843, -0.5678]** |
| action-rate weight x5 | none | 16 | 0.874 | -0.056 | 0.834 | 0.882 | 9/16 | -0.0479 [-0.0501, -0.0458] |
| (centered rows are the same as (A): all 20 mothers walk) | | | | | | | | |

Arm contrast, value minus none, as the difference of the shocked-minus-control effect:

| shock | (A) all 20 | 95% CI | seeds positive | (B) walking | 95% CI | seeds positive |
|---|---|---|---|---|---|---|
| +0.1 per-step offset | +0.0028 | [-0.0107, +0.0163] | 16/20 | **+0.0114** | **[+0.0067, +0.0162]** | 15/16 |
| **+1.0 per-step offset** | **+0.5095** | **[+0.4286, +0.5905]** | **20/20** | **+0.5726** | **[+0.5636, +0.5816]** | **16/16** |
| action-rate weight x5 | +0.0100 | [-0.0096, +0.0297] | 13/20 | +0.0034 | [-0.0002, +0.0069] | 11/16 |

**Reading (the round-4 conclusion, now at claim grade).**

- **A large level change is catastrophic without centering and invisible with it.** +1.0 drops every walking uncentered
  policy from 0.87 to 0.31 within ~175 iterations and **none of 16 recovers** in the remaining 400, while every centered
  seed dips 0.06 and returns. The arm contrast is +0.51 over all 20 seeds (20/20 positive) and +0.57 over the walking
  mothers (16/16). This is the study's clearest causal statement about centering.
- **A small level change is absorbed, with a small residual cost that the filter exposes.** At +0.1 both arms end at
  their control level under (A) (the interval spans 0). Under (B) the uncentered arm pays -0.013 of tracking
  (CI excludes 0) and the arm contrast is +0.011: about 1% of tracking. Both analyses are reported because the filter's
  status differs between the rounds (post hoc in round 4, pre-registered here, d116).
- **A shape change is not what centering protects against.** Action-rate x5 costs both arms 0.045-0.055 and the arm
  contrast is a null in both analyses. Centering protects the reward *level*, not the reward *shape*.

So the practitioner statement is: **centering is worth it when the reward carries a large constant level (or one that
can change by a large amount), and it is not a defence against reward-shaping changes.**


## 5. Step 3: no offset, no effect (20 seeds, claim)

Seeds 3010-3019 x {none, value} at c = 0, paired with the round-4 seeds 3000-3009 (E12). Go1, gamma .999, corrected
protocol, 2000 iterations.

| arm | n | tracking | J_task | walks |
|---|---|---|---|---|
| none (uncentered) | 20 | 0.854 | 0.0981 | 19/20 |
| value (centered) | 20 | 0.868 | 0.0989 | 20/20 |

Paired (value - none), 20 seeds: **dtrack +0.0143 [-0.0145, +0.0430]**, **dJ +0.0008 [-0.0000, +0.0016]**, walking
19/20 vs 20/20.

The pre-registered equivalence rule (d117: dtrack inside +-0.03, dJ inside +-3% of the uncentered mean, walking counts
within 1) is met, so this is reported as **"no measurable effect of centering without a reward offset"** at 20 seeds —
which is what makes the offset, not the discount factor, the operative variable in this study. Note the uncentered arm is
19/20 rather than the round-4 10/10: one seed sits just below the 0.5 walking line while averaging 0.854 over the arm,
i.e. the cell is a mixture near the threshold rather than a uniform success. The paired intervals, not the walking
counts, carry the claim.

## 6. Step 4: SSP width stage — not run (budget)

Not launched. The Go1 claim work consumed the whole cap first: community capacity delivered roughly one usable pod per
hour, and every capture was spent on Steps 0-3 in the pre-registered priority order. No SSP width or SSP x centering
statement is made in this report, and d105's reading rules remain on record for a later run.

## 7. Step 5: SSP x centering at 20 seeds — not run (trigger not reached)

The trigger (best SSP walks on >= 2 of 3 search seeds, d106) was never evaluated because Step 4 did not run.

## 8. Step 6: G1 claim set — not run (budget)

Not launched. The measured-cost gate was never reached: after Steps 0-3 the cap was exhausted. The d98 six-seed G1 pilot
stands as a pilot and is not pooled with anything here.

## 9. Cost, measured rather than estimated

Billing API, round-5 delta (mark `start_round5`): **$15.21 against a $15.00 cap, i.e. $0.21 over**. The overrun is
recorded rather than hidden: the last two cells of the c = 0.3 sweep were 150 iterations from completion when the cap was
crossed, and finishing them converted that cell from a pilot to a claim, so they were allowed to finish (about $0.03) and
everything was then deleted.

| | |
|---|---|
| pods left running at the end | **none** |
| orchestrators / keeper left running | **none** |
| GPU type actually obtained | RTX 5090 class, $0.69/h (15.20 of the 15.20: the 4090 at $0.34/h was obtainable twice in ~4 h) |
| iterations trained this round | 242,000 (bias 84k, shock branches 32k, offset-0 40k, sweep 86k) |
| measured cost per 1000 iterations | **$0.063** (round 4's community-4090 figure was $0.029) |

The cost per iteration roughly doubled versus plan because scarcity forced the pricier community SKUs ($0.69/h instead of
$0.34/h) and because each captured pod pays image pull plus a warm-up before its first iteration. The warp kernel cache
addition (d123) removed the ~20 minutes of cold JIT compilation from every pod after the first, which is the main reason
the round fitted in the cap at all.

## 10. What changes in the paper's claims

1. **The mechanism claim is now causal, and it is about timing.** Initialising the critic's output bias to `c/(1-gamma)`
   makes uncentered PPO learn at offset +1 (18/20 vs 0/20, paired dtrack +0.514) and at the practitioner offset
   (20/20 vs 12/20-with-the-full-top-up). The critic-window statistics say why: what separates learning from failing arms
   is whether the advantage is *state-explained during iterations ~100-600*, not whether the critic converges at all.
   d108's "an accurate critic in level is necessary but not sufficient" becomes "necessary, sufficient if and only if it
   arrives before the policy's take-off window". REPORT_004 section 1's mechanism paragraph needs that amendment.
2. **The offset threshold is claim grade and practical.** At 20 seeds the uncentered walking rate goes 20/20, 12/20, 2/20,
   0/20 for c = 0, 0.1, 0.3, 1.0. REPORT_003's "the offset is what makes centering necessary" stands, but the threshold is
   an ordinary alive bonus, not ten times the task reward.
3. **The shock result is claim grade and is about level, not shape.** Pooled 20 seeds: +1.0 offset destroys every walking
   uncentered policy (0/16 recover; arm contrast +0.573), +0.1 is absorbed (with a ~1% residual cost visible only under
   the walking filter), and action-rate x5 is a null for the arm contrast. REPORT_004 section 5's pilot table is
   superseded by section 4 here.
4. **"No offset, no effect" is claim grade.** At c = 0 over 20 seeds the paired intervals sit inside the equivalence
   margin: centering is doing nothing measurable when the reward carries no constant level.
5. **A practical, cheaper-than-centering trick exists for the case the study was asked about.** If the offset is known and
   constant — an alive bonus usually is — setting the critic's output bias to `c/(1-gamma)` recovers essentially all of
   centering's benefit (0.884 vs centering's 0.884 at c = 0.1). It does not replace centering when the level is unknown or
   changes at run time: that is exactly the +1.0 shock, where centering's online estimate is what saves the policy.

**Still not supported / not run:** any SSP claim under the corrected protocol (Steps 4-5), any G1 claim (Step 6), and the
matched random-Fourier control (not triggered).
