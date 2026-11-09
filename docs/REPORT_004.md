# Report 004: mechanism, offset size, shocks, and SSP width under the corrected protocol

Written 2026-11-03. Continues REPORT_003, whose numbers stand unless a section here supersedes one explicitly. Decisions
are cited as `d<N>` from `docs/mjlab_centering/02-decisions.md`; the round-4 decisions are d101-d107 (written
before their runs were launched) and d108-d112 (results, and corrections made after re-checking). Every figure is regenerated from the R2 mirrors by the script named beside it.

**Labelling note added in round 5 (d113).** The seed rule is 20 paired seeds for a claim. In this report the c = 1.0
sweep row, the Go1 main effect and the REPORT_003 numbers it cites are claim grade; **the 10-seed shock cells and the
9-10-seed offset-sweep cells (c = 0, 0.1, 0.3) are pilots**, and the section 5 headline "claim" label is withdrawn. The
topped-up 20-seed numbers are in `REPORT_005.md`, which supersedes the pilot tables here wherever they differ. Section 5
was also corrected in round 5 (the walking-mother count; see there).

## 0. Cost, measured rather than estimated

The RunPod billing API, not the orchestrator, is the source of truth from here on (d101).

| day | pods | network volume | total |
|---|---|---|---|
| 2026-10-31 | 4.33 | 0.47 | 4.80 |
| 2026-11-01 | 21.20 | — | 21.20 |
| 2026-11-02 | 31.61 | — | 31.61 |
| **study total to 11-02** | | | **57.14** |

The orchestrator reported $18.88 for the same window: it priced every pod at $0.34/h, while the 4090s were only
obtainable on secure cloud at $0.89/h. Measured rate on that path: 20 Go1 runs of 2000 iterations, 12.8 min each,
$0.185 per run. Community cloud (4090 $0.34/h, A6000 $0.33, 3090 $0.22) is preferred from here on and is retried for
three full rounds before secure is used; every job mirrors to R2, so a lost community pod costs at most one run.
Hard cap for this round: **$20**, checked against the billing API after every step.

Measured cost per 1000 training iterations on this task (2048 envs, one pod, six processes, Go1):

| GPU | cloud | $/h | aggregate it/s | $ per 1000 it |
|---|---|---|---|---|
| RTX 4090 | secure | 0.89 | 3.3 | **0.075** |
| A40 | secure | 0.59 | 1.8 | 0.091 |
| RTX 4090 | community | 0.34 | 3.3 | **0.029** |

Two consequences that changed the plan mid-round. First, the GPU saturates at about 3.2 iterations/s, so six
processes cost the same as three and parallelism has to come from more pods, not more processes. Second, the
cheaper-looking A40 is the *worse* buy once six processes are running: it is GPU-bound at 1.8 iterations/s. The
launcher therefore prefers the 4090 on secure cloud over the A40, and community stock first of all. One job per pod
(no concurrency) is never worth it: measured $0.149/1000 it on an A40 that way, against $0.091 with six.

## 1. The mechanism, measured on E11

`analysis/mechanism_e11.py`, figure `figures/report4/r4_mech_e11.png`. Go1, gamma .999, offset +1, 20 seeds per arm,
corrected protocol. Only the per-iteration diagnostics the trainer already logs are used.

| window | arm | mean V / analytic target | const_frac | explained variance | state-dependent error | action std |
|---|---|---|---|---|---|---|
| 100-300 | no centering | 0.181 | 0.803 | -2.191 | 9.844 (se 0.136) | 1.54 |
| 100-300 | value centering | n/a | 0.001 | 0.954 | 0.307 (se 0.004) | 0.65 |
| 600-1000 | no centering | 0.739 | 0.672 | -0.188 | 2.077 (se 0.207) | 1.25 |
| 600-1000 | value centering | n/a | 0.003 | 0.952 | 0.128 (se 0.013) | 0.54 |
| 1750-2000 | no centering | 0.986 | 0.165 | 0.867 | 0.689 (se 0.083) | 0.86 |
| 1750-2000 | value centering | n/a | 0.000 | 0.951 | 0.092 (se 0.014) | 0.42 |

State-dependent error, uncentered over centered: **32x** at iterations 100-300, 16x at 600-1000, **6.7x** at
1750-2000 (95% CI 4.7-9.5 on the 20 paired seeds).

**Reading.** The uncentered critic is at 18% of its analytic target early, with explained variance -2.19: the batch
advantages are dominated by a constant prediction error, i.e. they are close to noise, exactly while the centered arm
takes off (iterations 300-600). The uncentered critic only reaches 98.6% of its target at iterations 1750-2000, by
which time its action std has fallen from 1.54 to 0.86 and the policy never recovers. So centering here acts on the
*critic's transient*: it removes the constant part of the error in the first iterations and leaves the state-dependent
part that actually carries the policy gradient.

Two caveats stay attached to this. The logged error is the bootstrapped advantage (rsl-rl returns = advantages + the
critic's own values), not a Monte-Carlo value error, so "state-dependent error" is really the variance of the advantage
that is not explained by the state. And after about iteration 300-600 the two arms are running different policies;
iterations 100-300 are the clean like-for-like window.

## 2. Code corrections made before this round

- **Absolute offset shocks.** `extra.shock_delta` is now a per-step addition. The old form, offset x (factor - 1), is
  identically 0 when the pre-shock offset is 0, which is exactly the Step 3 mothers.
- **Iteration-1200 checkpoints** are uploaded to R2 (`ckpt_it<N>.pt`) when a run asks for them, so a shock branch can
  resume on a different pod than its mother.
- **Random-Fourier seed follows the run seed.** The RFF control used a fixed seed 0 in every run, so the three "seeds"
  of that arm shared one random draw.
- **`podrun` writes `<campaign>/runs/<job>/`**, like the worker path, skips jobs R2 already marks DONE, and fetches a
  mother checkpoint from R2 when the mother ran elsewhere. The old layout double-prefixed the keys
  (`mjlab_ssp_centering/mjlab_ssp_centering/...`), which is why no analysis could see those runs.
- **Community-first launching**, with three full rounds over the community candidates before secure is attempted.

Verified locally before any spend: unit tests for the manifests, the absolute shock (offset 0 -> 0.1 at the branch
point, control stays 0), the RFF seed, and an end-to-end mother-branch-control smoke run through `podrun` against R2,
including the checkpoint round trip.

## 3. Step 1: clip-off pilot, which doubles as a mechanism test

`manifests/clip1.json`, 3 seeds (3000-3002, paired with the E11 cells), corrected protocol, offset +1, gamma .999,
2000 iterations, no centering, PPO's value clipping off. 3 seeds: a **pilot**, not a claim.

| seed | tracking | J_task | walks |
|---|---|---|---|
| 3000 | 0.302 | 0.0676 | no |
| 3001 | 0.312 | 0.0534 | no |
| 3002 | 0.309 | 0.0551 | no |

**0 of 3 walk**, so decision d100's rule fires the other way: the 20-seed clip-off run is skipped and
"centering vs clip-off" is reported as legacy-only supporting evidence plus this pilot.

The value is in the critic windows (`figures/report4/r4_mech_clip1.png`), which were pre-registered as a mechanism test
(d102):

| window | arm | mean V / target | const_frac | ev | state part | ev_state | action std |
|---|---|---|---|---|---|---|---|
| 100-300 | clip-off | 0.773 | 0.804 | -3.226 | 0.968 | 0.186 | 1.22 |
| 100-300 | plain (E11) | 0.181 | 0.803 | -2.191 | 9.844 | 0.405 | 1.54 |
| 100-300 | centered (E11) | n/a | 0.001 | 0.954 | 0.307 | 0.954 | 0.65 |
| 600-1000 | clip-off | 0.997 | 0.010 | 0.952 | 0.370 | 0.953 | 0.79 |
| 1750-2000 | clip-off | 1.004 | 0.002 | 0.957 | 0.491 | 0.957 | 0.92 |

Clip-off's critic learns the *level* much faster than plain PPO's (77% of target at iterations 100-300 against 18%), and
by iterations 600-1000 it is essentially exact (99.7%, explained variance 0.95). The robot still spends all 2000
iterations at the standing floor. So a slow critic is **not** the whole mechanism, and neither is exploration collapse:
the centered arm has the lowest action std of the three and is the only one that learns.

What does separate them is the *state-explained share* of the advantage. PPO standardises advantages each iteration
(`(A - mean) / std`), so a constant component is divided out and cannot be the mechanism by itself; what matters is how
much of the standardised advantage is state signal. At iterations 100-300 that share is 0.954 (centered), 0.405 (plain)
and 0.186 (clip-off). Centering is the only intervention here that makes the advantage state-explained from the first
iterations, and it is the only one after which the robot walks.

## 4. Step 2: does offset size matter

Figure `figures/report4/r4_01_offset_sweep.png`. Go1, gamma .999, corrected protocol, 2000 iterations, c in
{0.1, 0.3} x {none, value}, seeds 3000-3009, paired within seed with the earlier c = 0 (E12) and c = 1 (E11) cells on
the same seeds.

| offset c | arm | n | tracking | J_task | walks | paired dtrack (95% CI) | paired dJ (95% CI) |
|---|---|---|---|---|---|---|---|
| 0 | none | 10 | 0.882 | 0.0993 | 10/10 | — | — |
| 0 | value | 10 | 0.884 | 0.1000 | 10/10 | +0.0019 [-0.0019, +0.0058] | +0.0007 [-0.0001, +0.0014] |
| 0.1 | none | 10 | 0.583 | 0.0797 | **5/10** | — | — |
| 0.1 | value | 9 | **0.885** | 0.0998 | **9/9** | **+0.3287 [+0.0984, +0.5589]** | **+0.0212 [+0.0078, +0.0345]** |
| 0.3 | none | 9 | 0.431 | 0.0627 | **2/9** | — | — |
| 0.3 | value | 9 | **0.777** | 0.0942 | **8/9** | **+0.3459 [+0.1294, +0.5624]** | **+0.0315 [+0.0170, +0.0460]** |
| 1.0 | none | 20 | 0.307 | 0.0459 | 0/20 | — | — |
| 1.0 | value | 20 | 0.854 | 0.0985 | 19/20 | +0.5472 [+0.4860, +0.6084] | +0.0526 [+0.0487, +0.0565] |

Three of the 40 cells did not finish before the cap (one centered cell at c = 0.1, one uncentered and one centered cell
at c = 0.3), so the c = 0.1 and c = 0.3 rows have 9-10 seeds and both contrasts are paired on 9. Seeds with a missing
partner are dropped from the paired contrast, not imputed. **Pilot:** the c = 0, 0.1 and 0.3 rows have 9-10 seeds, below
the 20-seed claim rule; only the c = 1.0 row is claim grade. The direction and size of the c = 0.1 effect were the reason
for the round-5 top-up.

**Reading.** The effect is graded in the offset, and it starts much smaller than the study's +1.

- At **c = 0** there is no effect at all: both arms walk on every seed (+0.0019 tracking).
- At **c = 0.1**, an alive bonus of about the size of the per-step task reward, uncentered PPO walks on **5 of 10**
  seeds and averages 0.583 (bimodal: some seeds find the walk, others sit at the floor), while centered walks **9 of 9**
  at 0.885. That is the practitioner-relevant cell: a 0.1 reward offset is a completely ordinary engineering choice,
  and it already halves the reliability of uncentered PPO.
- At **c = 0.3** uncentered is down to 2 of 9 and centered to 8 of 9 (0.777) — the centered arm starts to feel it too.
- At **c = 1.0** uncentered never walks (0 of 20).

So REPORT_003's "the offset is what makes centering necessary" survives, but the threshold is much lower than the
evidence there suggested: **an offset of about one per-step task reward is enough to break uncentered PPO on half the
seeds.** d103's top-up rule fired at both c = 0.1 (walking counts differ by 5, CI excludes 0) and c = 0.3 (counts differ
by 6, CI excludes 0); neither top-up was run, for budget (d110/d111).

## 5. Step 3: the proper shock test

Figure `figures/report4/r4_03_shock.png`. Mothers: Go1, **offset 0**, gamma .999, 1200 iterations, both arms, 10 seeds
(3000-3009). At iteration 1200 each mother branches into four 400-iteration continuations that share its weights: three
shocks and one unshocked control. This is the test REPORT_003 could not run, because there the uncentered arm had never
learned to walk and so there was nothing to shock. **Pilot** (10 seeds per cell, paired within seed; the 20-seed version is in REPORT_005).

**Two analyses, side by side.** (A) is **all 10 seeds, as pre-registered** (d104). (B) restricts each arm to the mothers
that had learned to walk by iteration 1200. **The restriction was decided after seeing the data** (round 4 dropped the
uncentered mothers that had not learned), so it is exploratory in this report. In round 5 it became a pre-registered rule
(d116: a mother walks iff the mean of its last 3 evals before iteration 1200 has `track_lin_frac` >= 0.8) and (B) below
applies that rule retrospectively. Both tables are recomputed by `analysis/shock_tables.py` from
`results/report4_shock.json`. "Final within 95%" means the mean of the last 5 evals is at least 95% of the pre-shock level.

**Correction (d116).** The first version of this section said two of the ten uncentered mothers "had not learned" (seed
3005 at 0.30, seed 3008 at 0.55) and then printed a 9-mother table: that table dropped only 3005 and kept 3008 (pre-shock
mean 0.842 = nine mothers including 0.55). Under the 0.8 rule both are excluded, so there are **8** walking uncentered
mothers, and (B) is computed on 8. The 9-mother table gave the same picture (+1.0 uncentered: end 0.308, 0/9 recovered).

(A) all 10 seeds, as pre-registered:

| shock | arm | n | pre-shock | worst dip | end (1599) | unshocked control end | final within 95% | shocked - control (95% CI) |
|---|---|---|---|---|---|---|---|---|
| +0.1 per-step offset | none | 10 | 0.788 | -0.041 | 0.813 | 0.812 | 10/10 | +0.0004 [-0.0233, +0.0240] |
| +0.1 per-step offset | value | 10 | 0.875 | -0.011 | 0.881 | 0.883 | 10/10 | -0.0020 [-0.0056, +0.0015] |
| +1.0 per-step offset | none | 10 | 0.788 | -0.506 | 0.307 | 0.812 | 1/10 * | **-0.5047 [-0.6317, -0.3778]** |
| +1.0 per-step offset | value | 10 | 0.875 | -0.056 | 0.881 | 0.883 | 10/10 | -0.0022 [-0.0048, +0.0004] |
| action-rate weight x5 | none | 10 | 0.788 | -0.056 | 0.752 | 0.812 | 5/10 | -0.0602 [-0.1007, -0.0197] |
| action-rate weight x5 | value | 10 | 0.875 | -0.056 | 0.836 | 0.883 | 9/10 | -0.0466 [-0.0500, -0.0431] |

\* the one "recovered" uncentered mother is seed 3005, which never learned (pre-shock 0.30), so "95% of its pre-shock
level" is trivially met at the floor. This is why the (A) recovery counts are read with (B).

(B) walking mothers only (>= 0.8; uncentered 8 of 10, centered 10 of 10), exploratory in this report:

| shock | arm | n | pre-shock | worst dip | end (1599) | unshocked control end | final within 95% | shocked - control (95% CI) |
|---|---|---|---|---|---|---|---|---|
| +0.1 per-step offset | none | 8 | 0.878 | -0.048 | 0.871 | 0.882 | 8/8 | -0.0109 [-0.0168, -0.0051] |
| +0.1 per-step offset | value | 10 | 0.875 | -0.011 | 0.881 | 0.883 | 10/10 | -0.0020 [-0.0056, +0.0015] |
| +1.0 per-step offset | none | 8 | 0.878 | **-0.593** | **0.307** | 0.882 | **0/8** | **-0.5748 [-0.5906, -0.5590]** |
| +1.0 per-step offset | value | 10 | 0.875 | -0.056 | 0.881 | 0.883 | 10/10 | -0.0022 [-0.0048, +0.0004] |
| action-rate weight x5 | none | 8 | 0.878 | -0.060 | 0.834 | 0.882 | 3/8 | -0.0485 [-0.0517, -0.0453] |
| action-rate weight x5 | value | 10 | 0.875 | -0.056 | 0.836 | 0.883 | 9/10 | -0.0466 [-0.0500, -0.0431] |

Arm contrast, the difference of the shocked-minus-control effect between arms (value minus none):

| shock | (A) all 10: value minus none | 95% CI | seeds positive | (B) walking: value minus none | 95% CI | seeds positive |
|---|---|---|---|---|---|---|
| +0.1 per-step offset | -0.0024 | [-0.0262, +0.0214] | 7/10 | +0.0088 | [+0.0011, +0.0165] | 7/8 |
| **+1.0 per-step offset** | **+0.5025** | **[+0.3757, +0.6294]** | **10/10** | **+0.5720** | **[+0.5554, +0.5886]** | **8/8** |
| action-rate weight x5 | +0.0136 | [-0.0267, +0.0539] | 6/10 | +0.0017 | [-0.0036, +0.0070] | 5/8 |

**Reading.** The answer to the practitioner question is conditional, and the condition is the size of the offset. The
+1.0 and action-rate conclusions are the same under (A) and (B); the +0.1 one is not quite.

- A **small** offset, of the order of the per-step task reward (+0.1), is absorbed by both arms: the worst dip is
  -0.04 to -0.05 (uncentered) and -0.01 (centered), and every mother ends within 95% of its pre-shock level. Under (A)
  there is no arm difference (CI spans 0). Under (B) a small but detectable cost appears for the uncentered arm (-0.011
  against its control, CI excludes 0) and the arm contrast is +0.009 [+0.001, +0.017]: about 1% of tracking, which is
  why this is stated as "absorbed" and not as "no effect", and why the filter's post-hoc status matters.
- A **large** offset (+1.0, about 10x the per-step task reward) is catastrophic without centering and invisible with it:
  every walking uncentered policy collapses from about 0.88 to 0.31 within about 175 iterations of the shock and none
  recovers in the 400 remaining (0/8 under (B)), while every centered seed dips by about 0.06 and returns to 0.88
  (10/10). The arm difference is +0.50 (A) and +0.57 (B) with every seed positive. The `r_bar` trace shows the mechanism
  directly: the centered arm's estimate climbs from 0.05 to 1.0 within ~200 iterations, so the new offset never reaches
  the critic; the uncentered critic has to absorb a step of 1.0 in its target value (~1000 at this discount) through its
  own slow update, and the policy is destroyed first.
- A **reward-weight** change (action-rate x5) costs both arms about the same, 0.05-0.06 of tracking, and neither is
  fully back to its control level after 400 iterations (final minus control -0.060 (A) / -0.049 (B) and -0.047; within 95%
  of the pre-shock level 5/10 (A), 3/8 (B) and 9/10). Centering does not protect against this, which is unsurprising: the
  change alters the shape of the reward, not its level (arm contrast a null in both analyses).

So the practical statement is narrower and more useful than "centering makes shocks safer": **centering protects against
changes in the reward level, and does nothing for changes in reward shape.**

**Where the danger sits for an ordinary offset (`d116` reading, carried into REPORT_005).** Put the offset sweep and the
shock test side by side and a timing asymmetry appears at small offsets. An offset of 0.1 *present from the start*
halves uncentered reliability (5 of 10 seeds walk, against 9 of 9 centered), but the same 0.1 *added to a policy that
already walks* costs it about 0.01 of tracking and nothing else. At +1.0 the asymmetry vanishes: the large offset is
catastrophic whenever it arrives (0 of 8 walking mothers recover, round 4). So the claim is conditional on size:
**at small offsets the damage is done in early learning, before the policy takes off; a large level change is dangerous
at any time.** That is exactly where the mechanism (section 1) puts it: the critic's transient from the offset is worst
in the first few hundred iterations, and by iteration 1750 it has caught up to within 1.4% of its target while the policy
has already stopped improving.

These are pilot numbers (10 seeds). The two top-ups the rule asked for (d104 fired for the +1.0 and the action-rate
shock) were not run in round 4 for budget (d110); round 5 ran them for seeds 3010-3019 and pools the two rounds only
under identical branch settings and the identical pre-registered walking filter (d116). See REPORT_005.

## 6. Step 4: SSP width stage — not completed (budget)

Stopped before it could run (d109). Three of its fifteen cells completed — the raw-input arm for each of the three
search seeds — and they sit at the usual trained level (tracking 0.885 / 0.884 / 0.889, J_task 0.1001 / 0.0999 / 0.1005), which confirms the
harness and the seed block but says nothing about SSP width. **No width claim is made, and none of the d105 reading
rules were applied.** The prepared arms were raw, `ssp_low` at 2x2 (33 features per term) and 4x4 (129), and a random
Fourier code matched to each width; the 8x8 pilot remained conditional on 4x4 winning.

## 7. Conditionals, what was not run, and where the money went

| conditional | trigger | what happened |
|---|---|---|
| clip-off at 20 seeds | >=1 of 3 pilot seeds walks (d100) | **not triggered**: 0 of 3 |
| offset-sweep top-ups (c = 0.1, c = 0.3) | d103: walk counts differ by >=3/10 or paired CI excludes 0 | **triggered, refused for budget** |
| shock top-ups (+1.0 offset, action-rate x5) | d104: arms differ | **triggered, refused for budget**; the revived campaigns ran 20 mothers each before being stopped, no branches |
| SSP x centering confirmation | best SSP walks 2 of 3 (d106) | **not reachable**: the width stage did not run |
| matched RFF control at 20 seeds | SSP ties or beats raw | **not reachable** |
| G1 claim set, fresh seeds 3000-3019 | budget remains after Steps 1-4 (d106) | **not run**: budget exhausted |
| Go1 offset-0 top-up (seeds 3010-3019) | budget remains | **not run**: budget exhausted |

Spend for the round, from the billing API: **$20.47** (4090 $11.34, A40 $9.04, 3090 $0.03, 5090 $0.07), against a $20
cap. The cap was crossed by $0.47, and the cause is worth recording: the supervisor's "extra campaign" loop restarted
two shock top-up campaigns that had been cancelled by hand, because cancellation only removed their orchestrators and
pods, not their entry in the supervisor's state. Those two campaigns then ran their 20 mothers (1200 iterations each,
~$3.5) overnight before the cap check pulled everything down. Their branches never ran, so the money bought mothers
whose branches do not exist — a straight loss, and the one process bug of this round. Fixed afterwards: retirement is
written to the state file and honoured on supervisor restart (d111).

Per-step costs were not separated on the invoice (steps ran concurrently, hourly buckets); the orchestrator ledgers, which over-count by about a third, put Steps 2 and 3 at roughly 40% of the spend each, the rest being the width stage that was cut, the clip pilot and the revived top-ups.

## 8. Conclusions

**Supported (this round)**

1. **The mechanism has two parts, and neither is "a slow critic" alone** (pilot evidence: the clip-off arm is 3 seeds;
   the E11 critic windows are 20 seeds). Clip-off reaches 77% of the analytic value
   target by iterations 100-300 (plain PPO: 18%) and 99.7% by 600-1000, and the robot never walks. What separates the
   arm that learns is that centering makes the *advantage* state-explained from the first iterations: 0.954 of the
   return variance explained by state at 100-300, against 0.405 (plain) and 0.186 (clip-off). An accurate critic in
   level is necessary but not sufficient (d108). Round 5 tests this causally (REPORT_005).
2. **Offset size matters, and the threshold is lower than REPORT_003 claimed** (**pilot** at c = 0, 0.1, 0.3: 9-10
   seeds; claim grade only at c = 1.0, 20 seeds). Uncentered Go1 walks 10/10 at c = 0,
   5/10 at c = 0.1, 2/9 at c = 0.3 and 0/20 at c = 1.0; centered is 10/10, 9/9, 8/9 and 19/20. An alive bonus of about
   one per-step task reward is already enough to halve uncentered reliability.
3. **The proper shock test, finally run, gives a conditional answer** (**pilot**, 10 seeds, with the walking filter exploratory;
   see section 5). A *level* change of +1.0 (about 10x the
   per-step task reward) destroys every walking uncentered policy (about 0.88 -> 0.31, 0/8 recover under the 0.8 filter)
   and the centered policy dips about 0.06 and returns (final 0.881, 10/10): arm difference +0.5025 over all 10 seeds, 95%
   CI [+0.3757, +0.6294], 10/10 seeds positive. A *level* change of +0.1 is
   absorbed by both arms (a ~1% cost appears for the uncentered arm only under the walking filter). A *shape* change
   (action-rate weight x5) costs both arms 0.05-0.06 and centering does not
   help. So centering protects against changes in the reward level, not against changes in reward shape.
4. **Methodology.** Two measurement corrections that change how this project should be planned: the orchestrator's
   spend estimate was 1.9x low because it priced pods at $0.34/h when only $0.89/h secure stock existed, and one GPU
   process per pod is 2x more expensive per iteration than six processes because the GPU saturates at ~3.2
   iterations/s. The billing API is the only trustworthy spend read (d101).

**Not supported / not run**

- Any SSP width or SSP x centering statement under the corrected protocol (Step 4 did not run).
- Any G1 number from this round (its claim set was cut for budget; the 6-seed pilot in d98 stands as a pilot).
- The offset-sweep and shock top-ups, though both rules fired.
- The 20 extra shock mothers (seeds 3010-3019) from the revived top-up campaigns: no branches ran in round 4, so nothing was
  extracted; their checkpoints stayed on R2 and round 5 branches from them (REPORT_005).

## 9. Limits

Two robots exist in the study but only Go1 ran this round. Cell sizes range from 20 seeds (c = 1) down to 9-10 (the sweep
cells at c = 0, 0.1, 0.3 and the shock cells): **all of those are pilots**. The shock test's mothers are 10 per arm, and the
walking filter in (B) is exploratory in this report. The shock test's 400-iteration recovery window is
short: the centered arm is flat throughout, so the window is not binding for the conclusion, but a slow partially
recovering uncentered run would not be distinguished from a dead one except by the final-level count (0/8 within 95% of
pre-shock under (B)). Total round spend $20.47 against a $20 cap, with $3.5 of it lost to the revived top-up campaigns
(the mothers' checkpoints turned out to be reusable, round 5).
