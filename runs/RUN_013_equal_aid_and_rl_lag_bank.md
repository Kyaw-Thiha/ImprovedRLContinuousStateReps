# RUN_013: equal aid, crossed axes, the prior-free memory, the lag bank, and RL (in progress)

Started 2026-10-14. Executes `handoffs/11-ssp-memory-module-deployable.md` (CURRENT PLAN, steps 1-4), plus the
lag-free supervised split (step 2b) and the lag-bank RL arm. Nothing is committed.

Machine discipline: every number below is a **pod** row unless it says local. Pod and local runs are never tabulated
together (RUN_011 0b).

## 0. Pass rules (written BEFORE any run in this file; never edited after seeing data)

1. **Equal aid, index T=32, budget 256, 10 seeds, pods only.**
   Compare `fwp_delta+gate+banks2` against `recipe+gate+banks2`, in the default column (configs=1) and the tuned column
   (configs=6, 3 search seeds, `--select aulc --min-conv 0.0`).
   - The index lead is **withdrawn** if `fwp_delta+gate+banks2` is within 5 points absolute of `recipe+gate+banks2` on
     converged accuracy AND reaches >= 0.95 on 10/10 seeds, in either column.
   - The lead **stands** only if `recipe+gate+banks2` is ahead by more than 5 points or `fwp_delta+gate+banks2` has
     fewer than 9/10 seeds at 0.95 in BOTH columns.
   - Anything between is reported as "unresolved" and the lead is not claimed.
2. **Delay (p14):** read for the same comparison at delay 1-4; reported, not gating.
3. **Crossed combinations (p16):** the prior-free target is >= 9/10 seeds at 0.95 with NO prior, on BOTH index (T=32)
   and delay (1-4). A cell that passes one and fails the other is a partial and is recorded as such.
4. **RL pilot (RepeatPreviousEasy, POPGym PPO, 3 seeds, ~1M steps):** thresholds +0.244 (50%) and +0.849 (90%) as in
   RUN_012. A candidate "learns" if >= 2/3 seeds reach +0.244 within 1M steps. Parity with the GRU needs the final
   return within 5 points of the GRU's under the same loop. A sample-efficiency win needs >= 2x fewer steps to +0.244
   than the GRU (median 361k in RUN_012), with 3 seeds treated as a pilot only.
5. **Spatial recall (p19, the robotics gate -- handoff 11 step 5 / decision 4).** Pods, 10 seeds, T=24, K=3 landmarks,
   1500 steps, budget 256. Metric is **MSE on the queried steps** (lower is better), on `id`, `extrap_T` (2x episode,
   i.e. twice the motion accumulated since each landmark was written) and `extrap_motion` (2x step size).
   - *Parity:* the SSP arm's MSE is within the decision-2 margin of the best of `gru` / `fwp_delta` on every split.
   - *The robotics gate:* if the SSP arm is worse than **both** `gru` and `fwp_delta` by more than that margin, pause
     the robotics argument and say so to the user; the general memory-module track continues elsewhere.
   - *Abledations must lose:* `shift=False` (no group action) and `roles="shared"` must both be clearly worse than the
     full arm, or the group action is not what is doing the work and the result must be reported that way.
   - The mechanism is already verified before training (phase-domain identity to 1.4e-06 shifted vs 2.000 unshifted,
     `test_tasks_e.py`), so a failure here is a *learning* failure, not a mis-specified task.
6. **Reporting:** token counts are quoted with batch (64 supervised; the RL loop's `train_batch_size` 32768) and
   checked against the expected tokens per update.

## 1. Starting state (verified 2026-10-14, before anything was launched)

- Local: `test_default_path_hash` six `ok`; `test_harness` 6/6, `test_fixes` 2/2, `test_learned` 12/12, `test_candidates`,
  `test_fwp_gates`, `test_fwp_delta`, `test_tasks_f` all pass. No `ssp-*` pods alive.
- p14: delay column complete (12 cells); index column had only `recipe_def` and `fwp_delta_tun` -- the rest were lost
  when the pods were stopped in RUN_012.
- p15 (`delay_distract`): no rows.

## 2. Result: the equal-aid test fires rule 1 -- **the index lead is WITHDRAWN**

`jobs/p14_equal_aid.json`, pods, index T=32, budget 256, one machine class, **hold-out rows only (n = 10)** --
`report_holdout.py` filters `phase == "final"`; the earlier version of this table counted the search-seed run and
pooled search configs into the tuned rows, which is corrected in section 13.

| arm (equal aid: gate + banks available to both) | conv (sd) | seeds >= 0.95 | t50 tokens | t90 tokens |
|---|---|---|---|---|
| SSP `recipe+gate+banks2` (default == tuned) | 1.000 (0.000) | 10/10 | 25.7k | **30.9k** |
| FWP `delta+gate+banks2` (default == tuned) | **1.000** (0.000) | **10/10** | 30.7k | 66.7k |
| SSP `recipe+gate` (no banks) | 1.000 (0.000) | 10/10 | 25.7k | 51.3k |
| SSP `recipe` (incumbent) | 0.957 (0.009) | 9/10 | 28.3k | 61.7k |
| FWP `delta+gate` (no banks) | 1.000 (0.000) | 10/10 | 38.5k | 77.0k |
| FWP `delta` alone | 0.744 (0.028) | 0/10 | 59.0k | never |

**Rule 1 fired: within 5 points and 10/10 seeds in both columns, so the index lead is withdrawn.** This confirms and
extends RUN_011/012's "`index` is a harness check, not a thesis vehicle".

Three things this round adds that were not visible before:

1. **The input-dependent gate is a *general* improvement, not an SSP one.** It moves the fast-weight arm from 0.744
   (0/10 at 0.9) to 1.000 (10/10 at 0.95) -- the largest single change measured on the fast-weight family in this
   project. RUN_012's "the fast-weight family is far behind on index (0/10 seeds at 0.9)" was measured *without* the
   gate; with equal aid the gap disappears. That is exactly the threat handoff 11 step 1 was written to test, and the
   handoff's own pass rule says what to do about it: withdraw the claim.
2. **What survives is speed at the 90% threshold, and nothing else.** At equal state, equal aid, equal batch and
   *equal accuracy* (both 1.000, every seed), SSP reaches 0.90 in **2.16x** fewer tokens (30.9k vs 66.7k) but reaches
   0.50 in only **1.20x** fewer (25.7k vs 30.7k). So the >= 2x of decision 2 holds **at t90 only**, and it holds on one
   task family: on `delay` the two arms are identical to the token (both 21.6k, 10/10 at 0.95, 1.000 vs 0.998).
3. **Delay is a tie, with equal aid** (SSP 1.000 vs FWP 0.998, both 10/10 at 0.95, the same 21.6k t50/t90), i.e.
   RUN_012's "`delay` is substitutable" confirmed under equal aid rather than inferred.
4. **A trade-off the first version of this section missed entirely: the prior-free configuration is *worse* at length
   extrapolation.** On the 4x-longer stream the incumbent `recipe` scores **0.457** while `recipe+gate` scores 0.338
   and `recipe+gate+banks2` 0.343 (and the K-heads cells 0.352-0.365). So the gates and banks buy accuracy and speed
   on the trained range and cost extrapolation past it -- the hand-set prior was doing real work on that axis.

## 3. Result: **rule 3 passes** -- the hand-set prior is removed on both families

`jobs/p16_cross.json`, pods, 1500 steps, **hold-out rows only (n = 10)**. Prior-free cells start from
`init="scrambled"` (no offset, no `eps`, no range rule, no schedule).

| prior-free cell | conv | seeds >= 0.95 | t50 | t90 |
|---|---|---|---|---|
| index T=32, `tied+gate+banks2` | **1.000** | **10/10** | 43.5k | 64.2k |
| index T=48, `tied+gate+banks2` | **1.000** | **10/10** | 76.8k | 103.7k |
| delay 1-4, `tied+gate+banks2` | 0.999 | **10/10** | 21.6k | 36.7k |
| delay 1-8, `tied+gate+banks2` | 0.999 | **10/10** | 18.4k | 36.9k |

**The pre-registered target (">= 9/10 seeds at 0.95 with NO prior, on both index and delay") is met on all four cells.**
RUN_012 recorded `tied+gate` at **0.488 (0/10) on delay** and 0.977 on index; adding multi-timescale banks repairs the
delay failure and lifts index T=48 to 10/10. Compare with the incumbent, which needs four hand-set ingredients:

| configuration | task-dependent constants | index T=32 | index T=48 |
|---|---|---|---|
| `recipe` (incumbent) | 4 (offset 0, `eps`, frequencies, ramp) | 0.957, 9/10 at 0.95, t90 61.7k | 0.973 (RUN_011), t90 77.7k |
| `tied+gate+banks2` | 2 (fixed frequencies; `banks=2`) | **1.000, 10/10**, t90 64.2k | **1.000, 10/10**, t90 103.7k |

Equal accuracy at T=32 for **two constants fewer**, and neither of the two is an assumption about where the read
address starts. It is slower at both lengths on t90 (64.2k vs 61.7k; 103.7k vs 77.7k) and it is **worse at length
extrapolation** (section 2 item 4). On the bitter-lesson axis this is still the round's main advance, but the price is
now stated: speed on the trained range and accuracy on the longer stream.

## 4. Result: gates x banks and gates x K heads are synergistic (the axis interactions)

The two axes had only ever been measured one change at a time. The crossed cells, **hold-out rows only (n = 10)**:

| cell | conv | seeds >= 0.95 | t50 | t90 | the same change measured ALONE (RUN_012) |
|---|---|---|---|---|---|
| index T=32, recipe+gate+banks2+K3 | 1.000 | 10/10 | 25.7k | **28.3k** | -- |
| index T=32, gate+banks2+K5 from an offset-1 start | 1.000 | 10/10 | 25.7k | **25.7k** | -- |
| index T=32, gate+banks2+K5 from an offset-3 start | 1.000 | **10/10** | 51.3k | 77.0k | K5 alone at offset 3: **0.496, 0/10** |
| delay 1-4, recipe+K5 from an offset-1 start | 1.000 | 10/10 | 21.6k | 21.6k | -- |
| index T=32, recipe+gate+banks2 (no K) | 1.000 | 10/10 | 25.7k | 30.9k | gates+banks2: 1.000, 10/10 |
| delay 1-4, `tied+gate+banks2` (no prior) | 0.999 | 10/10 | 21.6k | 36.7k | gates alone, no prior: **0.488, 0/10** |
| index T=48, recipe+K5 | 0.910 | 0/10 (7/10 at 0.90) | 38.4k | 264.9k | prior-free `tied+gate+banks2`: **1.000, 10/10** |

Two interactions, both invisible in the one-change-at-a-time design: **gates+banks rescues the offset-3 dead-zone
configuration** that K heads alone could not (0.496 -> 1.000, 10/10) and **banks rescues the prior-free delay case**
that gates alone broke (0.488 -> 0.999).
**These are ablation results, not mechanism claims.** An earlier version of this section said "two dead-zone-looking
failures were really missing-a-bank failures", which asserts a cause nobody measured. `diag_address.py` was listed as
run for the tied+gate delay cell and **was not run**; what exists is the ablation contrast above (with the bank
0.999/10-of-10, without it 0.488/0-of-10). The mechanism is untested and the claim is withdrawn pending a diagnostic
that logs the gate's write strength and decay -- not attempted this round on the no-local-compute rule. On the trained range K3 and K5-from-offset-1 give the round's fastest cells
(28.3k and 25.7k t90, against the no-K cell's 30.9k). **And K heads are not a general repair:** `recipe+K5` at index
T=48 finishes at 0.910 with 0 of 10 seeds at 0.95, against the prior-free `tied+gate+banks2`'s 1.000 at 10/10 on the
same range -- so K heads rescue a bad *start* at a fixed range, not a longer range, the same pattern as the `ls`
schedule and the affine prior.

## 5. Result: distractors -- the gates help in the predicted direction, but nothing solves it yet

`jobs/p15_distract.json`, pods, `delay_distract` at T=32, distractor rate 0.25, 800 steps, 10 seeds. The task asks the
memory to *choose what to store*: marked distractor symbols are never answers, so a memory that stores everything
dilutes itself.

| arm | conv | seeds >= 0.95 | note |
|---|---|---|---|
| recipe | 0.567 | 0/10 | incumbent |
| recipe + gate | 0.638 | 0/10 | gates help (+0.071) |
| **recipe + gate + banks2** | **0.690** | 0/10 | best SSP cell (+0.123 over the recipe) |
| tied (no prior) | 0.465 | 0/10 | weakest memory cell |
| tied + gate | 0.637 | 0/10 | gates recover most of the tied loss (+0.172) |
| `fwp_delta` | 0.564 | 0/10 | |
| **`fwp_delta` + gate** | **0.695** | 0/10 | best cell of the round |
| GRU | 0.128 | 0/10 | **at chance (0.125)**: it cannot do selective delay at all |
| LRU | 0.127 | 0/10 | at chance |

**The input-dependent gate moves every memory arm in the predicted direction** (+0.07 SSP, +0.16 tied, +0.14
fast-weight) but **no arm passes and the gate's benefit is not SSP-specific**. The clean part is the floor: the GRU and
LRU sit exactly at chance while every key-addressed memory sits at 0.48-0.70 -- a capability separation on a task that
asks for selective storage. The budget confound is resolved in 5b below, where the incumbent does not move.

### 5b. The distractor task at 3x the budget -- the picture holds, with more headroom

`jobs/p15b_distract_extend.json`, pods, `delay_distract` at T=32, **3000 steps** (vs 800 above), 10 seeds.

| arm | conv at 800 | conv at 3000 | seeds >= 0.90 at 3000 |
|---|---|---|---|
| recipe | 0.567 | **0.562** | 0/10 (flat: a capability limit, not a budget limit) |
| recipe + gate + banks2 | 0.690 | 0.733 | 0/10 |
| tied + gate (no prior) | 0.637 | 0.746 | 2/10 |
| `fwp_delta` + gate | 0.695 | **0.762** | 0/10 |
| GRU | 0.128 | 0.126 | 0/10 (exactly chance at both budgets) |

Tripling the budget moves the gated cells by +0.04 to +0.10 and leaves the incumbent **exactly where it was** (0.569 ->
0.564), so the section-5 ordering was not a budget artefact: gates and banks lift the ceiling on a task that asks for
selective storage, and the prior-based recipe is stuck at 0.56. Nobody passes, the benefit is still not SSP-specific,
and the GRU's floor (0.127 against chance 0.125) is the one clean separation.

## 5c. Result: the last unbuilt axis-2 candidate works -- content-plus-position keys

`jobs/p18_assoc_mixed.json`, pods, `assoc_mixed` (each query asks **either** by content **or** by index, decided only
by the mode column), T=24, 1200 steps, 10 seeds, chance 0.250.

| arm | conv | seeds >= 0.95 | what it means |
|---|---|---|---|
| **`AssocPosSSPArm` (content + position keys)** | **1.000** | **10/10** | solves both query modes at once |
| the same, `pos=False` | 0.626 | 0/10 | = 0.5*1.0 + 0.5*0.25: content solved, **index at chance** |
| the same, `bind=False` (additive bundle) | 0.253 | 0/10 | chance-ish: the binding is load-bearing |
| content-only SSP control (`assoc_ssp`) | 0.635 | 0/10 | cannot answer index-mode queries at all |
| GRU | 0.249 | 0/10 | chance |
| LRU | 0.612 | 0/10 | partial (content side only) |

Read with the arithmetic: `0.5 * 1.000 + 0.5 * 0.250 = 0.625`, which is exactly the `pos=False` and content-only scores.
So the ablations fail **precisely and only** on the mode they should, which is the strongest form this kind of evidence
takes here -- every arm is scored on both modes in the same episodes, and the failure is mode-specific rather than a
uniform drop in accuracy.

This closes the last unbuilt candidate from `docs/11` section 14.3 axis 2. The honest costs: the arm stores **two
bindings per pair**, and the position codes are a second fixed codebook, so the mixture is bought with capacity and one
more hand-set codebook rather than for free. It is a *capability* result like the pointer: no matched non-SSP control
passes it (GRU at chance), which is exactly the comparison the SSP-specificity rule warns about -- a capable arm on a
task the incumbent cannot do, not a win over a working incumbent.



## 6. Code and tests built this round (all verified locally)

- **`rl/ssp_tied_model.py`: `SSPTiedLagBankModel`** (+ registered as `tied_bank` / `tied_bank_aux` in `rl/common.py`).
  Reads at **every** lag in a fixed bank (default 1..8) and mixes the reads with a learned softmax selector; state is
  unchanged (bundle only, 256 floats), and at init the selector is exactly uniform so no task constant is injected.
  `rl/ssp_model.py` gained `_write_bundle` (a pure extraction of the write path, no behaviour change -- both RL test
  modules reproduce their old counts afterwards).
- **`rl/test_ssp_tied.py`: 5 new contract tests** (8 -> **13/13**). They cover the state/param separation, the uniform
  default selection, one-hot-recalls-exactly (lag 2/4/8 at accuracy 1.000), the aux + `custom_loss` path, that the
  gradient reaches the selector, and -- the claim itself -- that **the selector finds the right lag from the uniform
  default** (accuracy 0.469 -> 1.000 in 60 selector-only steps, `lag_eff` 4.50 -> 3.98 against a target of 4).
- **`tasks_f.py`: `DelayNoLagTask`** (variant `delay_nolag`), wired into `run_sweep.make_task` and
  `harness.splits_for`. It is the supervised mirror of `RepeatPrevious`: observations, targets and mask are
  bit-identical to `delay_distract` at `p_dist=0`, but the **lag feature is removed** (22 features instead of 23), so
  the arm must discover *which offset to read at* -- the half of the discovery problem no supervised task had tested.
  It is deliberately a separate variant, not a split, because removing a feature changes `in_dim`.
- **`test_tasks_f.py`: 3 new tests** (5 -> **8/8**): the two variants are the same task with the lag hidden (mutating
  the lag leaves the lag-free features untouched, while the control arm's features move); the alphabet and width stay
  fixed across splits; no leak (untrained MLP 0.123 against chance 0.125) while the mask stays non-vacuous.
- **`scripts/ssp_pods.py`**: the durable form of the lost `/tmp/reap_pods.py` (`list`, `kill --prefix`), needed because
  re-launching a job list resumes on R2 and re-creates pods for cells that are already complete locally.
- **`tasks_d.AssocMixedTask`** (variant `assoc_mixed`) + **`arms_assoc.AssocPosSSPArm`** (`assoc_pos_ssp`): the last
  unbuilt axis-2 candidate, "content-plus-position keys". Each pair is written under **two** keys -- its content key
  and its presentation position -- and a query is answered with whichever key the new ``mode`` column selects, through
  a gate derived from ``mode`` itself, so the scheme adds no parameter and no task constant beyond the codebooks. It
  costs **two bindings per pair** (reported, not hidden). `pos=False` (no position key) and `bind=False` (additive
  bundle) are the matched ablations, and the whole family is scored **per mode**, never pooled.
- **`test_tasks_d.py`: 5 new tests.** They check determinism and mode balance, that the ``mode`` column is
  load-bearing (a content-key oracle scores 1.000 on content-mode queries and **0.172** -- chance 0.250 -- on
  index-mode ones), that the alphabet and width are fixed across the mixed splits, that `assoc` itself is untouched
  (no new column, same width), and -- local, short-run, mechanism only -- that the full arm solves **both** modes
  (1.000 / 1.000) while `pos=False` loses index-mode (0.625) and `bind=False` loses content-mode (0.473).
  **Two construction bugs were found by these tests before any pod run**, which is what they are for:
  (a) the oracle initially gathered over the pair region as well as the query region and tripped an index bound;
  (b) far more seriously, the first version paired the two modes on the *same* pair on consecutive steps, so an
  index-mode query could be answered from the content-mode query just before it -- the `pos=False` ablation still
  scored **0.736** on index queries. Queries now draw their pair independently and never repeat the previous pair.

- `test_default_path_hash`: six `ok` after every edit to `run_sweep.py` / `harness.py`.

## 7. Result: with the lag genuinely hidden, **no arm solves the supervised delay task**

`jobs/p17b_nolag_fixed.json` (fresh tags after the bug in section 8), pods, `delay_nolag` at T=32, 800 steps, 10 seeds,
chance 0.125.

| arm | conv | seeds >= 0.95 | extrap_lag | note |
|---|---|---|---|---|
| recipe (the incumbent) | 0.348 | 0/10 | 0.125 | **drops from 1.000 in the buggy run** |
| recipe + gate + banks2 | 0.383 | 0/10 | 0.123 | |
| tied (no prior) | 0.396 | 0/10 | 0.136 | |
| tied + gate + banks2 (no prior) | 0.400 | 0/10 | 0.134 | |
| K5 read heads from an offset-1 start | **0.417** | 0/10 | 0.126 | best cell, and still nowhere near passing |
| GRU (hidden 64) | 0.397 | 0/10 | 0.125 | |
| LRU (hidden 64) | 0.400 | 0/10 | 0.117 | |
| `fwp_delta` (tied, delta rule) | **0.152** | 0/10 | 0.127 | **at chance (0.125)**: the fast-weight arm cannot do it at all |

**The clean statement: when the lag is not given, the supervised delay task is unsolved at this budget by every arm,
including the GRU and the LRU.** All of them learn something partial (0.35-0.42 against 0.125) and none of them
extrapolates at all. Two consequences:

1. This is the **supervised mirror of the RL failure**, reproduced for the price of a pod cell: `RepeatPreviousEasy`
   put the untied SSP arm at -0.163 and the tied arm at -0.133 while the GRU reached +0.993, and the reason is not
   that the SSP arms are weak -- it is that *lag selection from the stream alone* is a hard problem for every
   architecture here. The GROU's RL advantage must therefore come from something other than a better lag search
   (its recurrent state can integrate the stream, but so can these arms).
2. It kills the obvious cheap route to the RL question. The supervised harness cannot serve as a fast discriminator
   for a lag-selection candidate unless the candidate is far stronger than K5, so the RL lag-bank pilot (section 10)
   is the only direct test.

Caution on the level, not the ordering: 800 steps is short and every arm is still improving, so these numbers are a
floor; the *ordering* (all arms within 0.07 of each other, extrapolation at chance) is the finding.



## 8. A caught bug: `DelayNoLagTask` did not actually hide the lag (first p17 run invalid)

The first `jobs/p17_nolag.json` run returned `recipe` at **1.000, 11/11** on the supposedly lag-free delay task, and
`fwp_delta` at 0.997 -- both implausible for a task where the arm is not told the lag. The cause:

- **The arm does not read the lag from a feature column.** `SSPIndexLearned._read_addr` reads **`batch["lag"]`**
  directly (`arms_learned.py:284`), and `FWPArm._addr` does the same (`:548`). The first version of the variant zeroed
  only `batch["query"]` (the feature column), so every arm still received the true lag and the task was not lag-free at
  all. The 1.000 was the *ordinary* delay task wearing a different name.
- **My own test could not see it, because it asked the wrong object.** It asserted on `task.features(batch)` -- which
  never contained the lag -- and on `batch["query"]`. Both were correctly zero. The channel that mattered was a *third*
  tensor, `batch["lag"]`, which the test never looked at.

Fix: `DelayNoLagTask.batch` now zeroes **both** `lag` and `query` after `super()` has materialised the targets, and the
new test `test_delay_nolag_gives_the_arm_no_lag_channel` asks **the arms themselves** -- it builds the recipe arm and
the fast-weight arm and asserts their read addresses carry no lag (`_read_addr` is exactly zero; `FWPArm._addr` is
exactly `t` in the `read_abs` form and exactly zero in the lag-only form). `test_tasks_f` is 9/9 with the check in
place, against 8/8 before it.

Consequences: the first run's R2 objects and local copies were **deleted** (they are invalid, and the tag is the resume
key, so a re-run under the same tag would have been skipped as "already done" and the wrong numbers kept). The
corrected run is `jobs/p17b_nolag_fixed.json` with fresh tags.

**This is the most transferable lesson of the round**: a variant that hides an input must be tested at the *consumption
site*, not at the feature builder. Every "hide a feature" experiment in this project (`cue_drop`, `mask_hop_in_start`,
this one) has the same failure mode available.



## 10. RL: the lag bank learns the offset itself (complete: 6 runs x 1M steps)

Local POPGym runs (POPGym's own PPO loop, `minibatch_size=8192`, `workers=0 --envs-per-worker 32`, 3 seeds, 1M steps,
`rl/launch_bank.sh`). Random return -0.512, optimal +1.000, thresholds T50 = +0.244, T90 = +0.849. **Local rows; not
tabulated with any pod number.** The reference rows below are RUN_012's, same machine and loop.

| arm | what it is | T50 (steps to +0.244) | T90 (steps to +0.849) |
|---|---|---|---|
| `gru` | POPGym's incumbent, same loop | 360k / 393k / 360k (**median 360k**) | 557k / 557k / 524k (**median 557k**) |
| `fwp` | POPGym's own fast-weight model (no delta rule) | **never** (final -0.19 to -0.28 at 2M) | never |
| `tied` | tied, lag 0 = "read now" | **never** (up to 1.6M: +0.03 / +0.03 / -0.23) | never |
| `tied_lag4` | tied, offset **hand-set** to the task's lag | 196k / 196k / 459k | never / 459k / 426k |
| **`tied_bank`** | tied + **learned softmax selection over lags 1..8** | **229k / 229k / 229k** | **491k / 491k / 459k** |
| `tied_bank_aux` | the same with the aux on | never (still -0.38 to -0.48 at 524k) | never |

### 10.1 Final 1M-step numbers (all six runs complete, 1,016,583 steps each)

| arm | final return (last 10% of 1M) | T50 | T90 | AULC | steps/s |
|---|---|---|---|---|---|
| `gru` | **+0.993 ± 0.005** | 3/3, median 361k | 3/3, median 557k | +0.436 ± 0.024 | 264 |
| **`tied_bank`** | **+0.958 ± 0.003** | **3/3, median 230k** | **3/3, median 492k** | **+0.582 ± 0.004** | 101 |
| `tied_bank_aux` | -0.316 ± 0.046 | 0/3 | 0/3 | -0.425 ± 0.035 | 101 |
| `ssp` (untied, RUN_012) | -0.163 ± 0.117 | 0/3 | 0/3 | -0.338 ± 0.061 | 295 |

Applying the pre-registered rule 4 and decision 2, in order:

- **"Learns": yes.** 3 of 3 seeds clear both +0.244 and +0.849. No lag constant was injected.
- **Parity: yes.** +0.958 against the GRU's +0.993 is a **3.5-point** gap, inside the 5-point margin. Note this is
  *convergence* parity and it is real -- the bank's three seeds span 0.955-0.961, so it is a tight, reproducible
  plateau, not a lucky run.
- **A listed metric won by the required margin: no.** Sample efficiency is 1.57x at T50 and 1.13x at T90, both under
  the >= 2x the rule asks for. The **AULC is 33% higher** (+0.582 vs +0.436), which is the metric that best captures
  "learns faster overall" -- but area-under-the-curve is not one of the metrics decision 2 lists in advance, and it
  must not be quietly substituted for one that is.
- **A cost, stated plainly:** the bank runs at **101 steps/s against the GRU's 264** in this contended run (RUN_012
  measured 1.25 ms vs 0.64 ms per step uncontended), so per-step cost is *worse* at equal accuracy. That is the
  "state size or per-step cost at equal accuracy" metric from the decision-2 list, and the bank **loses** it.
### 10.2 The bank's *range* is a task constant, and it is now measured (test added)

`RepeatPreviousEasy` is constructed with `k=4`, so its recall lag is 3 (`k-1`, RUN_006), and the bank is the fixed set
1..8 -- it contains the answer. `test_ssp_tied.py::test_lag_bank_without_the_true_lag_cannot_solve_the_task` measures
what that range buys, same seed and same 60 selector-only steps:

| bank | contains lag 4? | accuracy at the true lag | `lag_eff` it reports |
|---|---|---|---|
| `[1..8]` | yes | **1.000** | 3.99 |
| `[1, 2, 5, 6]` | no | **0.375** | 4.03 |

Two things follow. First, **"no lag constant was injected" is only half the claim**: the *selector* is learned from a
uniform default, but the *range* is a hand-set constant, and outside it the arm scores 0.375 against chance 0.250. On
`RepeatPreviousMedium` (`k=32`) this bank would not contain the lag. Second, **`lag_eff` is a misleading readout**: both
banks report about 4.0, because the selector can mix lags 1 and 6 to make the mean look like 4 while holding no such
address -- so the mechanism check in section 10.1 should be read as "the selector concentrated on members of the bank",
not as "it found the lag". Accuracy is the honest readout, which is why the test scores that.


**So the honest final verdict is: parity at convergence, faster on both thresholds but not by the pre-registered
factor, better area under the curve, worse per-step cost, and a clear win over POPGym's own fast-weight model, which
never learns this task at 2M steps.** Under decision 2 as written that is "a working memory with no demonstrated
reason to prefer it" -- which is a real result (it is the first SSP memory arm in this project that works in RL at
all), but not the claim the rule was written to license.

**The verdict against the pre-registered rule 4.** `tied_bank` **learns**: 3 of 3 seeds clear +0.244, and -- unlike
`tied` and `tied_lag4` -- all three also clear **+0.849**. Speed: **1.57x** the GRU's median T50 (229k vs 360k) and
**1.13x** its median T90 (491k vs 557k). So it is *faster on both thresholds, and short of the 2x the decision-2 rule
asks for*. The honest statement is **parity plus a speed advantage, not a decisive sample-efficiency win** -- and the
mechanism (a bank of read offsets selected by a softmax, no lag constant, no task constant beyond the bank's range)
is the first configuration in this project with which an SSP memory arm learns this task at all.

**`tied_bank` clears the 50% threshold in all three seeds at 229k steps with no lag given** -- 1.57x faster than the
GRU's median T50 (360k), and more *reliable* than the diagnostic that is handed the offset (`tied_lag4` reached T50 in
2 of 3 seeds by 196k, the third only at 459k, and one seed never reached T90 at all). `tied_bank_aux` does not learn at all, which is the third independent
confirmation that the aux hurts this family (RUN_012: tied -0.133 vs tied+aux -0.202 in RL; and section 7 here).


**The mechanism check passes, and it is the point of the round.** `address_summary` logs the effective lag every
iteration. In `tied_bank`: `lag_eff` moves 4.50 (uniform bank, the default init) -> **2.99** and `sel_entropy` collapses
from 2.06 (uniform over 8) to **0.04-0.05** in all three seeds -- i.e. the selector *found a single lag and locked onto
it* within ~130k steps, without ever being told one. `tied_bank_aux` settles at 4.0-5.0 instead and learns much more
slowly, which is consistent with RUN_012's finding that the aux hurts this family.

An important convention note that the trajectory comparison settles: the task's constructor is `k=4` but its `step`
compares against `deck[-k]` *before* dealing, so the observation the model must recall is `k-1 = 3` steps back (RUN_006
already recorded "lag is k-1 not k"). The bank converged to 2.99, i.e. to the correct lag on this reading, and its
trajectory tracks `tied_lag4` -- the arm that is *handed* the offset -- rather than `tied` (lag 0, which never learns).
So at 196k the bank is where the hand-set-lag reference is, having paid no hand-set constant.

**What this does and does not establish.** It establishes that *selection* removes the dead zone that stopped the
continuous lag head (RUN_012: `|bias| <= 0.011` in six runs, -0.133 at 1M) -- and not just the dead zone: the bank
*learns the task*, 3 of 3 seeds past the 50% threshold, where `tied` never gets there even at 1.6M steps. It also
beats POPGym's own fast-weight model outright, which never learns on this task at 2M steps. It does **not** yet pass
decision 2: 1.57x at T50 and 1.13x at T90 against the GRU are both short of the >= 2x the rule asks for, so the
correct statement is "faster, not decisively so". It does beat POPGym's own fast-weight model outright, which never
learns this task at 2M steps, and it is the first SSP memory arm here that learns it at all.



## 12. The spatial recall task: the robotics gate, built and verified (not yet run on pods)

Handoff 11 step 5. `tasks_e.SpatialRecallTask` + `arms_spatial.SpatialSSPArm` + `test_tasks_e.py` (4 tests, all pass
locally). The task is the one `docs/11` section 14.4 names as potentially unique to SSP and which no testbed in this
repo covered: ``K`` landmarks are shown once, each as a **current-frame position**; every step carries an egocentric
**motion increment**; and a query asks for a landmark **in the current frame**. So the memory must hold ``K`` items and
apply one accumulated rigid motion to all of them at read time.

`SpatialSSPArm` does exactly that with the algebra: ``M = sum_k role_k (*) phi(x_k)``, ``M <- M (*) phi(-d_t)`` each
step (one element-wise multiplication for the whole memory, independent of ``K``), read as
``M (*) conj(role_q) = phi(x_q - (p_t - p_q))``.  Ablations wired: ``shift=False`` (no group action) and
``roles="shared"`` (all landmarks under one role).

**What is established before any training** (the tests are written to make the claim falsifiable, not to show the arm
trains):

| check | result |
|---|---|
| task deterministic; target == ``lpos[shown] - (p_t - p_shown)`` on every query | exact (all 168 queries, atol 1e-5) |
| **the group action reproduces the target's phase vector, with no parameters involved** | shifted ``max|z - phi(target)| = 1.4e-06``; **unshifted ``2.000``** |
| the arm trains in a 250-step local run (MSE) | **0.0003** |
| ``shift=False`` | 0.0068 (23x worse) |
| ``roles="shared"`` | 0.0843 (280x worse) |

The second row is the important one: it verifies the *algebra against the task* rather than the arm against the task,
so the experiment is well posed before a single pod is spent, and it shows the shift is load-bearing rather than
decorative. The arm reaching 0.0003 in 250 steps says the mechanism works in practice, not just in principle.

### 12.1 [SUPERSEDED -- see section 14] The untuned comparison appeared to show the SSP arm winning

> **Withdrawn.** Everything in this subsection is measured against untuned baselines. Section 14 tunes them (4 configs,
> 4000 steps) and the tuned GRU reaches MSE **0.021** against the SSP arm's **0.077** -- it *wins*. The numbers below
> are kept only as the record of the error and of what the untuned comparison looked like.

`jobs/p19_spatial.json`, pods, T=24, K=3, 1500 steps, budget 256, 10 seeds. The reported metric is **`-MSE`**
(`harness.loss_and_acc` negates the mean squared error for regression tasks), so **higher is better**; the raw MSE is
the negative of the printed number.

| arm | id `-MSE` (MSE) | extrap_T | extrap_motion (2x step) | rob_pos (25% position noise) |
|---|---|---|---|---|
| **SSP full** | **-0.0965 (MSE 0.097)** | **-0.1005** | **-0.5190** | **-0.7569** |
| SSP `shift=False` | -0.3062 (0.306) | -0.3070 | -1.0882 | -0.9820 |
| SSP `roles="shared"` | -1.1354 (1.135) | -1.1300 | -1.2326 | -1.2417 |
| LRU | -1.1070 (1.107) | -1.1030 | -1.2184 | -1.4514 |
| `pose_attn` | -1.3469 (1.347) | -1.3437 | -1.3830 | -1.3438 |
| GRU | -1.6594 (1.659) | -1.6654 | -2.6102 | -1.6535 |
| `fwp_delta` | -1.6395 (1.640) | -1.6476 | -2.5283 | -1.6350 |
| `mlp` (no-memory floor) | -1.6386 (1.639) | -1.6462 | -2.5255 | -1.6338 |
| `posmlp` | -1.6386 (1.639) | -1.6457 | -2.5265 | -1.6335 |

**Read plainly: the SSP arm's MSE is 0.097 against the GRU's 1.659 and `fwp_delta`'s 1.640 -- a factor of 17.** Even
more tellingly, **the GRU and `fwp_delta` are at the no-memory floor** (1.659 and 1.640 against the MLP's 1.639, a 1.3%
gap): neither learns the task at all. So this is not a close win, and it is not a parity result -- it is a capability
separation on the one task that directly tests the claim `docs/11` section 14.4 calls potentially unique to SSP (one
group action re-expressing every stored landmark at once).

**The ablations are load-bearing**, which is what makes the win attributable to the algebra rather than to capacity:
removing the group action costs 3.2x (0.097 -> 0.306) and sharing one role across all landmarks costs 11.8x
(0.097 -> 1.135, i.e. down to LRU level).

**The extrapolation columns are the interesting caveat.** Downstream of the trained regime the arm still leads by a
wide margin (extrap_T 0.101 vs 1.665; 2x step size 0.519 vs 2.610) but the 2x-step split degrades the arm 5.4x while
the baselines merely stay at their floor. With no learned motion scale the shift assumes the trained step size, so this
is the expected and *bounded* failure mode -- it is the honest limit to report, not a second win.

**Fairness note.** Each memory gets the query in the form it natively addresses by: the SSP arm a role one-hot, the
fast-weight arm a per-step `lag` (so its read lands on the step the landmark was written), and the recurrent/attention
arms both through the shared feature vector. The information content is the same; only its form differs, which is what
"matched aid" means here.

**This licenses the robotics direction under decision 4** -- the arm is ahead of both the GRU and `fwp_delta` by far
more than the parity margin, so the argument is not paused. It does **not** license an RL claim (see section 10: the
RL result is parity plus 1.57x, short of the 2x the rule asks for), and the 2x-step degradation is the cost to carry
into any deployment claim.

## 11. Traps found this round

- **A shell does not pass JSON safely to `--arm-cfg`.** Unquoted, bash strips the quotes and then **brace-expands**
  `{a,b,c}` into separate words, so `run_sweep.py` fails with "unrecognized arguments". Single-quote the JSON. The pod
  launcher is unaffected (it passes argv as a list).
- **`ssp_pod_batch.py launch` resumes on R2, not on `results/`.** Re-launching a partially-collected job list
  re-creates pods for cells whose rows are already on disk, so `status` (which reads R2) and the local files can
  disagree. Check `status` before reading local files, and kill redundant pods by prefix (`ssp_pods.py kill --prefix`).
- **`launch_bank.sh` filename expansion**: `${ENV}` inside the single-quoted `xargs bash -c` body is expanded by the
  inner shell, which does not have `ENV`; `--env` was correct (it sat outside the quotes) but the filenames came out as
  `rl_bank__<arm>_s<seed>.jsonl`. Fixed with `export ENV`; the current run's files keep the old name.
- Editing a doc while several edits overlap silently merges two sections (cost one rewrite of this file).

---

## 13. Correction log (appended 2026-10-14, from auditing this file's own tables)

Triggered by re-reading p14 and finding the numbers did not reproduce. Every claim in the round was then recomputed
from scratch with `report_holdout.py`, which keeps only `phase == "final"` rows and states `n` next to every figure.

**1. Every table used the wrong denominator, and the tuned columns were invalid.** A sweep file holds *search* rows
(the config-selection runs, one per config) and *final* rows (the 10 hold-out seeds). The original tables averaged the
whole file, so (a) every `configs=1` cell read "11/11" instead of the protocol's 10 hold-out seeds, and (b) every
tuned cell (`configs>1`) pooled **search runs of different configs** with the final runs -- the "17/17" and "28/28"
rows averaged configs that were never the reported one, and those two numbers are withdrawn outright. All tables in
sections 2-7, 5b, 5c and 12.1 now carry hold-out-only figures at `n = 10`.

**2. The speed claim was overstated, and the >= 2x holds at one threshold only.** The first version said SSP reaches
0.90 *and* 0.50 in "**2.2x / 2.1x**" fewer tokens than the equal-aid fast-weight arm. On hold-out rows the true
figures are **2.16x at t90** (30.9k vs 66.7k) and **1.20x at t50** (25.7k vs 30.7k). The t50 figure came from a
miscopied row. So the decision-2 sample-efficiency claim survives **at the 90% threshold on `index` only**; it does
not hold at t50, and `delay` is an exact tie at both thresholds (both 21.6k).

**3. A real trade-off that the tables contained all along and this file never reported: the prior-free configuration
is worse at length extrapolation.** On the 4x-longer stream, `conv_extrap_T` for index is **0.457 for the incumbent
`recipe`**, 0.338 for `recipe+gate`, 0.343 for `recipe+gate+banks2`, and 0.352-0.365 for the K-heads cells. The same
pattern holds on `delay` (`extrap_T`/`extrap_lag`: recipe 0.999/0.776 against `fwp_delta` 0.914/0.393). So gates and
banks buy accuracy and speed **on the trained range** and cost generalisation **past** it -- the hand-set prior was
doing work on that axis. Any statement of the form "the prior is gone at no cost" is wrong; the cost is here.

**4. Two bars changed in the K-heads cells.** `index T=32, gate+banks2+K5 from an offset-3 start` is **10/10** at
0.95 (recorded as 9/9 -- the 11th row was a search run), and `index T=48, recipe+K5` is **0.910 with 0 of 10 seeds**
at 0.95 (now complete, previously 5 rows). The offset-3 rescue therefore stands and the "K heads do not rescue a
longer range" negative is confirmed at full seed count.

**5. `rob_pairs = 1.000` for the mixed-keys arm is vacuous by construction, not a robustness result.** The task's
`cue_drop` corrupts the query and then recomputes the target for the *corrupted* query, so a correct answer is still
available; the column is reported for completeness and must not be cited as evidence of robustness. (RUN_012 flagged
the same vacuity in the content-keyed family.)

**6. The memory-zeroed ablation of the RL bank was not run, and can no longer be run from the saved state.** The
machine rebooted (uptime 1:27 at the time of the audit); `/tmp` held the RLlib checkpoints (`/tmp/rl_ckpt_bank/*`) and
the per-run logs, and both are gone. The **durable** artefacts survived intact and were re-verified: six run files in
`results/` with 31 iterations each and a summary row, all at 1,016,583 steps. So the RL *curve* numbers in section 10
stand, but the check that the return is actually carried by the memory state (`eval_ckpt.py --ablate-memory`) is
**missing for `tied_bank`**. RUN_012 has the equivalent measurement for the older arms (zeroing drops them to about
-0.50) and RUN_006 established that `RepeatPrevious` has no feed-forward shortcut, but neither substitutes for the
measurement on this arm. Remedy: one seed re-run with `--ckpt` (about 1.7 h) and then
`eval_ckpt.py tied_bank:<dir>`.

**7. Method note for the next round.** `report_holdout.py` is now the way to read a pod job list; a table built by
hand from a sweep file will make mistake 1 again. The `n` column exists so that a wrong denominator cannot pass
unnoticed, which is exactly how it was caught here.

---

## 14. The spatial gate and tuned baselines: the untuned comparison is invalid, and the tuned margin is smaller

`jobs/p21_spatial_tuned.json`, pods, hold-out rows only. Added because the p19 comparison (section 12.1) rested on
baselines that sat at the no-memory floor, which is the signature of "did not train" -- and docs/11 4.2 requires a
baseline to pass a trainability gate before its number is used.

**14.1 Trainability gate** (`spatial_easy`: one landmark, zero motion, so the answer is simply the position shown at
step 0 and the only requirement is to carry a 2-vector across 24 steps). Same protocol as p19: `configs=1`, 1500 steps,
10 seeds, MSE lower-is-better.

| arm | MSE (hold-out) | vs the no-memory floor |
|---|---|---|
| `mlp` (no memory) | 1.314 ± 0.055 | the floor |
| **GRU** | **0.630 ± 0.326** | 0.48x the floor -- **fails the pre-registered gate** (0.1x required) |
| `ssp_spatial` | **0.001 ± 0.000** | solves it exactly |

**The gate fails, and that is the finding: with the protocol p19 actually used, the GRU cannot even carry one 2-vector
across 24 steps.** Its p19 number (MSE 1.662, at the floor) is therefore a *training* failure, not a capability limit,
and -- exactly as the gate was written in advance -- **the p19 margin may not be quoted until the tuned run replaces
it.** The SSP arm solves the trivial variant to 0.001, so the arm itself is not the problem.

**14.2 Tuned baselines** (4000 steps, 4 configs x 2 search seeds, `--select aulc --min-conv -100`, 10 hold-out seeds).

| arm (tuned: 4000 steps, 4 configs x 2 search seeds) | MSE (hold-out) | untuned p19 number | change |
|---|---|---|---|
| **GRU** | **0.021 ± 0.001** | 1.662 | **79x better -- and better than the SSP arm** |
| `ssp_spatial` | 0.077 ± 0.028 | 0.096 | tuning helps slightly |
| LRU | 0.787 ± 0.055 | 1.108 | 1.4x better |
| `pose_attn` | 1.351 ± 0.026 | 1.351 | unchanged |
| `fwp_delta` | 1.643 ± 0.030 | 1.643 | unchanged |

**The pre-registered withdraw rule fires: tuned GRU MSE 0.021 <= 0.5, so the 17x is WITHDRAWN.** And the correction is
worse than "the margin shrinks" -- **the tuned GRU beats the SSP arm outright**, on every axis that was measured:

| split | GRU (tuned) | SSP (tuned) | who wins |
|---|---|---|---|
| in-distribution | **0.021** | 0.077 | GRU by **3.7x** |
| `extrap_T` (2x episode) | **0.021** | 0.080 | GRU by 3.8x |
| `extrap_motion` (2x step size) | **0.035** | 0.457 | GRU by **13x** |
| `rob_pos` (25% position noise) | 0.774 | **0.742** | tied (SSP marginally) |

**So the spatial recall task does not demonstrate an SSP advantage at all.** The 17x was entirely the untuned baseline:
the same GRU that sat at the no-memory floor with one default config reaches 0.021 with four configs and 4000 steps,
which is 3.7x *better* than the arm built for the task. The untuned p19 tables in section 12.1 are therefore invalid
for comparison purposes and its "the SSP arm wins by an order of magnitude" heading is false -- superseded by this
section. Under decision 4 the robotics argument is **not** supported: SSP is beaten by the tuned GRU by far more than
the parity margin (it does beat `fwp_delta` and `pose_attn`, but that is no longer the standard the rule sets).

**What survives.** Only the *internal* claims: the task is SSP-shaped by construction (the parameter-free phase
identity, 1.4e-06 shifted against 2.000 unshifted), and within the SSP arm the group action is load-bearing (untuned:
0.096 full against 0.303 with `shift=False` and 1.137 with shared roles). Both remain true and neither says the
algebra beats a generic memory. The tuned ablations were not run, so the ablation contrast is only known untuned.

---

## 15. Figures, and two caveats the figures exposed

Figures are in `experiments/ssp_algebra/figures/` (`make_report_figures.py` reads saved result files only; supervised
numbers are hold-out rows, n = 10). Fig 1 axis scoreboard; Fig 2 equal aid; Fig 3 interactions; Fig 4 capability tasks;
Fig 5 spatial reversal; Fig 6 RL.

**15.1 `fwp_delta` and `pose_attn` are identical tuned and untuned on the spatial task (1.643 and 1.351 both times).**
So even with 4 configs x 2 search seeds and 4000 steps neither ever trained: they remain at the no-memory floor (MLP
1.642). The earlier sentence that the SSP arm "does beat `fwp_delta` and `pose_attn`" is therefore *not* a valid
comparison -- neither is a gate-passing baseline. The only spatial comparison with trained opponents is SSP vs the tuned
GRU (0.077 vs 0.021, GRU wins) and tuned LRU (0.787, SSP wins).

**15.2 In the RL curves the GRU overtakes the lag bank at about 600k steps and keeps climbing.** The bank is faster
early (T50 230k vs 361k) and plateaus at +0.958; the GRU reaches +0.993. Any statement of the form "the bank is faster"
therefore holds only for budgets below ~600k steps; past that the GRU is both ahead and still improving. The 33% AULC
advantage is an early-budget effect. One PPO iteration is 32,793 steps, so the T90 gap (492k vs 557k) is two iterations,
which is within the measurement's resolution at 3 seeds.

**15.3 In the gated configurations the "banks" are NOT multi-timescale (found by re-reading `arms_learned.py` 369-390).**
With `gate="input"` the decay is `lam_t = sigmoid(g[:, 1])[:, None]` -- one value per batch row, broadcast over every bank
(`lam_t[:, :, None] * M`), so the per-bank learned horizons (`bank_lams`) are **not used when the gate is on**. They are
used only in the ungated path (`lams[None, :, None] * M`). Consequently:
- `banks=4` *without* a gate (0.962, t90 30.9k vs 61.7k) really is multi-timescale, and its speed gain is a
  multi-timescale result.
- `gate + banks2` and the prior-free `tied + gate + banks2` are **two sub-bundles of D/2 with the same gated decay,
  cleaned up separately and concatenated into a readout that sees 2x the features**. Their gain over `gate` alone
  (t90 51k -> 31k on index; delay 0.488 -> 0.999) therefore cannot be attributed to multiple timescales. What it is
  attributable to -- the state split, the separate cleanup, the doubled readout input -- was not separated. Every
  sentence in this file and in docs/11 that calls the gated configuration "multi-timescale" should be read as
  "split into banks".
