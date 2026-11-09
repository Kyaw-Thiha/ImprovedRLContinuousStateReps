# RUN_010 — speed-aware selection, baseline hardening, the recipe rule, and the pointer controls

**Status: COMPLETE for handoff 09 steps 1-4 (2026-10-11).** Every number is MEASURED unless marked
*inferred*. Every job listed in this record has finished; where a number came from a partial read the row says so.
Protocol: 10 hold-out seeds (1000-1009), dense
evaluation, converged = mean of the last 3 checkpoints, ± = standard error over seeds.

Scope: handoff 09 section 15.4 steps 1-4. Step 1 (speed-aware selection + baseline hardening), step 2 (a range-scaling
recipe rule), step 3 (reliable and transferable interface pretraining), step 4 (the pointer controls: equal-aid chain
baselines, a looped weight-tied transformer, an iterated fast-weight read, the rung-3 cue fix, and the stuck-seed
diagnosis).

---

## 1. Code changes (all with tests; nothing loosened)

| Change | File | Test |
|---|---|---|
| `--search-seeds K` (seeds `ci + 100*k`) and `--select {last5,aulc}` with `--min-conv` floor | `run_sweep.py` | `test_select.py` 10/10 (written first, red on the old code) |
| `weight_decay` {0,1e-4,1e-2} and `dropout` {0,0.1,0.3} searched for **baseline arms only**; `dropout` threaded through `ArmBase`/`GRUArm`/`MLP`/`LRU`/`TF`/`KernelAttn` and `build_arm` | `harness.py`, `arms.py` | `test_regularisation.py` 7/7 (default path bit-identical to before) |
| `recipes.py`: `address_range`, `continuous_address_cfg(R,a,b,c)` freezing **every** knob, `job_entry` | `recipes.py` | `test_recipes.py` 9/9 (pinned values; keys must be consumable) |
| `pretrain_mode` in {retrieval, probe, full} + `SSPIndexLearned.load_address_params` | `harness.py`, `arms_learned.py` | `test_pretrain.py` 6/6 |
| `PointerTask(chain=True)`: each query expanded into `max_hops` supervised sub-steps with a step marker; `ans_mask` so **accuracy is still the answer** | `tasks_b.py`, `harness.py` | `test_pointer.py` (prefix-solver scores 0.500, the prefix average would be 0.818) |
| `LoopedTFArm`: weight-tied looped transformer, per-iteration output, `aux_loss` against `inter` | `arms.py` | `test_pointer.py` (reads out the queried iteration; per-hop labels reach the tied block) |
| Rung 3 in `SSPPointerLearned`: `cue="features"` (query from the corruptible features) and `halting="learned"` | `arms_learned.py` | `test_pointer.py` (the retrieval responds to the cue; old path immune) |
| Iterated fast-weight read for pointer (`FWPArm`) | `arms_learned.py` | `test_pointer.py` (1-hop and 3-hop retrievals differ; index path unchanged) |
| `diag_pointer.py`: role-gate accuracy, query-map alignment, per-hop accuracy, role-gate basin sweep | new | — (diagnostic) |

Test suite at the time of writing: `test_harness` 6/6, `test_fixes` 2/2, `test_measurement` 6/6, `test_learned` 12/12,
`test_analysis` 9/9, `test_trainability` pass (gru/tf_full/fwp >= 0.95 on easy index), `selftest` pass, plus the five new
modules above.

## 2. Step 1 — speed claims under speed-aware selection

Selection: `--select aulc --search-seeds 3 --min-conv 0`, 6 configs, 1500 updates, budget 256 floats, 10 hold-out seeds.
Tokens to 0.90 of ceiling, and the converged score of the config that was actually selected:

index (R=14, 256 floats)

| arm | conv_id | ± | t50 tok | t90 tok | seeds>=0.90 | aulc |
|---|---|---|---|---|---|---|
| **ssp (rung 0)** | **0.996** | 0.001 | 12,888 | **24,447** | 10/10 | 758,178 |
| learned (`ws8 s375`) | 0.993 | 0.001 | 56,498 | 132,037 | 10/10 | 708,248 |
| tf_full | 0.995 | 0.001 | 173,041 | 287,197 | 10/10 | 617,442 |
| fwp | 0.936 | 0.003 | 27,000 | 89,687 | 10/10 | 693,884 |
| gru | 0.857 | 0.089 | 98,680 | 139,814 | 10/10 | 654,933 |
| lru | 0.547 | 0.007 | 348,822 | — | 0/10 | 355,550 |
| kernel_attn | 0.165 | 0.028 | — | — | 0/10 | 119,848 |
| tf_window | 0.227 | 0.031 | — | — | 0/10 | 149,457 |

pointer (n_nodes 8, 256 floats)

| arm | conv_id | ± | t90 tok | seeds>=0.90 |
|---|---|---|---|---|
| **ssp (rung 0)** | **0.999** | 0.001 | **12,800** | 10/10 |
| gru | 0.162 | 0.002 | — | 0/10 |
| lru | 0.160 | 0.003 | — | 0/10 |
| tf_full | 0.137 | 0.004 | — | 0/10 |
| kernel_attn | 0.138 | 0.003 | — | 0/10 |
| tf_window | 0.132 | 0.004 | — | 0/10 |

**Accuracy-matched (floored) re-run** — `--select aulc --min-conv 0.95`, so every arm is compared using a config that
actually converges (job `s1-idx-floor`, complete):

| arm | conv_id | ± | t50 tok | t90 tok | seeds >= 0.90 |
|---|---|---|---|---|---|
| **ssp (rung 0)** | 0.996 | 0.001 | **12,888** | **24,447** | 10/10 |
| fwp | 0.936 | 0.003 | 27,000 | 89,687 | 10/10 |
| learned (`ws8 s375`) | 0.994 | 0.001 | 56,498 | 132,037 | 10/10 |
| gru | **1.000** | 0.000 | 124,311 | 196,254 | 10/10 |
| tf_full | 0.995 | 0.002 | 173,041 | 281,997 | 10/10 |

**The rung-0 speed claim is confirmed at 8.0x** (24,447 vs 196,254 tokens to 0.90) against a GRU whose selected config
converges at 1.000, i.e. not against a weakened baseline. This replaces both RUN_008's convergence-selected "about 4x" and
section 2's unfloored number.

**The learned arm's floored number is 132,037 tokens** (`s1-idx-ws8-floor`, conv 0.994 — the same config the unfloored
selection chose, so the floor changed nothing for it). Against the floored GRU that is **1.49x faster**, and 5.4x slower
than rung 0. Under docs/11 4.5 that fails criterion (a) (it wants 2x, i.e. half the tokens), so the learned arm's thesis
case rests on criterion (b) — clearly higher low-data accuracy at N=256 (0.937 vs the GRU's 0.308) — and not on N=64,
where it collapses (0.233). That is a materially weaker position than "about 4x faster" implied, and it is the honest
post-selection reading.

1. **The rung-0 speed claim is confirmed, but only for rung 0.** Against the *floored* GRU (conv 1.000) the gap is
   8.0x to 0.90 (24,447 vs 196,254 tokens) and 11.5x against `tf_full`. The unfloored table above (5.7x) is superseded:
   it compared against a GRU config that gave up accuracy. This is the re-selected number that replaces RUN_008/009's
   convergence-selected one.
2. **The learned arm's floored re-run landed at 132,037 tokens to 0.90** (conv 0.994): 1.49x faster than the floored GRU,
   which fails docs/11 4.5's half-the-tokens criterion (2x). Its advantage over the GRU is converged accuracy under the
   unfloored protocol and low-data accuracy at N=256, not speed.
3. **The GRU's selected config converges at 0.857**, i.e. speed-aware selection without a floor buys speed by giving up
   accuracy. Its `t90` (139,814) is the first checkpoint that *touched* 0.90. A speed comparison must therefore be run
   under the conv floor (`--min-conv 0.95`), which is what the handoff's "AULC subject to converged >= x" means; that
   re-run is job `s1-idx-floor` (ssp, gru, fwp, tf_full), whose numbers are in the table above.
4. **Pointer: no generic arm learns it at this budget** (0.13-0.16 against chance 0.125), while rung 0 is at 0.999 after
   12,800 tokens. The learned pointer arm and the controls are step 4 (section 5).

## 3. Step 1 — low data (fixed pool, trained to convergence)

Selection for these runs is `--select last5` (the *converged* criterion), because the claim here is accuracy, not speed;
6 configs on 3 search seeds, 10 hold-out seeds, 1500 updates. "Pool N" = N training sequences generated once, minibatches
sampled from that pool, evaluated on fresh sequences.

| arm | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|
| **ssp (rung 0)** | **0.978** | **0.996** | **0.997** | **0.997** |
| learned (`ws8 s375`) | 0.233 | 0.937 | — | — |
| fwp (hardened) | 0.382 | 0.849 | 0.906 | 0.933 |
| gru (hardened) | 0.216 | 0.308 | 0.996 | 0.812 ±0.086 (6/10) |
| tf_full (hardened) | 0.199 | 0.224 | 0.994 | 0.995 |

(± for the pooled runs: ssp ±0.001 throughout; gru at N=4096 is 6/10 seeds above 0.9, i.e. unstable, and its N=4096 mean is
*lower* than its N=1024 mean — a training-stability artefact, not a data-efficiency claim in either direction.)

**Decisions under the handoff's step-1 rules (drop a claim if the gap < 2x, or if tuned baselines come within 5 points).**

- **Rung-0 low-data claim: survives.** N=64: 0.978 against the best hardened baseline (fwp 0.382) is a 60-point gap;
  N=256: 0.996 against 0.849 is 15 points. Hardening the baselines (weight decay, dropout, 3 search seeds) did not
  close it.
- **Rung-0 speed claim: survives and is confirmed at 8.0x** under the floored selection (section 2); the earlier 5.7x was
  measured against an unfloored GRU config that only converged at 0.857.
- **Learned-arm speed claim: fails the 2x rule** (1.49x vs the floored GRU: 132,037 vs 196,254 tokens to 0.90; section 2).
  It is also *slower* than the `fwp` control to 0.90 (89,687 tokens), which matters for any SSP-specificity claim.
- **Learned-arm low-data claim: survives only at N=256** (0.937 vs fwp 0.849, about 6 pooled se), and **fails at N=64**
  (0.233, 0/10 seeds): the collapse RUN_008 measured is confirmed under the hardened protocol. This is exactly what the
  pretraining study (section 6) is meant to fix, and it is the reason the learned arm cannot inherit rung 0's low-data
  claim.

## 4. Step 2 — the range-scaling recipe rule

T=16 grid (R=14, 10 hold-out seeds each, the rule's constants frozen at every grid point; 11 runs per condition):

| (a, b) | conv_id | ± | t90 tok | seeds >= 0.95 | aulc |
|---|---|---|---|---|---|
| 0.5, 24 | 0.984 | 0.003 | **130,728** | 10/10 | 285,238 |
| 0.5, 47 | 0.994 | 0.001 | 174,379 | 10/10 | 601,215 |
| 0.5, 94 | 0.994 | 0.001 | 242,220 | 10/10 | 1,254,468 |
| 1.0, 24 | 0.990 | 0.002 | 144,841 | 10/10 | 276,782 |
| 1.0, 47 | 0.993 | 0.001 | 240,969 | 10/10 | 581,335 |
| 1.0, 94 | 0.992 | 0.001 | 338,538 | 10/10 | 1,221,496 |
| 1.5, 24 | 0.986 | 0.002 | 148,694 | 10/10 | 269,328 |
| 1.5, 47 | 0.994 | 0.001 | 256,417 | 10/10 | 567,675 |
| 1.5, 94 | 0.995 | 0.001 | 404,974 | 10/10 | 1,195,929 |

T=32 grid (R=30, 256 floats, 10 hold-out seeds each; all nine conditions complete):

| (a, b) | start width | schedule | steps | conv_id | ± | t90 tok | seeds >= 0.95 |
|---|---|---|---|---|---|---|---|
| **0.5, 24** | 15 | 720 | 1440 | 0.993 | 0.001 | **251,262** | 10/10 |
| 0.5, 47 | 15 | 1410 | 2820 | 0.995 | 0.001 | 353,850 | 10/10 |
| 0.5, 94 | 15 | 2820 | 5640 | 0.993 | 0.001 | 512,389 | 10/10 |
| 1.0, 24 | 30 | 720 | 1440 | 0.993 | 0.001 | 302,641 | 10/10 |
| 1.0, 47 | 30 | 1410 | 2820 | 0.992 | 0.001 | 471,403 | 10/10 |
| 1.0, 94 | 30 | 2820 | 5640 | 0.992 | 0.002 | 748,126 | 10/10 |
| 1.5, 24 | 45 | 720 | 1440 | **0.996** | 0.001 | 312,913 | 10/10 |
| 1.5, 47 | 45 | 1410 | 2820 | 0.994 | 0.001 | 527,853 | 10/10 |
| 1.5, 94 | 45 | 2820 | 5640 | 0.994 | 0.002 | 871,053 | 10/10 |

**The rule is what was missing, not the constants.** At T=32 every grid point clears 0.95 on 10/10 seeds, where RUN_009's
*unscaled* constants (start 8, 750 updates) reached 0.665. Scaling the start width and the schedule length with the
address range is the load-bearing part.
**Frozen rule** (chosen by the pre-registered criterion: fastest t90 subject to conv >= 0.95 — on T=16 as well as T=32):
`start = 0.5 * R`, `schedule = 24 * R` updates, `steps = 48 * R`, `lr = 0.02`, `lam_init = 0.99`, `temp = 0.1`, i.e. the
defaults already in `recipes.py`; no constant was changed after seeing T=32. (Note: the raw AULC column cannot rank
conditions across different run lengths — a longer run accumulates a larger area — so t90 is the ranking quantity here.)

**Unseen-variant test, pre-registered (docs/11 4.7):** the frozen rule on index T=24/48/64 (R = 22/46/62; 512 floats at
T=48/64 so the D/4 capacity rule is not the limit) and on delay with lags 1-4 at T=16 and 1-8 at T=32, 10 hold-out seeds
each, **no retuning**; pass = at least 0.95 on at least 9 of 10 seeds on every variant; fail = two or more variants need
retuning (then the honest report is "a variant-specific trick", and the fallback is a multi-scale frequency bank).
Jobs `s2-un24`, `s2-un48`, `s2-un64`, `s2-un-dl4`, `s2-un-dl8` ran with these numbers, frozen, no retuning (10 hold-out
seeds each; 512 floats at T=48/64 so the D/4 capacity rule is not the limit):

| unseen variant | R | conv_id | ± | min seed | seeds >= 0.95 | t90 tok | verdict |
|---|---|---|---|---|---|---|---|
| index T=24 | 22 | 0.995 | 0.001 | 0.990 | 10/10 | 183,360 | pass |
| index T=48 | 46 | 1.000 | 0.000 | 1.000 | 10/10 | 348,758 | pass |
| index T=64 | 62 | 1.000 | 0.000 | 1.000 | 10/10 | 486,748 | pass |
| delay lags 1-8 | 8 | 0.999 | 0.000 | 0.997 | 10/10 | 121,374 | pass |
| delay lags 1-4 | 4 | 0.929 | 0.039 | **0.657** | **7/10** | 43,245 | **fail** |

**This is the first recipe in this study to reach the reliability protocol, with one failed variant.** Four of five unseen
variants clear 0.95 on 10/10 seeds with no retuning, including both stream lengths that RUN_009's constants failed at
(0.665 at T=32 then; T=48/64 are now 1.000). The failure is the shortest address range (R=4, where the rule's schedule is
only 96 updates and its start width 2).

Both readings must be stated, because the pre-registration contains two thresholds:

- docs/11 4.7 as written says "at least 0.95 accuracy on at least 9 of 10 seeds **on each** unseen variant": under that
  reading this is a **partial pass** (4/5 variants).
- handoff 09 section 15.4 step 2's kill rule is "if **two or more** unseen variants need retuning, report a
  variant-specific trick": under that reading the recipe **passes**, with one recorded failure.

No retuning was done on the failing variant, and none may be: retuning on a previously unseen variant converts the study
into per-problem tuning (docs/11 3.5). The hypothesis for the failure is *not* tested and is therefore a hypothesis only:
the rule is linear in R with no floor, so very short ranges get a schedule too short to discover the address, which is
what an additive constant or a minimum schedule length would fix. Testing that would be a new rule, evaluated on its own
unseen variants.

## 5. Pointer verdict, and the controls behind it

The handoff lists four conditions for the pointer pass to stop being provisional. Their status now:

| condition | status | evidence |
|---|---|---|
| (c) equal-aid baselines | **met** | chain protocol, 3 baselines at chance (0.154-0.183) with ~7.7M tokens, ~140x the SSP arm's 54k |
| (d) an arm that reads the cue from the corruptible features | **met** | rung 3: retrieval responds to the cue, corruption curve 0.972 -> 0.201 monotone |
| (b) a non-SSP iterated control | **not met, and not refuted** | `LoopedTFArm` exists and trains, but **fails its own trainability gate** (best 0.416 on the easy 1-hop variant at 6000 updates), so it cannot be used to argue either way about "iteration plus supervision" |
| (a) less reliance on the per-hop aid | **partially** | RUN_009: plain training succeeds on 6 of 10 seeds; this run's rung-3 arm reaches 0.972 in-distribution *with* the aid (and 0.361 on held-out hops, down from 1.000 with the hop count given) |

**So the pass remains provisional, and the reason is now precise:** the two conditions that could be met were met (equal aid
fails to rescue the baselines; the cue reaches the arm), but the control that would falsify "iteration plus supervision" is
itself untrainable on this task, and dropping the given hop count costs most of the extrapolation. What is solid is that
*no competent non-SSP arm exists on the full task within these budgets*: the only generic arm that learns anything is the
GRU at 0.478 after 6000 updates, against the SSP arm's 1.000 in about 54k tokens.

All controls are now complete, and they changed no verdict: the chain runs added three more chance-level baselines
(0.145-0.181), and the looped arm at 6000 updates reached only 0.227 (best seed 0.410).

## 5.1 The measurements


**Pointer trainability gate, complete (easy variant: 4 nodes, 1 hop, 6000 updates, 3 hold-out seeds; probability of a
random answer is 0.25):**

| arm | per-seed conv_id | best | gate |
|---|---|---|---|
| gru | 0.999, 0.256, 0.996 | 0.999 | **pass** (2 of 3 seeds) |
| looped_tf | 0.248, 0.416, 0.244 | 0.416 | **fail** |
| kernel_attn | 0.350, 0.259, 0.245 | 0.350 | **fail** |
| tf_full | 0.260, 0.236, 0.256 | 0.260 | **fail** (at chance) |

**This changes which pointer numbers may be used at all.** By the study's own rule (a baseline's number is not used until
it clears a trivially representable version of the task), `tf_full`, `kernel_attn` and `looped_tf` **fail the pointer gate**
— `tf_full` is at chance (0.25 for 4 nodes) after 6000 updates — so their pointer numbers carry no claim, exactly as
RUN_007 excluded `tf_full` on index. The pointer comparison is therefore SSP against **the GRU** (0.478 at 6000 updates on
the full task, 0.999 on the easy variant for 2 of 3 seeds) and `fwp` (0.176).

**Consequence for the non-SSP iterated control:** the looped transformer's 0.151 on the full task is not evidence that
"iteration plus supervision" fails to explain the pointer pass, because the arm cannot even do the easy 1-hop version
(0.416 best). It is a control that failed its own gate — so the "iteration" explanation is neither confirmed nor refuted by
it; the equal-aid test therefore rests on the chain-protocol baselines, all six of which are at chance.

**Full pointer task at 6000 updates (8 nodes, trained on hops 1-2; complete, 10 hold-out seeds each):**

| arm | conv_id | ± | best seed | best value anywhere in the curve |
|---|---|---|---|---|
| gru | 0.478 | 0.015 | 0.539 | 0.542 |
| fwp | 0.176 | 0.003 | 0.189 | 0.207 |

No phase transition on the full task within 6000 updates, for either arm (the GRU's easy-variant transition does not
happen here). Worth recording about the easy variant: the GRU's winning config sat at ~0.25 for the first 31 of 73
checkpoints (about 2.5M tokens) and then jumped to ~0.99 within three checkpoints, and the same config's *search* run
stayed at 0.25 to the end — so on this task "has not learned it at budget X" and "cannot learn it" are different
statements, and a single search seed can miss a config that would have solved the task (the selection hazard RUN_009
found for speed, now for learnability itself).

So the pointer claim is stated as: the SSP arm reaches 1.000 in about 54k tokens (rung 1 + per-hop labels,
RUN_008) and 0.999 at rung 0 in 12.8k tokens, while the best generic arm is at 0.47 after 6000 updates (~25M tokens) — a
separation in both tokens and final accuracy, but not "the baselines cannot learn this at any budget", which the easy
variant's transition disproves.


**Looped weight-tied transformer (the non-SSP iterated control), pointer, 10 hold-out seeds, per-hop labels as the aid:**

| arm | in-distribution | held-out hops 4-8 | seeds >= 0.9 | t90 tok |
|---|---|---|---|---|
| looped_tf + per-hop labels (1500 updates) | **0.151** | 0.204 | 0/10 | — |
| looped_tf plain (1500 updates) | 0.156 | 0.203 | 0/10 | — |
| reference: rung-1 SSP + per-hop labels (RUN_008) | 1.000 | 1.000 | 5/5 | ~54,000 |
| reference: rung-3 SSP (this run) | 0.972 | 0.361 | — | 199,111 |

The per-hop labels change nothing for the looped arm (0.151 vs 0.156), so its failure is not a supervision-shape problem,
and it is the same arm that fails the easy-variant gate (best 0.416 at 6000 updates).

**At the matched 6000-update budget it is still not competent:** 0.227 mean, best seed 0.410 (`s4-loop-auxlong`, 10
seeds). So it does improve slowly with the labels (0.151 -> 0.227 from 1500 to 6000 updates) but remains far from the task
and far below its own gate, which means it cannot serve as the non-SSP iterated control that would falsify "iteration plus
supervision". That is why the pointer verdict's condition (b) is recorded as *not met and not refuted* rather than as
evidence either way.


**Equal-aid chain baselines, complete (full task, 8 nodes, trained on hops 1-2, 1500 updates):** the chain protocol gives
each baseline what the SSP arm gets as per-hop supervision — every query is expanded into its hop count of sub-steps, each
with a supervised target `f^k(s)` — and only the *answer* step is scored, so the aid cannot inflate the metric.

| arm | conv_id | ± | best seed | held-out hops 4-8 | seeds >= 0.5 |
|---|---|---|---|---|---|
| gru | 0.183 | 0.003 | 0.197 | 0.165 | 0/10 |
| lru | 0.181 | 0.004 | 0.205 | 0.163 | 0/10 |
| looped_tf | 0.181 | — | 0.200 | — | 0/10 |
| tf_full | 0.154 | 0.004 | 0.168 | 0.194 | 0/10 |
| kernel_attn | 0.145 | — | 0.156 | — | 0/10 |
| tf_window | 0.145 | — | 0.171 | — | 0/10 |
| reference: rung-1 SSP + per-hop labels | 1.000 | — | — | 1.000 | 5/5 |
| reference: rung-0 SSP | 0.999 | — | — | — | 10/10 |

**The equal aid does not rescue the baselines.** All three are at chance (0.125) — and the chain protocol is *expensive*:
each update sees an 80-step episode instead of 24, so 1500 chain updates is roughly 7.7M tokens, about **140x the SSP
arm's 54k tokens** for the same task. One caveat keeps this honest: the chain protocol changes the input sequence, and the
GRU's easy-variant solve was a delayed phase transition, so "no transition within this budget" is what is measured, not
"cannot". The chain protocol gives the same answer for every baseline tested: all six are at 0.145-0.183.


**Rung 3 — no given hop count, query decoded from the corruptible features (10 hold-out seeds, per-hop aux as the aid):**

| split | conv_id | ± |
|---|---|---|
| in-distribution | 0.972 | 0.022 |
| held-out hops 4-8 | 0.361 | 0.008 |
| cue drop 25% | 0.606 | 0.010 |
| cue drop 50% | 0.357 | 0.005 |
| cue drop 75% | 0.201 | 0.006 |
| input noise 10% | 0.714 | 0.019 |
| input noise 25% | 0.465 | 0.009 |

- **The cue-drop split is finally non-vacuous**: the arm now reads the start and hop count from the same features the
  baselines get, so corruption reaches it (RUN_008 section 7 item 2 is fixed for this arm, and the test in
  `test_pointer.py` pins it: the retrieval, not just the output, changes).
- **Degradation is monotone, but it is not "graceful" as pre-registered** (the handoff's example criterion is "accuracy at
  50% corruption >= 0.8 of clean"): 0.357 at 50% is 0.37 of the clean 0.972. The honest statement is "well above chance
  through 50% corruption, failing the pre-registered gracefulness rule". A baseline comparison is uninformative here
  because the generic pointer arms are at chance even on the clean task (section 2).
- **Removing the hints costs extrapolation far more than in-distribution accuracy**: rung 1 with the same per-hop aid
  reached 1.000 on held-out hops 4-8 with the hop count given; rung 3 reaches 0.361. The learned halting distribution has
  to generalise to hop counts it was never trained on, and it does not.

**Stuck-seed diagnostic (`diag_pointer.py`, local, 4 seeds, 1200 updates, lr 1e-2, n_nodes 6, budget 256):**

| seed | id | extrap (hops 4-8) | role-gate acc vs parity | qmap identity | stuck |
|---|---|---|---|---|---|
| 0-3 | 1.000 | 1.000 | **0.500** | 1.00 | no |

- **No stuck seeds at this configuration**: at lr 1e-2 all four seeds solve both the in-distribution split and held-out
  hops 4-8. So the ~40% stuck rate RUN_009 estimated is a property of *the configs the search drew*, not of the arm: the
  lr range (3e-4 … 3e-2) is wide and the arm needs the top of it.
- **The role gate never learns parity (0.500) even on seeds that solve the task**, and the query map does learn the
  identity (1.00). That is a load-bearing-free component: the same pattern as RUN_007/008's ablations, now visible inside
  the learned arm. Consequence for step 4: an intervention on the role gate (initialising it, scheduling it) is unlikely
  to be the fix for stuck seeds, because the arm does not need the gate at all.
- **The basin sweep is not informative as run**: it used 400 steps (a third of the training budget), where every sigma
  sits near 0.60 and only `sigma=2.0` reaches 0.80 (one seed at 1.0). It needs full-length runs before it can be read as
  "the basin is wide/narrow".

## 6. Step 3 — interface pretraining

20 hold-out seeds, pool 64, the same frozen width schedule in every condition; "aligned" = the seed reaches 0.9 converged
accuracy at N=64, measured *because* the N=64 collapse is the address-learning failure (RUN_009 section 1.3):

| condition | conv_id | min seed | seeds >= 0.9 | Wilson 95% | pretrain tokens |
|---|---|---|---|---|---|
| **probe** (throwaway readout on the memory) | **0.960** | 0.860 | **19/20** | 0.76-0.99 | 255,952 |
| retrieval loss (the previous mode) | 0.376 | 0.141 | 2/20 | 0.03-0.30 | 255,952 |
| no pretraining (control) | 0.257 | 0.171 | 0/20 | 0.00-0.16 | 0 |

**Reading (the first two adoption conditions).** The probe mode fixes the low-data collapse: trained through a throwaway
readout on the memory's own features, the interface aligns on 19 of 20 seeds, where the retrieval-loss variant
(RUN_009's 2-of-5) manages 2 of 20 and the control 0 of 20. The two conditions the handoff sets — at least 18 of 20 seeds
aligned, and N=64 at least 0.9 — are both met. The **third condition, transfer, is the measurement below**, so the mode is
adopted *conditionally* until that lands. Its 256k pretraining tokens are reported separately from the task tokens, as
required.

**Transfer protocol (the third adoption condition).** `transfer_test.py` implements the two-stage protocol per seed: pretrain the interface at the source shape with the probe mode, copy only the interface
(`load_address_params`) into a destination arm of a *different* shape, freeze it, and train **only the readout** on the
destination task. Two axes: T=16 -> T=32 (longer stream) and 8 -> 16 items (a different codebook size). Reported next to
the in-place destination result from the recipe run (0.995 at T=24, 1.000 at T=48/64).

**Transfer result: FAILS on both axes** (10 seeds each, frozen interface copied into the destination arm, only the readout
trained; `need` = 9 of 10 seeds, the same threshold as the recipe protocol):

| axis | readout steps | dst conv_id | min | seeds >= 0.9 | in-place reference | task tokens |
|---|---|---|---|---|---|---|
| T=16 -> T=32 | 800 | 0.452 | 0.381 | 0/10 | 1.000 (T=48/64), 0.995 (T=24) | 819,521 |
| T=16 -> T=32 | 2,976 | 0.441 | 0.374 | 0/10 | as above | 3,048,726 |
| 8 -> 16 items | 800 | 0.697 | 0.639 | 0/10 | (not measured) | 409,572 |
| 8 -> 16 items | 2,976 | 0.691 | 0.645 | 0/10 | (not measured) | 1,523,337 |

- **More readout training does not help** (0.452 -> 0.441 and 0.697 -> 0.691 with 3.7x the updates), so the limitation is
  the transferred interface itself, not an under-trained readout. The pretraining tokens (256k per seed) are therefore
  *not* amortised.
- **Verdict under the pre-registered rule**: alignment passes (19/20), the N=64 target passes (0.960), transfer **fails**
  -> the pretraining stage is reported as **"more training data", not a reusable interface**. It fixes the low-data
  collapse for the shape it was pretrained on; it does not carry to a longer stream or a larger alphabet.
- Partial signal worth noting, not a claim: the alphabet axis (0.69) transfers much better than the length axis (0.44),
  which is consistent with the address heads being *position*-based (a codebook-size change does not move the address
  range) while a longer stream moves the address range the read head was fitted on. *Inferred, untested.*
- The missing no-transfer control from the first run is reported directly below (random frozen interface, same budget).

**No-transfer control (the missing baseline above): same destination shapes, same readout budget, but with a random
*frozen* interface and no pretraining at all:**

| axis | readout steps | transfer | no-transfer control | in-place recipe |
|---|---|---|---|---|
| T=16 -> T=32 | 2,976 | 0.441 | **0.182** (0/10 seeds, min 0.133) | 1.000 |
| 8 -> 16 items | 2,976 | 0.691 | **0.198** (0/10 seeds, min 0.168) | (not measured) |

So the interface does transfer *something* — roughly 2.4x the no-transfer baseline on the length axis and more on the
alphabet axis — but it reaches neither the pass threshold nor the in-place result. The verdict is unchanged and now
properly controlled: not a reusable interface; useful only for the shape it was pretrained on.

Per the pre-registered rule the mode is therefore adopted *conditionally*: it works for the shape it was trained on and
must not be quoted as amortised.

## 7. Errors, corrections and limitations

1. **`--select aulc` without a conv floor selects a config that gives up accuracy** (the GRU at 0.857). The rule the
   handoff states is "AULC *subject to* converged >= x", so the floored re-run is the number to quote; the unfloored
   table above is retained only to show the selection's effect.
2. **RUN_008/009's speed numbers are superseded** for index and pointer rung 0 by section 2: use "8.0x to 0.90, rung 0,
   floored selection" and drop any earlier "about 4x faster" phrasing.
3. The GRU/RUN_008 comparison of 0.999 converged is not contradicted: it came from a different (unregularised) search
   space selected on converged accuracy. Search-space changes are not comparable (RUN_009 limitation), hence new tags.
4. **Correction to RUN_008 section 7 item 2** (`test_pointer.py::test_cue_fix...`): the old pointer arm's *retrieval*
   is immune to cue corruption, but its *readout input* is not, so "the arm is immune by construction" was too strong.
5. Chain mode changes the task's `ep_len` and `in_dim`, so chain-protocol numbers are only comparable with each other;
   the SSP arm cannot run on a chain batch (its query-step indexing assumes one step per query), so chain jobs contain
   baseline arms only.
5b. **Job-list row counts must be `len(arms) * (configs * search_seeds + seeds)`.** Two of my lists declared the count
   with `search_seeds` folded away (`s4-ptrlong` said 112 for a 56-row job), which made `status` report the finished job as
   incomplete and `collect` refuse to download it; the data was on R2 and the script had simply produced exactly its full
   set. The pod had self-deleted because it succeeded. No data was lost, but the completion check is only as good as that
   number.
6. The looped transformer is an unlimited-state class arm (like `tf_full`); its state accounting is reported, not matched.

7. **The pretraining stage does not transfer** (section 6): alignment 19/20 and the N=64 target are met, but the interface
   learned at T=16 does not carry to T=32 (0.44) or to 16 items (0.69), and 3.7x more readout updates does not change
   that. It is reported as more training data, not as an amortised interface, per the pre-registered rule.
8. **The no-transfer control was added after the first transfer run** (section 6): 0.182 (length axis) and 0.198
   (alphabet axis) against 0.441 and 0.691 with the transferred interface, so the interface carries information but not
   enough to pass.
9. **The looped-transformer control is provisional** (section 5): it is at chance at 1500 updates with per-hop labels; the
   6000-update run is the fair statement and it is still not competent (0.227).
10. **The pointer trainability gate is complete** (GRU passes, tf_full/kernel_attn/looped_tf fail); no pointer
   baseline number may be used until it passes.
11. **The stuck-seed diagnostic used 4 seeds at one learning rate** and found no stuck seeds, and its basin sweep ran a
    third of the training budget. It is a lead (the stuck seeds are config-dependent), not a measured stuck rate.
12. **Batch size is fixed at 64 for every arm and was never searched.** The learned arm's schedule runs in *updates*, so
    its token cost depends on the batch; baselines might also move. Every "tokens to 0.90" number here is therefore
    conditional on batch 64. Handoff 10 Phase 0 tests this.
13. **`fwp` never cleared the 0.95 converged floor** in the floored speed run (0.936), so its row is the best-available
    fallback; its pointer variant, `lru` and `tf_window` were not put through the pointer trainability gate.
14. **Only one task family drives the recipe** (index and delay, two address ranges tuned, five unseen). Nothing here
    says the rule transfers to discrete-key or continuous-state tasks. The number of conditions tried while designing the
    rule: 18 grid points plus 5 unseen variants.

## 8. Do not do

- Do not quote the unfloored speed table as the speed result; the floored table is the claim (8.0x rung 0, 1.49x learned).
- Do not claim the *learned* index arm passes the thesis rule on speed: 1.49x fails the 2x half-the-tokens criterion, and
  its low-data case rests on N=256 alone (N=64 collapses at 0.233).
- Do not quote the pretraining stage as amortised: transfer fails on both axes.
- Do not compare chain-protocol baseline numbers with standard-protocol numbers, and do not run the SSP pointer arms on a
  chain batch (their query-step indexing assumes one step per query).
- Do not read the T=16 grid differences as a rule fit: at T=16 every grid point passes; the constants come from T=32.
- Do not cite any pointer baseline other than the GRU as a competent comparison: `tf_full`, `kernel_attn` and `looped_tf`
  fail the pointer trainability gate, and `fwp`, `lru` and `tf_window` were never gated.
- Do not retune the frozen rule on the delay-1-4 variant that failed the unseen test: retuning on an unseen variant is
  what makes the study per-problem tuning.

## 9. Literature check (added 2026-10-11) and what it implies

Searched alphaXiv and the web after this run. Coverage is incomplete: alphaXiv indexes arXiv only, and several items were
seen as snippets (marked). **No paper was found that studies why gradient descent fails to find the read address of an
SSP/FHRR memory, builds a validated training recipe for it, or compares it at matched state against GRU, transformer and a
fast-weight control.** Details and the full list are in `docs/handoffs/10-ssp-discoverability-and-rl-port-prep.md`.

- The coarse-to-fine width schedule is **not novel as a technique**: it is the progressive frequency activation of BARF
  (2021) and FA-BARF (2025, which targets the slowdown such a schedule causes: the same bottleneck measured here) (snippets).
  What this study adds is the diagnosis (basin width tracks kernel width), the range-scaling rule validated on unseen
  variants, and the matched-budget evaluation.
- Capacity, not the algebra, limited multi-hop composition in a holographic memory with 14.5k entities (arXiv 2606.24948,
  single-author preprint). The pointer pass here uses 8 nodes at D=256 and has not been scaled.
- Training through HRR operations found frozen random vectors better than learned ones (Ganesan et al. 2021, seen secondhand
  through arXiv 2609.28409): consistent with the learned-codes ablation matching the full arm here.
- A GRU discovers path integration and object-location binding on its own (arXiv 2602.03490): the generic competitor is
  capable, which is consistent with the GRU's delayed phase transition on the easy pointer variant.
