# Report on SSP/VSA state and memory experiments, with next steps

Written 2026-10-31. Last experiment activity was 2026-10-14 (RUN_013). Nothing is committed.

Method:
- **Re-derived from raw files:** the 25 POPGym files, the RL summaries and per-iteration rows, the RUN_013 supervised tables (via `report_holdout.py`), the analysis/transfer JSONs, and the `tied_bank` curves.
- **Taken from the run records:** the CartPole numbers (their raw data is on R2, not in the repo) and RUN_005-012 details, summarised by delegated readers and spot-checked.
- Section 8 lists the discrepancies found.

---

# 1. Executive summary

There are three lines of work.

- **CartPole, SSP as state encoder:** negative to neutral.
  - SSP is below tile coding on the greedy score under DQN and A2C.
  - It is far better under action delay.
  - The legacy claim that SSP beats tile coding under A2C was an artifact of the online metric.
- **SSP/VSA as memory:** the algebra works, but it has not beaten a properly tuned generic memory.
  - Several earlier "wins" were withdrawn once baselines were given equal aid or tuning.
  - The input-dependent gate gave the largest single gain, and it helps fast-weight memories as much as SSP.
  - The spatial-recall result, the best-looking SSP niche, reversed: a tuned GRU is 3.7x better.
- **RL on POPGym `RepeatPreviousEasy`:** one real positive.
  - The lag-bank arm `tied_bank` learned the task in 3 of 3 seeds with no lag constant.
  - It finished at +0.958 against the GRU's +0.993, inside the 5-point parity margin.
  - It hit the 50% and 90% thresholds 1.57x and 1.13x sooner than the GRU.
  - It is worse per step, and the GRU overtakes it at about 600k steps.

**Success rule** (decision 2 in doc 11): parity, plus better on at least one pre-listed metric.
- Parity holds in RL and on index/delay.
- The >=2x sample-efficiency margin holds only on one task family at one threshold (index, t90).
- Prior-free configurations are worse at length extrapolation.
- Per-step cost is worse.
- So SSP is currently "a working memory with no demonstrated reason to prefer it".

---

# 2. Timeline

| Date | Run | What |
|---|---|---|
| to Sep 2026 | RUN_001-003 | DQN/A2C search on CartPole: 3 representations x 3 centerings x linear/deep (18 DQN cells, 9 A2C), Optuna, greedy re-evaluation |
| 2026-10-06 | RUN_004, docs 08-10 | Robustness suite, mechanism tests, re-scope: CartPole becomes a negative control, direction becomes memory |
| 2026-10-10 | RUN_005, RUN_006 | Encoder-only battery, POPGym first pass (in-repo A2C) |
| 2026-10-11 | RUN_007-010 | Supervised harness: hand-wired, then learned structure; recipe diagnostics |
| 2026-10-12 | RUN_011, RUN_012 | Discoverability mechanism, halting, practitioner baselines, candidate scoreboard |
| 2026-10-13/14 | RUN_013 | Equal aid, prior-free memory, lag bank in RL, spatial gate (failed) |

---

# 3. CartPole encoder track (RUN_001-004)

**Protocol.**
- 1000 episodes per run, greedy evaluation, score = mean of the last 5 of 15 checkpoints, "solved" >= 475.
- Seed blocks: search 0-499, filter 1000+, final 2000-2019, replicate 3000-3019.

## 3.1 DQN (20 seeds, mean +- std)

| Cell | RUN_003 | RUN_004 replicate | Solved | Dim |
|---|---|---|---|---|
| tile/none/linear | 498.6 +- 3.6 | 499.64 +- 0.99 | 100% | 32,768 |
| tile/simple/linear | 496.3 +- 7.1 | 496.31 +- 7.02 | 100% | 131,072 |
| tile/value/linear | 495.4 +- 6.6 | 493.04 +- 9.58 | 90% | 65,536 |
| tile/value/deep | 498.0 +- 2.8 | 491.52 +- 13.26 | 80% | 65,536 |
| tile/none/deep | 489.7 +- 20.1 | 489.46 +- 11.85 | 85% | 32,768 |
| ssp/none/deep | 486.4 +- 21.4 | 489.34 +- 21.69 | 85% | 951 |
| ssp/value/deep | 474.6 +- 26.7 | 478.55 +- 24.97 | 70% | 201 |
| tile/simple/deep | 479.5 +- 18.4 | 472.97 +- 26.09 | 60% | 65,536 |
| ssp/value/linear | 468.7 +- 34.6 | 470.21 +- 28.06 | 60% | 1,001 |
| ssp/none/linear | 469.0 +- 30.7 | 461.65 +- 35.88 | 45% | 1,001 |
| ssp/simple/deep | 363.6 +- 67.3 | 369.87 +- 65.52 | 0% | 651 |
| ssp/simple/linear | 311.5 +- 101.6 | 313.27 +- 86.73 | 5% | 251 |
| discrete (6 cells) | 220.8-265.8 | 230.5-267.2 | 0% | 14k-130k |

## 3.2 A2C (NumPy port, 20 seeds)

| Cell | Mean +- std | Solved |
|---|---|---|
| tile/value | 499.70 +- 1.30 | 100% |
| tile/none | 499.45 +- 1.38 | 100% |
| tile/simple | 483.46 +- 16.07 | 80% |
| ssp/value | 473.86 +- 71.86 | 85% |
| ssp/none | 470.00 +- 60.31 | 80% |
| ssp/simple | 459.92 +- 50.86 | 50% |
| discrete (simple / value / none) | 300.46 / 173.31 / 131.95 | 0% |

The legacy table said SSP >= tile under A2C (498.4 vs 495.0 for no centering). It used the online metric, and the winners' exploration epsilons differ by about 10x. Those configs also match no surviving Optuna trial, so that comparison cannot be tested.

## 3.3 Centering, robustness and mechanism

- **Centering.**
  - It is applied naively to an episodic +1-per-step task, which removes the failure signal.
  - `simple` centering hurt SSP by 10.1 points, tile coding by 16.0, and helped discrete by 168.5.
  - Its beta winners sit at the 1e-5 lower bound, so those cells mostly measure "centering switched off".
  - `value` centering was harmless to mildly positive.
- **Robustness**, retained score, primary cells, DQN:

| Cell | action_delay | random_action | pole_length | obs_noise |
|---|---|---|---|---|
| tile/none/linear | 0.542 | 0.977 | 0.938 | 0.990 |
| tile/none/deep | 0.303 | 0.856 | 0.931 | 0.932 |
| tile/value/deep | 0.278 | 0.818 | 0.892 | 0.947 |
| ssp/none/linear | 0.923 | 0.997 | 0.855 | 0.956 |
| ssp/none/deep | 0.988 | 1.001 | 0.840 | 0.973 |
| ssp/value/deep | 0.993 | 1.003 | 0.847 | 0.971 |

  - SSP wins `action_delay` in all 4 pairs by 153-342 points, Holm-significant.
  - Under A2C the delay pattern reverses, so it is not representation-general.
  - Ordering by the policy's feedback gain `dQ/dx` fits retention in 4 of 4 cells, which is a correlation across 4 points.
- **Settings test.** A target update of 500 is catastrophic everywhere. The SSP collapse is not a replay or target-network artifact.

## 3.4 Why SSP was worse (RUN_005, encoder-only)

- A single isotropic `length_scale` cannot resolve theta (range 0.418) and omega (range 20) at once.
  - Resolving theta at the failure cliff needs `length_scale` = 0.0417.
  - That is 2.4x below the search floor.
  - Per-dimension scaling fixes theta inside the searched range (not run as a learning experiment).
- Interference (H1): SSP disturbs a probe set about 12x less than tile coding at a common learning rate. The hypothesis that dense SSP interferes more is not supported.
  - Tile coding's hash table is full, with 81-98% of lookups colliding, so it is not a clean local basis.
  - The deep-pair numbers are contaminated by value centering.
- Headroom: SSP linear 461.6 vs tile 499.6, so at most about 38 points. The best case is parity.

**Open and cheap:**
- H2 (per-dimension normalization x `length_scale`).
- Centering under the De Asis termination-corrected form.
- A2C winners re-derived from the Optuna DBs.

---

# 4. Encoder-only memory battery (RUN_005)

| Test | Result |
|---|---|
| T0 algebra exactness | bind 2.7e-15, unbind 3.2e-15 |
| T1 kernel floor | mean floor < 0.05 at Dc~1000 (1/sqrt(D)) |
| T4 capacity (depth 1-3, 10 seeds) | reliable up to m ~ Dc/4. Real dim 201: m=10 0.89, m=20 0.55. Real dim 1001: m=50 0.90-0.94, m=100 0.55. Chained unbinding is free |
| T6 cue dropout | collapses at Dc~200 (p=0.75: 0.62-0.69), 1.00 at Dc >= 1000 |
| T7 entity count | works only when `length_scale` matches the grid step |

T4 passes everything and discriminates nothing. The report warns it must not be cited as evidence for the memory thesis.

---

# 5. POPGym first pass, `experiments/popgym_vsa/` (RUN_006)

This used the in-repo A2C, one seed (seed 0), CPU, discrete observations. The VSA memory parameters are fixed (only the MLP trunk trains), and `vsa_delay` is handed the lag k-1.

**Headroom** (100 evaluation episodes, `headroom.json`):

| Env | random | FF(step, obs) | FF(obs) | oracle |
|---|---|---|---|---|
| RepeatPrevious Easy / Medium / Hard | -0.513 / -0.505 / -0.496 | -0.511 / -0.498 / -0.502 | -0.498 / -0.490 / -0.494 | +1.000 |
| CountRecall Easy / Medium / Hard | -0.931 / -0.922 / -0.886 | -0.436 / -0.523 / -0.443 | -0.921 / -0.924 / -0.884 | +1.000 |

RepeatPrevious has no feedforward shortcut. CountRecall has one worth about half the gap.

**Readout sufficiency** (`selftest.json`, readout used as the policy):

| Code | Task | Easy / Medium / Hard return |
|---|---|---|
| random | RepeatPrevious | +0.982 / +0.856 / +0.705 |
| random | CountRecall | +0.016 / +0.915 / -0.261 |
| block | CountRecall | +1.000 / +1.000 / +1.000 |

- Random-code count cross-talk breaks Easy and Hard at D=256. Block codes make the count readout exact.
- Finding: the lag is k-1, not k (hit rate 0.991 vs 0.247 chance).
- RepeatPrevious Hard is at 0.705 because k=64 sits near the D/4 limit.

**Results**, all 25 files (eval = mean +- std over 20 greedy episodes):

| Group | Files | Result |
|---|---|---|
| 2-update assertions (4) and smoke runs (2), plus the 5-update GRU stub | `assert_*` x4, `smoke_*` x2, `RepeatPreviousEasy_gru_s0` | Plumbing only, all at or near random. `smoke_cleanup` eval of exactly -1.000 is a constant wrong action, meaningless |
| 150 updates (~122k steps), unpinned threads, ~2,100-2,430 s each | 7 files | `vsa_delay` RP Easy +0.575 +- 0.086. mlp -0.458, posmlp -0.477. All CountRecall arms at chance (-0.92 to -0.93 Easy). CR Hard vsa_count -0.780, treated as noise |
| 600 updates (~490k steps), pinned, 50-80 s | 9 files | See below |
| 2000 updates (~1.63M steps) | 2 files | CR Easy gru -0.771, vsa_count (block) -0.718, both still rising |

The 600-update results:

| File | Arm | Eval |
|---|---|---|
| `rp_easy_vsa` | vsa_delay | +0.990 +- 0.029 (train +0.889) |
| `rp_easy_gru` | gru | -0.108 +- 0.094 |
| `rp_easy_posmlp` | posmlp | -0.481 +- 0.128 |
| `cr_easy_gru` / `mlp` / `posmlp` | | -0.910 / -0.918 / -0.729 |
| `cr_easy_vsa` / `vsa_cleanup` | vsa_count (block) | -0.888 / -0.835 |
| `cr_hard_vsa_cleanup` | | -0.629 (random -0.886, FF ceiling -0.443) |

**What this means.**
- `vsa_delay` solving RepeatPreviousEasy was largely a badly trained baseline. POPGym's own PPO trains a GRU to +0.42 at 296k steps, and later to +0.99.
- The result is "structural prior only", not evidence that the algebra is learned.
- No arm learns CountRecall, so the stop rule fires vacuously.
- The README and RUN_006 describe the cleanup decode as "wrong direction". The numbers show eval -0.888 -> -0.835 (better) and train -0.893 -> -0.907 (worse). That is a wash.

**Not run:** RP Medium and CR Medium training, any Hard GRU/MLP baseline, and more than one seed.

---

# 6. SSP algebra supervised track (RUN_007-013)

**Setup.**
- Tasks: `index` (recall the j-th item), `delay` (lag recall), `pointer` (chained lookup).
- Default state budget is 256 carried floats.
- Hold-out seeds are used for claims, with 10 seeds for pass/fail.
- **Rung ladder.** Rung 0 is fully hand-wired, rung 1 has learned address heads, rung 2 also learns frequencies and codes, rung 3 has no hop count.

## 6.1 Rung 0, hand-wired upper bound (RUN_008 `r0fix`, 5 seeds)

| Arm | index id | 4x stream | tokens to 50% |
|---|---|---|---|
| gru | 0.999 | 0.469 | 113k |
| ssp rung 0 | 0.998 | 0.626 | 26k |
| tf_full | 0.972 | 0.384 | 133k |
| fwp (plain linear attention) | 0.945 | 0.409 | 26k |
| lru / tf_window / kernel_attn | 0.548 / 0.414 / 0.406 | | |
| mlp / posmlp | 0.133 / 0.126 | | |

Low data (accuracy at N training sequences):

| Arm | N=64 | N=256 | N=1024 | N=4096 |
|---|---|---|---|---|
| ssp | 0.966 | 0.996 | 0.995 | 0.994 |
| fwp | 0.354 | 0.852 | 0.921 | 0.937 |
| gru | 0.195 | 0.320 | 0.988 | 1.000 |

- Delay: lru 1.000, gru 0.998, ssp 0.976 (slowest). Verdict `algebra_buys_nothing`.
- No ablation (position table, no cleanup, no decay) removed the index result, so nothing SSP-specific is load-bearing.
- **Speed re-selected on floored area under the curve** (RUN_010), tokens to 0.90: SSP rung 0 24k, `fwp` 90k, GRU 196k, `tf_full` 281k.
  - Rung 0 is 8.0x faster than the GRU.
  - The learned arm needed 132k, which is 1.49x and misses the 2x rule.

## 6.2 Learning the structure, and the discoverability mechanism

- **Plain learning stalls.**
  - A random read head gives index accuracy 0.40 (rung 1) and 0.616 (rung 2).
  - Identity initialisation gives 0.998, so the structure is expressible and discovery is the problem.
- **Offset grid** (index T=32, converged accuracy, seeds >= 0.95 of 10):

| Slope error \ offset | 0 | 0.5 | 1 | 3 | 7 |
|---|---|---|---|---|---|
| 0 | 0.957 (9/10) | 0.956 (9/10) | 0.341 (0/10) | 0.266 (0/10) | 0.265 (0/10) |
| 0.1 | 0.955 (9/10) | 0.791 (6/10) | 0.338 (0/10) | - | 0.252 (0/10) |

  - Any offset of 1 or more gives a flat loss that never moves. Tolerance is narrow in origin and wide in scale.
  - Three sign cells were predicted before running, and all landed.
- **Resulting recipe:** a time-ramp read head with about 3% scale error, offset exactly 0, fixed frequencies and no schedule. It is an aid with one tuned constant (`eps = 0.03`).

| Variant | Recipe conv (seeds >= 0.95) | t90 | Width schedule |
|---|---|---|---|
| index T=16 | 0.996 (10/10) | 25.7k | 0.984, 102.6k |
| index T=24 | 0.976 (10/10) | 38.2k | 0.871 (3/10 at 0.90) |
| index T=32 | 0.956 (9/10) | 52.0k | 0.833 (0/10 at 0.90) |
| index T=48 | 0.973 (10/10) | 77.7k | 0.775 |
| index T=64, 1024 floats | 0.991 (10/10) | - | 0.690 (at 512 floats) |
| index T=64, 512 floats | 0.952 (6/10) | 152.7k | - |
| delay 1-4 / 1-8 | 1.000 / 0.997 (10/10 each) | 21.6k / 18.4k | 0.929 (7/10) / 0.999 |

- The T=64 limit is capacity (D/4). The exact address scores the same, 0.952.
- The T=64 at 1024 floats result has no local raw file (see section 8).
- **Failed routes to remove the schedule:**
  - multi-scale frequencies: 0.738 / 0.813 / 0.893;
  - fine-heavy hybrids: 0.41-0.79;
  - relative affine prior: 0.35-0.46 at T=24-48.
  - The multi-scale failure is a discovery failure: with the address given it reaches 0.975.
- **Interface pretraining.**
  - Aligns on 19 of 20 seeds (N=64: 0.960) for 256k pretraining tokens.
  - Transfer fails. Verified from `transfer*.json`: T=16->32 gives 0.441-0.452, against a no-transfer control of 0.182 and a pass bar of 0.9.
  - The in-training probe auxiliary is a no-op (0.274 vs 0.269).
- **Batch size** is not a speed lever. Reaching 0.90 takes a batch of 64 for the learned arm (0/10 seeds at batch 8, 5/10 at 16, 9/10 at 32, 10/10 at 64).

## 6.3 Equal-aid audit, which withdrew the index lead (RUN_013, re-derived)

Index T=32, budget 256, 10 hold-out seeds:

| Arm | conv | >=0.95 | t50 | t90 | extrap_T |
|---|---|---|---|---|---|
| SSP recipe + gate + banks2 | 1.000 | 10/10 | 25.7k | 30.9k | 0.343 |
| fast-weight delta + gate + banks2 | 1.000 | 10/10 | 30.7k | 66.7k | 0.380 |
| SSP recipe + gate | 1.000 | 10/10 | 25.7k | 51.3k | 0.338 |
| fast-weight delta + gate | 1.000 | 10/10 | 38.5k | 77.0k | 0.397 |
| SSP recipe (incumbent) | 0.957 | 9/10 | 28.3k | 61.7k | 0.457 |
| fast-weight delta alone | 0.744 | 0/10 | 59.0k | never | 0.357 |

- The pre-registered rule fired, so the index lead is withdrawn. The missing aid was the input-dependent gate, not the memory.
- Delay: every arm sits at the same t50 = t90 = 21.6k.
- **Observation:** evaluation happens every 25 updates, and 21.6k (delay) and 25.7k (index T=32) are the first checkpoint after initialisation.
  - The delay "tie" therefore means all arms were already solved by update 25, so the metric cannot discriminate.
  - The t50 ratio of 1.20x has the same floor problem.
  - The SSP speed advantage is resolution-limited on both sides.
- **Banks are not multi-timescale when the gate is on** (RUN_013 s15.3). Decay is one gated value shared across banks, so the "banks2" gain is a state split plus separate cleanup.
- Gate + banks cuts extrapolation: `conv_extrap_T` falls from 0.457 to about 0.34.

## 6.4 Candidate scoreboard (RUN_012, index T=32, 10 seeds, recipe prior unless noted)

| Candidate | conv | >=0.95 | t90 | Verdict |
|---|---|---|---|---|
| recipe (reference) | 0.957 | 9/10 | 52.0k | reference |
| input-dependent gates | 1.000 | 10/10 | 51.3k | positive |
| gates + banks(2) | 1.000 | 10/10 | 25.8k | best cell |
| banks(4) | 0.962 | 9/10 | 25.8k | positive |
| K=5 read heads, offset 1 | 0.962 | 9/10 | 51.3k | repairs a bad start |
| K=5, offset 3 | 0.496 | 0/10 | - | boundary |
| tied relative address, no prior | 0.930 | 1/10 | 970k | removes prior, slow |
| tied + prior | 0.960 | 8/10 | 51.4k | best accuracy |
| gates + tied, no prior | 0.977 | 9/10 | 231k | prior-free best on index; delay 0.488 |
| rung 2 (learned freqs + codes) | 0.999 | 10/10 | 51.4k | neutral-positive |
| delta write, banks2, interp readout | 0.951-0.961 | 8-9/10 | 51-52k | neutral |
| cauchy kernel / learned width | 0.882 / 0.947 | 0/10 / 5/10 | 205k / 77k | negative |

The dead zone is not SSP-specific. Both fast-weight forms degrade with read offset (0.726 / 0.744 at offset 0, 0.397 / 0.321 at offset 1).

## 6.5 Prior-free memory, the round's real advance (p16, re-derived)

`tied + gate + banks2`, random start, no offset, no eps, no schedule, no range rule:

| Cell | conv | >=0.95 | t90 |
|---|---|---|---|
| index T=32 | 1.000 | 10/10 | 64.2k |
| index T=48 | 1.000 | 10/10 | 103.7k |
| delay 1-4 | 0.999 | 10/10 | 36.7k |
| delay 1-8 | 0.999 | 10/10 | 36.9k |

- The incumbent needs four hand-set ingredients, this needs two (fixed frequencies and `banks=2`).
- It is slower on t90 (64.2k vs 61.7k at T=32; 103.7k vs 77.7k at T=48) and worse at extrapolation.
- Gates + banks rescue the offset-3 start (0.496 -> 1.000, 10/10).
- Recipe + K5 at index T=48 gives 0.910 with 0/10 seeds at 0.95. K heads rescue a bad start, not a longer range.
- The mechanism behind the interaction is untested. The "really a missing-bank failure" reading was withdrawn.

## 6.6 Capability tasks (where generic memories fail)

| Task | Result | Caveat |
|---|---|---|
| **Pointer, rung 0** | 0.999 in 12.8k tokens. GRU 0.162, others 0.13-0.16 | Baselines fail their own gate |
| Pointer, rung 1 with per-hop labels | 1.000 id and held-out hops (5/5 seeds) | Plain: 6/10 seeds (0.805) |
| Pointer, one hop of training, per-hop aux | 1.000 on hops 4-8 (10/10) | Without aux 0.660 |
| Pointer, rung 3 (no hop count), `cue_hop` | 1.000 on held-out hops (10/10) | The hop scale is handed in; cue-dropout curve is 0.62 / 0.36 / 0.20 (not graceful) |
| Pointer, node scaling | 8 nodes 1.000, 16 nodes 0.997-1.000, **32 nodes ~0.205 at 512 floats** | Limit is iteration, not bundle capacity. Raw rows absent locally |
| GRU / equal-aid baselines on pointer | 0.18-0.48, at or near chance | A delayed phase transition may exist, so "not learned" is not "cannot" |
| **assoc_mixed** (content or index query), re-derived | full arm 1.000 (10/10); no-pos 0.626; no-bind 0.253; content-only 0.635; GRU 0.249; LRU 0.612 | Ablations fail mode-specifically (0.5x1.0 + 0.5x0.25 = 0.625). Capability, not a win over a working incumbent |
| `assoc` (content-keyed, local) | VSA 1.000, outer-product 1.000, GRU 0.249, LRU 0.281 | Under 4 distractor pairs the outer product wins (0.794 vs 0.589) |
| Distractors (p15, 800 steps) | recipe 0.567, +gate 0.638, +gate+banks2 0.690, fwp-delta+gate 0.695, GRU 0.128 (chance 0.125) | Benefit is not SSP-specific |
| Distractors at 3000 steps | recipe flat 0.562, gated 0.73-0.76, GRU 0.126 | The incumbent's ceiling is a capability limit |
| **delay with the lag hidden** (p17b, re-derived) | recipe 0.348, tied+gate+banks2 0.400, K5 0.417, GRU 0.397, LRU 0.400, fwp-delta 0.152; extrap_lag all ~0.125 | Lag selection is unsolved by every arm at 800 steps |

## 6.7 Negative results

**Spatial recall** (the "SSP-shaped" robotics gate; MSE, lower is better, 10 seeds):

| Arm | untuned (p19) | tuned (p21, 4 configs x 2 search seeds, 4000 steps) |
|---|---|---|
| SSP spatial | 0.097 | 0.077 |
| GRU | 1.659 | **0.021** |
| LRU | 1.107 | 0.787 |
| fwp-delta | 1.640 | 1.643 (never trained) |
| pose_attn | 1.347 | 1.351 (never trained) |

- The "17x win" was an untrained GRU. The trainability gate caught it: GRU 0.630 against a floor of 1.314 on a trivial variant.
- Tuned GRU beats SSP on:
  - in-distribution (3.7x);
  - 2x episode length (3.8x);
  - 2x step size (13x: 0.035 vs 0.457).
- Only noise robustness ties (0.774 vs 0.742).
- The SSP arm had a fixed motion scale, and a learned action->phase map (rung 1 of the ladder) was never built.
- What survives is internal: the phase identity is exact (1.4e-6 shifted vs 2.000 unshifted), and the group action matters within the arm.

**Variant A, path integration with landmarks** (RUN_011 s7b, light budget): GRU MSE 0.049, `pose_attn` 0.101, LRU 0.135, hand-wired SSP 0.147. Dropped.

---

# 7. RL track: POPGym PPO on `RepeatPreviousEasy`

Setup:
- POPGym's own PPO and models, `minibatch_size = 8192`, 3 seeds, 1M steps.
- Random -0.512, optimal +1.000, T50 = +0.244, T90 = +0.849.
- RLlib's `sgd_minibatch_size` is silently dropped, so older reference numbers used a 128-step minibatch.

Re-derived from the summary JSONs and raw rows:

| Arm | Final | T50 (k steps, per seed) | T90 | AULC |
|---|---|---|---|---|
| GRU | +0.993 +- 0.005 | 360, 393, 360 | 557, 557, 524 | +0.436 |
| POPGym fast-weight | -0.396 +- 0.073 (-0.262 at 2M) | never | never | -0.455 |
| SSP untied | -0.163 +- 0.117 (-0.057 at 3M) | never | never | -0.338 |
| SSP untied + aux | -0.268 +- 0.107 (-0.097 at 3M) | never | never | -0.406 |
| tied | -0.133 +- 0.136 | never | never | -0.316 |
| tied + aux | -0.202 +- 0.026 | never | never | -0.359 |
| untied, scaled features | -0.116 +- 0.063 | never | never | -0.336 |
| tied, lag **hand-set** | +0.832 +- 0.209 | 459, 196, 196 | never, 459, 426 | +0.467 |
| **tied_bank** (learned softmax over lags 1..8) | +0.958 (0.954-0.960) | 229, 229, 229 | 491, 491, 459 | +0.582 |
| tied_bank + aux | -0.316 +- 0.046 | never | never | -0.425 |

**Mechanism checks on raw rows.**
- The `tied_bank` selector locks onto lag 3 within about 164k steps. The effective lag is 3.00 and its entropy falls to 0.0003. Lag 3 is k-1, the correct convention.
- The selector locks on while return is still -0.19, so selection precedes policy learning.
- **New observation:** `tied_bank_aux` also collapses to a single lag (entropy ~0.001), but the wrong one: lag 5, 4, 5 in the three seeds. The label-free auxiliary appears to cause premature wrong-lag collapse, which would explain its failure.

**Verdict under the success rule.**
- Parity holds (3.5-point gap).
- The speed ratios are 1.57x (T50) and 1.13x (T90), both short of 2x.
- AULC is 33% higher, but it is not a pre-listed metric.
- Cost: 101 vs 264 steps/s contended, and 1.25 vs 0.64 ms per step uncontended.
- The GRU overtakes it at about 600k steps. T90 differs by 2 PPO iterations, which is within 3-seed resolution.
- The bank's lag range 1..8 is a hand-set constant. A bank without lag 4 scores 0.375 against chance 0.250, and `RepeatPreviousMedium` (k=32) would not be covered.
- The memory-zeroed ablation for `tied_bank` is missing (the checkpoints were lost with `/tmp`). Older arms drop to about -0.50 when zeroed.

---

# 8. Corrections, withdrawals and audit

**Withdrawn or superseded** (these still appear in some files):
- RUN_006 "vsa_delay +0.990 vs GRU -0.108": a badly trained baseline.
- RUN_007 `proceed_to_rl` verdicts and all pointer numbers from the broken arm.
- RUN_008 "about 4x faster", replaced by 8.0x for rung 0 and 1.49x for the learned arm.
- RUN_010 "4 of 5 unseen variants pass": the index jobs omitted `--ep-len` and ran at T=16.
- RUN_011 "schedule is the only recipe with a pass".
- RUN_012 "fast-weight family far behind on index" (missing gate), and "fast-weight loses on delay" (address convention).
- RUN_013 s12.1 "SSP wins spatial 17x": the file marks it superseded but keeps it.
- RUN_013 tables with 11 rows per cell and pooled tuned columns, corrected to hold-out only.
- The fwp `invalid_buggy_*` files.

**Additional checks.**
1. **Local raw-file coverage.**
   - 823 files, and most job lists have all cells locally.
   - Missing: `p5cap_nodes` (32/64 pointer nodes), `p5g_t64_*` (T=64 at 1024 floats), `p3_short_nodes`, `p6_gate`, `p9_assoc` (pod run that crashed), `p17` (deleted on purpose).
   - So the 32-node limit and the "T=64 passes at 1024 floats" claim rest on the run text alone.
   - The 32-node 1024-float value is quoted as 0.208 (doc 11) and 0.304 (RUN_011 table). Not reconciled.
2. **Eval-cadence floor** on the speed claims (section 6.3).
3. **p14 "default" vs "tuned" columns are bit-identical** in all index cells. Either the search picked the default config every time or tuning had no effect, so the "two columns" are effectively one.
4. **Stale docs:**
   - Handoff 11's "CURRENT PLAN" block sits below the newer STATUS block.
   - The CartPole docs cite the referenced paper as ICCM 2022 where RUN_004 says 2023.
   - The popgym README says "nothing learns CountRecall" and omits the 2k runs.

---

# 9. Literature check (alphaXiv)

**Closest neighbours to what was built:**
- [2610.13197](https://www.alphaxiv.org/abs/2610.13197), *Does Learning Recurrent Networks Matter in Partially Observable RL?*
  - Frozen random GRU/LRU plus a trainable MLP readout matches trainable counterparts on POPGym.
  - Their Frozen GRU and Frozen LRU reach about 1.00 on RepeatPrevious.
  - This is the right control for any "fixed SSP memory + learned readout" claim, and it is not in the baseline set.
- [2605.31261](https://www.alphaxiv.org/abs/2605.31261), *Why Linear Recurrent Memory Works in POMDPs* (ICML 2026).
  - Diagonal unit-circle linear memories are provably sufficient for deterministic or nearly deterministic HMMs, including action-controlled ones, where the filter is time-varying with action-dependent eigenvalues.
  - That is the theoretical home of SSP path integration, and it also says the time-as-SSP memory is substitutable by LRU.
- [2609.38598](https://www.alphaxiv.org/abs/2609.38598), *RL with Complex-valued Memories* (EWRL 2026): unitary recurrent PPO reaches 2-3x baseline reward on RockSample and Craftax.
- [2601.15086](https://www.alphaxiv.org/abs/2601.15086), *Memory Retention Is Not Enough*.
  - New benchmarks require rewriting (Endless T-Maze, Color-Cubes).
  - PPO-LSTM with a forget gate wins. SHM, FFM and GTrXL fail.
  - All baselines score zero on Color-Cubes Medium and Extreme.
  - This matches the finding that the input-dependent gate was the largest lever.
- [2410.10132](https://www.alphaxiv.org/abs/2410.10132), Stable Hadamard Memory: an element-wise multiplicative matrix memory, close to FHRR binding.
- [2608.28334](https://www.alphaxiv.org/abs/2608.28334), real-valued hyperdimensional sequence representations with shift equivariance, from the same FPE family.
- Also relevant:
  - [2512.06204](https://www.alphaxiv.org/abs/2512.06204), Temporal Range, a memory-use metric.
  - [2610.40030](https://www.alphaxiv.org/abs/2610.40030), a cheap retrieval-credit idea.
  - [2610.00592](https://www.alphaxiv.org/abs/2610.00592), learnable experience rewriting.

**Reading.**
- The structured phase memory idea is already inside the linear-recurrent/complex-memory line.
- The distinctive SSP ingredients are explicit bind/unbind/cleanup, one group action on the whole bundle, and iterated queries.
- Of those, only iterated queries has produced a result no generic baseline matched.

---

# 10. Where the thesis stands

| Claim | Status |
|---|---|
| Algebra is exact, capacity ~ D/4, chained queries free | Established |
| Discovery of the read address is the bottleneck, with a mechanism (offset dead zone) | Established, generalises to fast-weight memories |
| A learner can find the structure with an aid | Yes (recipe, K heads, gates+banks); prior-free version works with 2 constants |
| SSP matches generic memories at convergence | Yes on index and delay; yes in RL on RepeatPreviousEasy (0.958 vs 0.993) |
| SSP is more sample-efficient | Only on index at t90 (2.16x), resolution-limited. RL 1.57x / 1.13x. No second task family |
| SSP is more robust or extrapolates better | No. Prior-free is worse at extrapolation. Cue-dropout is not graceful |
| SSP is distinctive on spatial / path integration | No (tuned GRU 3.7x better; variant A lost) |
| SSP is distinctive on iterated or content-keyed tasks | Capability yes, but baselines fail their gates, and 32 nodes breaks |
| Deployable | RL parity yes; cost per step worse; the bank range is a constant |

---

# 11. What to do next

## Option 0: close the cheap gaps first (recommended, days)

1. **Memory-zeroed ablation for `tied_bank`.** One seed, about 1.7 h. Without it the RL result has no proof the return is carried by memory.
2. **Scale-free bank on RepeatPrevious Medium (k=32) and Hard (k=64).**
   - Use a geometric lag set (1, 2, 4, ..., 64) or K heads with scale-free offsets.
   - This removes the 1..8 constant and tests the regime where lag selection and credit assignment actually hurt generic memories.
3. **Add the missing baselines in the same PPO loop:** LRU, frozen GRU/LRU (per 2610.13197), and an equal-aid fast-weight arm with bank selection. If a frozen LRU matches `tied_bank`, the SSP-specific RL claim is gone.
4. **Fix the measurement.**
   - Evaluate every 5 updates early.
   - Run at least 5 seeds.
   - Report at two thresholds on at least two tasks.
   - Re-fetch or re-run the missing raw cells (32-node, T=64 at 1024 floats).
5. **Diagnose the aux wrong-lag collapse** (it locks onto lag 4-5). A temperature or selector-freeze fix may revive the label-free signal, which RL needs.

**Gate:** if the scale-free bank is at parity and at least 2x faster than the best of GRU/LRU on Medium with 5 seeds, the memory-module line is worth continuing. If not, stop claiming general memory superiority.

## Option 1: pivot to rewriting and gated key-value memory (best new direction)

- Tasks: Endless T-Maze and Color-Cubes (2601.15086).
- Why it fits.
  - Published baselines (LSTM, GTrXL, SHM, FFM) score 0 on Color-Cubes Medium/Extreme, so headroom is large and the failing baselines are not ours.
  - Color-Cubes needs bind, overwrite and "which item moved", which is a delta-write bundle with a learned gate.
  - The best measured lever is the gate.
- Arms: gated delta-write SSP bundle with cleanup, vs LSTM, GRU, LRU and an equal-aid fast-weight memory. Every baseline must pass its trainability gate first.
- Risk: the gate benefit is not SSP-specific, so the fast-weight arm may tie.

## Option 2: action-conditioned phase memory for localization POMDPs

- Basis: 2605.31261 gives a theoretical anchor, with action-dependent eigenvalues on ring and torus worlds, RockSample position tracking and masked navigation.
- Required change: build a **learned action->phase map**, which was never built. The spatial failure was largely a fixed motion scale (13x worse on 2x step size).
- Warning: the tuned GRU already extrapolates to 2x length without loss (0.021). Only try this on a regime where the GRU provably degrades (long horizons with motion noise, many landmarks), and tune every baseline.
- Rank: medium-low. This direction already has one strike.

## Option 3: cheap encoder-side results (parallel, near-zero cost)

- H2 (about $2): per-dimension normalization x `length_scale` in [0.01, 1]. It closes "why was SSP worse on CartPole".
- Centering x approximator interaction (doc 10 s7.1): does centering help less when the basis has a bias term? The theory is written down and the data already points the right way (SSP +8.6, tile -6.6). It takes hours and gives a practitioner rule either way.
- Continuing Reacher / Swimmer-with-wrapped-angles (doc 09 Tier 1) would need a separate MuJoCo repo.

## Option 4: write up what exists (always valid)

A rigorous mixed result with transferable methodology:
- the offset dead-zone mechanism, shown to hold across fast-weight memories;
- the equal-aid reversal of the index lead;
- the untuned-baseline reversal in spatial recall;
- the trainability-gate and hold-out-only protocol;
- the first RL SSP memory that learns RepeatPreviousEasy, and the lag-bank mechanism.

## Lower priority

- **VSA world model:** crowded by equivariant models with proofs (doc 10 s9).
- **On-device or continual learning:** per-step cost is currently worse than a GRU.
- **MIKASA RGB tasks:** the single-item tasks are a weak SSP fit, and the set tasks have no published solver.

## Suggested order

1. Option 0, steps 1-4 (days). Decide by the gate.
2. In parallel, Option 3's two cheap tests.
3. If the gate passes, run the confirmatory 5-seed RL study. If it fails, switch to Option 1.
4. Write up (Option 4) either way.

**Process rules to keep:**
- Tune every baseline and run its trainability gate first.
- Never tabulate pod and local rows together.
- Report hold-out seeds only.
- Pre-register pass rules.
- Set `--ep-len` explicitly.
- Test hidden-input variants at the consumption site.
