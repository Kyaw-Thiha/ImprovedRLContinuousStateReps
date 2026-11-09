# RUN_011: discoverability, halting and the RL port (handoff 10, phases 0-5)

Written 2026-10-12. Executes `handoffs/10-ssp-discoverability-and-rl-port-prep.md`. Nothing is committed (the user said
not to). Conventions as in RUN_005-010: every number is labelled **[measured]**, **[inferred]** or **[taken from an
earlier run]**; bugs and traps are recorded so they are not rediscovered.

**Status (final, 2026-10-12):** every cell is complete (no pods remain; nothing is in flight). Phases 0-3 and 5 have measured results; phase 4
has the comparable interface hooks for the non-SSP control and its results (section 7); phase 5.1 (POPGym reference) is done (0.683 at 632k steps);
5.2 (RL memory stub) is done and tested; 5.3 (first RL test) is **not** licensed -- the gate in handoff 10 section 5 fails on conditions (1) and (2).
Section 12 (appended) states the conclusions and the redirection that follows; the next agent starts at `handoffs/11-ssp-memory-module-deployable.md`.

---

## 0b. Correction (found 2026-10-12, after the tables below were written)

Two findings from auditing my own numbers. The first invalidates a claim in RUN_010; the second is a methodological
rule that applies to every comparison in this folder.

**1. RUN_010's index "unseen variants" ran at T=16, not at T=24/32/48/64.**  Those jobs never passed `--ep-len`, and
`run_sweep`'s default is 16.  The stored rows prove it: they carry **511-514 tokens per update**, which is T=16 (T=32
is ~1025, T=64 ~2049 -- my own `p1_idxt32_*` rows measure 1025 and my `p1d_idxt64_a05` row 2049).  So:

- the range-scaling rule's "4 of 5 unseen variants pass" and "T=48/64 at 1.000" were **T=16 numbers** and are withdrawn;
- of the genuinely unseen variants, only delay lags 1-8 pass, delay 1-4 fails (7/10), and the first real test at
  index T=32 (my `p1_idxt32_sched`, 1025 tokens/update) gives **0.833 with 0 of 10 seeds at 0.90**;
- **no recipe is validated for the continuous-address family**, so the statement in section 4.3 and in docs/11 that the
  schedule "stays as the only recipe with a pass" is withdrawn.  `jobs/p5_rule_real.json` re-runs the frozen rule at the
  real lengths T=24/32/48/64 with `--ep-len` set explicitly (verified in the emitted argument string).

**2. Pod and local runs are not numerically identical, so they must not be mixed.**  `replay_check.py` regenerates
stored rows through today's code.  Locally, four stored RUN_010 rows differed by 3.9e-3 to 4.3e-1 (the largest was the
GRU's mid-training accuracy, while its converged value matched).  The same replay **on a pod reproduced the stored rows
exactly (max |delta| = 0.000e+00, four rows including the rung-0 `ssp`, the `gru`, the learned-schedule index arm and
the rung-3 pointer arm)**.  Therefore:

- **my edits introduced no regression** -- the earlier "DIFF" verdict was a machine artefact (local BLAS/instruction set
  differs from the pod image), not a code change;
- local and pod results differ at the ~1e-3 level in accuracy, which is enough to flip individual seeds near a
  decision boundary, so a comparison table must be built from one machine class, and every tag records where it ran.

This matters for one open question in section 4.4: the eps-versus-`affine_abs` inconsistency at T=16 is *not* a
local-versus-pod artefact (the eps=0.2/0.4 and the `affine_abs`=0.5 rows all ran on pods), so it still needs the basin
map.  Both facts are recorded here rather than edited into the tables, so the sequence of what was believed when stays
readable.

## 0. Summary of the session's findings

1. **Batch size is not a speed lever (phase 0.1). [measured]**  The learned arm is *more* fragile at small batches
   (0/10 seeds reach 0.90 at batch 8, 5/10 at 16, 9/10 at 32, 10/10 at 64), and tokens fall only because the token
   counter scales with the batch.  The pre-registered target "tokens to 0.90 <= 50k" cannot be reached by batch tuning
   at convergence >= 0.99.  Batch 64 stays the headline for every arm.
2. **Rung 3's held-out-hop failure is the *halting*, not the iteration (phase 0.2). [measured]**  The unrolled
   unbind-clean-up chain reads the right node after h iterations at *every* h = 1..8 (per-hop accuracy 1.000); the
   arm's *answer* is a halt-weighted ensemble whose peak cannot leave the trained hops.
3. **A learned halting scale does not extrapolate; a fixed one from the cue does (phase 0.2). [measured]**  Three
   learned halting mechanisms were built and all fail on held-out hops (0.28-0.52 over 10 seeds), for a reason that is
   now pinned: with only hops 1-2 trained, a whole slab of (intercept, slope) pairs is loss-free, so the decode stops
   compressive (measured slope 0.15 with a hard gate, 0.30 with a soft quadratic).  Reading the cue's own hop channel
   with a fixed scale plus the same hard gate reaches **1.000 on held-out hops 4-8 on all 10 seeds** (incumbent 0.362,
   reproduced), with a real cue-dropout curve, and the per-hop auxiliary is load-bearing (0.574 without it).
4. **The discovery obstacle is the *offset* of the initial read address, not its size (phases 0.3, 1). [measured]**
   Three schedule-removal routes failed (multi-scale frequencies 0.738-0.893; fine/hybrid 0.409-0.793; the *relative*
   affine prior 0.35-0.46 at T=24-48), and the audit that followed located the real constraint.  In a controlled grid
   (`err(j) = slope_err * j + offset`, index T=32, 10 seeds) **every offset-zero cell passes (9/10 seeds at >= 0.95)
   for scale errors 0, 3% and 10%, while every offset of 1 or more fails 0/10 and never moves at all** (flat at
   0.25-0.34 for 1500 steps).  Three purpose-built sign cells then confirmed the rule: a dead-zone offset is rescued
   exactly when the drift carries the error through zero *inside* the stream (both such cells 9/10 as predicted; the
   cell whose crossing comes only at the last index scored 0.637, also as predicted).  So the tolerance is *narrow* in
   the origin coordinate and *wide* in the scale coordinate -- a single "basin radius" is the wrong description.
5. **That mechanism yields a working, scale-free recipe (phase 1). [measured]**  Read head = time ramp with a ~3% *scale*
   error and **offset exactly 0**, fixed frequencies, no schedule, no address-range constant:

| variant | conv | seeds >= 0.95 | tokens to 0.90 | the incumbent schedule recipe |
|---|---|---|---|---|
| index T=16 | 0.996 (min 0.991) | **10/10** | **25.7k (50 updates)** | 0.984, 10/10, 102.6k |
| index T=24 | 0.976 (min 0.954) | **10/10** | 38.2k | **0.871, 0/10 at 0.95** |
| index T=32 | 0.956 (min 0.934) | **9/10** | 52.0k | **0.833, 0/10 at 0.90** |
| index T=48 | 0.973 (min 0.961) | **10/10** | 77.7k | **0.775, 0/10 at 0.90** |
| index T=64 (1024 floats) | 0.991 (min 0.985) | **10/10** | -- | **0.690, 0/10** (at 512 floats) |
| delay 1-4 | 1.000 (min 1.000) | **10/10** | 21.6k | **0.929, 7/10** |
| delay 1-8 | 0.997 (min 0.996) | **10/10** | 18.4k | 0.999, 10/10, 128.4k |

   It beats the incumbent at every variant both can be run on, is 4x cheaper on tokens at T=16, and is flat with length
   where the schedule decays monotonically (0.984 -> 0.871 -> 0.833 -> 0.775 -> 0.690 across T=16/24/32/48/64).  Its one
   apparent failure, index T=64 at a 512-float budget (0.952, 6/10), is **the task's capacity, not the prior**: the exact
   rung-0 address scores the same there (0.952, 5/10), and at 1024 floats both reach 0.99 with all ten seeds clearing
   0.95.  So the record is **"passes every variant run inside the memory's capacity"**.  It remains a **strong aid**
   (monotone, correctly-centred) and `eps = 0.03` is a tuned constant whose search cost counts (section 3.3).
6. **Shape-randomised interface pretraining scales with its budget but is still expensive (phase 2). [measured]**
   N=64 pool accuracy: 0.269 (none) / 0.303 (50 steps) / 0.436 (100) / 0.630 (200) / 0.845 (500); it does not transfer
   (0.40-0.46 on the length axis), and the in-training probe auxiliary is a **no-op** (0.274 / 0.275 against 0.269) --
   the one form the RL port could have used.
7. **Pointer: the iteration extrapolates, the halting and the start decode both had to be fixed. [measured]**  Trained
   on a *single* hop, the arm answers 8-hop queries at 1.000; rung 3 with no hop count handed over reaches **1.000 on
   held-out hops 4-8 on all 10 seeds** once (a) the halting reads the cue's own hop channel with a fixed scale and
   (b) the start decoder is masked from that channel (`mask_hop_in_start`; without it the same arm scored 0.807, and
   the 2x2 showed the loss was entirely the cue-based start decode, not the halting).  The per-hop auxiliary stays
   load-bearing (0.513 without it).
8. **Methodology: pod and local runs are not numerically interchangeable. [measured]**  A four-row replay of RUN_010
   reproduces stored results **bit-for-bit on a pod** but differs locally by up to 4.3e-1; that both clears my edits of
   any regression and forbids tabulating the two machine classes together.  Two of my own earlier claims are corrected
   in section 0b as a result.
9. **Two limits are now located, and neither is what it looked like (phases 3, 5). [measured]**  (a) *Pointer's node
   limit is between 16 and 32, and it is not the bundle's capacity*: 32 nodes scores 0.205 at 512 floats and 0.208 at
   1024 floats, where `D/4 ~ 128` items would comfortably allow 32 nodes, so the limit is the *iteration* at that node
   count (32 items along an 8-hop chain), not how much one vector can hold.  (b) *The recipe's one failure, index T=64,
   is the task's capacity at that budget*: at 512 floats the address range 62 sits at `D/4 ~ 64`, and the **exact**
   rung-0 address scores the same 0.952 (5/10 seeds) as the recipe (6/10), while at 1024 floats both jump to ~0.99 with
   every seed clearing 0.95 (exact address 0.990, recipe 0.989, 4/4 seeds each so far).  So the recipe's record is
   **"passes every variant run inside the memory's capacity"**, and the pointer arm's usable node count is 16 at 512
   floats.
10. **No generic baseline for pointer exists at these budgets. [measured]**  The GRU on the *easy* variant (4 nodes, 1
   hop) sits at **0.247 over 10 seeds at 15000 steps** (0/10 near the 0.95 bar; RUN_010 measured 0.478 at 6000), and
   `fwp`/`lru`/`tf_window` are at 0/30 seeds on the same gate at 3000 steps.  The pointer result is therefore a
   *capability* demonstration, not a matched-baseline win -- which is the honest characterisation the thesis rules
   require.

---

## 1. Phase 0.1 -- batch size as a speed lever

Protocol: `ssp_index_learned` (frozen range rule, `recipes.continuous_address_cfg(14)`), `gru`, `fwp`, `tf_full`;
index T=16, budget 256 floats, `--select aulc --min-conv 0.95 --search-seeds 3 --seeds 10 --steps 1500`, batch in
{8, 16, 32, 64}; jobs `jobs/p0_batch.json`; 10 hold-out seeds per arm and batch. **[measured]**

| arm | batch | conv | min | last ckpt >= 0.90 | updates to 0.90 | tokens to 0.90 |
|---|---|---|---|---|---|---|
| learned SSP | 8 | 0.752 | -- | 0/10 | -- | -- |
| learned SSP | 16 | 0.860 | -- | 5/10 | 592 | 75.8k |
| learned SSP | 32 | 0.956 | -- | 9/10 | 277 | 70.9k |
| **learned SSP** | **64** | **0.986** | | **10/10** | 251 | 128.3k |
| GRU | 8 | 0.955 | | 10/10 | 1205 | 77.1k |
| GRU | 16 | 0.991 | | 10/10 | 702 | 89.8k |
| GRU | 32 | 0.913 | | 10/10 | 387 | 99.1k |
| **GRU** | **64** | **1.000** | | 10/10 | 377 | 192.8k |
| fwp | 8 | 0.914 | | 10/10 | 323 | 20.7k |
| fwp | 16 | 0.924 | | 10/10 | 699 | 89.5k |
| fwp | 32 | 0.930 | | 10/10 | 214 | 54.7k |
| fwp | 64 | 0.936 | | 10/10 | 163 | 83.4k |
| tf_full | 8 | 0.486 | | 0/10 | -- | -- |
| tf_full | 16 | 0.940 | | 10/10 | 1102 | 141.1k |
| tf_full | 32 | 0.989 | | 10/10 | 705 | 180.4k |
| tf_full | 64 | 0.995 | | 10/10 | 502 | 257.0k |

**Verdict.**  "Update-bound" as handoff 10 section 0.1 defines it (learned t90 falls >= 2x at some batch with
conv >= 0.99) is **not** satisfied: at batch 32 the learned arm's updates to 0.90 fall only from 251 to 277 and its
converged mean is 0.956, below the 0.99 floor.  Smaller batches *break* discovery (0/10 at batch 8).  `tf_full` at
batch 8 fails its own trainability gate and must not be quoted there.  **Batch 64 is the headline batch for every
arm**, and every token count in this record is at batch 64 unless stated.

**Token unit (trap).**  `tokens` counts *supervised positions*: for index T=16 that is `updates x batch x 8`, so it is
not a batch-independent measure.  The pre-registered "<= 50k tokens" at batch 64 means <= 97 updates.  RUN_010's
132k / 196k / 90k / 282k numbers reproduce (learned 251-128.3k, GRU 377-192.8k, fwp 163-83.4k, tf_full
502-257.0k).

## 2. Phase 0.2 -- halting: three mechanisms, one that works

Protocol: `ssp_pointer_learned`, rung 3 (`cue="features"`, no hop count handed over), `aux_weight=0.5`, n_nodes 6-8,
max_hops 8, trained on hops (1, 2) and tested on hops 4-8.  Unit-test evidence at 1500 steps (batch 64, lr 3e-3)
in `test_halting.py`; the 10-seed pod protocol (6000 steps, `--robust`) is `jobs/p0_halt.json`.

### 2.1 The iteration is not the problem. [measured]

`diag_pointer.per_hop_accuracy` reads `_hop_scores[:, :, h-1]` -- the cleanup *after h iterations of the chain* --
independently of how the arm stops.  For the rung-3 arm trained on hops (1, 2) that is **1.000 at every h = 1..8**.
So docs/11 section 1.2 unique #2 ("chained algebraic queries do not compound") survives in a learning setting: the
same unbind-clean-up step, iterated, reads the right node at 8 hops.  The failure RUN_010 reported (0.361) is entirely
the *stopping*.

### 2.2 The three mechanisms (all in `SSPPointerLearned`). [measured]

| `halting=` | mechanism | in-distribution | held-out hops 4-8 | halting peak on the queried hop |
|---|---|---|---|---|
| `learned` (RUN_010's) | free embedding row per hop, ensemble over hops | 1.000 | 0.45 | 1.00, 1.00, 0, 0, 0, 0, 0, 0 |
| `counter` (first form) | free MLP on [cue, k/max_hops] | -- | 0.17 (broken) | broad monotone, peak 0.36 on its own hop at h=4 |
| `counter` (soft quadratic, index units) | -2*(hop_hat - k)^2 | 1.000 | 0.45 | decode slope 0.30 |
| `counter` (hard straight-through gate) | gate the answer at `hop_hat` | 1.000 | 0.45 | decode slope 0.15 |
| **`cue_hop`** | same gate, hop read from the cue with a *fixed* scale | **1.000** | **1.000** | 1.00 at every h = 1..8 |

Cue-dropout curve for the rung-3 arm (fraction of query dimensions zeroed), hops 4-8: `cue_hop` 0.62 / 0.34 / 0.21 at
25 / 50 / 75 %; `counter` 0.39 / 0.27 / 0.21.  Both are monotone but not graceful, and both are *real* curves (the
rung-3 arm reads the cue, unlike the rung-0/rung-1 arms which read `batch["start"]` and are immune by construction).

### 2.2b The 10-seed protocol (60 queued runs, `jobs/p0_halt.json`). [measured]

rung 3, `cue="features"`, `max_hops` 8, trained on hops (1, 2), 6000 steps, batch 64, floored AULC selection, 10
hold-out seeds, `--robust`:

| arm | id (hops 1-2) | seeds >= 0.9 | **held-out hops (4, 8)** | seeds >= 0.9 there | tokens to 0.90 | cue-drop 25/50/75% |
|---|---|---|---|---|---|---|
| **`cue_hop` + aux** | **1.000** (min 1.000) | **10/10** | **1.000** | **10/10** | 89.6k | 0.62 / 0.36 / 0.20 |
| `cue_hop`, no aux | 0.879 (min 0.669) | 5/10 | 0.574 | 1/10 | 1305.6k | 0.55 / 0.32 / 0.18 |
| `counter` (learned decode) + aux | 1.000 | 10/10 | **0.282** | 0/10 | 102.4k | 0.60 / 0.35 / 0.19 |
| `learned` (RUN_010's arm, re-run) | 0.987 | 10/10 | **0.362** | 0/10 | 160.0k | 0.61 / 0.35 / 0.20 |

**The pre-registered target (held-out hops >= 0.9) is met, on all 10 seeds**, and RUN_010's 0.361 is reproduced at
0.362 by an independent re-run of its exact configuration -- so the 0.362 -> 1.000 difference is the mechanism, not
noise.  Two qualifications belong with it:

* the per-hop auxiliary is load-bearing (0.574 without it), so the result is an **aided** one, costed at its own
  sample budget;
* the cue-drop curves of all four arms are nearly identical (0.55-0.62 / 0.32-0.36 / 0.18-0.20), which says the
  *halting* is not what robustness is limited by here: when the cue's hop channel is zeroed every arm halts wrongly,
  `cue_hop` included (it degrades to "halt at 1").  A confidence-weighted fallback (halt at the hop whose score is
  highest when the decoded hop is unreliable) is the obvious next design and was not implemented.

### 2.3 Why the learned halting scale cannot be learned here. [measured, and it is the interesting part]

Training asks for hops 1 and 2 only.  A monotone decode `hop_hat = a + b * h` is in-distribution-correct on a *slab* of
(a, b) pairs: with the hard gate, any `hop_hat(1) in [1, 2)` and `hop_hat(2) in [2, 3)` has zero task loss.  Gradient
descent stops at the first point inside the slab, which is `a ~ 2 - b` -- i.e. it prefers a *compressed* slope.
Measured: with the soft quadratic the slope is 0.30 (hop_hat 1.35 -> 3.47 over h = 1..8), with the hard gate 0.15
(1.86 -> 2.93).  Nothing in the task loss distinguishes those from slope 1, so held-out hops stay broken.

`cue_hop` sidesteps the identifiability problem rather than solving it: the *observation* carries `hop / max_hops`
linearly (it is part of the cue the arm is already given), so the scale is fixed, and the cue-dropout split still
corrupts it.  **What has to be given is the hop *scale*; what is learned stays everything else** (the query code, the
start node, the role gate, the readout).  This is a rung-ladder result, not a trick: docs/11 section 3.4 asks at each
rung what the learner is handed, and this one is handed the counter's units.

**Bugs found here (both silent, both costing a run each):**

1. `gate = g_hard + (g_surr - g_hard).detach()` detaches *the surrogate too*, so the decode never moves (measured:
   `hop_hat` pinned at 1.00 for every queried hop, id 0.634).  The correct straight-through is
   `gate = g_hard + g_surr - g_surr.detach()`.
2. Scoring the halt in *fraction* units with the same sharpness constant gives 64x larger gradients (256 vs 4) and the
   head oscillates instead of converging; index units with c = 2 are stable.

## 3. Phase 0.3 -- how much prior does the learned arm need?

`init="affine"` (new): the read address starts as `(1 + eps*N(0,1)) * j + eps*N(0,1)*R/2`, i.e. a slope and an offset
error in units of the address range R; `eps = 0` is exactly `init="identity"` (verified bit-for-bit).  Sweep
eps in {0, 0.02, 0.05, 0.1, 0.2, 0.4} x {fixed width 1, schedule}; index T=16, 10 seeds, frozen lr 0.02, steps 1000.
**[measured, partially collected]**

| mode | eps | conv | >= 0.95 | tokens to 0.90 |
|---|---|---|---|---|
| fixed width 1 | 0.00 | 0.994 | 10/10 | 25.7k (50 updates) |
| fixed width 1 | 0.02 | 0.995 | 10/10 | 25.7k |
| fixed width 1 | 0.05 | 0.995 | 10/10 | 25.7k |
| fixed width 1 | 0.10 | 0.994 | 10/10 | 25.7k |
| fixed width 1 | 0.20 | (queued) | | |
| fixed width 1 | 0.40 | (queued) | | |
| schedule | 0.00 | 0.993 | 10/10 | 102.6k |
| schedule | 0.02 | 0.994 | 10/10 | 102.6k |
| schedule | 0.05 | 0.994 | 10/10 | 102.6k |
| schedule | 0.10 | 0.994 | 10/10 | 102.6k |
| schedule | 0.20 / 0.40 | (queued) | | |

Two readings follow, and the second one corrects the theory:

1. **eps = 0 reproduces RUN_009's "identity init reaches 0.90 in 50 updates" exactly** (25.7k tokens = 50 x 64 x 8), and
   even with a correct address the *scheduled* arm costs 4x that (102.6k = 200 updates) because the schedule spends its
   length narrowing the kernel.  So the 50k-token target was reachable in principle and phase 1's failure is about
   discovery, not about a floor.
2. **The basin is wider than the sinc argument says, and the width does not slow the arm down.**  At eps = 0.1 the
   initial address error is measured at **1.70 indices RMS with 0% of queries inside 0.5**, i.e. well outside what
   RUN_008 called "the basin of about one index" -- and the arm still converges to 0.994 in the same 50 updates as
   eps = 0, on every seed.  **[measured]**  This is consistent with the negative result in section 0.7 (the gradient
   norm does *not* vanish beyond one index): the exact sinc zero at integer offsets is masked by the finite-D
   crosstalk of size ~1/sqrt(D), so the read head always has *some* gradient, and the "basin" is a **speed/quality
   gradient rather than a hard wall**.  **[inferred]**  Consequences: the phase-1 question is not "is there a gradient" but "does the read head start *monotone in the index*" -- see
section 4.4, which turns this diagnostic into a better recipe than the schedule.

### 3.1 The audit finding: the failures look dead-zone shaped, not size shaped. [measured + pre-registered]

The eps-versus-`affine_abs` inconsistency (two priors of nearly the same size, 10/10 versus 8/10 seeds) is resolved in
the direction of *neither* of my earlier explanations.  Auditing the per-seed priors against the per-seed outcome
(the arm draws both from a generator seeded `seed + 11`, so the same seeds give the same underlying draws in both runs):

| run (pod) | worst prior in the run | outcome |
|---|---|---|
| `affine_eps = 0.4` | seed 1008: slope error **-0.958** (an 18-index error at j = 13) | conv **0.991**, converged by step 25 |
| `affine_abs = 0.5` | seed 1008: **2.1-index** error | conv 0.995 (passes) |
| `affine_abs = 0.5` | seed 1000: 2.2-index error, seed 1002: 1.9-index error | conv **0.385 / 0.337** |

So the seeds that fail are **not** the ones with the largest prior, and the arm recovers from an 18-index error.  The
curves show why it is not instability either: the two failing seeds *never* find the address (a slow creep from 0.12 to
0.33-0.38 over all 1500 steps), while passing seeds jump to ~0.9 by step 25.  Reading the two failure modes off the
draws:

* the failures are **offset-dominated** runs whose offset lands at 1.39-1.46, i.e. in the dead zone of
  ``kappa(d) = sinc(pi d)`` at ``length_scale = 1``, which is zero at every integer offset -- so *every* position is
  equally uninformative and the read head gets no direction;
* the fast-converging runs are **slope-dominated**: their error varies with the index (seed 1008 at eps = 0.4 is
  wrong by 18 indices at j = 13 but passes through zero around j = 12), so a slice of positions is nearly correct and
  supplies the gradient.

That is a *dead-zone* explanation, not a basin-width one, and it predicts that what matters is whether the initial
address is near-correct for *some* positions -- not the size of the error.

**Pre-registered predictions** for the cells running now (`jobs/p5_basin.json` with offsets {0, 1, 3, 7}, and the
discriminating `jobs/p5_basin_frac.json` with fractional offsets, both at index T=32, width 1, 10 seeds):

| cell | dead-zone prediction | size prediction |
|---|---|---|
| offset 0 | pass | pass |
| offset 0.5 (kappa 0.64) | pass | pass |
| offset 1, 2, 3, 7 (kappa 0) | **fail** | fail (3 and 7 much worse than 1) |
| offset 1.5 (kappa -0.21), 2.5 (kappa 0.13) | borderline | fail |
| any offset with slope error 0.1 | **pass** (the error sweeps through zero along the stream) | fail |

**Correction, after the first two cells: that last row was mis-specified, and the first cells refute it as written.**
The address error is ``err(j) = slope_err * j + offset``, so a *positive* slope error with a *positive* offset moves
**away** from zero and never crosses it; my "any slope error rescues" row ignored the signs.  Measured:

| cell | result | reading |
|---|---|---|
| `(slope 0, offset 0)` -- the identity address | conv **0.957**, 9/10 seeds >= 0.95, t90 **52.0k tokens** (101 updates) | passes; the reference cell |
| `(slope +0.1, offset +1.0)` | conv **0.338**, 0/10 seeds at 0.90 | stuck: err runs 1.0 -> 4.0, never near zero |

So the *corrected* prediction is sign-aware: a dead-zone offset is rescued only when the drift carries the error
**through zero inside the stream**.  That is now tested directly by three opposing-sign cells (`jobs/p5_basin_sign.json`,
launched with the prediction written into their names before running):

| cell | err(j) = slope*j + offset | prediction |
|---|---|---|
| `(slope -0.1, offset +1.0)` | 1.0 -> -2.0, crosses at j = 10 | **pass** |
| `(slope +0.1, offset -1.0)` | -1.0 -> +2.0, crosses at j = 10 | **pass** |
| `(slope -0.1, offset +3.0)` | 3.0 -> 0.0, crosses only at j = 30 | borderline, likely fail |

If the first two pass and the third does not, the mechanism is established quantitatively: what a read-address prior has
to guarantee is that *some* positions sit inside the kernel's informative range, not that the error is small overall.
That, in turn, is what a deployable rule should enforce (for example a slope error whose sign is *anti-correlated* with
the offset, or an offset drawn only from the interval where `sinc` is healthy).  **[pre-registered; results not yet in]**

### 3.2 The grid result: only the *offset* coordinate is constrained. [measured]

First batch of cells (index T=32, width 1, fixed hyperparameters, 10 seeds, `jobs/p5_basin.json` and
`jobs/p5_basin_frac.json`; `err(j) = slope_err * j + offset`, and `kappa = sinc(pi * err)`):

| slope error | offset 0 | offset 0.5 | offset 1 | offset 1.5 | offset 2.5 | offset 3 | offset 7 |
|---|---|---|---|---|---|---|---|
| 0 | **0.957, 9/10 >= 0.95** | **0.956, 9/10** | 0.341, 0/10 | 0.341, 0/10 | 0.266, 0/10 | 0.266, 0/10 | 0.265, 0/10 |
| 0.03 | **0.957, 9/10** | -- | 0.341, 0/10 | -- | -- | 0.266, 0/10 | 0.261, 0/10 |
| 0.1 | **0.955, 9/10** (t90 102.9k) | 0.791, 6/10 | 0.338, 0/10 | 0.350, 0/10 | -- | -- | 0.252, 0/10 |

All twelve planned cells plus the fractional and sign cells are in.  Two details worth reading off it:

* **offset 0.5 passes as well as offset 0** (0.956, 9/10) -- consistent with `kappa(0.5) = sinc(pi/2) = 0.64` still
  being informative, so the constraint is not "offset must be zero" but "the offset must leave the positions inside the
  informative range";
* **slope +0.1 *plus* offset +0.5 is much worse than offset 0.5 alone** (0.791, 6/10 against 0.956, 9/10), because the
  two errors *reinforce* each other (`err` runs 0.5 -> 3.5 instead of staying near 0.5) -- the sign-aware form of the
  mechanism, confirmed on a cell I did not design for it.

**The slope-only recipe, measured so far (10 seeds each, frozen hyperparameters):**

| variant | conv | seeds >= 0.95 | tokens to 0.90 |
|---|---|---|---|
| index T=16 | **0.994** (min 0.989) | **10/10** | 25.7k (50 updates) |
| delay lags 1-4 | **1.000** (min 1.000) | **10/10** | 21.6k (42 updates) |
| delay lags 1-8 | **0.997** (min 0.995) | **10/10** | 18.4k (36 updates) |
| index T=24 | 0.915 (min 0.380) | **9/10** | 38.3k |
| index T=32 | 0.896 (min 0.375) | **7/10** | 76.3k |
| index T=48, T=64 | not run at `eps` 0.1 (superseded by 0.03) | | |

For comparison, the *incumbent* recipe at the same variants: index T=16 0.984 (10/10, 102.6k tokens), delay 1-8
0.999 (10/10, 128.4k), delay 1-4 **0.929 (7/10)**, index T=24 **0.871 (3/10 at 0.90)**, index T=32 **0.833 (0/10 at
0.90)**.  So the slope-only prior is faster, and it passes the two delay cases and the two real index lengths where the
schedule recipe fails or is marginal.  `jobs/p5e_slopeonly.json`'s remaining index lengths decide whether it clears
**The crossing mechanism is confirmed on purpose-built cells.**  Three cells were launched with the prediction written
beforehand, and all three landed as predicted:

| cell | err(j) = slope*j + offset | prediction | measured |
|---|---|---|---|
| `(slope -0.1, offset +1.0)` | 1.0 -> -2.0, crosses zero at j = 10 | pass | **0.957, 9/10 >= 0.95** |
| `(slope +0.1, offset -1.0)` | -1.0 -> +2.0, crosses at j = 10 | pass | **0.956, 9/10 >= 0.95** |
| `(slope -0.1, offset +3.0)` | 3.0 -> 0.0, crosses only at j = 30 | borderline, likely fail | **0.637, 4/10** |

So the rule is quantitative, not descriptive: **an offset that would be a dead zone is rescued exactly when the drift
carries the address error through zero inside the stream**; a crossing at the very last index is too late.  This is what
explains the individual `affine_abs` seeds (section 3.1) and the slope-plus-offset cell above.

**Where the slope-only recipe stands, all lengths measured (eps = 0.1, offset 0, 10 seeds):**

| variant | conv | seeds >= 0.95 | verdict | the schedule recipe there |
|---|---|---|---|---|
| index T=16 | 0.994 (min 0.989) | **10/10** | pass | 0.984, 10/10, 102.6k tokens |
| index T=24 | 0.915 (min 0.380) | **9/10** | pass (marginal) | **0.871, 3/10 at 0.90** |
| index T=32 | 0.896 (min 0.375) | **7/10** | **fail** | **0.833, 0/10 at 0.90** |
| delay lags 1-4 | 1.000 (min 1.000) | **10/10** | pass | **0.929, 7/10** |
| delay lags 1-8 | 0.997 (min 0.995) | **10/10** | pass | 0.999, 10/10, 128.4k tokens |
| index T=48 | not run at `eps` 0.1 | **0.973, 10/10**, 77.7k tok | **0.775, 0/10** |
| index T=64 | not run at `eps` 0.1 | **0.991, 10/10 at 1024 floats** (0.952, 6/10 at 512) | **0.690, 0/10** (512 floats) |

Two things stand out, and both are honest limits rather than wins:

1. **It beats the schedule everywhere they can be compared** -- same or better reliability at every length, and 4-5x
   fewer tokens (25.7k against 102.6k at T=16) -- but it still **fails docs/11 4.7 at T=32** (7/10 at the 0.95 bar).
2. **The reliability depends on the *tail* of the slope-error draw, not just its typical size.**  The grid's
   deterministic `slope_err = +0.1` cell passed at T=32 (0.955, 9/10) while the recipe's *random* draw with the same
   typical magnitude fails there (7/10): the extra failures are seeds whose draw is large enough that the error no
   longer crosses zero inside the stream in time.  The grid's `slope_err = +0.03` cell also passed at T=32 (0.957,
   9/10), which suggests the constant should be smaller.

`jobs/p5f_slopeonly003.json` tests `eps = 0.03` at every length (index T=16/24/32/48/64 and delay 1-4/1-8), and it
fixes exactly the marginality the tail of the `eps = 0.1` draw caused:

| variant | `eps = 0.1` | `eps = 0.03` | the schedule recipe there |
|---|---|---|---|
| index T=16 | 0.994, 10/10, 25.7k tok | **0.996, 10/10, min 0.991**, 25.7k tok | 0.984, 10/10, 102.6k tok |
| index T=24 | 0.915, **9/10**, min 0.380 | **0.976, 10/10, min 0.954**, 38.2k tok | **0.871, 3/10 at 0.90** |
| index T=32 | 0.896, **7/10**, min 0.375 | **0.956, 9/10, min 0.934**, 52.0k tok | **0.833, 0/10 at 0.90** |
| index T=48 | not run at `eps` 0.1 | **0.973, 10/10, min 0.961**, 77.7k tok | **0.775, 0/10 at 0.90** |
| index T=64 | not run at `eps` 0.1 | **0.952, 6/10 at 512 floats; 0.991, 10/10 at 1024** | **0.690, 0/10** |
| delay 1-4 | 1.000, 10/10 | **1.000, 10/10**, 21.6k tok | **0.929, 7/10** |
| delay 1-8 | 0.997, 10/10 | **0.997, 10/10**, 18.4k tok | 0.999, 10/10, 128.4k tok |

**The two recipes decay in opposite directions with stream length.**  The schedule rule's converged accuracy falls
monotonically as the stream gets longer -- 0.984 (T=16), 0.871 (T=24), 0.833 (T=32), **0.775 (T=48)** -- and it is below
0.90 at *every* length past the one it was tuned on.  The slope-only prior is almost flat: 0.996, 0.976, 0.956 (9/10),
0.973.  Whatever the schedule's coarse-to-fine kernel is doing, it does not scale with the address range even though its
constants are supposed to (`start width = 0.5R`, `schedule = 24R` updates).  **[measured, T=16..64]**

Reporting this as a **constant search, not as a recipe** in the strict sense: the conditions tried for this family are now
`eps` in {0.4 relative, 0.5 absolute, 0.1, 0.03} plus twelve deterministic grid cells plus three sign cells -- and per
docs/11 4.7 that search is part of the cost.  The frozen-constant test decides whether "the read head starts as a ramp
with a small *scale* error and no offset" is a recipe (>= 0.95 on >= 9/10 seeds on every unseen variant) or a tuned
technique, and it now has an answer for six of seven variants: **pass at T=16 (10/10), T=24 (10/10), T=32 (9/10),
T=48 (10/10), delay 1-4 (10/10) and delay 1-8 (10/10); fail at index T=64 (6/10)** -- where every seed lands in
0.941-0.97, a uniform ceiling rather than the dead-zone stalls seen elsewhere.  That shape, plus the arithmetic (address
range 62 against `D/4 ~ 64` at a 512-float budget), makes a **capacity** limit the leading explanation, and two controls
running now decide it: the exact rung-0 address at T=64 under 512 and 1024 floats, and the recipe at T=64 under 1024
floats.  If the exact address also caps near 0.95 at 512 and rises at 1024, the T=64 failure is the task's budget and the
recipe's record is "passes every variant inside the memory's capacity".  Otherwise the recipe fails T=64 and that is
recorded as its limit.  **[measured]**

**Confirmed, at 10 seeds: the T=64 ceiling is the budget, not the prior.**

| arm at index T=64 | 512 floats | 1024 floats |
|---|---|---|
| exact rung-0 address (`init="identity"`) | 0.952, **5/10** seeds >= 0.95 | **0.992 (0.980-0.997), 10/10** |
| the recipe (`eps = 0.03`, offset 0) | 0.952, **6/10** | **0.991 (0.985-0.997), 10/10** |

At a 512-float budget the address range (62) sits at the `D/4 ~ 64` capacity limit and *no* initialisation clears 0.95 --
the exact address performs identically to the recipe there.  At 1024 floats both clear it on every seed.  So the recipe's
record is **"passes every variant run inside the memory's capacity"**: index T=16/24/32/48 (10/10, 10/10, 9/10, 10/10),
index T=64 at 1024 floats (10/10), and delay 1-4/1-8 (10/10, 10/10).  T=64 at 512 floats is recorded as a
capacity-limited task rather than a failure of the prior.

**Conclusion about the mechanism, and a correction to my own earlier wording.**  The discoverable set is *not* a ball in
address-error space: the two coordinates behave completely differently.

- **The offset is the binding constraint**, and its tolerance is exactly the `sinc` structure: at
  ``length_scale = 1`` a constant offset of 1, 1.5, 2.5 or 7 leaves *every* position equally far from the kernel's
  informative range, so the read head receives no direction at all.  Tolerance to offset is therefore *narrow*, not
  wide -- the opposite of what the earlier wording in section 3 item 2 said (it treated the initial address error as a
  single scalar and concluded the basin was wide).
- **The slope error is almost free**, including a 10% scale error that is 3 indices wrong at j = 30, because the error
  *varies* with the index and the small-index positions stay nearly correct, which is enough to give the gradient a
  direction.  Tolerance to slope error is *wide*.

That resolves the eps-versus-`affine_abs` inconsistency: the `affine_abs` form draws an **offset** (σ = a) as well as a
slope, so its bad seeds are the ones whose offset lands in the dead zone (measured: the two stuck seeds had offsets
-1.46 and -1.39 with tiny slope errors, while passing seed 1008 had offset +1.03 with slope error -0.086, whose drift
carries the error through zero at j ≈ 12).

**Design consequence, now under validation.**  A deployable prior should *control the offset* and may freely perturb the
scale.  `jobs/p5e_slopeonly.json` runs exactly that arm -- read head = ramp x (1 + 0.1*N(0,1)), offset exactly 0, no
schedule, no R-dependent constant -- at index T=16/24/32/48/64 and delay 1-4/1-8, 10 seeds each, as a candidate recipe
under docs/11 4.7.  The cells above already show it passing at T=32 (0.955, 9/10 >= 0.95); the run tests whether that
holds at every length, including the two lengths where the frozen range rule fails (T=24: 0.871 and T=32: 0.833).
**[measured at T=32; the full validation is in section 3.3]**

### 3.3 The real-length rule result, and why it makes the slope-only prior the candidate

Measured while the grid was running (Step 1, `jobs/p5_rule_real.json`, index at its **true** length, 10 seeds):

| arm | index T=16 (the tuned length) | index T=24 (first real unseen length) | index T=32 (real) |
|---|---|---|---|
| frozen range rule (schedule) | 0.984, **10/10** seeds >= 0.95 | 0.871, **0/10** >= 0.95, 3/10 >= 0.90 | 0.833, **0/10** >= 0.90 (reproduced twice) |
| `affine_abs = 0.5` | 0.869 (8/10 >= 0.95) | 0.861, **8/10 >= 0.95**, min 0.362 | 0.825, 7/10 >= 0.95 |
| **slope-only prior** (`eps` 0.1, offset 0) | **0.994, 10/10, t90 25.7k tokens (50 updates)** | 0.915, **9/10** (min 0.380) | 0.896, **7/10** |
| per-seed shape at T=24 | rule: 0.97-1.00 | rule: 0.74-0.93 (uniformly mediocre); affine: `0.42 0.98 0.36 0.97 0.99 0.97 0.98 0.98 0.99 0.99` | -- |

Three things follow.

1. **The frozen range rule does not generalise past T=16**, and this is now confirmed at two lengths by independent pod
   runs (T=24: 0.871 with 3/10 seeds at 0.90; T=32: 0.833 with 0/10 at 0.90, reproduced exactly).  RUN_010's "4 of 5
   unseen variants" is withdrawn in full -- the only genuinely unseen cases it tested were the delay lags, and there it
   passes 1-8 and fails 1-4.
2. **The affine prior fails differently**: its *ceiling* is higher (eight seeds at 0.97-0.99 at T=24, better than the
   rule's best) and its problem is a **bimodal discovery failure** -- two seeds that never start moving, which the grid
   in section 3.2 attributes to the **offset draw** landing in a `sinc` dead zone.
3. **The fix that follows from the mechanism works at two lengths so far**: forcing the offset to zero and keeping only
   the scale perturbation (`affine_eps` 0.1 + `affine_offset_zero`) gives **0.994 with 10/10 seeds at T=16 in 25.7k
   tokens** -- the same 50-update class as the exact answer, but from a genuinely perturbed prior -- and 0.955 with 9/10
   at T=32.  `jobs/p5e_slopeonly.json` is validating it at T=24/48/64 and delay 1-4/1-8.

**How strong is this prior, stated plainly (do not overclaim it).**  The arm is told three things: the read address is
*monotone* in the index, its scale is right to ~10%, and its origin is exactly right.  That is much closer to the answer
than a curriculum is, and docs/11 3.4 lists "near-identity initialisation of address heads" as an *aid* with its own
cost -- so this must be reported as an aid, not as discovery.  What makes it interesting is not that it is weak but that
(a) it is **scale-free** (no R-dependent constant, unlike the range rule), (b) it costs **4x fewer tokens** than the
schedule at T=16, and (c) it is derived from a measured mechanism rather than searched.  Whether it is *cheating* is
answerable: the perturbation is a 10% scale error with zero offset, so it is not the rung-0 address, and the grid shows
the same arm tolerating scale errors up to 10% while failing outright on any offset.

### 4.1 The multi-scale set, first form (log-uniform mixture). [measured]

`omega_kind="logmix"` with `ls_hi = 0.5 * R` (the frozen rule's constant a = 0.5, so nothing is retuned per variant),
`ls_mode="fixed"` (no schedule), frozen lr 0.02 / lam_init 0.99 / temp 0.1, compared in the same runs against the
schedule recipe.  `jobs/p1_multiscale.json`.

| variant | multi-scale conv (>=0.95 seeds) | schedule conv (>=0.95 seeds) | verdict |
|---|---|---|---|
| index T=16 | 0.738 (0/10) | 0.984 (10/10) | FAIL |
| delay lags 1-4 | 0.813 (5/10) | 0.929 (7/10) | FAIL |
| delay lags 1-8 | 0.893 (0/10) | 0.999 (10/10) | FAIL |

Scope note: the third kind (`residue`, a geometric ladder of scales with jitter against exact commensurability) was
checked at the *kernel* level only (`test_multiscale.py`: unit modulus, determinism, no dead region, no aliasing) and
not trained, because the two kinds that were trained came out this far behind.  Its kernel has isolated near-zero
offsets (5, 8, 13, 14 for R=14) where the logmix has one (11), so the training result would be expected to be no
better.  **[inferred]**

### 4.2 It is a *discovery* failure, not a resolution failure. [measured]

Same frequency set, with the address **given** by `init="identity"` (rung 0's address) so only the readout trains:

| arm | conv | >= 0.95 |
|---|---|---|
| logmix `ls_hi = 0.5R`, no schedule, identity init | 0.975 | 10/10 |
| logmix + `fine_frac = 0.7`, identity init | 0.991 | 10/10 |

So the mixture's *ceiling* is fine -- much better than the 0.83 a single global width of 8 would give **[taken from
RUN_008]** -- and the loss-landscape probe agrees that the retrieval loss keeps rising with the address error out to
4-6 indices where the single scale is flat (0.38 -> ~3.0 plateau for the uniform kind; 0.82 -> 2.84 for logmix).
What fails is that plain gradient descent from a *random* read head still does not converge to the right address with
a mixture, whereas the schedule does.  **[inferred]**  The mixture widens the basin but flattens the *direction* of the
gradient: the wide components make many addresses nearly as good as the right one, so the descent has a long plateau
followed by a narrow final approach that the schedule instead walks down in one direction.

### 4.3 The hybrid / fine-heavy fallback, and the phase-1 verdict. [measured]

`fine_frac` pins that fraction of dimensions to the finest scale, keeping the resolution while a minority supplies the
basin.  Index T=16, 10 seeds, frozen hyperparameters (batches `jobs/p1b_hybrids.json`):

| arm | conv | min | >= 0.95 | ceiling (identity init) |
|---|---|---|---|---|
| `fine_frac = 0.7`, no schedule | 0.409 | 0.258 | 0/10 | 0.991 |
| `fine_frac = 0.9`, no schedule | 0.410 | 0.340 | 0/10 | 0.994 |
| hybrid: `fine_frac = 0.7` + schedule 6R from width 0.5R | 0.793 | 0.738 | 0/10 | -- |
| hybrid: `fine_frac = 0.7` + schedule 6R from width 2 | 0.554 | 0.451 | 0/10 | -- |
| **the incumbent: coarse-to-fine schedule, uniform frequencies** | **0.984** | 0.970 | **10/10** | 0.997 |

**Phase 1 verdict: FAIL, and the schedule stays.**  Every schedule-free and hybrid variant is far below the incumbent
(0.409-0.793 against 0.984), and none clears the pre-registered 9/10-seeds-at-0.95 bar.  The trade-off is not a
tuning problem: the fine-heavy mixtures that have the *best ceilings* (0.991, 0.994) are the *worst* at discovery
(0.409), while the plain mixture with the weakest ceiling (0.975) discovers best (0.738).  A minority of wide
dimensions supplies a weak basin; a majority destroys the gradient's direction.  Per handoff 10 section 4 the fallback
has been measured and reported, and per docs/11 3.6 the honest reading is: **the coarse-to-fine width schedule is a
costed aid, carried by the frozen range rule, and the "aid or natural learning" question (handoff 10 section 9 item 2)
is answered -- aid.**  The 50k-token target is not reached (the schedule's own floor is 102.6k, section 3).

*What would be needed to try again* (not attempted here): a mechanism that produces a *directional* gradient from a
wide basin rather than a flat one -- e.g. an address loss with a monotone target (the supervised variant of the
schedule), or an explicit coarse-to-fine *network* rather than a coarse-to-fine *kernel*.

### 4.4 The attempt that nearly removed the schedule: an affine prior on the read head

The phase-0.3 diagnostic was intended as a bound, and it produced a *nearly*-working alternative: at T=16 it removes
the schedule and is four times faster, and it fails to generalise.  Reading the sweep as a 2x2
(initial read head x kernel-width mode; index T=16, 10 seeds, floored AULC selection, frozen lr 0.02): **[measured]**

| read head | kernel width | conv | seeds >= 0.9 | **tokens to 0.90** |
|---|---|---|---|---|
| random (`init="random"`) | fixed width 1 | 0.40-0.62 **[taken from RUN_008]** | -- | never |
| random | schedule (frozen range rule) | 0.984 | 10/10 | 102.6k (200 updates) |
| **affine (`init="affine"`, eps <= 0.4)** | **fixed width 1 (no schedule)** | **0.994** | **10/10** | **25.7k (50 updates)** |
| affine (eps = 0.4) | schedule | 0.994 | 10/10 | 102.6k |

Three things follow, all measured at T=16:

1. **With an affine read head there is no discovery problem at all at T=16**: no schedule, fixed single-scale
   frequencies, converged 0.994 on every seed, in **50 updates** -- and this holds for eps = 0, 0.02, 0.05, 0.1, 0.2
   **and 0.4**, i.e. even when the initial address is wrong by 6.8 indices RMS (section 3).
2. **Given the affine head, the schedule is pure overhead at T=16**: 102.6k vs 25.7k tokens at the same accuracy.  Its job was
   to make a *random* read head discoverable, and it is not needed once the head starts monotone.
3. **The learned arm's headline number changes from 132k to 25.7k tokens to 0.90 at T=16** -- 5.1x faster than the
   schedule recipe, 7.6x faster than the GRU (196k) and 3.5x faster than `fwp` (90k), and inside the pre-registered
   ``<= 50k`` target.  **[measured at T=16; the cross-variant validation is described below]**

What the prior is, stated exactly so it is not overclaimed: the read head's weight vector is initialised to the time
*ramp* (the same shape rung 1 uses to write) scaled per dimension by ``1 + eps*N(0,1)``, plus an offset of
``eps * N(0,1) * R/2`` in address units.  That is a statement about the *shape* of the read address -- monotone in the
index, roughly unit slope -- not about the answer; at eps = 0.4 the initial address is off by ~7 indices of 14.  It is
the "near-identity initialisation of address heads" aid that docs/11 section 3.4 already lists, promoted from a
stabiliser to the mechanism that replaces the curriculum, and it must therefore be reported as a **costed aid with its
own prior cost** (a design decision, not extra samples).

**Cross-variant validation, first form: FAIL, and the failure is informative. [measured]**  With `affine_eps = 0.4`
(a *relative* prior, i.e. an error that scales with the address range) the recipe does **not** transfer:

| variant (unseen) | affine_eps = 0.4: conv | seeds >= 0.95 |
|---|---|---|
| index T=24 | 0.438 | 1/10 |
| index T=32 | 0.464 | 3/10 |
| index T=48 | 0.346 | 2/10 |
| delay lags 1-4 (R=4) | 0.886 | 7/10 |
| delay lags 1-8 (R=8) | 0.757 | 5/10 |

The pattern is exactly the one the *absolute*-error reading predicts: the arms with the **smallest address range** do
best (delay R=4 and R=8) and the arms with the largest do worst (T=48), while the *relative* error is constant by
construction.  At T=16 a 6.8-index initial error is recoverable (section 3) but at T=32 an error of ~14 indices is not,
so what matters is the absolute error relative to the kernel's ~1-index width at `length_scale = 1`, not its fraction
of R.  **The prior needs a rule that holds the absolute error fixed** -- `slope = 1 + a*N/R`, `offset = a*N` with `a`
in index units -- which is the same *kind* of rule the schedule recipe used, with the opposite scaling (the schedule
scaled *up* with R because a wide kernel is what it needed; the prior holds *constant* because resolution, not basin
width, is what limits it here).  Verified: `affine_abs` 0.5 gives 1.09 / 1.08 / 1.10 indices RMS at T = 16 / 32 / 64.
**[measured]**  **Cross-variant validation, second form (scale-free `affine_abs`): also FAIL. [measured]**  With the absolute-error rule
(`slope = 1 + a*N/R`, `offset = a*N`, `a` in index units) at `a = 0.5`:

| variant | conv | min | seeds >= 0.95 | tokens to 0.90 (updates) |
|---|---|---|---|---|
| index T=16 | 0.869 | 0.337 | 8/10 | 25.6k (50) |
| index T=24 | 0.861 | 0.362 | 8/10 | 38.1k (74) |
| index T=32 | 0.825 | 0.294 | 7/10 | 64.2k (125) |
| index T=48 | 0.842 | 0.247 | 8/10 | 77.7k (152) |
| index T=64 | 0.814 | 0.208 | 5/10 | 152.7k (298) |
| delay lags 1-4 | 0.957 | 0.568 | 9/10 | 21.6k (42) |
| delay lags 1-8 | 0.885 | 0.430 | 8/10 | 18.4k (36) |
| index T=16, `a = 4.0` | 0.478 | 0.298 | 1/10 | 25.7k (50) |
| index T=32, `a = 4.0` | 0.367 | 0.240 | 1/10 | 77.6k (151) |

Two clear readings.  (i) The **absolute** form removes the length-dependence the relative form had: 0.81-0.87 at every
index length instead of collapsing to 0.35-0.46, and it is *fast* (36-50 updates to 0.90 at R <= 14).  (ii) It is never
**reliable** -- 5 to 9 of 10 seeds clear the bar on each variant, never 9 on the harder ones -- and the larger-error
probe (`a = 4.0`) fails badly, so the useful window is narrow and sits near the identity prior rather than at any
comfortable distance from it.

**Phase-1 final verdict, corrected:** *neither* route to removing the schedule produced a recipe that passes docs/11
4.7.  The multi-scale set fails on convergence (0.738-0.893), the fine-heavy/hybrid families fail worse (0.409-0.793),
and the affine prior -- which is genuinely *fast* at T=16 (25.7k tokens, 10/10) -- fails on reliability everywhere else
(0.81-0.96, at most 9/10).  The incumbent coarse-to-fine width schedule still passes 4 of 5 unseen variants
**[taken from RUN_010]**, so **the schedule stays**, and the affine prior is recorded as a *fast but unreliable*
alternative whose speed does not transfer.  Conditions tried while designing it, all part of the search cost per 4.7:
eps in {0, 0.02, 0.05, 0.1, 0.2, 0.4} x {fixed, scheduled} (12) plus `affine_abs` in {0.5, 4.0} over 7 variants (9), plus
the two failed relative-form validations (5 more) -- 26 conditions, none of which can be reported as a recipe.

What the affine work *does* leave behind is the mechanism, which is worth more than the recipe: the difficulty is the
read head's **shape** (a random 13-feature linear map is non-monotone in the index), not a one-index basin -- the arm
recovers from a 6.8-index error at T=16 (section 3) and the gradient never vanishes (finite-D crosstalk ~1/sqrt(D)).  A
future fix should therefore beat the schedule by *parameterising* the read address monotonically (e.g. a monotone
network, or a cumulative-softplus ramp) rather than by widening a kernel.

## 5. Phase 2 -- make the interface amortised

### 5.1 Shape-randomised pretraining (2.1)

`--pretrain-shapes LO,HI` draws a new stream length per pretraining step (verified: 17 distinct lengths in [8, 64] over
20 steps), so the interface cannot fit one shape.  The two-axis transfer test (`transfer_test.py`) with
`--pretrain-shapes 8,64`, 500 pretraining steps, 800 readout steps: **[measured, 3 seeds; run stopped early]**

| axis | seeds | destination accuracy | no-transfer control | pass rule |
|---|---|---|---|---|
| length T=16 -> T=32 | 0, 1, 2 | 0.453, 0.404, 0.457 | 0.182 **[taken from RUN_010]** | >= 9/10 seeds at >= 0.9 |

The run was stopped after 3 seeds **because the criterion is already unreachable**: the rule needs at least 9 of 10
seeds at 0.9, so 2 failures would do, and 3 seeds are all far below.  The transfer claim fails with 3x the
no-transfer control and less than half the required accuracy; the alphabet axis was not reached.  Compare the
*fixed-shape* interface of RUN_010: 0.960 at N=64 on its own shape, 0.441 on T=32.  Randomising the shape costs the
in-shape accuracy (0.845 vs 0.960 at N=64, section 5.2) and buys no transfer, so the verdict is the one 4.7 warns
about: **more training data, not an amortised interface.**

### 5.2 Pretraining budget curve (2.2). [measured]

N=64 pool, index T=16, shape-randomised pretraining, then readout training on the pool; 10 seeds.

| pretraining updates | conv (N=64) | >= 0.9 seeds |
|---|---|---|
| 0 (control) | 0.269 | 0/10 |
| 50 | 0.303 | 0/10 |
| 100 | 0.436 | 0/10 |
| 200 | 0.630 | 0/10 |
| 500 | 0.845 | 3/10 |

Monotone and real, and still short of the 0.9 bar at 500 steps; the *cost* is 500 x 64 x 8 = 256k pretraining tokens,
reported separately from the task tokens.  Compare RUN_010's *fixed-shape* probe at the same 500 steps: 0.960 with
19/20 seeds aligned on its own shape.  So randomising the shape costs in-shape accuracy to buy a transfer property
that (on 5.1's evidence) is not there.  **[inferred]**  This is exactly the "more training data" verdict 4.7 warns
about, now measured on both axes.

### 5.3 The in-training probe auxiliary (2.3): a no-op. [measured]

A throwaway probe trained jointly from step 0 (no staging, no freezing, no extra tokens -- verified: the token count is
identical to the plain run), with `probe_aux_weight` in {0.5, 1.0}:

| arm | conv (N=64) |
|---|---|
| plain (no aid) | 0.269 |
| in-training probe, weight 0.5 | 0.274 |
| in-training probe, weight 1.0 | 0.275 |

**This is the negative result that matters for the RL port**: the one form of the alignment aid that would have been
usable inside an RL loop (labels are the stream's own observations, no separate stage, no extra tokens) does not move
the needle at N=64.  Whatever fixes the N=64 collapse has to be a *stage* with its own cost, or an architectural
change.

## 6. Phase 3 -- pointer

### 6.0 The 0.807 combined gap is the cue-based start decode, not the halting. [measured]

The 2x2 diagnostic planned for this (`cue` in {batch, features} x `halting` in {given, cue_hop}, trained on hop 1 only,
max_hops 8, 1200 steps, seed 0, local):

| cue | halting | id (hop 1) | hop 4 | hop 8 |
|---|---|---|---|---|
| batch | given | 1.000 | 1.000 | **1.000** |
| batch | cue_hop | 1.000 | 1.000 | **1.000** |
| features | given | 1.000 | 1.000 | **0.802** |
| features | cue_hop | 1.000 | 1.000 | **0.802** |

The two `halting` rows are *identical*, so **halting is not implicated at all**; the whole loss is `cue="features"`
(1.000 -> 0.802), which is exactly the arm rung 3 requires.  That isolates the learned **start decoder** (`start_head`,
an MLP over the whole `feat` vector) as the cause, and it matches the mechanism predicted before the run: trained with
the hop channel pinned at 1/8, it sees 0.5 and 1.0 at test and its decode drifts, with the damage appearing at the
longest hop (hop 4 is still 1.000).  The fix that follows is to remove the hop channel from `start_head`'s input
(`mask_hop_in_start`), which is a *task-input* change, not a hint: the hop count stays out of the start decode, and the
halting mechanism keeps reading it from the cue as before.

Implemented and tested (`arms_learned.SSPPointerLearned.mask_hop_in_start`, default False so every earlier result is
unchanged; `test_halting.py::test_mask_hop_in_start_makes_the_start_decode_hop_invariant`): with the flag on, changing
the hop channel on the *query steps* alone leaves the retrieval bit-identical, and without it the retrieval must change.
Two things were learned while writing that test, both recorded because they are easy to get wrong again: the mask must
apply to a **copy** of the feature slice (it is a view of the caller's tensor, so zeroing in place would corrupt the
readout's own input), and the perturbation must be restricted to the query steps -- the role gate legitimately reads the
stream steps, so changing the hop channel there moves the memory itself and is not a test of the start decode.
`jobs/p4_combined_fixed.json` re-runs the combined job (rung 3, `cue_hop`, trained on hop 1 only, with and without the
aux) with the mask on.  **[measured: 1.000 on held-out hops 4-8 with 10/10 seeds when the aux is on; 0.513 without it]**

### 6.1 Is the iteration free?  Train on **one hop**, test on 4-8. [measured]

`ssp_pointer_learned` rung 1 (hop count *given*), `--train-hops 1 --max-hops 8 --n-nodes 8`, 3000 steps, 10 hold-out
seeds, `jobs/p3_pointer.json`.  Because the 1-hop target *is* the value stored under the queried key, this run is also
the label-free retrieval signal handoff 10 section 3.3 asks for: no hop labels beyond the first step are used.

| arm | id (1 hop) | held-out hops (4, 8) | tokens to 0.90 |
|---|---|---|---|
| 1-hop training, **per-hop aux on hop 1** | **1.000** (10/10) | **1.000** | 76.8k |
| 1-hop training, no aux | 0.939 (6/10) | 0.660 | 460.8k |

**This is the cleanest support the thesis has for docs/11 section 1.2 unique #2.**  The arm is trained on a *single*
iteration of unbind-clean-up and answers 8-hop questions at 1.000 on every seed; the per-hop supervision is what makes
the single step reliable (10/10 vs 6/10).  Combined with section 2.1 (the chain reads the right node after h
iterations at every h), the "chained algebraic queries do not compound" claim holds in a *learning* setting, not just
an encoder-only one, and the 6x lower token cost (76.8k vs 460.8k) is the measurement of it.

*Caveat*: the hop count is still given at evaluation (this is rung 1, not rung 3).  Rung 3 needs the halting mechanism
of section 2, and with `cue_hop` it now has one; the combination (1-hop training + `cue_hop` halting) is the natural
next run and was not executed here.

### 6.1b The combination: no hop count given and only one hop trained. [measured]

`rung=3`, `halting='cue_hop'`, trained on hops (1,) only, `max_hops` 8, 3000 steps, 10 seeds
(`jobs/p3_combined.json`):

| arm | id (1 hop) | held-out hops (4, 8) |
|---|---|---|
| 1-hop training + `cue_hop` + aux | 1.000 (10/10) | **0.807** |
| 1-hop training + `cue_hop`, no aux | 0.988 (9/10) | 0.405 |

Each half of the mechanism on its own reaches 1.000 (6.1 with the hop count given; section 2.2b with the hop count
read from the cue but *both* hops trained).  Combining them -- the arm is trained on a single iteration and must read
the hop count out of the cue itself -- lands at 0.807: the two difficulties interact, and 0.807 is short of the 0.9 bar.
The interaction is the honest headline here: neither "the iteration extrapolates" nor "the halting extrapolates" is by
itself the hard part, but *both at once* is.

### 6.2 Node scaling and the baseline gates. [measured; the 32-node number is preliminary]

Node-count scaling with the halting fix that works (`rung=3`, `halting='cue_hop'`, per-hop aux), 3000 steps, 10 seeds,
budget 256 floats unless stated.  **[measured, where finals exist]**

| nodes | budget | conv | seeds >= 0.9 | held-out hops (4, 8) |
|---|---|---|---|---|
| 8 | 256 | 1.000 | 10/10 | 1.000 |
| 8 | 512 | 1.000 | 4/4 (partial) | 1.000 |
| 16 | 256 | 0.997 | 4/4 (partial) | 0.993 |
| 16 | 512 | 1.000 | 10/10 | 1.000 |
| 32 | 512 | **0.205 (0.062-0.321), 10 seeds** | **0/10** | -- |
| 32 | **1024** | **0.304 (0.078-0.610), 10 seeds** | **0/10** | -- |
| 64 | 512, 1024 | **stopped** (see below) | | |

**The node count at which pointer accuracy falls below 0.9 is between 16 and 32**, measured three times by different
protocols (0.164 with a two-lr frozen protocol; **0.205** and, at double the budget, **0.208** with the searched 16-node
winner config).  Two things are now settled about *why*: the 32-node failure is not a bad learning rate (the same config
solves 16 nodes at 1.000), and **it is not the bundle's capacity either** -- doubling the budget to 1024 floats doubles
`D/4` to about 128 items, far above 32 nodes, and accuracy stays at chance.  The limit is therefore in the *iteration*
at that node count (32 items to distinguish along an 8-hop chain), not in how much one vector can hold.  The 64-node
cells were **stopped and deleted before finishing**: the limit was already located, they are the most expensive cells in
the batch (`ep_len` 136 x 8 hops, over 40 minutes without producing a final), and the budget was better spent on the
1024-float cell at 32 nodes, whose answer is above.  **[measured at 512 and 1024]**

**The GRU on the easy variant (4 nodes, 1 hop), 15000 steps, 3 searched configs, 10 seeds** (the gate that must pass
before any pointer comparison counts): conv **0.247** over all **10 seeds** (range 0.238-0.267), i.e. **0/10 anywhere
near the 0.95 bar**.  RUN_010 measured this gate at 0.478 after 6000 updates, so tripling the budget did not move the GRU
toward it.  **The pointer pass therefore has no valid generic baseline**, and the pointer result must be worded as a
*capability* demonstration rather than a matched-baseline win: the arm does something no generic sequence model in this
harness does (8-hop chained queries at 1.000), and the comparison the thesis would need cannot be run until some
baseline passes its own gate.  **[measured, 10 seeds]**

**The never-run trainability gate, now run** (`fwp`, `lru`, `tf_window` on the *easy* variant: 4 nodes, 1 hop, max_hops
1, 3000 steps, 10 seeds each): conv 0.329, **0 of 30 seeds reach 0.9**.  Caveat on fairness: RUN_010 needed 6000
updates for the GRU to pass this gate (0.478), so 3000 steps is half that budget; the honest statement is that these
three arms do not pass the gate *at half the GRU's required budget*, and they remain invalid generic comparisons on
pointer until a longer run says otherwise.  **[measured]**

**The GRU on the standard pointer task, 12000 steps with the hop curriculum, 20 seeds**: after 4 seeds the mean id
accuracy is 0.178, i.e. still at chance (8 nodes) -- the ordinary recurrent memory does not solve the task at this
budget either.  If that holds for all seeds it is decisive for how the pointer result must be worded: the SSP arm's
1.000 has **no valid generic baseline to beat**, so the "pointer pass" cannot be claimed as a comparison, only as a
capability demonstration.  **[measured, 4 of 20 seeds]**

### 6.3 The one clean positive, restated

Taken together -- the iteration extrapolates from 1 hop to 8 at 1.000 (6.1), the halting fix makes rung 3 reach 1.000
on held-out hops (section 2), the node count holds to 16 (6.2) -- the pointer variant is a *capability* result: an SSP
memory with learned query codes, a learned start decode and a cue-read halting counter performs 8-hop chained
algebraic queries that no generic sequence model in this harness solves.  What it is **not** is a matched-baseline win,
because every generic baseline fails its own trainability gate (6.2).

## 7. Phase 4 -- SSP-specificity: the non-SSP control with the same aid

`FWPArm` previously had no interface hooks, so "give the control the same aid" was not implemented: the pretraining
machinery needs `address_params`, `freeze_address` and `memory_features`.  Those are now implemented (the control's
interface = its position-key head and its read-key head, the counterparts of the SSP arm's `w_time`/`read_head`;
`v_head` is left out, as the codebook is for the SSP arm), and the pointer path was refactored into
`_pointer_features` + `mix` so the readout vector is available to the probe.  Verified: the index/delay forward output
is unchanged, the pointer path still runs, and `pretrain_mode="probe"` now works on `fwp` (id 0.837 vs 0.790 plain in
a 60-step smoke run).

Comparison at index T=16, budget 256 floats, 10 hold-out seeds, identical pretraining aid (probe, 500 steps,
shape-randomised) where shown: **[measured]**

| arm | aid | N=64 conv | full-data conv | tokens to 0.90 |
|---|---|---|---|---|
| SSP (`ssp_index_learned`) | none | 0.269 | 0.986 | 128.3k **[taken from RUN_010]** |
| SSP | probe pretraining | 0.845 | **0.927** (8/10 seeds >= 0.9) | -- (collect) |
| **fwp** (non-SSP outer product) | **none** | **0.363** | 0.936 | 83.4k **[taken from RUN_010]** |
| **fwp** | **probe pretraining** | **0.821** | **0.914** (9/10) | **128.1k** |

**Verdict: the index-recall advantage is not SSP-specific, now on both axes and with the aid equalised.**  Without any
aid the generic outer-product memory already beats the SSP arm at low data (0.363 vs 0.269); with the *same* aid it is
a statistical tie at low data (0.821 vs 0.845) and an exact tie on speed (128.1k vs 132k tokens).  The schedule-free
"SSP structure" is doing nothing that a fixed-code outer-product memory with the same learned write/read heads cannot
do -- which is precisely the competing explanation handoff 09 section 1 set out to rule out, and it is not ruled out.
By docs/11 4.5 this is the "it only matches the substitutes: the algebra buys nothing in that variant" branch, and the
index variant should be reported as a *harness check*, not a thesis vehicle (answering handoff 10 section 9 item 1 in
the direction the working assumption expected, with numbers now).

## 7b. Variant A (`path`): the redesign, and a rung-0 check that fails

The proxy RUN_007 used was uninformative (no-memory floor MSE 0.024 vs best 0.013) because a query was usually about a
landmark seen at `t-1`, so the answer was just minus the previous action.  The redesign (handoff 09 section 15.4 step 6)
is now in `tasks_a.py`: **a minimum query age of 8 steps** (the target is a displacement integrated over at least 8
steps), `step_scale` 0.5 and `arena` 4.0.  Measured on the redesigned task: 13.3 queries per 32-step episode and a
target RMS of **1.36** (zero-floor MSE 1.86), against the old 0.024 floor -- i.e. the task now has signal.  **[measured]**

Rung 0 on the redesigned task, 800 steps, 1 seed, no tuning (MSE, lower is better; floor 1.86; the harness reports
`id` as negative MSE): **[measured]**

| arm | id (-MSE) | extrap_T (4x episode) | extrap_step (2x step scale/arena) | state floats |
|---|---|---|---|---|
| mlp (no memory) | 1.68 | 5.71 | 6.57 | 0 |
| gru | 0.11 | 2.57 | 0.64 | 256 |
| ssp (hand-wired path arm) | 0.25 | 3.37 | 2.55 | 256 |
| **`pose_attn` (the arm the design specifies)** | **0.098** | -- | -- | 256 |

**The 10-seed rung-0 table (3000 steps, 4-6 configs each, floored AULC selection, ep_len 32; MSE, floor 1.86;
`jobs/p4_variantA.json`, `jobs/p4_variantA_pose.json`): [measured]**

| arm | id (-MSE) | extrap_T (4x episode) | state floats |
|---|---|---|---|
| mlp (no memory) | 1.702 | 4.795 | 0 |
| **gru** | **0.049** | **1.683** | 256 |
| `pose_attn` (K=32 stored pairs, learned integrator) | 0.101 | 3.660 | 256 |
| lru | 0.135 | 5.441 | 256 |
| **ssp (hand-wired path arm)** | **0.147** | 2.434 | 256 |

**Variant A's rung 0 fails decisively: the hand-wired SSP path arm is the worst memory arm on the task it was designed
for** -- 3x the GRU's error, beaten by the LRU and by the attention-over-stored-pairs competitor, and mid-pack on
extrapolation.  By docs/11 4.6's stop rule the variant is dropped: no rung ladder, no SO(2) heading, no RL.  The
per-slot competitor is also the *cheapest* (1285 parameters against the path arm's ~2x that for its codebook).
Caveats: 6 configs per arm and 3000 steps is a light tuning budget for this task, and `pose_attn`'s own hyperparameters
were not searched beyond its `hidden`; the honest summary is "no advantage at a matched light budget", not "proven
inferior at any budget".

## 8. Phase 5 -- the RL port

- **5.1 POPGym reference: DONE.**  Rebuilt `/tmp/popgym_ref` (ray 2.59.0, torch 2.14.1+cpu, popgym 1.0.7) and ran
  POPGym's own PPO + GRU on `RepeatPreviousEasy` to 632k steps (4 workers x 8 envs).  **Result: mean episode return
  0.683, last-5 mean 0.658** (`ref_ppo/ref_gru_600k.json`), against the published reference of +0.42 at 296k steps
  **[taken from RUN_007 section 3]** and our in-repo A2C GRU's -0.108 at 490k **[taken from RUN_006]**.  This is the
  ungated repair of RUN_006's comparison and it confirms the earlier reading: the 0.99-vs--0.108 gap was the training
  loop, not the task.  Nothing about the SSP claims rests on it.
- **5.2 RL memory stub: done and tested** (`rl_memory.py`, `test_rl_memory.py`).  Contract: `reset`, `write`, `read`,
  `read_from(hidden, t)` (the address comes from the policy's hidden state plus time features -- the one genuinely new
  design question in the port), `aux_loss` (label-free; labels are the stream's own observations), `freeze_interface`,
  `state_floats` (2D floats; the codebook is a constant table, not carried state).  Tests pass: round trip 1.000 at
  dim 128 with 8 items written; the address head receives gradient before freezing and none after; `aux_loss` finite;
  `reset` clears the memory (1.00 -> 0.00).  Traps recorded: `load_state_dict` copies buffers under `no_grad` and drops
  the memory's graph -- do not use it to set up a gradient test.
- **5.3 First RL test: still gated.**  The gate (handoff 10 section 5) needs (1) a recipe passing all unseen variants
  at <= 50k tokens -- **fails**: phase 1 does not produce one, and the schedule recipe's learned arm is 132k; and
  (2) an amortised interface or an in-loop auxiliary that keeps it aligned -- **fails**: 5.1 and 5.3 above.  Running
  RL now would produce a result nobody could interpret, which is the situation the gate exists to prevent.

## 9. Collection status and what remains open

| item | state |
|---|---|
| 0.1 batch sweep | complete (section 1) |
| 0.2 halting, 10-seed protocol | **complete (section 2.2b): `cue_hop` + aux 1.000 on held-out hops, 10/10 seeds** |
| 0.3 affine init + the offset mechanism | **complete**: the grid (section 3.2) shows only the *offset* coordinate constrains discovery; the crossing rule was confirmed on purpose-built cells (section 3.1) |
| 1 the schedule-removal question | **complete, with an unexpected win**: multi-scale and hybrid routes fail (sections 4.1-4.3), but the *slope-only prior* passes 6 of 7 variants and is 4x faster (section 3.3) |
| 2 interface | complete: pretraining does not transfer and the in-loop probe is a no-op (section 5) |
| 3.1 1-hop iteration | **complete: 1.000 on held-out hops with the aux (0.660 without)** |
| 3.2 pointer node scaling and gates | **complete**: 8/16 nodes pass (1.000/0.997-1.000), 32 nodes fails at both budgets (0.205 at 512 floats, 0.304 at 1024 floats, 0/10 each), 64 nodes stopped on cost once the limit was located; `fwp`/`lru`/`tf_window` 0/30 seeds on the easy gate; the GRU 0.247 over 10 seeds at 15000 steps (section 6.2) |
| 4 the combined pointer arm | **complete: 1.000 on held-out hops 4-8 on all 10 seeds** once the halting and the start decode are both fixed (section 6.0); the aux stays load-bearing (0.513 without) |
| 5 specificity | complete: the non-SSP `fwp` ties SSP with the same aid (section 7) |
| variant A | complete and dropped: the hand-wired path arm is the worst memory arm (section 7b) |
| 5.1 POPGym reference | complete: POPGym's own PPO + GRU reaches 0.683 at 632k steps |
| 5.2 RL memory stub | complete and tested |
| 5.3 RL test | **not licensed** (gate conditions (1) and (2) fail) |
| T=64 ceiling controls | **complete**: the exact rung-0 address and the recipe both score 0.952 at 512 floats (5/10 and 6/10 seeds) and both reach ~0.99 with 10/10 at 1024 floats, so the T=64 limit is the task's capacity (section 3.3) |
| test suite | **all 17 modules pass** (12 pre-existing plus `test_halting`, `test_multiscale`, `test_affine`, `test_pretrain_shapes`, `test_rl_memory`), with no existing test or threshold edited |

**Next steps, in priority order, if this work continues:**

1. **Finish the T=64 question.**  If the exact address also caps near 0.95 at 512 floats and rises at 1024, restate the
   recipe's record as "passes every variant inside the memory's capacity, and the T=64 task needs 1024 floats"; if not,
   record T=64 as the recipe's limit.  Either way the recipe is the first in this project to clear 4.7 on the variants
   inside capacity.
2. **Cheapen the recipe's prior.**  It currently tells the learner "monotone, ~3% scale, **offset exactly 0**".  The
   mechanism says the origin must be right, but not that it must be exact: a rule that draws the origin from the
   healthy part of `sinc` (|offset| <= 0.4 or so) would be a strictly weaker prior doing the same job.
3. **Make the pointer claim fair.**  The GRU gate decides it; if the GRU cannot pass the easy variant at 15000 steps
   there is no valid generic baseline and pointer stays a capability demonstration.
4. **Settle the node count** with the capacity cells now running, and report the count where accuracy falls below 0.9.
5. **Do not** spend more on interface pretraining (three negative verdicts) or on the index variant's specificity (tied).

**Scope notes on what is *not* covered by the above (so nothing reads as done that was not run):**

- **The multi-scale arm was not run at index T=48/64 or at the N=64/256 low-data pools.**  It was run at index
  T=16/32 and delay 1-4/1-8, and it fails the pass rule there (0.738 / 0.570 / 0.813 / 0.893 against the schedule's
  0.984 / 0.833 / 0.929 / 0.999).  Because 4.7's rule requires >= 0.95 on >= 9/10 seeds on *every* variant, two
  failures already settle it; the unrun variants would add evidence, not change the verdict.  The `residue` kind is
  kernel-checked only (section 4.1).
- **`fwp` was not compared against SSP on the `path` variant.**  `FWPArm`'s write/read key heads are functions of the
  index/lag address (`batch["index"]`, `batch["lag"]`), which the path task does not provide, so the control cannot run
  there without new code.  The matched-aid specificity comparison was therefore done on **index** (section 7), where it
  is decisive, and the path variant was compared against GRU, LRU and the purpose-built `pose_attn` (section 7b).
- **The RL gate was not attempted.**  Conditions (1) and (2) fail (sections 4.4, 5), so 5.3 was not licensed by the
  pre-registered gate -- that is a decision the gate made, not an omission.

**Recommended next steps (in order), given the above:**

1. Finish the halting 10-seed protocol and the pointer scaling, then re-issue the pointer pass/fail: the held-out-hop
   criterion is now *reachable* (1.000 with `cue_hop`) and the iteration extrapolates (1.000 for the chain), so the
   question is whether the equal-aid baselines can do the same with the same halting mechanism.
2. Attack the *discovery* question in phase 1 from the other side: the measured failure is the gradient's direction,
   not the ceiling, so a supervised address signal (the `retrieval_loss` arm already exists) or an explicitly
   coarse-to-fine *network* is the next honest attempt, with the schedule as the barrier to beat.
3. Implement `pose_attn` and finish variant A: it is the only variant where the algebra has a *structural* argument
   (one group action on the whole memory) that no key-value memory reproduces.
4. Do not spend more on the interface: three attempts (fixed-shape probe, shape-randomised probe, in-training probe)
   now have measured verdicts, all negative for amortisation.

## 10. Handoff 10's open decisions, as answered by these measurements

1. **Is `index` a thesis vehicle or a harness check?**  A harness check.  With the aid equalised, the non-SSP outer-product memory ties the SSP arm at low
   data and on speed (section 7), and without any aid it is better at low data.  Claims should be made on pointer (section 6.1 is the one clean positive) and on
   variant A if it is ever finished.
2. **If phase 1 fails, is the width schedule a costed aid or ordinary training?**  A costed aid carried by the frozen range rule: it failed to be removed twice
   (RUN_010's constants, RUN_011's multi-scale set and hybrids), it has direct precedent in neural fields, and the evidence here is that discovery without it does not
   happen at any mixture tested.
3. **Keep the thesis-pass thresholds?**  Kept -- they were used exactly as pre-registered, and the phase-1/2 failures are not marginal (0.409-0.845 against 0.95; 0.40-0.46
   against 0.9), so no threshold change would rescue them but the thin `t90 <= 50k` margin.
4. **RL design: POPGym `ModelV2` first vs a purpose-built pointer environment?**  Unresolved and deliberately so: no RL was licensed, so the decision is still open.
5. **Where does this folder live, and commit it?**  Not committed, per instruction; the folder-local question is unchanged from handoff 10.

## 11. Do not do

- **Do not add tests with blind global string replacements.**  Adding `test_slope_only_recipe_...` to `test_recipes.py`
  went wrong twice in one sitting: a `s.replace("L.build_learned(\"ssp_index_learned\"", ...)` also rewrote an
  *existing* test's call site, and an appended function landed after `if __name__ == "__main__":` so the runner never
  saw it (it reported 9/9 while the file had 10 tests).  Both were caught only by running the whole module, which is why
  that is the rule: after touching a test file, run the full module and check the *count*, not just the pass line.
- Do not quote a token count without its batch: `tokens = updates x batch x 8` for index T=16 (section 1).
- Do not use `tf_full` at batch 8 (it fails its own trainability gate there).
- Do not present the coarse-to-fine width schedule as novel, and do not present either pretraining mode as amortised.
- Do not report `per_hop_accuracy` as the arm's answer for a *gated* arm: it reads the always-advancing chain, which is
  deliberately kept clean (section 2.1); the arm's answer is `query_scores()`.
- Do not use the read-head gradient norm as a basin probe (section 0.7): it does not discriminate.
- Do not `load_state_dict` into an SSP module when setting up a gradient test (it detaches buffers).
- Do not start RL or a larger-model study while the section-5 gate fails.
- Do not retune the frozen range rule on delay 1-4.

---

## 12. Conclusions and what they mean for the redirected goal (appended 2026-10-12)

All numbers below are from the sections above (10 seeds, one machine class per table, pods unless stated). Nothing here is new data.

## 12.1 Results at a glance

| question | result | where |
|---|---|---|
| Is batch size a speed lever? | No: learned arm 0/10 reaches 0.90 at batch 8, 5/10 at 16, 9/10 at 32, 10/10 at 64 | 1 |
| Why did rung-3 pointer fail on held-out hops? | The halting (and then the cue-based start decode), not the iteration (per-hop 1.000 at h = 1..8) | 2, 6.0 |
| Held-out hops 4-8 after one hop of training, no hop count given | **1.000, 10/10 seeds** with `cue_hop` halting + masked start + per-hop aux 0.5; 0.513 without the aux | 6.1b |
| Do multi-scale frequencies replace the schedule? | No: 0.738 / 0.813 / 0.893 (index T=16, delay 1-4, delay 1-8); fine-heavy and hybrid 0.409-0.793 | 4 |
| What limits discovery of the read address? | The **offset**, not the size: every offset >= 1 fails 0/10 and never moves; every offset-0 cell passes (9/10); a dead-zone offset is rescued only if the drift crosses zero inside the stream (3 pre-registered sign cells all landed) | 3.1, 3.2 |
| Does the range-scaling schedule rule generalise? | No: 0.984 / 0.871 / 0.833 / 0.775 / 0.690 at index T=16/24/32/48/64 (0/10 seeds at 0.95 past T=16) | 3.3, 0b |
| Is there a working recipe? | Yes, for the continuous-address family: ramp x (1 + 0.03 N(0,1)), offset 0, fixed frequencies; passes every variant inside the memory's capacity (table in section 0 item 5) | 3.3 |
| Is the T=64 failure the prior? | No, capacity: the exact address scores the same at 512 floats (0.952, 5/10) and both reach ~0.99, 10/10 at 1024 floats | 3.3, 9 |
| Does interface pretraining amortise? | No: scales with budget (N=64: 0.269 -> 0.845 over 0-500 steps) but fails transfer (0.40-0.46); the in-training probe aux is a no-op | 5 |
| Pointer node limit | 16 nodes pass (1.000); 32 fail (0.205 at 512, 0.304 at 1024 floats, 0/10) -- iteration-bound, not capacity-bound; 64 stopped on cost | 6.2 |
| Does any generic baseline pass the pointer gate? | No: GRU 0.247 over 10 seeds at 15000 steps; `fwp`/`lru`/`tf_window` 0/30 | 6.2 |
| Is SSP specifically better on index? | No: `fwp` ties with equal aid | 7 |
| Variant A (path) | rung 0 fails: MSE SSP path 0.147 vs GRU 0.049, `pose_attn` 0.101, LRU 0.135; dropped | 7b |
| POPGym PPO + GRU on `RepeatPreviousEasy` | 0.683 at 632k steps (not converged) | 8 |

## 12.2 What can be concluded (and what cannot)

- **Concluded.** (1) The hard part of training an SSP read address is *discovery*, specifically the origin of the initial read head; capacity and speed are not the
  obstacle. (2) The mechanism predicts (three cells forecast before running). (3) A mechanism-derived recipe exists for positional address recall and is flat in
  stream length, unlike the schedule. (4) The unbind-clean-up iteration composes: one hop of training answers 8 hops, once halting and the start decode are not
  learned from two hops. (5) The memory has a capacity (`D/4`) outside which no initialisation reaches the bar. (6) Process: RUN_010's unseen-variant claim was a
  T=16 artefact; pod and local runs differ numerically.
- **Not concluded.** (a) That SSP is more sample-efficient than generic memories: no matched, trainable, incumbent baseline was compared (the `fwp` arm is plain
  linear attention, not a delta-rule or gated fast-weight memory; pointer baselines fail their gates). (b) That the recipe is deployable: it hands the model the
  address shape (offset 0, scale to ~3%, a time ramp) and has a tuned constant. (c) That any of this helps in RL: no RL run; interface does not transfer.
  (d) Anything about robustness beyond a monotone, non-graceful cue-drop curve.
- **The recipe against the redirected bar** (docs/11 section 14): it is a *research finding* about discoverability, not yet a module. It keeps four hand-set
  ingredients (offset, `eps`, fixed frequencies, time ramp). It is the reference to beat, not the product.

## 12.3 What the redirection changes

The user restated the goal: a **deployable, learnable, simple SSP memory module for RL agents**, judged by the bitter lesson (few task-dependent constants; learn
what can be learned), real-world utility (one default configuration, a plug-in policy memory) and parity-or-better against generic memories with better sample
efficiency and possibly robustness. So the next round is not "more characterisation" but **removing the hand-set rows** (axis 1: learnable parts; axis 2:
architecture), with matched incumbent baselines, a constants scoreboard, and an RL track (POPGym `RepeatPrevious`) plus a spatial recall gate. Design, pass rules and
traps are in docs/11 section 14 and `handoffs/11-ssp-memory-module-deployable.md`.

Ideas ranked for that goal (all untested): (1) tied encoder with relative addressing -- SSP's shift is a binding, so the read origin is removed by construction;
(2) a kernel with no dead zone; (3) K random read heads (compute for a constant; still needs an offset range); (4) a learned initialisation across tasks (risk:
pretraining did not transfer); (5) learned frequencies (only after 1 or 2); (6) attention as a teacher (two-stage; low rank). Architecture additions judged mostly
as parity with fast-weight practice: delta-rule write, learned write and forget gates, multi-timescale banks; the distinctive ones are relative addressing and
iterated reads (the node limit of 16-vs-32 is the open question there). A first diagnostic that matters for all of them: do *plain and delta-rule `fwp`* show the same
offset dead zone? If so the discovery problem is general to key-addressed memories, not an SSP flaw.
