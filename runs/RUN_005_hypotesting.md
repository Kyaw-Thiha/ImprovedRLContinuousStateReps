# RUN_005 — Stage-0 VSA memory battery, CartPole length-scale diagnostic, and the H1 interference test

**Status: COMPLETE (2026-10-10). Encoder-only. No RL training, no environment loop, no GPU.** Every number below
is MEASURED in this session on the local machine (RTX 3050 Laptop 4 GB, 16 cores, 15 GB RAM, repo `.venv`:
Python 3.14, numpy 2.5.3, scipy 1.18.1, torch 2.14.0+cu130, `sspspace` installed). Nothing is committed or
pushed. Total compute across all three parts: **~130 s wall, peak RSS 516 MB, CPU only.**

This run answers the question left open by `docs/10-vsa-memory-and-composition-direction.md` §6 (the "cheap
deciders before spending anything") and §11-14 (the Stage-0/1/2 plan), and then tests hypothesis **H1** from
`docs/08-ssp-vs-tile-coding-theory-and-hypotheses.md` §4 on the saved eval_v3 policies.

**Artifacts (all new, all inside one folder):**

```
experiments/vsa_memory_stage0/vsa.py             FHRR/SSP primitives + the three frequency designs
experiments/vsa_memory_stage0/stage0.py          T0/T1/T4/T6/T7 battery (CLI)
experiments/vsa_memory_stage0/cartpole_ls.py     CartPole length-scale / resolution diagnostic
experiments/vsa_memory_stage0/interference.py    H1 interference measure on saved policies
experiments/vsa_memory_stage0/extras.py          supplementary measurements (T4 endpoints, T7 naive vs matched, winner fills)
experiments/vsa_memory_stage0/results.json
experiments/vsa_memory_stage0/cartpole_ls.json
experiments/vsa_memory_stage0/interference.json
experiments/vsa_memory_stage0/results_extras.json
```

**Reproduce:**

```bash
.venv/bin/python experiments/vsa_memory_stage0/stage0.py          # T0,T1,T4,T6,T7  (~107 s)
.venv/bin/python experiments/vsa_memory_stage0/cartpole_ls.py     # (~0.8 s)
.venv/bin/python experiments/vsa_memory_stage0/interference.py    # (~21 s)
.venv/bin/python experiments/vsa_memory_stage0/extras.py          # (~10 s)
```

**Scope note.** The encoder design is shared with the repo: `legacy` = `rl/representations/ssp_legacy.py`
(n_scales 8, scales linspace(0.5, 2), the A2C-era encoder) reused directly by file-path import; `current` =
upstream `sspspace.HexagonalSSPSpace` (n_scales 5, what the DQN `SSPRep` uses); `rff` = dense Gaussian-spectrum
control. No repo file was modified; `git status` shows only the seven new files above.

---

## 1. Protocol

### 1.1 What was run

| Test | Question | Protocol |
|---|---|---|
| **T0** | Is the algebra exact? | `phi(x)phi(y) == phi(x+y)`, `phi(y)conj(phi(x)) == phi(y-x)`, and kappa depends only on x-y. All three designs, d in {1,2,4} |
| **T1** | Kernel shape and long-range floor | kappa(delta) over delta in [0, 40] with random unit directions, 1-D and 4-D. Reports decay half-width, first sub-0.1 radius, and the floor over the last decile |
| **T4** | **Query capacity (the crux)** | Bundle m chains of `depth` binds, unbind with the query keys, clean up against a fixed 512-entry codebook. m in {1,2,5,10,20,50,100}, depth in {1,2,3}, D at the repo's range endpoints and above, 10 seeds |
| **T6** | Degraded-cue recall | Zero out a fraction p of the query dimensions before retrieval; p in {0,0.1,0.25,0.5,0.75,0.9}, m=10, depth 1 |
| **T7** | Variable entity count (structured working memory) | N spatial entities bound to roles in one fixed-size vector, each retrieved individually via cleanup against a 21x21 position grid; N in {1,2,4,8} |
| **CartPole diagnostic** | Is the SSP deficit a length-scale artefact (doc 08 M3a)? | Per-axis kappa along each CartPole axis at every *fitted* length_scale from the repo's own winners; the single-scalar feasibility solve; the per-dimension (normalised) variant |
| **H1 interference** | Do dense features interfere more? | Change in squared TD residual on a fixed 256-transition probe set per `DQN.update`, on saved eval_v3 policies, SSP vs tile coding |

Deferred, and why: **T2** (resolution), **T3** (action separation) and **T5** (group structure of the task) all
require a chosen dynamics and its state/action ranges, so they are not runnable without picking a task. T2/T3 for
CartPole specifically *were* run, as the CartPole diagnostic in section 3.

### 1.2 Dimension convention (read this before the T4 tables)

The Stage-0 CLI takes and reports **complex** FHRR dimension `Dc`. The repo's convention is the **real**
`ssp_dim = 2*Dc + 1`. For d=4 and n_scales=5 the repo's search range is `ssp_dim = 50*n_rotates + 1` with
n_rotates in [4, 20], i.e. **real 201-1001 = complex 100-500** (`handoffs/03` §2).

So the CLI's nominal `{201, 1001, 1601}` are complex, i.e. real **403 / 2003 / 3203** — the middle and top sit
*above* the repo's search range. To cover the actual endpoints, T4 was additionally run at **Dc=100 (real 201)**
and **Dc=500 (real 1001)**, and the legacy design lands on its scale-ladder granularity (Dc=120 -> real 241,
Dc=520 -> real 1041). Both sets are reported.

---

## 2. Results — Stage-0 encoder-only battery

Wall-clock per test, 10 seeds, three designs: T0 0.5 s, T1 1.6 s, **T4 80.9 s**, T6 21.4 s, T7 2.0 s.
Total **107 s wall, 26 min CPU, peak RSS 516 MB.**

### 2.1 T0 — algebra parity: PASS

Maximum relative error across all designs and dimensions:

- bind (`phi(x)phi(y)` vs `phi(x+y)`): **2.7e-15**
- unbind (`phi(y)conj(phi(x))` vs `phi(y-x)`): **3.2e-15**
- kappa depends only on x-y: **1.1e-16**

Threshold was 1e-12. A failure here would have invalidated everything downstream (cf. the legacy A2C lesson in
`handoffs/03` §4); it did not fail. Note the design comparison: the `legacy` and `current` designs are
*real-valued* SSP matrices in the 2n+1 Hermitian layout, so the complex frequencies are extracted by
`half_spectrum()`, which asserts the layout's Hermitian symmetry before use.

### 2.2 T1 — kernel shape and the long-range floor: PASS on the doc's criterion

kappa(delta) at the fitted-style setting `length_scale = 1`, sweep to delta = 40:

| design | d | Dc | half-width (kappa=0.5) | first delta < 0.1 | kappa at delta=1 | floor mean | floor max |
|---|---|---|---|---|---|---|---|
| legacy | 1 | 142 | 0.90 | 1.20 | 0.286 | 0.0118 | 0.0275 |
| legacy | 4 | 1040 | 1.80 | 3.71 | 0.798 | 0.0032 | 0.0060 |
| current | 1 | 212 | 1.90 | 2.91 | 0.842 | 0.0186 | 0.0327 |
| current | 4 | 1025 | 4.41 | 12.13 | 0.963 | 0.0239 | 0.0374 |
| rff | 1 | 1001 | 1.30 | 2.21 | 0.626 | 0.0280 | 0.0516 |
| rff | 4 | 1001 | 1.20 | 2.21 | 0.605 | 0.0007 | 0.0017 |

**Floor verdict.** Every design's long-range floor is **below 0.05** (mean) at Dc~1000, matching the ~1/sqrt(D)
prediction of `docs/10` §10.1 item 3. The doc's kill condition for the OOD/hallucination claim — "floor
dominates well inside the task's working range" — **does not fire**. That claim survives at this dimension.

**But the working range is the live issue, not the floor.** At `length_scale=1` the `current` design's kernel is
still 96% self-similar one state-unit away (half-width 4.41). A unit-scale state space is, to this encoder, very
nearly a single point. `length_scale` must be matched to the task's range before any of this is usable — which is
what section 3 quantifies for CartPole.

### 2.3 T4 — query capacity (the crux): PASS, and the margin is large

Retrieval accuracy vs bundle size m, 10 seeds. Depth 1 and depth 3 are **identical within noise** in every cell
(shown for the endpoints), so only depth 1 is tabulated:

| configuration | real dim | m=1 | m=2 | m=5 | m=10 | m=20 | m=50 | m=100 |
|---|---|---|---|---|---|---|---|---|
| legacy, Dc=120 | 241 | 1.00 | 1.00 | 1.00 | 0.98 | 0.72 | 0.20 | 0.09 |
| current, Dc=100 | **201** | 1.00 | 1.00 | 1.00 | 0.89 | 0.55 | 0.14 | 0.07 |
| legacy, Dc=520 | **1041** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.94 | 0.56 |
| current, Dc=500 | **1001** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.55 |
| legacy, Dc=1640 | 3281 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |
| current, Dc=1625 | 3251 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |

**Capacity rule (empirical).** Reliable retrieval (>0.9) holds up to roughly **m ~ Dc/4**, i.e. m ~ 20 at the
repo's smallest real dimension (201), m ~ 50 at its largest (1001), and m ~ 100 at real ~2000 and above.

**Chained unbinding is free.** depth 3 costs nothing relative to depth 1 in any cell (e.g. current Dc=500:
m=50 0.90 vs 0.92; legacy Dc=120: m=20 0.72 vs 0.69). The doc's only genuinely unique capability — answering
a query by algebra on a bundle rather than by forward simulation — is real and its cost does not compound.

**The crux test does not discriminate for the robotics shortlist.** `docs/10` §14.4 identifies
BunchOfColors/SeqOfColors/ChainOfColors 3/5/7 as the only MIKASA tasks where SSP's unique capability is
structurally load-bearing. Those hold 3-7 items. At the repo's *smallest* dimension (real 201) the encoder
already retrieves 5 items at 1.00 and 10 at 0.89. So T4 passes with margin and therefore **cannot** serve as
evidence for the "fewer episodes / tasks others cannot solve" hypothesis; it only clears the preconditions. The
decision moves entirely to learning curves (Stage 1/2).

### 2.4 T6 — degraded-cue recall: PASS at Dc>=1000, collapses at Dc~200

Accuracy vs fraction p of query dimensions zeroed, m=10, depth 1:

| configuration | p=0 | p=0.1 | p=0.25 | p=0.5 | p=0.75 | p=0.9 |
|---|---|---|---|---|---|---|
| current, real 225 | 1.00 | 1.00 | 1.00 | 0.96 | 0.69 | 0.22 |
| legacy, real 240 | 1.00 | 1.00 | 0.99 | 0.98 | 0.62 | 0.25 |
| rff, real 201 | 1.00 | 1.00 | 1.00 | 0.88 | 0.64 | 0.15 |
| current, real 1625 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| legacy, real 1640 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| rff, real 1001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |

Degradation is graceful and dimension-dependent: at Dc~200 it breaks down past p=0.5; at the repo's top dimension
it survives losing 90% of the query. **The POMDP / sensor-dropout claim needs the top of the range, not the
bottom.**

### 2.5 T7 — variable entity count: PASS, but only under length-scale matching

Two configurations were measured, and the contrast is the finding.

**Unmatched (`length_scale=1`, the naive setting).** Every design fails at N>=2. "Dc" is the achieved complex
dimension (the nominal request in brackets, since the hexagonal ladder quantises):

| design | Dc (nominal) | off-diagonal mean kappa over the 441-cell grid | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|---|---|
| legacy | 120 (100) | 0.850 | 1.00 | 0.10 | 0.00 | 0.01 |
| legacy | 520 (500) | 0.850 | 1.00 | 0.70 | 0.25 | 0.01 |
| current | 105 (100) | **0.972** | 1.00 | 0.00 | 0.00 | 0.00 |
| current | 510 (500) | **0.972** | 1.00 | 0.00 | 0.05 | 0.00 |
| rff | 100 (100) | 0.850 | 1.00 | 0.00 | 0.03 | 0.01 |
| rff | 500 (500) | 0.848 | 1.00 | 0.35 | 0.17 | 0.03 |

**Resolution-matched.** With `length_scale` set per design so that one grid step equals that design's kernel
half-width (kappa ~ 0.5 between adjacent cells): **N=8 retrieval is 0.99-1.00 for every design at both dimensions**
(legacy ls=0.0415 -> 0.99 at Dc=120 and 1.00 at Dc=520; current ls=0.0166 -> 1.00 both; rff ls=0.0415/0.0355 ->
1.00 both).

**Interpretation.** Spatial multi-entity retrieval is dominated by **scale matching**, not by the frequency
design. The hexagonal `current` design is somewhat more correlated (0.972 vs 0.850 mean off-diagonal kappa) but
the failure mode is shared, and matching equalises all three. This **corrects** an earlier reading in this
session that the `current` design's scale ladder "aliases over a unit box while the others do not": that reading
came from a run made before the `length_scale` bug fix in section 5.1 (in which `current` ignored
`length_scale` entirely), and it is not supported post-fix.

---

## 3. Results — CartPole length-scale / resolution diagnostic (doc 08 M3a)

Encoder-only, 0.8 s. CartPole ranges: x +-4.8, x_dot +-10, theta +-0.418, omega +-10. The repo feeds SSP the
**raw** state (`normalize_state` is `False` in every cell and `domain_bounds_` is never passed — `handoffs/06`
§2(b)). The value function has a hard edge at `|theta| > 0.209`.

### 3.1 The theta axis is unresolved at every fitted length_scale

kappa along the theta axis alone, at each winner's own `length_scale`:

| winner | encoder | length_scale | kappa_theta at the cliff (0.209) | kappa_theta across the full range (0.418) | half-width along theta | half-width along omega |
|---|---|---|---|---|---|---|
| DQN ssp/none/linear | current | 0.1477 | **0.927** | 0.747 | never < 0.5 | 0.652 |
| DQN ssp/value/deep | current | 0.1631 | **0.940** | 0.785 | never < 0.5 | 0.727 |
| A2C ssp/none | legacy | 0.2477 | **0.851** | 0.518 | never < 0.5 | 0.476 |
| A2C ssp/value | legacy | 0.3433 | **0.920** | 0.713 | never < 0.5 | 0.652 |
| A2C NNI representation | legacy | 0.5276 | **0.965** | 0.867 | never < 0.5 | 0.977 |

**Two states on opposite sides of the cliff are 85-97% "the same" to the readout**, and along theta the kernel
never reaches kappa = 0.5 anywhere inside the axis's own range. Meanwhile omega is *over*-resolved (half-width
0.48-0.73 against a tile-coding bin width of 10/12 = 0.83 and a per-step change of about 1). One isotropic scalar
spends its resolution on x and omega and runs out before theta.

### 3.2 The single scalar cannot serve both axes (quantified)

kappa's half-width scales linearly with `length_scale`. Along theta at `length_scale=1` the half-width is
**5.013**, so forcing it down to the cliff width requires:

```
length_scale = 0.209 / 5.013 = 0.0417
```

which is 6x below the lowest fitted winner (0.2477) and 2.4x below the search's lower bound (0.1). At that
setting:

```
theta half-width        : 0.193   (cliff now resolved)
omega half-width        : 0.201
kappa_omega at delta=1  : 0.136   (omega is ~random at one-step separation)
kappa_x     at delta=1  : -0.125
```

So the trade-off is not "the search did not go low enough" — it is that **no value of one scalar resolves theta
at the cliff while keeping omega and x usable.** That is a stronger and more specific statement than doc 08 M3a
as written, and it explains the `length_scale` floor crowding (two-thirds of trials pinned below 0.14 in two
cells, `handoffs/06` §2): the search was pressing against a wall created by the isotropic parameterisation, not
by a mis-set bound.

### 3.3 Per-dimension scaling repairs it, inside the range the search already had

Equivalent per-dimension effective `length_scale_i = ls * range_i` (i.e. normalise each axis by its range):

| ls | theta half-width | theta resolved? | x | omega |
|---|---|---|---|---|
| 0.163 | **0.118** | yes | 1.420 | 3.008 |
| 0.248 | **0.180** | yes | 2.153 | 4.561 |
| 0.500 | **0.362** | yes | 4.343 | 9.173 |

theta's half-width drops below the cliff width at `length_scale` values **already inside the repo's search
space**, while omega becomes coarse (which is what the grid does by construction). This is exactly the H2
prediction in doc 08 §4, now with a specific predicted signature: normalisation improves theta resolution and
coarsens omega; the sweep should be flat if the deficit is not a scale artefact.

### 3.4 The caveat that bounds this whole section

"The encoder cannot place the edge" is a statement about the **linear** readout. A smooth isotropic basis cannot
represent a step at any bandwidth, but a 64x64 ReLU head can *construct* one from smooth features — it has to
learn it, which costs samples. This is why H4 (nonlinear readout) is the companion test rather than an optional
one, and it is consistent with the observation that the deep SSP arm (478.5) sits above the linear SSP arm
(461.6) in section 4.2.

---

## 4. Results — H1 interference test on the saved eval_v3 policies

21 s. Uses the repo's own `rl.networks.dqnBasic.DQN.update`, so the "update" is the real training update. Seeds
3000, policies from `outputs/eval_v3/`.

### 4.1 Design, and the two corrections it needed

The first implementation was **discarded as invalid** for two reasons, both recorded because they will recur:

1. It drew probe and batch transitions from a *uniform-random* policy. On those states the trained nets are far
   from their Bellman fixed point, so the probe residual was dominated by genuine learning progress rather than
   by representation crosstalk. Fixed by using a hand-coded CartPole controller (arm-neutral, mean episode
   length **500.0**), so the data is in-distribution and neither arm is privileged.
2. It used the online net as its own bootstrap target, so a residual change mixed "the predictor moved" with
   "the target moved". Fixed by computing the regression target `r + gamma*max Q0(s')` once from the loaded net
   and holding it fixed for the whole loop.

Final design: shared raw transitions in shared order, identical probe subset, identical RNG seed before every
update (so `ReplayBuffer.sample` draws identical indices), 20 000 transitions, 256 probes, batch 64,
200 updates, gamma 0.99. Metric: mean change in squared residual on the probe set, reported raw and per unit
parameter step `||dTheta||`, plus the signed drift.

### 4.2 The linear pair — the trustworthy result

The linear winners use `reward_center_mode: none`, so their Q is a raw-return value function and the residual is
interpretable. Baseline scores (from `train_summary.json`, 20 seeds):

| cell | score_mean | score_std | share_solved_lastk | final_policy_eval_mean |
|---|---|---|---|---|
| `dqn_ssp_none` (linear) | 461.646 | 35.880 | 0.45 | 455.160 |
| `dqn_tile_coding_none` (linear) | **499.639** | 0.989 | 1.00 | 499.091 |

Interference:

| arm | lr | mean abs dR^2 | signed dR^2 | residual level | \|\|dTheta\|\| | abs dR^2 per step | signed dR^2 per step |
|---|---|---|---|---|---|---|---|
| ssp / own lr | 0.03664 | 0.1336 | 0.0305 | 3.935 | 0.46139 | **0.290** | 0.066 |
| tile / own lr | 0.00194 | 0.2124 | 0.1568 | 16.013 | 0.02776 | **7.653** | 5.649 |
| ssp / lr 1e-3 | 0.00100 | 0.0080 | -0.0051 | 1.247 | 0.03837 | **0.209** | -0.134 |
| tile / lr 1e-3 | 0.00100 | 0.0963 | 0.0449 | 7.811 | 0.01586 | **6.067** | 2.828 |

Precision note: `abs dR^2 per step` is `interference_meanabs / param_step_norm`; 0.2895 rounds to 0.290 at three
decimals in the JSON's own representation but prints as 0.289 under `%.3f`. The ratios quoted in the text
(12x raw at common lr, ~29x per step) come from the unrounded values.

At a common learning rate — same data, same batch indices, only the representation differs — SSP disturbs the
probe set **12x less** in raw terms and **~29x less** per unit parameter step. The signed drift agrees (SSP
+0.066 vs tile +5.65 per unit step at own lr). **The dense representation is the calmer one**, which is the
opposite of what M1 predicts.

Note also that tile coding's probe residual is *higher* (16.0 vs 3.9) while it performs *better* (499.6 vs
461.6): TD-error-based interference decouples from control performance, so this metric should never be used
alone.

### 4.3 The deep pair — CONTAMINATED, do not use these numbers

| arm | lr | mean abs dR^2 | signed dR^2 | residual level | \|\|dTheta\|\| | abs dR^2 per step |
|---|---|---|---|---|---|---|
| ssp / own lr | 0.00071 | 7.6866 | 7.2957 | 419.501 | 0.03953 | 194.459 |
| tile / own lr | 0.00011 | 2.4028 | 1.5834 | 185.148 | 0.00498 | 482.482 |
| ssp / lr 1e-3 | 0.00100 | 8.2587 | 7.2989 | 419.819 | 0.04466 | 184.916 |
| tile / lr 1e-3 | 0.00100 | 10.5716 | 1.3806 | 143.916 | 0.02804 | 377.051 |

**Both deep winners use `reward_center_mode: value`**, so their Q approximates a *centred* value, offset from
the raw-reward regression target by roughly `avg_reward/(1-gamma)`. That inflates the absolute residual level
(419 and 185, versus 3.9 and 16.0 for the linear pair) and contaminates the per-step figures. A
centering-aware re-derivation is required before the deep arm can be compared. Recorded so it is not
rediscovered as a result.

### 4.4 Verdict on M1

On the only pair that is cleanly measured, **M1 as stated is not supported**: per unit parameter movement, the
SSP representation interferes *less* than tile coding, not more. This is consistent with doc 08 §6's finding that
tile coding in this repo is IHT-collision-dominated (dictionary 100% full, 81-98% of lookups colliding), i.e. it
is not the clean local basis that M1's contrast assumes. The implication is that **the comparator itself needs
repair before the interference question can be settled either way**.

### 4.5 Caveats on the metric

This is a proxy for Liu et al.'s statistic, not their exact measure. Sparse-versus-dense step normalisation is
genuinely ambiguous (`||dTheta||` understates per-coordinate movement for a sparse arm, since only the tiles hit
by a batch receive gradients), which is why both the common-lr raw number and the per-step number are reported
rather than one headline.

---

## 5. Corrections and additions recorded during the run

These belong in the record because each would otherwise have produced a wrong number.

1. **Bug: the `current` design ignored `length_scale`.** `make_omega(..., "current")` returned the upstream
   phase matrix without dividing by `length_scale`, while the `legacy` and `rff` branches did divide. All
   `current` results before the fix were implicitly at `length_scale=1`. This is what made the `current` design
   look like it had a 0.48 long-range floor and a degenerate spatial kernel; post-fix its floor is 0.024 mean and
   it behaves like the others. Caught by a per-axis diagnostic, not by a test.
2. **T1's sweep was too short.** At 5x the state range, the `current` design's half-width (4.41) had not been
   reached, so mid-decay values were being reported as the "floor" (0.48 instead of 0.024). Fixed by sweeping to
   40x and computing the floor over the last decile. The doc's "sweep to 5x" is insufficient for these designs.
3. **T4's value codes were inconsistent with the encoder's domain.** An early version encoded 2-D grid points
   with a 4-D `Omega` and crashed on shape mismatch. Fixed by making T4 a pure item-memory test (random
   unit-modulus codes), which is also the right design: it isolates superposition capacity from kernel
   resolution, and it makes the test design-independent by construction.
4. **T7 needed resolution matching to be a capacity test at all.** With `length_scale=1` every design fails at
   N>=2 for a reason that says nothing about capacity (adjacent grid cells have kappa 0.85-0.97). Fixed by
   setting `length_scale = grid_step / half_width` per design. Consequence: T7 is a test of capacity *given*
   scale matching, and the scale matching is a controlled choice, not a result.
5. **The single-scalar solve was inverted** in the first version of the CartPole diagnostic (longer
   `length_scale` gives a *coarser* kernel, so the target is `target/half_width_at_1`, not its reciprocal). It
   reported a NaN and would have reported a nonsense conclusion.
6. **T4/T6/T7 are design-independent.** They use random unit-modulus item codes, so the three-design comparison
   carries information only in T1 and T7. Stated here so the three-design tables in section 2 are not
   over-read.

---

## 6. Limitations

1. **No learning loop anywhere in this run.** Nothing here measures episodes-to-competence. The hypothesis under
   test ("robots learn in fewer episodes") is untouched; only preconditions were measured.
2. **T4's crux test does not discriminate for the robotics shortlist** (section 2.3). It passes with margin at
   the repo's smallest dimension, so it can neither support nor refute the memory thesis.
3. **The temporal floor comparison is at an arbitrary `length_scale`.** T1 fixes `length_scale=1` for all
   designs, which is a fair comparison of kernel *shape* but not of tuned performance. The fitted values are
   used in section 3 instead.
4. **Scene: the CartPole diagnostic is encoder-only.** It shows the features are too smooth at the cliff for a
   linear readout. It does not show that a better scale would improve the score; that is H2's job.
5. **The interference comparison is on one seed** (3000) and one probe set, so the numbers in section 4 are
   point estimates with no error bars. Given the 12-29x separation this is unlikely to flip, but it is not a
   distribution.
6. **Stage 1 and Stage 2 were not run.** Dependency sizing only: `popgym` core resolves to 9 packages with no
   torch; with baselines (`ray[rllib]`, `torch`) 155 packages (ray 2.51.2, torch 2.8.0). `mikasa-robo-suite==0.0.5`
   needs `--prerelease=allow` and resolves to **391 packages** (mani-skill 3.0.0b15, sapien 3.0.0b1, torch 2.2.1,
   numpy 1.23.5); `1.0.0` (VLA) is 441 packages. Both MIKASA lines are incompatible with this venv and need
   their own environment; sapien renders through Vulkan and the ranked tasks need RGB, so the 4 GB GPU is
   likely too small for GPU-parallel envs.

---

## 7. Conclusions

1. **The algebra is exact and the preconditions hold.** T0 passes at 3e-15. Capacity is m ~ Dc/4 with chained
   queries costing nothing (T4). Dropout is graceful above Dc~1000 (T6). Spatial multi-entity retrieval works
   (T7). The long-range floor is below 0.05 at Dc~1000, so the doc's kill condition for the OOD claim does not
   fire. **Stage 0's purpose was to be able to kill the direction; it did not kill it.**

2. **It also did not validate it.** The crux test (T4) passes with a large margin at the smallest dimension the
   repo uses, and the robotics tasks that structurally need SSP's unique capability hold 3-7 items. So Stage 0
   clears the runway and moves the decision entirely onto learning curves.

3. **The CartPole deficit is substantially a parameterisation artefact, and the mechanism is the isotropic
   scalar.** At every fitted `length_scale`, in both encoders, the theta axis — which carries most of the
   control-relevant information — is unresolved (kappa at the cliff 0.85-0.97). Forcing theta resolution
   requires `length_scale = 0.0417`, at which omega and x become near-random at one-step separation. Per-dimension
   scaling resolves theta at `length_scale` values **already in the search space**, and this is the explanation
   for the observed floor crowding.

4. **But this does not restore an advantage, and the ceiling is low.** CartPole caps at 500; linear SSP is at
   461.6 against tile coding's 499.6, so the whole headroom is <=38 points (~8% of the metric). The deep arm
   already recovered about half of it (461.6 -> 478.5, confounded by a different centering mode), which is
   directional evidence that the readout is as much the binding constraint as the scale (H4). A grid places a
   hard edge *by construction*; SSP needs per-dimension tuning to do what the grid does for free, so the best
   case after a full fix is **parity, not a lead**.

5. **M1 (interference) is not supported on the cleanly-measured pair.** Per unit parameter step, SSP's probe
   disturbance is 0.290 in the linear pair against tile coding's 7.653 (own lr) and 0.209 against 6.067 at a
   common lr. Combined with doc 08 §6 (tile coding is IHT-collision-dominated here, so it is not a clean local
   basis), the correct conclusion is that **the interference contrast in doc 08 was never cleanly testable in
   this configuration**, not that SSP is interference-free.

6. **Net effect on the encoder thread's narrative.** It should not be "SSP loses because dense features
   interfere". It should be "SSP's kernel is isotropic over four axes spanning 50x, so it cannot place the
   CartPole cliff; per-dimension scaling fixes the linear arm, and what remains is the sample cost of learning
   the edge in a nonlinear readout". That is sharper and more defensible than a ranking — and it is still a
   parity story on a structureless task, which is why the memory direction remains the one with upside.

---

## 8. Next steps

Ordered by information per unit cost, with the necessity judgement stated.

1. **H2 — normalisation x extended `length_scale` (necessary only if the CartPole negative is a publication
   deliverable).** Normalisation on/off x `length_scale` log-spaced over [0.01, 1.0] x {isotropic,
   per-dimension}, at each cell's winner, 10 seeds. Hours, ~$2. It now has a specific predicted signature
   (theta resolution improves, omega coarsens) and a stated asymmetry: holding `lr` fixed biases against the
   changed encoder, so a **positive is trustworthy and a null is inconclusive** (doc 08 §4). This is the only
   test needed to close the "why was SSP worse?" question.
2. **POPGym — Stage 1 (necessary for the memory direction; nothing else on any list is).** Write the VSA memory
   module once against POPGym's `MemoryModel` interface (`memory_forward` / `initial_state`), then run
   `Count Recall` and `Autoencode` (the direct superposition-capacity proxies, i.e. exactly what T4/T7 measured)
   plus `Concentration` (cue-based recall). Baselines MLP, positional-MLP, GRU. Score as **steps-to-competence,
   mean-of-last-k, not MMER**. ~2 GPU-h per run; the envs are vector-observation, so a first pass fits on this
   CPU-only machine. The same module object is what drops into MIKASA's PPO-LSTM slot later, so this is one
   implementation, not a pipeline.

Explicitly **not** next, with reasons:

- **H1 with a repaired local basis.** It would re-establish a mechanism this session's data already failed to
  support. The correct response to doc 08 §6 is a footnote, not a training budget.
- **Wrapped-angle Swimmer and Reacher shifted-target (doc 09 Tier 1).** These are encoder-thread tests; doc 10
  already concluded the encoder thread is substitutable (Fourier, learned encoders), and both need new
  environment setup. They matter only if the encoder becomes a deliverable in its own right — decision-log
  entries 31/32 already answered no.
- **H4 (random-ReLU readout) as a gate.** Optional. The deep arm gives directional support already; run it only
  if "the readout is the binding constraint" needs to be a load-bearing sentence.
- **Running the CartPole diagnostics and POPGym in parallel.** The CartPole work can only produce a parity
  result; it should not compete for attention with the only test that can produce a positive one.

---

## 9. Do not do

- Do not cite T4 as evidence for the memory hypothesis. It passes with margin and therefore discriminates
  nothing about the robotics shortlist.
- Do not use the section 4.3 deep-pair numbers. They are contaminated by `value` reward centering.
- Do not compare dimensions across this file and the repo without converting: **here `Dc` is complex,
  `ssp_dim = 2*Dc+1` is real**, and the repo's search range is real 201-1001.
- Do not read T7's success as a property of the frequency design. It is a property of matching `length_scale` to
  the grid step; the same capability fails for every design without that matching.
- Do not report `length_scale` sweeps that stop at 5x the state range for these encoders — the `current`
  design's half-width at `length_scale=1` is 4.41.
- Do not run the remaining CartPole diagnostics before deciding whether the encoder result is a deliverable.
