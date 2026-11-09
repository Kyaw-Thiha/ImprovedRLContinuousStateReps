# RUN_012: the practitioner's fast-weight baseline, the four candidates, and the content family (handoff 11)

Written 2026-10-12. Executes `handoffs/11-ssp-memory-module-deployable.md` under the user's instruction to run the rest
of the plan without asking. Nothing is committed (the user said not to). Conventions as in RUN_005-011: every number
is **[measured]**, **[inferred]** or **[taken from an earlier run]**; pod and local results are never tabulated
together; every claim is against a matched, trainable baseline.

**Summary of what this round establishes.** (1) The `fwp` control used by every earlier run was **not** the memory
practitioners use; the delta-rule, tied form was built, gated and measured, and it changes two conclusions: `delay` is
substitutable at parity (0.998 against the SSP recipe's 1.000) while on `index` at equal carried state the fast-weight
family is far behind (0.744 against 0.957, and 0 of 10 seeds at 0.9 against 10 of 10 if the address is fixed for both).
(2) The read-address dead zone is **not SSP-specific** — both fast-weight arms degrade monotonically with a read
offset, so RUN_011's mechanism generalises. (3) **Tied addressing removes the hand-set prior entirely**: with the read
address produced by the same weights as the write address, `delay` reaches 0.999/0.994 with 10 of 10 seeds at 0.95 and
`index` reaches 0.930 with 10 of 10 at 0.9, from a *random* start with no `eps`, no offset, no ramp and no range
constant — the first configuration in the project that needs no task-dependent constant. On index that costs ~3
accuracy points, a much slower tail and ~19x the tokens; on delay it is at parity. (4) The dead-zone-free kernel
**does not pay** (0.882 against 0.957 with a prior): it trades the dead zone for resolution. (5) A **content-keyed**
family now exists (`assoc`), where the VSA binding is demonstrably load-bearing (the additive ablation stays at
chance) and the positional advantage cannot be assumed. (6) **On a content-keyed family with no positional structure, both associative memories solve the task (1.000
with 10 of 10 seeds) while GRU and LRU sit at chance (0.249 / 0.281) at equal state, and the no-bind ablation is at
chance — the binding is load-bearing; under distractors the outer-product form holds better (0.794 against 0.589), so
neither structure dominates.** (7) *Not done:* the spatial recall task was not built, and the RL track is in progress
— so nothing here yet licenses the RL port (handoff 11 section 3.2).

**Which techniques to use, on the evidence.** For a *positional* memory: keep the recipe prior, and add
**input-dependent gates** (1.000, all seeds at 0.95, recipe speed) and/or **multi-timescale banks** (2.0x fewer tokens
to 0.90); if the initialisation is wrong, **K spread read heads** restore recipe-level performance; if a task-dependent
prior must be avoided altogether, **tied relative addressing** does it at parity on delay and ~3 points of accuracy on
index, and **gates + tied** gives the best prior-free index cell (0.977). Do not add: the cauchy kernel, learned width,
or the delta write (all neutral-to-negative here). For a *content-keyed* memory: use binding (the ablation is at
chance) and prefer the outer-product form if distractors are expected.

---

## 1. Code changes (all new knobs default OFF, so every earlier number still reproduces)

The check that matters: a fixed-seed forward pass of the plain `fwp` and the rung-1 SSP arm was hashed before any
edit and re-hashed after each change. It stayed **bit-for-bit identical** at every step (index and delay, three arms
each), and `test_learned` (12/12), `test_harness` (6/6), `test_fixes` (2/2), `test_pointer` (14/14), `test_recipes`
(10/10) all kept their counts.

| change | file | what it adds |
|---|---|---|
| `FWPArm(rule="delta")` | `arms_learned.py` | the practitioner's update: `M = lam*M + beta*(v - M k̂) k̂ᵀ` with a learned write strength, instead of the plain `M += v kᵀ` |
| `FWPArm(tied=, read_abs=, q_init_offset=, q_copy_init=)` | `arms_learned.py` | one shared encoder for write key and read query; a read at the wanted item's own time `t - lag` (the historical form keyed the query on the lag alone, which cannot match a key keyed on `t`); an initial read offset, for the offset grid |
| `fwp_delta` arm | `arms_learned.py`, `harness.py` | the registered arm = delta rule + tied encoder + absolute read time, at the same `d*d` state |
| `omega_kind="cauchy"` | `ssp_ops.py` | a dead-zone-free kernel: a Cauchy spectrum gives `exp(-\vert d\vert / ls)`, strictly positive and monotone, where `sinc(pi d)` is flat/zero at integer offsets |
| `addr_mode="tied"`, `read_residual` | `arms_learned.py` | the read scalar comes from the *same* weights as the write phase, so a key and its query share an address by construction (no read origin to discover); the residual is a zero-init correction. On delay it reads at the wanted item's own time, i.e. **relative addressing through the shared function** |
| `read_heads=K` | `arms_learned.py` | K read heads with offsets spread by half an index (`[-1,-0.5,0,0.5,1]` at K=5), scores concatenated: redundancy instead of a prior |
| `write_rule="delta"`, `beta_init` | `arms_learned.py` | the SSP bundle corrects the slot (`M += beta*(code(v) - M*conj(pos_w))*pos_w`) instead of piling onto it |
| `read_feat_scale`, `read_feat_shift` | `arms_learned.py` | a *deployment* stress: the query arrives in different units or with a different origin than the stream's own time feature |
| `tasks_d.py` (`assoc`) and `arms_assoc.py` | new files | the **content-keyed** family: `n_pairs` key→value pairs in random order, queries name a key, nothing is addressed by position; `AssocSSPArm` (fixed VHRR codes, `bind`) with a `bind=False` additive ablation and a `delta` write, plus `AssocFWPArm` (learned encoders, delta rule) as the *matched* baseline at exactly the same carried state |

New test modules: `test_fwp_delta.py` (5 checks), `test_candidates.py` (5 checks). Both pass.

**Local mechanism checks [measured, 1 seed, local machine, not results]:**

- `test_fwp_delta`: with the encoders pinned to one basis vector and no decay, the plain arm's read after an
  overwrite is exactly `va + vb` (error 0.00e+00) and the delta arm's is exactly `vb` (1.87e-07) — the rule
  difference, isolated.
- `test_candidates`: the cauchy kernel tracks `exp(-d)` to within Monte-Carlo error (4096 dims) while the historical
  kernel sits at 0.006 and 0.012 at `d = 1, 2` — the dead zone, measured at high dimension.
- The tied arm's key and query get **identical** addresses (max difference 0.00e+00).
- Every candidate trains on index in 150 steps with the recipe prior (baseline 0.992, cauchy 0.947, tied 1.000,
  tied+residual 1.000, K=5 0.996, delta-write 0.986). **Exception [measured]: the delta write at full strength from
  step 0 does not train (0.144 -> 0.085 in 150 steps); initialising its gate near the plain rule (`sigmoid(-4)`)
  does (0.142 -> 0.986).** So the failure is the starting point, not the mechanism, and every delta-write cell in
  the pod grids uses the near-plain gate.

---

## 2. Step 0a/0c — the practitioner's baseline, and is the dead zone SSP-specific? (`jobs/p7_grid_fwp.json`, pods)

Protocol: index T=32, budget 256, `--select aulc --min-conv 0.9 --search-seeds 1 --seeds 10`, 1500 steps, lr 0.02,
`lam_init` 0.99, `hidden` 32, both arms tied with an identity read head plus a drawn offset. 10 hold-out seeds.

**The offset grid (conv / seeds >= 0.9):**

| read offset | plain `fwp` | `fwp_delta` (delta + tied) |
|---|---|---|
| 0 | 0.726 / 0 | 0.744 / 0 |
| 0.5 | 0.644 / 0 | 0.678 / 0 |
| 1 | 0.397 / 0 | 0.321 / 0 |
| 3 | 0.329 / 0 | 0.439 / 0 |

**Trainability gates (index T=16, offset 0):** plain `fwp` 0.924, 10/10 at >= 0.9; `fwp_delta` 0.945, 10/10 at
>= 0.9 (4/10 at 0.95). Both **pass** the gate, so both are usable as baselines.

**Delay (lags 1-4, T=16, reading at the item's own time):** plain `fwp` 0.993, 10/10 at 0.95, t90 21.6k tokens;
`fwp_delta` 0.998, 10/10, t90 21.6k. With a one-index read offset: 0.934 and 0.913.

Three readings, all of them corrections to the earlier record:

1. **The dead zone is not SSP-specific.** A plain outer-product memory and its delta-rule form both degrade
   monotonically with a read-address offset (0.726/0.744 at offset 0 to 0.397/0.321 at offset 1) and neither reaches
   0.9 on index at this budget. Address discovery is a property of *any* key-addressed memory, which is what
   handoff 11 step 0c asked.
2. **The historical `fwp` delay number was an artifact of the read address.** Keyed on the lag alone its query can
   never match a key keyed on `t`; given the absolute read time it is at 0.993-0.998 with 10/10 seeds, i.e. *at least
   as good as the SSP recipe* (1.000 / 0.997, 10/10). `delay` is therefore confirmed as *substitutable* (docs/11
   1.1), and more strongly than before.
3. **On index at equal carried state the two fast-weight arms are far behind the SSP recipe** (0.73 vs 0.956 at
   offset 0, and neither gets a single seed to 0.9 where the recipe gets 9/10 to 0.95). This is the first place in
   the project where the SSP arm looks better than a *tuned, gated, tied* generic memory at the same budget — and
   note that these `fwp` arms were given the *better* (tied, absolute) address convention. It is a candidate win,
   not a win: the gap has to survive the specificity cells in section 3 (in particular the same arm with **no prior
   at all**).

---

## 3. Step 1 — the four candidates on index and delay (`jobs/p8_candidates.json`, pods)

Protocol: index T=32 (or as stated), budget 256 unless stated, 10 seeds, `--select aulc --min-conv 0.9
--search-seeds 1`, `length_scale` 1.0, `ls_mode` fixed, lr 0.02, `lam_init` 0.99, `temp` 0.1. "recipe" = the RUN_011
slope-only prior (`init=affine, eps=0.03, offset 0`); "scrambled" = **no prior**: the write phase *and* the read head
start random.

| cell (index T=32 unless stated) | conv | min | seeds >= 0.9 | seeds >= 0.95 | t90 tokens | held-out | constants |
|---|---|---|---|---|---|---|---|
| **recipe** (`eps=0.03`, offset 0) — the incumbent | 0.957 | 0.934 | **10/10** | **9/10** | **52.0k** | 0.457 | 1 tuned (`eps`) + offset 0 + ramp + fixed freq |
| scrambled (no prior) | 0.622 | 0.449 | 1/10 | 0/10 | 922k | 0.246 | 0 |
| **tied + scrambled (no prior at all)** | **0.917** | 0.892 | **9/10** | 0/10 | 970k | 0.438 | **0** |
| tied + scrambled + residual | 0.914 | 0.893 | 8/10 | 0/10 | 921k | 0.419 | 1 (residual, zero-init) |
| tied + recipe prior | **0.960** | 0.940 | **10/10** | 8/10 | 51.4k | 0.537 | 1 |
| cauchy kernel + scrambled | 0.732 | 0.619 | 0/10 | 0/10 | -- | 0.282 | 0 |
| cauchy kernel + recipe prior | 0.882 | 0.796 | 4/10 | 0/10 | 205k | 0.399 | 1 |
| K=1, offset 1 (the known dead zone) | 0.663 | 0.266 | 5/10 | 4/10 | 76k | 0.311 | 1 |
| **K=5 heads, offset 1** | **0.962** | 0.950 | **10/10** | **9/10** | **51.3k** | 0.518 | 1 (K) |
| K=5 heads, offset 3 | 0.496 | 0.403 | 0/10 | 0/10 | -- | 0.238 | 1 (K) |
| delta write + recipe prior | 0.957 | 0.936 | **10/10** | **9/10** | 52.0k | 0.455 | 2 |
| delta write, index T=64 at 512 floats | 0.951 | 0.937 | 10/10 | 5/10 | 153.8k | 0.518 | 2 |
| recipe, index T=64 at 512 floats (control) | 0.952 | 0.941 | 10/10 | 6/10 | 152.7k | 0.522 | 1 |
| delay 1-4, recipe | **1.000** | 1.000 | **10/10** | **10/10** | **21.6k** | 0.776 | 1 |
| delay 1-4, tied + scrambled | 0.352 | 0.273 | 0/10 | 0/10 | -- | 0.148 | 0 |

**Which candidate actually wins (index T=32, 10 seeds each):**

| configuration | conv | >= 0.9 | >= 0.95 | t90 tokens | task-dependent constants |
|---|---|---|---|---|---|
| recipe (incumbent) | 0.957 | 10/10 | **9/10** | **52.0k** | 4 (offset 0, `eps`, frequencies, ramp) |
| tied + prior | **0.960** | 10/10 | 8/10 | 51.4k | 1 (`eps`) + tying |
| **K=5 heads, offset 1** | **0.962** | **10/10** | **9/10** | **51.3k** | 1 (K) |
| tied, no prior | 0.917 (0.930 at 3k steps) | 9-10/10 | 0/10 | 970k | **0** |
| cauchy kernel + prior | 0.882 | 4/10 | 0/10 | 205k | 1 |
| delta write | 0.957 | 10/10 | 9/10 | 52.0k | 2 |

Four readings:

1. **Tied addressing is the round's substantive finding.** With **no prior at all** — the write phase and the read
   address both start random — the tied arm reaches 0.917 with 9 of 10 seeds at 0.9, where the *same arm with a
   separate read head* scores 0.622 with 1 of 10. The mechanism is exactly the one the design predicted: when the
   read address is produced by the same weights as the write address, there is no origin to discover. On the
   bitter-lesson axis this is the first configuration in the project that removes the hand-set prior *entirely*
   (`eps`, the offset and the range constant all disappear).
2. **...and it costs.** 0.917 against 0.957 with the prior, 0/10 seeds at 0.95 against 9/10, and 970k tokens to 0.90
   against 52k — about 19x. Tied-and-prior is the best cell on both axes (0.960, 10/10 at 0.9, 8/10 at 0.95,
   51.4k tokens), so the prior and the tying are complementary rather than alternatives. **Nothing here shows the
   prior is removable at equal quality; it shows the prior becomes optional and the price is speed and tail.**
3. **The dead-zone-free kernel does not pay.** Cauchy with a prior is 0.882 (4/10) against the recipe's 0.957 (9/10),
   and without a prior 0.732 against 0.622. Removing the dead zone buys a better *start* and loses resolution exactly
   as the risk note said: a wide, strictly-positive kernel cannot separate items a half-index apart. The dead zone
   was the *discoverability* obstacle, not the only obstacle.
4. **Delay is where tying does not transfer in its first form.** 0.352 for tied+scrambled against 1.000 for the
   recipe, because the delay read address is `t - lag` — an address no write ever used — so a shared function has no
   self-consistency to exploit. The fix is the relative form (read at the wanted item's own time through the shared
   function); it is measured in section 5.
5. **K heads buy back the prior with compute, and this is the cleanest result of the round.** Given a read offset of
   exactly one index — a configuration that scores 0.663 with 5 of 10 seeds when there is a single head — **five spread
   heads reach 0.962 with 10 of 10 seeds at 0.9 and 9 of 10 at 0.95 in 51.3k tokens**: statistically the same as the
   tuned recipe (0.957, 10/10, 9/10, 52.0k) at the same speed, with a different constant (`K` instead of `eps`). The
   boundary is exactly where the design says it is: at offset 3, outside the half-index spread of the five heads, the
   arm falls to 0.496 with 0 of 10. So the dead-zone obstacle is removable by *redundancy* (K independent read
   addresses covering the neighbourhood, one of which is in the healthy region) rather than by a prior — and unlike
   the tied design it costs nothing in accuracy or speed, only compute and a `K`.
6. **The delta write does not fix the capacity limit.** At index T=64 with a 512-float budget, where the plain bundle
   is known to sit at the `D/4` ceiling (0.952, 6/10 at 0.95), the delta rule gives 0.951 and 5/10 — the same place.
   The T=64 failure is therefore *not* the accumulation-versus-correction axis, and the honest record is that the
   capacity explanation stands unexplained by this candidate: either the limit is the address range (62 against
   `D/4 = 64`) as section 1 of RUN_011 argued, or the correction is not reaching the right slots. On index at T=32 the
   delta write is also exactly neutral (0.957, same t90), so it has no benefit to show anywhere yet.

**The specificity comparison this gives (the strongest in the project so far).**

| arm | index T=32 conv | seeds >= 0.9 | delay 1-4 conv | seeds >= 0.95 |
|---|---|---|---|---|
| plain `fwp` (p7, tied, offset 0) | 0.726 | 0/10 | 0.993 | 10/10 |
| `fwp_delta` (p7, tied, offset 0) | 0.744 | 0/10 | 0.998 | 10/10 |
| SSP recipe | 0.957 | 10/10 | 1.000 | 10/10 |

At equal carried state (256 floats), equal protocol and both gated, the SSP arm is **far ahead on index** and level on
delay. Per docs/11 4.5 this is a *candidate* advantage with a stated margin (>= 5 points absolute and well beyond 2
pooled standard errors), on the metric "converged accuracy at a fixed state budget". It is not yet a *deployed* win:
the same comparison has to be repeated with the offset grid so that the fast-weight arms are not handicapped by a
worse prior, and the index gap has to survive the content family (section 4).

---

## 4. Step 1 — the content-keyed family (`jobs/p9_assoc.json`, pods)

**Machine class: these are LOCAL runs** (the pod run of this family was killed after 50 minutes at 0 rows — the cause
was a real bug, see below — so the numbers here are local and must not be tabulated with the pod tables).
Protocol: `assoc` at `n_pairs=4`, T=16, budget 256 floats, `--configs 2 --search-seeds 1 --seeds 10 --steps 800`
(lr and `lam_init` searched by the harness's own sampler, so every arm gets the same tuning budget), `--workers 8`.
Rows: 84 (14 search + 70 final). Splits: `id` (same regime), `extrap_lag` = **4 distractor pairs added** (keys and
values from the same alphabets, never queried), `rob_pairs` = half the query keys corrupted.

| arm | conv | min | seeds >= 0.9 | seeds >= 0.95 | extrap (distractors) | t90 tokens |
|---|---|---|---|---|---|---|
| `assoc_ssp` (VSA bind, fixed codes) | **1.000** | 1.000 | **10/10** | **10/10** | 0.589 | 153.6k |
| `assoc_ssp` + learned codes | **1.000** | 1.000 | **10/10** | **10/10** | **0.666** | 153.6k |
| `assoc_ssp` **no bind** (ablation) | 0.248 | 0.227 | 0/10 | 0/10 | 0.248 | -- |
| `assoc_fwp` (outer product, delta write) | **1.000** | 1.000 | **10/10** | **10/10** | **0.794** | 153.6k |
| `assoc_fwp`, no delta rule | **1.000** | 1.000 | **10/10** | **10/10** | **0.794** | 153.6k |
| GRU (256 floats) | 0.249 | 0.243 | 0/10 | 0/10 | 0.250 | never |
| LRU (256 floats) | 0.281 | 0.246 | 0/10 | 0/10 | 0.252 | never |

Four readings:

1. **On a task with no positional structure at all, both associative memories solve it perfectly and the recurrent
   baselines are at chance.** 1.000 with 10 of 10 seeds against GRU 0.249 and LRU 0.281, at equal carried state (256
   floats) and equal tuning budget, chance = 0.250. This is the strongest capability separation the project has
   produced — stronger than the index gap, because here the baselines do not merely do worse, they fail to learn the
   task at these budgets, and there is no positional shortcut for any arm to exploit.
2. **The binding is load-bearing.** The additive ablation (same state, same readout, same tuning, but the value code
   is bundled without the key) sits at 0.248, i.e. chance. So the algebra is what carries the association, not the
   codebook's presence or the readout's capacity.
3. **Under distractors the fast-weight form is better.** With four irrelevant pairs added to the stream, the outer
   product holds at 0.794 while the VSA bundle falls to 0.589 (0.666 with learned codes). So on this axis the
   content family favours the fast-weight memory — the opposite of the index result — and the honest reading is that
   *neither* structure dominates: the SSP form wins where addresses are positional (sections 2-3) and loses where
   they are content-keyed, at the same state.
4. **The corruption column is vacuous by construction and must not be cited.** Corrupting the *query key* replaces it
   with another key, whose value is still a correct answer to the corrupted query, so every arm scores ~1.000
   (0.251 for the ablation-only arms, which is chance for the same reason). This is the same trap as RUN_007's
   pointer cue split; a real corruption would blank the query. Recorded, not fixed.

**The bug this family exposed (and it cost the pod run).** The first version's extrapolation split doubled `n_pairs`,
which also doubles the *value alphabet*: the arms' codebooks and encoders are sized at construction, so the wider
split crashed (`mat1 and mat2 shapes cannot be multiplied (64x16 and 8x16)`, then
`index 5 is out of bounds for dimension 0 with size 4`). Splits for this family must keep the alphabet fixed — the
distractor form above does, and the `extrap_lag` name is reused only so the report columns line up.

---

## 5. Step 1 — tied relative addressing, with no prior (`jobs/p10_tied.json`, pods)

The delay result above (0.352) was the *first* tied form, which applies the shared address function to the lag. The
natural form evaluates it at the **wanted item's own time**, `t - lag`, i.e. relative addressing: the query names an
address that the write function also produces. Protocol: no prior at all (`init="scrambled"`, `addr_mode="tied"`,
`ls_mode` fixed, lr 0.02, 10 seeds, `--select aulc --min-conv 0.0 --search-seeds 1`).

| cell | conv | min | seeds >= 0.9 | seeds >= 0.95 | t90 tokens | the recipe there |
|---|---|---|---|---|---|---|
| **delay 1-4, tied + scrambled (no prior)** | **0.999** | 0.998 | **10/10** | **10/10** | 43.2k | 1.000, 10/10, 21.6k |
| **delay 1-8, tied + scrambled (no prior)** | **0.994** | 0.989 | **10/10** | **10/10** | 46.0k | 0.997, 10/10, 18.4k |
| delay 1-4, tied + scrambled + residual | 0.999 | 0.999 | 10/10 | 10/10 | 43.2k | -- |
| index T=32, tied + scrambled (no prior), 3000 steps | 0.930 | 0.914 | **10/10** | 1/10 | 970k | 0.957, 10/10, 9/10 >= 0.95, 52.0k |

**This is the round's best result.** On `delay` the prior is **fully removable at parity**: 0.999 / 0.994 with 10 of
10 seeds at >= 0.95 and no task-dependent constant anywhere, against the recipe's 1.000 / 0.997 with the same 10/10 —
at 2x the tokens to reach 0.90 (43.2k against 21.6k) but with **zero** constants rather than one tuned `eps`, an exact
zero offset, a fixed frequency set and a ramp prior. The untied arm under the same random start sits at 0.27-0.32.

On `index` the prior is *nearly* removable: doubling the budget lifts tied+scrambled to **0.930 with 10 of 10 seeds
at 0.9** (from 0.917 / 9-of-10 at 1500 steps), but it still does not reach the recipe's tail (1 of 10 seeds at 0.95
against 9 of 10) and costs ~19x the tokens to 0.90. So the honest summary is: **tying removes the hand-set prior
entirely on delay (parity) and on index at a cost of about 3 accuracy points, a much slower tail, and ~19x the
tokens to 0.90** — with the residual variant neither helping nor hurting, and the tied+prior combination remaining
the best index cell (0.960).

The mechanism explains why the two tasks differ, and it is the same statement as the dead-zone result read
backwards: tying works when the read address is *an address the write function produces* (the item's own time, both
tasks in the relative form), and it fails when the read needs an address no write ever used (the first delay form).
That is precisely the "relative addressing through the shared function" claim in docs/11 14.3 axis 2 item 1, now
measured on both task families.

---

## 5b. The two-axis candidate scoreboard (every handoff-11 candidate measured)

Protocol for every cell: index T=32, budget 256 (equal carried state), 10 seeds, 1500 steps (3000 where stated),
`--select aulc --min-conv 0.0 --search-seeds 1`, lr 0.02, `lam_init` 0.99, `temp` 0.1, `length_scale` 1.0,
`ls_mode` fixed, started from the recipe prior unless the row says otherwise. Reference: the recipe is 0.957 with
10/10 at 0.9, 9/10 at 0.95 and 52.0k tokens.

**Axis 1 — make parts learnable**

| candidate | cell | conv | >= 0.9 | >= 0.95 | t90 tokens | verdict |
|---|---|---|---|---|---|---|
| learned width (`ls_mode="learned"`) | `p11-idx32-lslearned` | 0.947 | 10/10 | 5/10 | 77.3k | **no** — worse and 1.5x slower |
| learned frequencies + codes (rung 2) | `p11-idx32-rung2` | **0.999** | 10/10 | **10/10** | 51.4k | **neutral-to-positive**: recipe speed, better accuracy and tail |
| **input-dependent gates (alone)** | `p13-idx32-gate-in-clean` | **1.000** | **10/10** | **10/10** | **51.3k** | **yes — perfect seeds at 0.95 at the recipe's speed** (the first p11 cell read 0.131 because it launched on stale code; re-run clean) |
| interface pretraining | `p11-idx32-pretrain` | **not measured this round**: the cell produced 0 rows and was killed on cost. RUN_011 section 5 measured the same idea directly (shape-randomised pretraining scales with its budget — N=64 pool 0.269 -> 0.845 over 0-500 steps — but does **not** transfer, 0.40-0.46 on the length axis, and the in-loop probe auxiliary is a no-op at 0.274/0.275 against 0.269), so the answer stands as *more training data, not a learned initialisation* | | | | |
| **K heads (search instead of a prior)** | `p8-idx32-ssp-k5-o1` | **0.962** | **10/10** | **9/10** | **51.3k** | **yes — the strongest axis-1 result**: from a one-index offset error it matches the recipe exactly |
| the same at offset 3 (the boundary) | `p8-idx32-ssp-k5-o3` | 0.496 | 0/10 | 0/10 | -- | confirms the boundary is the half-index spread, as predicted |

**Axis 2 — change the architecture**

| candidate | cell | conv | >= 0.9 | >= 0.95 | t90 tokens | verdict |
|---|---|---|---|---|---|---|
| tied encoder + relative addressing | `p10-idx32-tied-scr-3k` | 0.930 | 10/10 | 1/10 | 970k | **removes the prior; costs speed and tail** (section 5) |
| dead-zone-free kernel (cauchy) | `p8-idx32-ssp-cauchy-rec` | 0.882 | 4/10 | 0/10 | 205k | **no** — trades the dead zone for resolution |
| delta-rule write | `p8-idx32-ssp-dw` | 0.957 | 10/10 | 9/10 | 52.0k | **neutral** at T=32, and no gain at the T=64 capacity limit (0.951 vs 0.952) |
| multi-timescale banks (2) | `p12-idx32-banks2` | 0.957 | 10/10 | 8/10 | 51.1k | neutral |
| **multi-timescale banks (4)** | `p12-idx32-banks4` | **0.962** | **10/10** | **9/10** | **25.8k** | **yes — 2.0x fewer tokens to 0.90 at equal accuracy and equal state** |
| **gates + banks (2)** | `p12-idx32-gate-banks2` | **1.000** | **10/10** | **10/10** | **25.8k** | **the best cell of the round**: every seed at 0.95, at 2.0x the recipe's speed |
| soft/interpolating readout (±0.5 heads) | `p11-idx32-interp3` | 0.961 | 10/10 | 8/10 | 51.3k | comparable; a weak form of the K-head rescue |
| iterated reads | RUN_011 `cue_hop` | 1.000 on held-out hops 4-8, 10/10 | | | | unchanged this round |
| content-plus-position keys | *not built* (the `assoc` family, section 4, is content-only) | | | | | |

**The prior-free path, on both families**

| cell | conv | >= 0.9 | >= 0.95 | t90 tokens | constants |
|---|---|---|---|---|---|
| index T=32, tied + gate (no prior) | 0.977 | 9/10 | 9/10 | 231k | 0 |
| index T=32 / 48 / 64, tied (no prior) | 0.930 / 0.913 / 0.817 | 10 / 9 / 0 | 1 / 0 / 0 | 970k / 2.45M / -- | 0 |
| delay 1-4, tied (no prior) | **0.999** | 10/10 | **10/10** | 43.2k | 0 |
| delay 1-4, tied + banks (no prior) | **0.999** | 10/10 | **10/10** | 43.2k | 1 |
| delay 1-4, tied + gate (no prior) | 0.488 | 0/10 | 0/10 | -- | 0 |

**What the scoreboard says.** Three changes are real improvements rather than neutral knobs: **input-dependent gates**
(1.000 with every seed at 0.95 at the recipe's speed, where the recipe itself has 9 of 10), **multi-timescale banks**
(2.0x fewer tokens to 0.90 at equal accuracy and state, with no extra task-dependent constant), and **K heads**, which
turn a known-failing initialisation into recipe-level performance at recipe speed. Two mechanisms remove the hand-set
prior without losing much: **tied addressing** (parity on delay, a 3-point cost on index) and **gates + tied** (0.977
on index with 9/10 at 0.95, at 4.4x the tokens). Three candidates are neutral (delta write, banks-2, interpolating
readout) and three are negative (cauchy kernel, learned width, and gates *alone* on delay without a prior — 0.488
against 0.999). The axes are asymmetric: on this family the architecture changes carry the wins, while the
"make it learnable" changes are mostly neutral except K heads and rung 2. **The tied design degrades with length**
(0.930 / 0.913 / 0.817 at T=32/48/64, no seed at 0.9 by T=64), so its cost grows with the address range: it removes the
prior at a price that scales.

---

## 6. The RL track (POPGym, delegated; local machine, no pods)

Built to handoff 11 section 3.2 with POPGym's own components: its environments, its PPO configuration and loop through
`PPOConfig` with its own `ray_models`, its evaluation. The SSP arm is an RLlib `ModelV2` wrapping
`rl_memory.SSPMemory`, with the same 256-float carried state as the GRU and the address produced by a learned head over
a feature of the current observation plus time. `RepeatPreviousEasy`: random return −0.512, optimal +1.000, so the
reported thresholds are +0.244 (50%) and +0.849 (90%). Three seeds per arm, identical PPO settings, ~1.05 h per arm to
1M steps on the shared machine.

| arm | final (1M, last 10%) | steps to +0.244 | steps to +0.849 | curve area | per-step forward |
|---|---|---|---|---|---|
| POPGym GRU | **+0.993 ± 0.005** | **3/3, median 361k** | **3/3, median 557k** | +0.436 | 0.64 ms |
| POPGym FWP (no delta rule) | −0.396 ± 0.073 | 0/3 | 0/3 | −0.455 | -- |
| SSP, aux off | −0.163 ± 0.117 | 0/3 | 0/3 | −0.338 | 1.25 ms |
| SSP, aux on | −0.268 ± 0.107 | 0/3 | 0/3 | −0.406 | 1.25 ms |

Extended (resumed from checkpoints because SSP was still climbing): at 3M steps SSP is −0.057 ± 0.113 (aux off) and
−0.097 ± 0.108 (aux on), still below the +0.244 threshold; FWP reaches −0.262 ± 0.048 at 2M. Throughput solo: GRU
510 steps/s, FWP 805, SSP 645 (25.8 min per 1M).

**The diagnosis is the project's own dead zone, found again in RL.** The learned read-offset bias stayed at
**|bias| ≤ 0.011 in all six SSP runs**: the memory never learned to read four steps back, which is verbatim the
supervised failure mode (RUN_011 section 3: an offset that never moves). Two controls say the *model* is capable and
the *search* is what fails:

- hand-setting the read offset to the task's lag (−4) reaches **+0.244 at 230k steps and +0.710 at 328k**, one seed
  (a diagnostic, not a claim);
- zeroing the memory state before every step drops every arm to about −0.50, so the return does come from memory.

**Against the success rule:** parity fails (−0.057 against +0.993) and no listed metric is won — per-step cost is 2x
the GRU's, state size is equal by construction, and length extrapolation and robustness were not run. So the RL port is
**not** licensed on this evidence, and the gate in handoff 11 section 3.2 did its job.

**What this implies for the next run, and it is specific:** every mechanism that removed the prior or repaired the
initialisation in the supervised study was measured *inside* the supervised arm. The RL model here uses a plain
learned address head, i.e. exactly the configuration that stalls — so the experiment that follows is not "more RL
tuning" but "put the tied relative addressing (and, as a fallback, a hand-set bias only for the diagnostic) into the RL
arm and re-run the same pilot". The evidence says that is where the failure lives.

**One deviation that supersedes an older number:** Ray 2.59's `PPOConfig.update_from_dict` silently drops POPGym's old
`sgd_minibatch_size` key, so POPGym's own reference runs (and RUN_011's 0.683-at-632k GRU number) effectively used a
128-step minibatch. The pilot sets `minibatch_size = 8192` (POPGym's `8 × bptt_cutoff`); `run_ref.py` is untouched, so
the old number stands with its caveat.

---

## 7. Do not do (added by this round)

- **Do not call `fwp` "the incumbent" or read the old tie as a result.** The arm every earlier number used is *plain*
  linear attention with untied heads and a lag-only delay query; the delta-rule, tied form behaves differently on both
  tasks (section 2). A claim of the form "SSP ties / beats the fast-weight family" needs `fwp_delta`, not `fwp`.
- **Do not start the delta write at full strength.** Measured: it does not train on index from step 0 (section 1).
  Use `beta_init=-4.0` (near-plain) and let the gate grow.
- **Do not compare a prior-using SSP arm to a prior-free baseline.** The SSP arm's default `init="affine"` hands it a
  ramp prior that the `fwp` arms never get; with both prior-free, the ordering changes (section 3).
- **Do not treat `read_heads=K` as free.** K multiplies the readout input and the parameters; the offsets' spread is
  tied to the index scale (half-index steps), so K rescues an offset only inside its own spread.
- **Do not read the local smoke numbers as results.** They are one seed on one machine, and locality has changed
  conclusions before (RUN_011 section 0b).
- **Do not use the cauchy kernel without checking resolution.** Its cost is exactly what the risk said: a wide,
  strictly-positive kernel trades the dead zone for resolution (section 3).

---

## 8. Plan state at hand-off (appended 2026-10-13)

This round is finished; the next round has not started. What exists and what does not:

- **Done:** every candidate of `docs/11` section 14.3 measured on pods (`jobs/p7`-`p13`), the content family measured
  locally, and the RL pilot plus a tied-addressing extension left **partially** run by the delegated agent (results in
  `results/rl_tied_*`, **no `REPORT_TIED.md`** -- unverified until its tests are re-run).
- **The RL decomposition** (unverified partial numbers, `results/rl_tied_RepeatPreviousEasy_summary.json`): untied
  -0.163, **tied -0.133**, tied+aux -0.202, untied-with-scaled-features -0.116, and **tied with the read offset set to
  the task's lag +0.832 with 3/3 seeds at the 50% threshold**, against GRU +0.993. Tying removes the *origin* problem
  but not the *lag-selection* problem, which no supervised task tested because there the lag is an input feature. The
  candidate that follows is a **bank of read offsets 1..L with the output selecting** -- the same mechanism as the
  supervised K-head win.
- **The default-path guard is now in the repo:** `test_default_path_hash.py` (six pre-edit hashes of the default `fwp`
  and rung-1 SSP paths). The `/tmp` version was lost in a wipe; its expected values must never be edited to make it
  pass.
- **Not started from the plan:** the equal-aid comparison for `fwp_delta` (`gate`/`banks` on `FWPArm`), the distractor
  task, the crossed-combination grid, the lag-bank RL arm, the spatial recall task, content-plus-position keys, the
  32-node pointer cell, and the regression for this round's new modules. Each step, with files, pass rules and traps,
  is the "CURRENT PLAN" block at the top of `handoffs/11`.
