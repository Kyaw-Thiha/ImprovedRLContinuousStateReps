# SSP algebra controller: what is substitutable, what is not, three variants, and the tests

Written 2026-10-10. Companion to [`10-vsa-memory-and-composition-direction.md`](10-vsa-memory-and-composition-direction.md),
[`08-ssp-vs-tile-coding-theory-and-hypotheses.md`](08-ssp-vs-tile-coding-theory-and-hypotheses.md) and the run records
`runs/RUN_005_hypotesting.md` (encoder-only battery) and `runs/RUN_006_popgym_first_pass.md` (POPGym first pass).
**Status (updated 2026-10-11):** sections 3-5 were executed once, in a *hand-wired* form ("rung 0", section 3.4), in
`runs/RUN_007_ssp_algebra_controller.md`, and this document is updated to the plan that follows from that run. The
handoff for the next agent is `handoffs/09-ssp-learned-structure-sample-efficiency.md`. Confidence labels say which
statements are measured, taken from earlier docs, or my inference.

**Update (2026-10-11, later):** the learned rungs were executed in `runs/RUN_008_ssp_learned_structure.md` (decision log in
section 8) and the recipe diagnostics in `runs/RUN_009_recipe_diagnostics.md` (decision log in section 9). The lessons for
this plan: plain training does not reliably discover the structure; a curriculum fitted to the failure mechanism does, but its
constants do not transfer across address ranges. Section 3.5 states the "one recipe per problem family" idea (with its rule
requirement), 4.7 how a recipe is validated, and 4.8 what must hold before the learned-arm robustness study, RL or larger
models.

**Update (2026-10-11, RUN_010):** `runs/RUN_010_recipe_and_controls.md` ran handoff 09 steps 1-4 (decision log in section 10). Net: the
range-scaling rule makes the width-schedule recipe work on 4 of 5 unseen variants; the learned arm is faster than the GRU and the
transformer but **not by the 2x the thesis rule asks for, and slower than `fwp`**; interface pretraining fixes the N=64 collapse but
**does not transfer**; the pointer pass stays provisional. The next work (schedule-free discoverability, amortised pretraining, pointer
scaling, RL port preparation) is in section 3.6 and **`handoffs/10-ssp-discoverability-and-rl-port-prep.md`: start there.**

**Update (2026-10-12, RUN_011 and the redirection after it):** handoff 10 was executed (`runs/RUN_011_discoverability_and_halting.md`; sections 11-13 below). The
headline: the read-address *offset* is the discovery constraint; a mechanism-derived recipe passes every variant inside the memory's capacity; the pointer
iterates to 8 hops from one hop of training; the RL port stays gated; and **no matched-baseline win exists yet**. The project's goal has since been restated
(section 14): not "characterise SSP" but **find a deployable, learnable, simple SSP memory module for RL agents**, judged by the bitter lesson, by real-world
utility, and by sample efficiency and robustness against generic memories. Two axes of change (learnable parts; architecture) and the experiment design are in
section 14; the handoff is **`handoffs/11-ssp-memory-module-deployable.md`: start there.**

Motivation and thesis: SSP gives the learner *structure it can interact with* (bind, unbind, shift, bundle), so that a
learner can **discover and use** that structure and therefore learn from fewer samples and shorter episodes, stay
robust (cue corruption, longer streams, distractors), and generalise (longer lags, more hops, more entities), **while
remaining competitive with generic methods at convergence**. The claim is deliberately *not* that SSP beats a
transformer or GRU given unlimited samples and state: those can model nearly anything. The claim is about the
cost of getting there (samples, episode length, state budget) and the failure modes (robustness, extrapolation).
RUN_006 showed that its pipeline did not test this (bind/unbind hard-wired, position codes a random table, no SSP
proper). RUN_007 then showed the same limitation in a cleaner setting: every SSP arm was hand-wired, so it tested
what the algebra can do when *operated correctly*, not whether a learner can *learn to operate it*.

---

## 1. What SSP offers: substitutable and not

Each property is sorted by whether an equivalent exists. A claim can only rest on the second list.

## 1.1 Substitutable (cannot carry a claim)

| Property | Substitute | Source |
|---|---|---|
| Smoothness / stationary kernel | random Fourier features | doc 10 §10; RUN_005 T1 (kappa floor and half-width measured) |
| Capacity | learned encoders | doc 10 §10 |
| Variable item count in a fixed vector | Deep Sets, attention | doc 10 §10 |
| Pure temporal delay, `pos(t) = exp(i*Omega*t)` with a rotating memory | diagonal linear recurrences with fixed unit-modulus eigenvalues: S4D, LRU; the Legendre Memory Unit (LMU) is a continuous-time delay memory from the same research group | my inference from the math; doc 10 §13.1 lists S4D and LMU among memories that lost to a plain GRU in POPGym (not re-verified) |
| Key-value association | fast-weight programmers and linear attention (matrix memory, outer products) | doc 10 §13.1 (same caveat) |
| Continuous learned address | soft attention with a Gaussian positional kernel centred at `t - delta` | my inference; to be tested by the control in section 4 |

Consequence: **time-as-SSP is a sanity arm, not the thesis.** The same applies to any claim that rests on
smoothness, capacity or recency.

## 1.2 What I believe is not substitutable

1. **One group action on the whole memory.** `M <- M * phi(Delta)` shifts every stored item's address at once, in
   O(D) and independent of how many items are stored. Exact to ~1e-15 (RUN_005 T0, measured). For a spatial map this
   is one multiplication to re-centre the world when the agent moves.
2. **Chained algebraic queries that do not compound.** Unbind, clean up, unbind again. Depth 1 to 3 cost nothing
   at the dimensions tested (RUN_005 T4, measured, random codes against a 512-entry codebook). If a controller
   learns to apply the operation once, iterating it extrapolates to more hops.
3. **A fixed-size streaming state with graceful degradation.** O(D) for an arbitrarily long stream. Recall survives
   losing 90% of the cue at D around 1000 (RUN_005 T6, measured); at D around 200 it breaks down past 50%.

These match doc 12 ranks 1 and 3 (content-addressable memory; spatial/relational algebra) and doc 10 §10.2
("relative coordinates, group-structured, moderate resolution"), which were arrived at independently.

## 1.3 Limits that cut across everything

- **Capacity is about D/4** for random codes (RUN_005 T4, measured). A plain bundle never forgets, so a stream
  much longer than D/4 destroys itself. **The memory therefore needs forgetting** (leaky decay `M <- lambda * M` or a
  gated write), otherwise unique #3 cannot be tested. This is a design axis in every variant.
- **Cross-talk is a property of the codebook.** Random codes carry error proportional to the total number of items
  ever bundled; block-orthogonal codes remove it (RUN_006 §3, measured). Whether choosing the codebook that way is
  legitimate is a modelling decision.
- **Kernel resolution must be matched to the task** (RUN_005 T7, measured: every design fails at N>=2 entities
  until `length_scale` is set to the grid step).
- **A perfect readout does not make RL learn.** With an oracle-exact readout, no arm learned CountRecall in 2000
  updates (RUN_006 §4, measured). RL discovery is a separate bottleneck, which is why section 4 starts supervised.

---

## 2. What has already been measured (so it is not redone)

- Algebra exact; capacity; chained queries free; graceful degradation: RUN_005 §2.
- POPGym `RepeatPrevious` has no feedforward shortcut; `CountRecall` has a partial one (headroom table in
  RUN_006 §2, deterministic across runs).
- Hand-wired delay memory (random position table, lag = k-1): +0.990 greedy eval on RepeatPreviousEasy at 490k steps
  against GRU -0.108; +0.575 at 122k steps. One seed, own A2C loop, no tuning. **A structural-prior result, not a
  claim about learning the algebra.**
- CountRecall: no arm learns it at these budgets; decode shape ruled out (RUN_006 §4).
- Bugs already fixed and not to be rediscovered: lag is k-1 not k; GRU state-mask broadcast; unseeded
  `action_space.sample()` (RUN_006 §5).
- **RUN_007 (rung 0, hand-wired, supervised, 256-float budget, 5 hold-out seeds, 1500 updates):**
  - Index-addressed recall: SSP and GRU tie at convergence (0.999); SSP is far faster early (>=0.998 at 300 updates vs
    GRU 0.786, GRU >=0.9 at 600). Extrapolation to a 4x longer stream favours SSP (0.803 vs ~0.16) but is confounded
    by input encoding. The SSP *time parameterisation* is not what wins (random position table 0.998).
  - Pointer chasing: SSP leads early and on held-out hops (0.240 vs 0.151); the GRU is better at convergence
    in-distribution (0.359 vs 0.312); no arm is competent (chance 0.125).
  - Delay: SSP converges to 1.000 but is slower than GRU/LRU; substitutable as predicted.
  - Landmark return (variant A proxy): uninformative (no-memory floor 0.024 vs best 0.013).
  - Baselines (notably `tf_full`: 0.37 on index, 0.54 on delay) were undertrained, so no "upper bound" exists yet.
  - The earlier `proceed_to_rl` verdicts for index and pointer are **withdrawn**; they came from a metric that blended
    speed and convergence and from a fragility check that could not see hand-wiring.
- POPGym's own PPO + GRU reaches +0.42 on `RepeatPreviousEasy` at 296k steps (our A2C GRU: -0.108 at 490k), so the
  in-repo loop, not the task, explained that gap; RUN_006's hand-wired-vs-GRU comparison needs re-running against a
  properly trained GRU. `CountRecallEasy` is still at chance at ~100k steps under POPGym's own loop (partial).

---

## 3. The three variants

Common design: a **controller** (GRU or MLP) plus an **SSP memory with algebraic operations**: `write(item *
address)`, `shift(Delta)`, `read(query)` (unbind), `cleanup` (nearest neighbour in a codebook). The controller emits
continuous addresses, shifts and queries from heads, all differentiable. Forgetting (decay or gated write) is part
of the memory module in all three.

**Note:** RUN_007 did *not* build this. Its controller did not emit addresses, shifts or queries; those were
hand-wired. See 3.4 for the ladder that replaces hand-wiring with learned heads.

## 3.1 Variant A: path-integration map (group action on the memory; unique #1)

- Pose is an SSP: `p_t = p_{t-1} * phi(a_t)` (exact path integration).
- Landmarks are bundled as `landmark * p_t` at observation time.
- Query: unbind the landmark to get its location, unbind the pose to get the **relative** vector, and a small readout
  maps that to an action.
- Extension: heading as a circular SSP (SO(2)), so rotating the whole map is the same one-multiplication trick; ties
  to the wrap-seam result in doc 09 (Swimmer).
- Task (needs building, a small 2-D simulator): visit landmarks, then act relative to a remembered one ("return to
  the red landmark", "go to the point halfway between"). Doc 09 Tier 2 #5 already noted no existing testbed covers
  this.
- Honest limit: a transformer that learns to subtract stored poses can be exact too. The expected advantage is
  sample efficiency, state size and extrapolation, not accuracy at unlimited state.

## 3.2 Variant B: iterated query (chained queries; unique #2)

- Pointer-chasing over bound pairs: "what is the value of the key that this value maps to".
- The controller applies unbind-then-cleanup in a loop; the loop count is part of the task input.
- Task (synthetic, no simulator): train at 1-2 hops, test at 4-8.
- This is the most likely place for a clean extrapolation result, because the algebraic step is identical at every
  hop.
- Cue-corruption axis (RUN_005 T6 in a learning setting): zero out a fraction of the query.

## 3.3 Variant C: learned-delta read and index-addressed recall (continuous address; substitutable)

- The controller outputs a continuous offset `delta`; the read is `M * pos(-delta)`. "Which lag?" becomes one
  continuous parameter, learned by gradient.
- Index-addressed recall (the third item, the last one in order) is the same variant with an absolute index; covers
  the Chain/Seq structure from MIKASA (doc 10 §14).
- Tasks: POPGym `RepeatPrevious` (already built), plus `num_decks` and `k` constructor arguments for a
  train-short/test-long split.
- Role: the cheapest warm-up and the validation of the harness. Per section 1.1 it is *expected* to be matched by
  S4D and kernel attention, and that outcome is informative rather than a failure.

## 3.4 Hand-wired versus learned structure: the learnability ladder

The thesis requires that the *learner* finds and uses the structure. Rungs, each removing hand-wiring:

- **Rung 0 (done, RUN_007):** fixed random frequencies `Omega`, fixed random item codes, hand-set write address,
  read address, key/value roles, hop loop and pose integration. Learned: a small readout, the forgetting rate, and
  (delay only) a lag-to-offset map. This is an *existence proof and upper bound*: it shows what the algebra can do
  when operated correctly.
- **Rung 1:** the controller emits the operations from learned heads: write address, read address/query, key/value
  role, and shift (action-to-phase map). `Omega` and codes still fixed.
- **Rung 2:** also learn the codes and `Omega` (initialised from the SSP prior; ablate against random init).
- **Rung 3:** remove task-visible hints: no hop count (the model learns to halt), query selection learned rather
  than given.

What rung 0 hand-wired, per variant, and what rung 1 replaces it with:

| variant | hand-wired in RUN_007 | rung 1 |
|---|---|---|
| C delay | lag given; write address `pos(t)` fixed (learned: lag-to-offset linear map) | learned write phase scale; offset head from content |
| C index | read address `min(query*T, t)` computed by me from `j/T`; write address `pos(t)` fixed | learned map from a raw absolute-index feature to read phase; learned write phase |
| B pointer | key/value roles by step parity; start key and hop count supplied; loop count from the hop count; query steps = last positions | learned role gate from token and position features; learned query code; hop count still given (rung 3 removes it) |
| A path | action fed as `phi(action)` with fixed `Omega` (exact integration given); landmark picked by argmax on a supplied one-hot | learned action-to-phase map; learned query code (soft selection over stored landmark codes) |

The algebra's own operations (the product that composes phases, the conjugate that unbinds) stay fixed in every rung:
the claim is that the controller learns what to feed them, not that it re-derives multiplication.

**Training aids**, applied only where plain training fails, reported as plain vs aided (the aid's own sample cost
counts): curriculum (short lags and 1 hop first); intermediate supervision (per-hop cleanup targets, pose targets,
address auxiliary losses); near-identity initialisation of address heads; cleanup-temperature annealing; longer
training and larger batches.

**Interpretations:** a rung-1/2 arm near rung 0 with fewer samples than generic arms *is* the thesis; a rung that
falls to GRU level means the structure exists but is not discoverable by gradient descent, and the question becomes
which aid closes the gap; if none does, that is the finding.

## 3.5 Training recipes by problem family (curricula)

**Why.** RUN_008 (measured): plain training from a random read address stalls on index recall (0.40-0.62), because
learning only succeeds when the read address starts within about one index of the write address. A coarse-to-fine
kernel-width schedule fixes it (0.993). The same trick is meaningless for the pointer arm, which has no continuous
address; there per-hop supervision is what made learning reliable. A trick that works for one task after its
diagnostics were read is not a recipe. The target is **one recipe per family of problems**, where a family is defined by
the *kind of structure the learner must discover*. Different families may have different recipes.

A recipe is (i) a schedule or curriculum, (ii) any aids, with their sample cost, (iii) optimiser settings, and (iv) a rule
that sets its constants from a task descriptor (for example the address range), so that no per-task retuning is needed.

**Continuous-address family**: index, delay (learned offset), path integration, and anything where a read phase must
align with a write phase through a finite-width similarity kernel `sinc(pi d / ls)`.
- *Measured on index (RUN_008):* the basin of attraction is about one index at `ls=1`. Widening the kernel widens it
  roughly in proportion (initial error about 5.3: accuracy 0.455 at `ls=1`, 0.886 at `ls=4`) and lowers the ceiling
  (0.997 / 0.987 / 0.947 / 0.831 for `ls` = 1 / 2 / 4 / 8). Scheduling `ls` from 8 to 1 over 750 updates reaches 0.993;
  a learned width gets 0.778; the index-range curriculum does nothing; address supervision alone gets 0.76-0.86; a
  4x learning rate alone does nothing.
- *Candidate recipe (RUN_009, measured where stated):* coarse-to-fine width **with a start width and schedule length that
  scale with the address range**. Measured at T=16: start 8 with 375-750 updates works (0.992-0.993); at T=32 the same
  constants give 0.665, start = T/2 gives 0.849, start = T gives 0.930, a 2x longer schedule 0.938; a learning rate of about
  0.02 (0.05 is no better, 0.1 is worse). *Refuted:* plateau-driven adaptive narrowing (no better than a fixed schedule), a
  lower address learning rate or a frozen write phase (no effect once aligned), light address supervision (slower).
  *Promising but unreliable:* label-free alignment pretraining of the address heads (aligns on 2 of 5 seeds; where it
  aligns, N=64 is 0.91-0.98).
- *Measured (RUN_010, 10 hold-out seeds):* the **range-scaling rule** `start width = 0.5*R`, `schedule = 24*R` updates, `steps = 48*R`, `lr = 0.02`
  (`lam_init 0.99`, `temp 0.1`; `experiments/ssp_algebra/recipes.py`) was chosen by fastest t90 subject to converged >= 0.95 on T=16 and T=32 (all 18
  grid points passed, so accuracy could not choose; speed did) and then frozen. On unseen variants: index T=24/48/64 and delay lags 1-8 pass
  (10/10 seeds at >= 0.95; T=48/64 at 1.000, where the unscaled constants gave 0.665 at T=32); **delay lags 1-4 fails (7/10)**. Under 4.7 as written
  that is a partial pass; under the "two or more variants need retuning" rule it passes. *Hypothesis, untested:* the rule is linear in R with no
  floor, so very short ranges get a schedule too short to discover the address.
- *Measured (RUN_010) speed:* floored AULC selection, index T=16: rung 0 reaches 0.90 in 24k tokens (GRU 196k, `fwp` 90k, transformer 282k); the
  **learned arm needs 132k** (1.49x the GRU, 2.1x the transformer, slower than `fwp`). Low data: learned arm 0.233 at N=64, 0.937 at N=256.
- *Prior art:* coarse-to-fine frequency scheduling is established in neural fields (BARF, FA-BARF); the schedule is not a contribution, the
  diagnosis and the rule are.

**Discrete-key family**: pointer chasing, key-value recall, counting; items are matched by fixed discrete codes and the
learner must find the roles (which token is a key, which a value) and the query.
- *Measured on pointer (RUN_008):* plain training 0.803 (3 of 5 seeds reach 1.0, two stall near 0.5); a hop curriculum
  alone changes nothing; per-hop supervision gives 1.000 on all five seeds.
- *Candidate recipe:* per-hop (per-step) supervision as a costed aid (measured: 5 of 5 seeds reach 1.000). A temperature
  schedule on the cleanup did nothing (RUN_009: 3 of 5 successes with the same stuck seeds); plain training succeeds on 6 of
  10 seeds. *Still hypotheses:* a schedule on the role gate, label-free key-to-value retrieval pretraining, restarts, and a
  role-gate initialisation chosen from a basin sweep.
- *Measured (RUN_010):* rung 3 (query read from the corruptible features, learned halting, per-hop aux) reaches 0.972 in-distribution but only
  **0.361 on held-out hops** (rung 1 with the hop count given: 1.000); cue-drop curve 0.972 / 0.606 / 0.357 / 0.201 at 0 / 25 / 50 / 75%, monotone
  but not "graceful" (4.5). The learned role gate stays at 0.500 accuracy against parity even on solved seeds (it is not load-bearing); at lr 1e-2
  4 of 4 diagnostic seeds solve the task, so the earlier stuck seeds were config-dependent. Only the GRU passes the pointer trainability gate
  (`tf_full`, `kernel_attn`, `looped_tf` fail it on the easy variant); chain-protocol equal-aid baselines are all at chance (0.145-0.183).
- *Hypothesis, untested:* the held-out-hop failure is the per-hop halting embedding (rows for untrained hops), fixable by a counter feature.

**Mixed tasks** (path integration with landmarks) need both; deferred until variant A is redesigned.

Only the *measured* items above are results; every remaining "candidate" is a hypothesis. Non-recipes so far: the index-range
curriculum and the hop curriculum alone (no effect), a learned kernel width (does not tune itself), and temperature
annealing on the index arm (a no-op there: that arm has no softmax).

**Recipe versus per-problem tuning.** The *form* of the continuous-address recipe (coarse-to-fine kernel width) carries to a
second stream length; its *constants* do not. If every new problem needs its own search, the result is a technique that is
tuned per problem, not a recipe: the search runs and their labelled data are then part of the sample cost, and the baselines
must be charged the same search budget. What turns it into a recipe is a **rule** that sets the constants from a task
descriptor (for example the address range). Three points per axis suggest that the start width and schedule length grow with
the range, but that is an inference, not a fitted rule. 4.7 specifies the test.

**Pretraining accounting.** Staging the learning (interface first on cheap synthetic streams, readout second on the task
data) restores rung-0-like data efficiency *where the interface aligns* (RUN_009). Its pretraining tokens are not task samples
only if the interface is reused across tasks, lengths or alphabets, which has not been tested; until then report task tokens
and task-plus-pretraining tokens separately. It also needs a self-supervised signal (for index recall the labels are the
stream's own tokens), which not every task offers.

*Measured (RUN_010):* probe pretraining (a throwaway readout trained on the memory's features) aligns the interface on 19 of 20 seeds
(N=64: 0.960; control 0/20; the old retrieval-loss mode 2/20), costing 256k pretraining tokens. **It does not transfer**: the interface copied
to T=32 gives 0.441 and to 16 items 0.691 (no-transfer controls 0.182 / 0.198), so by 4.7 it is "more training data".

## 3.6 Next recipe candidates: discoverability by design (planned, not measured)

The remaining gap is *discovery*: a linear width schedule has a speed floor (accuracy follows the current width's ceiling), the N=64 collapse is the
same failure, and the interface is shape-specific. Candidates, in priority order (details, tests and kill rules in handoff 10 section 4):
1. **Multi-scale frequency set** (log-uniform scales, residue-style periods, or band split with learned gains) in place of the schedule, so a wide
   component gives the basin from step 0. Removes the schedule constants and the "aid or natural" question. Fallback: a hybrid, then stop.
2. **Shape-randomised probe pretraining** (random lengths and alphabets) so the interface can amortise; the in-training probe loss as a label-free
   auxiliary (the form RL can use).
3. **Pointer fixes**: counter-based halting, 1-hop-only supervision, a self-supervised retrieval auxiliary, and a node-count scaling test
   (a holographic-memory study, arXiv 2606.24948, found capacity, not the algebra, limited multi-hop composition).
4. **Cheap diagnostics first**: batch size as a speed lever (the schedule counts updates, batch was fixed at 64 for every arm), an affine-init
   curve (how much prior a learned arm needs), the halting fix.
Pre-registered targets are in handoff 10 section 2 (tokens to 0.90 <= 50k, N=64 >= 0.9 without pretraining, 5 of 5 unseen variants, transferable
interface, held-out hops >= 0.9). Anything here that works is a recipe only after 4.7.

---

## 4. Test protocol

## 4.1 Supervised probing first, RL second

RL discovery confounded CountRecall. So each variant is first trained with **teacher-forced oracle labels** (no
exploration, no credit assignment), which isolates what the memory can represent and how fast it learns from
examples. RL is run only for arms that survive, and only for tasks where the supervised result discriminates. This
is the same kill-before-spending pattern as doc 10 §6.

## 4.2 Arms (same for all three variants)

- **mlp, posmlp:** no-memory floors (posmlp is mandatory, per POPGym's own finding).
- **GRU** (state size matched), **S4D/LRU** (state size matched; the time-SSP substitute).
- **Transformer, full context:** the upper bound with unlimited state.
- **Transformer, windowed:** window sized to the same number of state floats as the SSP memory.
- **Kernel-attention** with a learned `delta` (the control that isolates the algebra in C).
- **Fast-weight / linear-attention memory (`fwp`):** an outer-product matrix memory with learned write/read key heads,
  state `d*d` floats (d=16 at 256 floats). This is the non-SSP associative-memory control: RUN_007 showed the SSP
  *time code* is not what wins on index recall, so any benefit must be shown to exceed what a generic key-value
  memory gets from the same learned heads and inputs.
- **SSP controller:** reference configuration, the ablations in 4.4, and the rungs of 3.4 (rung 0 is the existing
  hand-wired arm and is reported alongside as an upper bound).
- **Baseline trainability gate:** every baseline must reach >=0.95 of ceiling on an easy, trivially representable
  version of the task before its number is used (RUN_007's `tf_full` failed this on index recall).

## 4.3 Axes and metrics

- **State size matched** in floats across arms (the budget-matched claim), as a **budget curve** (e.g. 128, 256, 512
  floats) rather than one point, so "GRU catches up with more state" is visible.
- **Sample efficiency (primary).** Evaluate every ~25 updates early and sparser later. Report (a) updates/tokens to
  50% and 90% of ceiling and (b) area under the learning curve, **labelled as such**. Add a **low-data regime**: a
  fixed pool of N training sequences (N = 64, 256, 1024, 4096) trained to convergence and tested on fresh
  sequences, which is different from "few updates".
- **Converged performance (reported separately, never blended with speed).** Final checkpoint at a generous budget;
  the claim is *within a stated tolerance of the best matched-state baseline*, not superiority.
- **Shorter episodes / extrapolation:** train short (episode length, lag, hop count, index range), test long, with
  **fair absolute encodings for every arm** (raw index and sinusoidal features; no arm is handed a de-normalisation
  the others lack).
- **Robustness:** cue corruption fraction (0 to 90%), distractor tokens, observation noise, stream length well above
  D/4 with and without forgetting.
- **Generalisation:** held-out lags, hops, index ranges and entity counts.
- **Matched tuning budget** per arm (same number of random configs; arm-specific knobs such as hidden size or `dim`
  count toward the budget); select on training-side seeds, report hold-out seeds only (`06-related-work.md`); never
  max-over-epochs. **Do not average over checkpoints and call it accuracy** (RUN_007's `last5` did, section 7.13).
- **Select on the quantity being claimed.** Configs are chosen on search seeds by converged accuracy (`last5_id`); under ties
  near ceiling that choice is arbitrary with respect to speed (RUN_009: the same arm needed 225 or 50 updates to reach 0.90
  depending on which tied config won). Speed claims (updates or tokens to 50%/90%, area under the curve) must therefore use a
  speed-aware selection (area under the learning curve on at least 3 search seeds per config, ties broken by converged
  accuracy); converged-accuracy claims keep the converged criterion. Done in RUN_010 (floored: `--select aulc --min-conv 0.95`, 3 search seeds); every earlier speed number is superseded.
- **Tokens depend on batch size.** Updates are the unit of the width schedule, and batch size (64) was fixed for every arm and never searched. State
  the batch with every token count; for a speed claim sweep batch per arm (same sweep for all arms) and report each arm's best. Not yet done
  (handoff 10, Phase 0.1).

## 4.4 Ablations, one change at a time from a reference configuration

- random position table vs SSP time (C)
- fixed vs learned codes (level 3)
- cleanup on vs off
- decay or gating on vs off
- (A) shift-on-memory vs re-encoding every stored item with the new pose; same answer, different cost, so the
  comparison is on state and compute, not accuracy

Not a full factorial: each variant changes one factor at a time.

Ablations from a hand-wired reference test *what is load-bearing inside the module*, not whether it is learnable;
the ladder in 3.4 is what tests learnability.

## 4.5 Decision rules (stated before running)

- **Thesis pass (for a learned rung):** within a stated tolerance of the best matched-state baseline at convergence
  (proposed: within 5 points absolute, to be fixed before running) **and** at least one of: (a) reaches 50%/90% of
  ceiling in at most half the tokens of the best matched-state baseline; (b) clearly higher accuracy at the smaller
  low-data N; (c) a separation of at least 2 pooled standard errors (and a meaningful absolute margin) on a fairly
  encoded extrapolation or robustness split. Seeds: 5 hold-out minimum.
- **Rung rule:** report the highest rung that gets within tolerance of rung 0, at what sample cost. If no rung does,
  even with the training aids, the finding is "structure present but not discoverable by gradient descent".
- **Proceed to RL for an arm** only if a **learned** (rung >= 1) arm passes the thesis rule at matched state. A
  hand-wired win alone never licenses RL.
- **It only matches S4D, GRU or kernel attention at convergence and is not faster or more robust:** the algebra buys
  nothing over the substitutes in that variant; drop it and report (doc 10's own kill condition).
- **It wins only with hand-wired operations and loses with learned heads:** report as a structural-prior result, not
  as learning the algebra (this is what RUN_007's `index` and `pointer` are).
- **A full-context transformer is expected to match accuracy at unlimited state:** a win over it is claimed only
  under a stated constraint (state budget, samples, extrapolation, stream longer than context), and only after it
  passes the trainability gate.
- **Never claimed:** results about continuous-state SSP encoders, or comparability to POPGym's published numbers
  (different loop, budget and machine), or any win against a baseline that failed its trainability gate.

## 4.6 Kill conditions (stated before running)

- Stop before RL if, once the measurement fixes are in, rung 0 shows no advantage in speed, low-data accuracy or
  fair-encoded extrapolation over matched-state baselines.
- Stop before RL if no rung >= 1 reaches the tolerance of 4.5 even with the training aids; report it as the result.

## 4.7 Recipe reliability protocol (stated before running)

- A recipe is declared per family and tested on **variants** of that family: for the continuous-address family index at
  several stream lengths and item counts and delay with different lag ranges; for the discrete-key family pointer at
  several node counts and hop ranges.
- Tune on two variants (selection on search seeds), then **freeze every hyperparameter**, including the constants of any
  start-width rule, and evaluate on **unseen** variants.
- **Pass:** at least 0.95 accuracy on at least 9 of 10 seeds on each unseen variant, with the sample cost reported
  (updates and tokens to 0.9, plus any pretraining or aid cost) next to the best matched-state baseline's.
- A recipe that needs per-variant retuning to pass is reported as a variant-specific trick, not as a recipe.
- **Fairness:** aids applied to the SSP arm are reported as aids with their cost; baselines receive an equivalent aid
  where one exists (for example per-hop supervision for pointer).
- **Tuning cost counts.** If a recipe's constants must be searched per variant, the search runs are part of the sample cost
  and the baselines are charged the same search budget. A *rule* that sets the constants from a task descriptor (address
  range, number of nodes) is what makes it a recipe: fit the rule on the two tuning variants and freeze it.
- **A pretraining stage is part of the recipe only if it transfers.** Report task tokens and task-plus-pretraining tokens
  separately. Pretraining counts as amortised only if the learned interface is reused on a different task, length or alphabet
  (train the interface once, then train just the readout elsewhere); otherwise it is simply more training data. Alignment
  reliability is the fraction of seeds aligned (at least 18 of 20) with a Wilson interval.
- **Statistics.** 10 seeds for every pass/fail claim, and report how many conditions were tried while designing the recipe.
- The thresholds (0.95, 9 of 10, 10 seeds) are proposals and must be fixed before running. A recipe study does not by
  itself license RL (4.5).
- **Status (RUN_010):** the continuous-address range rule has been frozen and tested on five unseen variants (4 pass at 10/10 seeds, delay 1-4 fails at
  7/10; 18 grid points plus 5 unseen conditions were tried while designing it). Interface pretraining meets the alignment criterion (19/20) and fails the
  transfer criterion, so it is "more training data". No discrete-key recipe has been validated.

## 4.8 Gating: what must hold before learned-arm robustness, RL and larger models (stated before running)

- **Robustness of the structure** (hand-wired upper bound, baselines) can be run at any time. The pointer cue-corruption split
  first needs an arm that reads the cue from the corruptible features: the current pointer arms read `batch["start"]` and
  `batch["hops"]` and are immune by construction.
- **Robustness of a learned arm** is run only once its recipe is reliable (at least 9 of 10 seeds); otherwise optimisation
  failures and fragility are indistinguishable.
- **RL or a larger-model study** requires all of: (1) at least one family has a recipe that passes 4.7; (2) its speed claims
  survive speed-aware config selection (4.3); (3) the pointer pass is non-provisional (equal-aid baselines and a non-SSP
  iterated control, handoff 09 section 15 step 4); (4) some robustness evidence on a learned arm; (5) the SSP-specificity
  caveat has been addressed (the `fwp` control, the position table and learned codes already match much of the index
  advantage). A hand-wired win never licenses RL (4.5).
- **Not gated:** re-running POPGym's PPO + GRU on `RepeatPreviousEasy` to at least 500k steps. It repairs RUN_006's comparison
  and does not test the SSP learning claims.
- **Status of the 4.8 conditions (RUN_010):** (1) a family recipe passes 4.7: *partial* (4 of 5 unseen variants); (2) speed claims survive speed-aware selection:
  *yes for rung 0 (8.0x), not for the learned arm (1.49x, fails the 2x rule)*; (3) the pointer pass is non-provisional: *no* (equal-aid and cue conditions met, but
  the non-SSP iterated control fails its own trainability gate); (4) some learned-arm robustness evidence: *yes for the rung-3 pointer arm* (not graceful);
  (5) SSP-specificity: *no* (`fwp` is faster to 0.90 than the learned arm). RL stays gated; handoff 10 section 5 states the revised gate.

---

## 5. Order of work

0. **Done (RUN_007):** reference PPO (settled that the in-repo A2C loop, not the task, explained the RepeatPrevious
   gap), the supervised harness and eight arms, and rung 0 of variants C, B and a weak A proxy.
1. **Measurement fixes** (cheap, local): fine evaluation cadence, fair absolute index encoding, baseline
   trainability gates, converged-vs-area reporting, state-budget curve, low-data regime, fix `analysis.py`'s verdict
   logic.
2. **Learned structure on index (C) and pointer (B):** rungs 1 and 2, then rung 3 (no hop count); training aids only
   if plain training fails.
3. **Redesign variant A** (larger steps, old queries, continuous targets; competitor = attention over stored
   (pose, landmark) pairs), then its rungs. SO(2) heading afterwards.
4. **Robustness and generalisation battery** (cue corruption, distractors, stream above D/4, held-out hops and
   entity counts) on the arms that survive 2-3.
5. **Re-run RUN_006's comparison** against a properly trained GRU (POPGym's PPO, >=500k steps).
6. **RL for surviving learned arms only** (this is where remote pods become useful: many seeds x arms x tasks; close
   them after every batch).
7. **Recipe study (3.5, 3.6, 4.7, 4.8).** Handoff 09 section 15 steps 1-4 are done (RUN_010). The continuing plan is
   `handoffs/10-ssp-discoverability-and-rl-port-prep.md`: Phase 0 diagnostics (batch size, halting counter, affine-init curve); Phase 1 schedule-free
   multi-scale discoverability; Phase 2 amortised interface pretraining; Phase 3 pointer fixes and scaling; Phase 4 SSP-specificity and variant A;
   Phase 5 RL port preparation (POPGym PPO + GRU to 500k steps now, memory-module interface, first RL test after the gate). Diagnostics already run:
   RUN_009, RUN_010.

## 6. Deferred (real variants, different thesis)

- Zero-shot goal composition (doc 10 §2, §5): competitor set is successor features.
- World-model transition composition (doc 10 §8-9): overlaps A's path integration; the equivariant world-model line
  is mature.
- Continual learning and reward-drift tracking (doc 10 §7.2): brings in centering.
- Multi-agent membership (doc 12 rank 4): weak fit.
- Quantised or on-device cost (qFHRR): a deployment axis, after anything works.

## 7. Open items

- ~~Whether ray 2.59 runs POPGym's `ray_models`~~ **Resolved (RUN_007 section 3):** `ppo.py` does not, but driving
  `PPOConfig` directly with POPGym's models does; see `experiments/ssp_algebra/ref_ppo/`.
- Doc 10 §13.1's claims about S4D, LMU, FWP and linear attention losing to GRU are taken from that doc and were not
  re-checked against the POPGym paper.
- Forgetting parameterisation: the leak is now a searched knob (`lam_init`), and RUN_007 showed the best horizon
  depends on the task (short helps delay, hurts index). Fixed decay vs learned gate is still undecided.
- State accounting rule (decided in RUN_007): floats carried between steps; windowed transformer fitted within 5% of
  the SSP state. Whether this is the right accounting for a transformer's KV cache is open to challenge.
- The thesis-pass thresholds in 4.5 are proposals and must be fixed before any run.
- Repo question: this is a different thesis from the repo's SSP-as-encoder work. The POPGym folder is self-contained
  and can be moved; the decision is deferred until a learned arm passes the thesis rule.
- Recipe reliability: the continuous-address range rule passes 4 of 5 unseen variants (RUN_010); the discrete-key family has no validated recipe. The width
  schedule itself is a borrowed technique (BARF-style) and handoff 10 Phase 1 tries to remove it.
- Speed claims: re-selected on floored AULC in RUN_010 (rung 0 8.0x, learned arm 1.49x); every earlier t50/t90 number (RUN_007-009) is superseded. Token counts
  depend on batch size, which was never searched (4.3).
- The pointer cue-corruption split was vacuous for the rung-0/rung-1 pointer arms; the rung-3 arm (RUN_010) reads the cue from the corruptible features and
  gives a real curve.
- `temp` annealing is a no-op in the index learned arm (no softmax there); "aided" on index therefore means address
  supervision plus the small-index curriculum.
- Whether label-free interface pretraining transfers: measured in RUN_010, **it does not** (T=16 to 32: 0.441; 8 to 16 items: 0.691). Open: whether
  shape-randomised pretraining or an in-training probe loss fixes that (handoff 10 Phase 2).

---

## 8. Decision log (appended 2026-10-11, after `runs/RUN_008_ssp_learned_structure.md`)

Mechanical outputs of `analysis.py` (5 hold-out seeds, thresholds as proposed in 4.5), with the judgment stated:

- **Stop rule (4.6): did not fire.** Hand-wired rung 0 separates from passing baselines on speed, low-data and fair-encoded extrapolation (index, pointer).
- **C `delay`: `algebra_buys_nothing`** (as pre-registered in 1.1).
- **C `index`: `structural_prior_only`.** The learned rung needed a coarse-to-fine kernel-width schedule to be discovered at all (plain 0.40-0.62; scheduled 0.993).
  It keeps a low-data edge at N=256 but not at N=64 or on long streams, and it is not separated from the `fwp` control; the position-table, learned-code and
  no-cleanup ablations match the full arm. What helps is a fixed-code addressable memory, not SSP-specific structure.
- **B `pointer`: `thesis_pass_candidate`, provisional.** Learned rung 1 reaches 1.000 on held-out hops with per-hop supervision (3/5 seeds without it); the GRU is at
  0.45 after 6000 updates. Conditions before RL: reduce reliance on the aid, add a non-SSP iterated-structure control, give baselines a comparable aid, and build an arm
  that reads the cue from the same (corruptible) features so robustness can be tested.
- **A `path`: still untested** (proxy too easy); deferred.
- Open items 7 updated: the learnability question is now answered *for index* (basin width, a curriculum on the kernel) and *for pointer* (discoverable with aux).
  The thesis-pass thresholds remain proposals.

---

## 9. Decision log (appended 2026-10-11, after `runs/RUN_009_recipe_diagnostics.md`)

- **The learned index arm is not slow after alignment:** identity-initialised at lr 0.0225 it reaches 0.90 in 50 updates (RUN_008's
  225 was a config tie-break artifact); the whole cost is discovery. The "Adam jitter" explanation is refuted.
- **Curriculum details measured:** a 375-update width schedule beats 750 at T=16 (0.90 at 250 vs 350 updates, same accuracy);
  adaptive narrowing, a higher lr on top, and light address supervision do not help; label-free retrieval pretraining aligns 2 of
  5 seeds and gives rung-0-like N=64 where it does.
- **Family recipe status:** the *form* (coarse-to-fine kernel width) carries to T=32, the *constants* do not (0.665 unchanged;
  0.93-0.94 with start = T or a 2x longer schedule). No recipe has passed 4.7; the next test is a frozen range-scaling rule on
  unseen lengths and on delay.
- **New caveat on earlier claims:** speed numbers for all arms come from convergence-selected configs, so the headline "about 4x
  faster to 50%" is unproven until re-selected on speed (4.3).
- **Pointer:** a temperature schedule does nothing; plain training succeeds on 6 of 10 seeds; the stuck seeds (about 0.5) are
  undiagnosed. `thesis_pass_candidate` remains provisional.
- RL, larger models and learned-arm robustness remain gated by 4.8.

---

## 10. Decision log (appended 2026-10-11, after `runs/RUN_010_recipe_and_controls.md`)

- **Speed, re-selected (floored AULC, 3 search seeds):** rung 0 reaches 0.90 in 24k tokens versus 196k for the GRU (8.0x); the learned arm needs 132k (1.49x the
  GRU, 2.1x the transformer, slower than `fwp` at 90k). It fails the 2x half-the-tokens rule of 4.5(a); its case rests on low data at N=256 (0.937), not on N=64 (0.233).
- **Recipe:** the range-scaling rule passes 4 of 5 unseen variants at 10/10 seeds, delay 1-4 fails (7/10). It is a recipe under the two-variant kill rule and a partial pass
  under the literal 4.7 wording; it was not retuned.
- **Pretraining:** probe mode aligns 19/20 seeds, N=64 0.960; transfer fails on both axes, with no-transfer controls at 0.182 / 0.198. Reported as more training data.
- **Pointer:** `tf_full`, `kernel_attn` and `looped_tf` fail the trainability gate; only the GRU is a valid generic comparison (0.478 at 6000 updates). Equal-aid chain baselines are at
  chance. Rung 3 gives a real cue-corruption curve (monotone, not graceful) and loses held-out-hop extrapolation (0.361). The pass stays provisional.
- **Literature:** nothing found that studies discoverability of the SSP read address; the width schedule has direct precedent in neural fields, so it is not a contribution.
- **Direction:** make discovery cheap by design and amortise the interface, then port to RL as a memory component (3.6; handoff 10). RL remains gated by 4.8.

---

## 11. Decision log (appended 2026-10-12, after `runs/RUN_011_discoverability_and_halting.md`)

Executes handoff 10. Full detail, tables and traps are in the run record; this is the judgment.

- **Batch size is not a speed lever (0.1).** The learned arm is *more* fragile at small batches (0/10 seeds reach 0.90 at batch 8, 5/10 at 16, 9/10 at
  32, 10/10 at 64); tokens fall only because the token counter scales with the batch (`updates x batch x 8` for index T=16). Batch 64 stays the headline; `tf_full`
  at batch 8 fails its own trainability gate. The <= 50k-token target cannot be reached by batch tuning.
- **Rung 3's held-out-hop failure is the halting, not the iteration (0.2).** The unrolled unbind-clean-up chain reads the right node after h iterations at
  *every* h = 1..8 (per-hop 1.000), so 1.2 unique #2 survives in a learning setting. Three learned halting mechanisms all fail on held-out hops (0.36-0.52), for a
  reason now pinned: with hops 1-2 trained, a whole slab of (intercept, slope) decodes is loss-free, so SGD stops compressive (measured slope 0.15 with a hard gate,
  0.30 with a soft quadratic). Reading the cue's own hop channel with a *fixed* scale plus the same gate reaches 1.000 on hops 4-8 with a real cue-dropout curve
  (0.62 / 0.34 / 0.21 at 25 / 50 / 75 %). The hop *scale* is the thing that must be given; everything else stays learned.
- **Schedule-free multi-scale frequencies: FAIL (1).** logmix at `ls_hi = 0.5R` reaches conv 0.738 (index T=16, 0/10 seeds), 0.813 (delay 1-4, 5/10) and 0.893
  (delay 1-8, 0/10) against the incumbent schedule's 0.984 / 0.929 / 0.999. It is a *discovery* failure, not resolution: with the address given the same frequency set
  has a ceiling of 0.975 (0.991 fine-heavy). Fine-heavy mixtures have the best ceilings and the worst discovery (0.409), and no hybrid clears the bar (best 0.793).
- **The schedule was NOT removed by the affine prior either -- but the attempt located the obstacle (0.3 -> 1).** Initialising the read head to the time ramp
  plus noise, with a **fixed** frequency set and no schedule, gives conv 0.994 in **25.7k tokens (50 updates), 10/10 seeds at T=16**, for every `eps` in
  {0, 0.02, 0.05, 0.1, 0.2, 0.4} -- i.e. the prior *removes the need for the schedule on the variant it was tuned on*, at 4x the schedule's speed (the same arm
  *with* the schedule costs 102.6k). But neither parameterisation passes 4.7 on unseen variants: with the error set as a fixed *fraction* of the address range it
  collapses on longer streams (0.44 / 0.46 / 0.35 at T=24 / 32 / 48) while doing best at the shortest range (delay R=4: 0.886); with the error set in *absolute*
  index units (`slope = 1 + a*N/R`, `offset = a*N`, verified constant at 1.09 / 1.08 / 1.10 indices RMS for T = 16 / 32 / 64) it is consistent but never reliable
  (index 0.869 / 0.861 / 0.825 / 0.842 / 0.814 for T = 16 / 24 / 32 / 48 / 64, delay 0.957 / 0.885; at most 9 of 10 seeds, and `a = 4.0` fails badly at
  0.478 / 0.367). So **the coarse-to-fine width schedule stays** as the only recipe with a pass (4 of 5 unseen variants, RUN_010), and the affine prior is recorded
  as fast-but-unreliable; 26 conditions were tried while designing it and none is a recipe (4.7's search-cost rule applies).
  **What the attempt is worth: the mechanism.** Plain training fails from a *random* read head (RUN_008: 0.40-0.62) and succeeds from a wrong-but-monotone one --
  even from a 6.8-of-14-index error -- and the gradient never vanishes (finite-D crosstalk ~1/sqrt(D): 3.15e-03 at a 5.3-index error). What cannot be found is
  therefore a read head that is **monotone in the index**, not a read address within one index of the answer; a future fix should parameterise the read address
  monotonically (a monotone network, a cumulative-softplus ramp) rather than widen a kernel.
- **Interface: not amortised, in either form (2).** Shape-randomised pretraining scales with its budget (N=64: 0.269 / 0.303 / 0.436 / 0.630 / 0.845 at 0 / 50 /
  100 / 200 / 500 steps, 256k pretraining tokens) and still fails the transfer bar (0.45-0.40 on the length axis against a 0.9 pass). The **in-training probe auxiliary
  is a no-op** (0.274 / 0.275 vs 0.269) -- and that was the only aid form the RL port could have used. Reported as more training data, again.
- **SSP-specificity: not established, now on both axes (4).** With the interface aid equalised, the non-SSP outer-product memory (`fwp`) is a tie with the SSP arm at
  low data (0.821 vs 0.845 at N=64) and on speed (128.1k vs 132k tokens), and without any aid it is *better* at low data (0.363 vs 0.269). By 4.5 this is the
  "only matches the substitutes" branch: `index` is a harness check, not a thesis vehicle (handoff 10 section 9 item 1).
- **RL stays gated (5).** Handoff 10 section 5 conditions (1) and (2) fail, so phase 5.3 is not run. Phase 5.1 (POPGym's own PPO + GRU on `RepeatPreviousEasy`, to
  600k steps) is running as the ungated repair of RUN_006. Phase 5.2 is done: `rl_memory.py` + its contract tests (round trip 1.000; `freeze_interface` stops the
  address gradient; `aux_loss` finite and label-free; `reset` clears).
- **Open after this run:** the accuracy-vs-token trade-off above 200 pretraining steps is unmeasured, and the RL gate still fails on the interface condition.
  Everything else from that list has since been done: variant A's rung-0 check was run and the variant dropped (the hand-wired path arm is the worst memory arm; the
  `pose_attn` competitor was written for it), the pointer 1-hop extrapolation reaches 1.000 on held-out hops, the node scaling is resolved (the limit is between 16
  and 32 nodes and it is the iteration, not the bundle), and the baseline gates all fail (the GRU at 0.247 over 10 seeds at 15000 steps).

---

## 12. Correction log (appended 2026-10-12, from auditing RUN_011's own numbers)

- **RUN_010's index "unseen variants" were T=16, not T=24/32/48/64.**  Those job lists omitted `--ep-len` and the default is 16; the stored rows carry 511-514 tokens
  per update (T=32 is ~1025, T=64 ~2049).  Withdrawn: the range-scaling rule's "4 of 5 unseen variants pass" and "T=48/64 at 1.000", and the statement that the schedule
  "stays as the only recipe with a pass".  The first real test at index T=32 scores 0.833 with 0 of 10 seeds at 0.90.  **No recipe is validated for the continuous-address
  family**; `jobs/p5_rule_real.json` re-runs the frozen rule at the true lengths with `--ep-len` explicit.
- **Pod and local runs are not numerically identical (>40x 1e-3 in accuracy; the GRU's mid-training accuracy moved by 0.43 locally), so results from the two machine
  classes must never be tabulated together.**  Established by `replay_check.py`: locally four stored RUN_010 rows differ, and the same replay on a pod reproduces all four
  **bit-for-bit (max |delta| = 0.000e+00)** -- which also clears RUN_011's edits of any regression and shows the earlier local "DIFF" was a BLAS/instruction-set artefact.
- **Two of RUN_011's own explanatory claims are therefore still open** and are being tested rather than asserted: whether discovery past T=16 is limited by the *absolute*
  address error, by its *slope* component, or by the stream length (basin map, `jobs/p5_basin.json`), and whether the 0.807 combined-pointer result is the start decoder
  seeing an unseen hop channel (2x2 diagnostic, step 4).

---

## 13. Discoverability mechanism and a working recipe (appended 2026-10-12, continuing RUN_011)

Follow-up to section 12, after the audit that section describes.  Two results, one mechanism and one recipe.

**The mechanism: discovery is limited by the *offset* of the initial read address, not by its size.**  In a controlled grid at index T=32 (10 seeds each;
`err(j) = slope_err * j + offset`, and `kappa = sinc(pi * err)` at `length_scale = 1`):

| slope error | offset 0 | offset 0.5 | offset 1 | offset 1.5 | offset 2.5 | offset 3 | offset 7 |
|---|---|---|---|---|---|---|---|
| 0 | **0.957, 9/10** | **0.956, 9/10** | 0.341 | 0.341 | 0.266 | 0.266 | 0.265 |
| 0.03 | **0.957, 9/10** | -- | 0.341 | -- | -- | 0.266 | 0.261 |
| 0.1 | **0.955, 9/10** | 0.791, 6/10 | 0.338 | 0.350 | -- | -- | 0.252 |

Every offset-zero cell passes; every offset of 1 or more fails 0/10 *and never moves* (flat at 0.25-0.34 for 1500 steps).  The reason is the kernel: a constant offset
leaves every stream position the same distance from `kappa`'s informative range, so the read head receives no direction, while a *scale* error leaves the
small-index positions nearly correct and therefore has a direction.  Tolerance is **narrow in the origin coordinate and wide in the scale coordinate** -- the "basin is
one index wide" phrasing (RUN_008) and my own "the read head must be monotone" (RUN_011 4.4) are both superseded by this.

Three purpose-built cells confirmed the sharp form of the rule (predictions written before running): an offset that would be a dead zone **is rescued exactly when the
drift carries the error through zero inside the stream** -- `(slope -0.1, offset +1)` and `(slope +0.1, offset -1)` (crossings at j = 10) both scored 0.957 / 0.956 with
9/10 seeds at >= 0.95 as predicted, while `(slope -0.1, offset +3)` (crossing only at j = 30) scored 0.637 with 4/10.

**The recipe: a ramp with a small scale error and no offset.**  Read head = `ramp * (1 + 0.03*N(0,1))`, **offset exactly 0**, fixed single-scale frequencies, no
schedule, no address-range constant.  Converged accuracy and reliability, 10 seeds, frozen hyperparameters:

| variant | recipe | seeds >= 0.95 | tokens to 0.90 | the frozen range rule (schedule) |
|---|---|---|---|---|
| index T=16 | 0.996 | **10/10** | **25.7k (50 updates)** | 0.984, 10/10, 102.6k |
| index T=24 | 0.976 | **10/10** | 38.2k | 0.871, 3/10 at 0.90 |
| index T=32 | 0.956 | **9/10** | 52.0k | 0.833, 0/10 at 0.90 |
| index T=48 | 0.973 | **10/10** | 77.7k | 0.775, 0/10 at 0.90 |
| index T=64 (1024 floats) | 0.991 | **10/10** | -- | 0.690, 0/10 (512 floats) |
| index T=64 (512 floats) | 0.952 | 6/10 | 152.7k | -- |
| delay 1-4 | 1.000 | **10/10** | 21.6k | 0.929, 7/10 |
| delay 1-8 | 0.997 | **10/10** | 18.4k | 0.999, 10/10, 128.4k |

It beats the incumbent everywhere both can run, is 4x cheaper on tokens at T=16, and is flat with length where the schedule decays monotonically
(0.984 -> 0.871 -> 0.833 -> 0.775 -> **0.690** at T=16/24/32/48/64 against the recipe's 0.996 -> 0.976 -> 0.956 -> 0.973 -> 0.952).
The one failure, index T=64, is **not the prior's fault**: the exact rung-0 read address at the same 512-float budget scores 0.952 with 5/10 seeds, so
no initialisation clears the 4.7 bar there -- the address range (62) is at the `D/4 ~ 64` capacity limit at that budget.  So the recipe's record is
"passes every variant inside the memory's capacity", confirmed at 1024 floats: the exact address 0.992 (10/10) and the recipe 0.991 (10/10).

**Two honest qualifications**: it is a *strong aid* (a monotone, correctly-centred address prior, which 3.4 lists as an aid), and `eps = 0.03` is a tuned
constant -- the conditions tried for this family now number `eps` in {0.4 relative, 0.5 absolute, 0.1, 0.03} plus twelve grid cells plus three sign cells, and per 4.7
that search is part of the cost.

**Two limits located, and neither is what it looked like (RUN_011 sections 6.2, 3.3).**

- **Pointer's node limit is between 16 and 32, and it is not the bundle's capacity.**  8 nodes: 1.000; 16 nodes: 0.997-1.000 (10 seeds at 512 floats);
  32 nodes: **0.205 (10 seeds) at 512 floats and 0.208 at 1024 floats**, where `D/4 ~ 128` items would comfortably allow 32 nodes.  So the limit is in the
  *iteration* at that node count -- 32 items to keep separate along an 8-hop chain -- not in how much one vector can hold.  (The 64-node cells were stopped
  unfinished on cost grounds once the limit was located.)
- **The recipe's one failure (index T=64) is the task's capacity at that budget.**  At 512 floats the address range 62 sits at `D/4 ~ 64`, and the *exact*
  rung-0 address scores the same 0.952 (5/10 seeds) as the recipe (6/10).  At 1024 floats both reach about 0.990 with every measured seed clearing 0.95
  (exact address 0.990, recipe 0.989).  So the recipe's record is **"passes every variant run inside the memory's capacity"**, which is the form the claim
  should keep: the memory has a capacity, and outside it no initialisation -- learned, prior or exact -- reaches the bar.
- **And no generic baseline for pointer exists at these budgets**: the GRU on the easy variant (4 nodes, 1 hop) sits at 0.247 over 10 seeds at 15000 steps
  (0/10 near the bar; RUN_010 saw 0.478 at 6000), with `fwp`/`lru`/`tf_window` at 0/30 seeds on the same gate.  The pointer result is a *capability*
  demonstration -- 8-hop chained queries at 1.000 with no hop count given and one hop of training -- and not a matched-baseline win.

---

## 14. Redirection: from characterising SSP to a deployable memory module (appended 2026-10-12)

This section changes what the project is for, and therefore what counts as a result. It does not change any measurement above. Written after the user
restated the goal; the plan that follows is in `handoffs/11-ssp-memory-module-deployable.md`.

## 14.1 The goal and the three constraints (stated by the user; carry them into every decision)

**Goal.** Find an SSP memory design and training recipe that is *capable*: usable as a memory module inside larger models doing real problem solving (RL
agents on robotics-like tasks first). The experiments are **not** only to find out how SSP behaves; they exist to find a solution that makes SSP usable as
a memory module. The user expects SSP to **match the performance of other memory modules while being more sample-efficient and perhaps more robust**; any
further advantage is a bonus, and the experiments so far have not found one.

The three constraints, in the user's words and in operational form:

1. **Sutton's bitter lesson: do not over-handcraft.** Prefer general methods that scale with data and compute (learning, search, self-supervision) over
   hand-set structure. Operationally: the number of **task-dependent constants** a practitioner must set is a reported cost of every method, and a method
   that removes constants at a small accuracy cost beats one that keeps them. Parts of the model may be made learnable.
2. **Real-world utility: deployable as a memory unit in models that solve actual problems.** Operationally: one default configuration across tasks; works
   from a default initialisation; plugs into a policy (PPO and a recurrent-memory interface, `rl_memory.py`); fixed small state and constant per-step cost;
   no per-task tuning. A result that holds only on tasks hand-matched to SSP does not count.
3. **Match generic memories, be more sample-efficient, maybe more robust.** Operationally: every claim is against **matched, trainable, incumbent**
   baselines (GRU, LRU, and the fast-weight family *as it is actually used*, i.e. with a delta-rule write, gating and tied keys and queries), at equal state,
   equal tuning budget, equal aid. Beating SSP's own previous recipe is not a result.

**Where the current recipe stands against these** (it is a research finding, not yet a module):

| ingredient of `recipes.slope_only_cfg` | hand-set? | what a practitioner must know |
|---|---|---|
| offset exactly 0 | yes | that reads and writes share an origin |
| `eps = 0.03` | yes, tuned once | the scale error to inject |
| fixed frequencies | yes | the address range, which sets capacity |
| time ramp as read head | yes | that the task is positional |
| `lr`, `lam_init`, `temp` | partly | defaults found by search |

Any new idea must remove rows from this table, not add to it, and it counts only if it passes **from a default initialisation**: passing only at offset 0
repeats what is already known.

## 14.2 What the evidence supports, said plainly

| expectation | evidence so far |
|---|---|
| matches other memory modules | `fwp` ties SSP on index; on the pointer the generic baselines fail their own gates (so there is no comparison); on variant A the hand-wired SSP path arm lost to the GRU (MSE 0.147 vs 0.049) |
| more sample-efficient | only against SSP's own earlier schedule (25.7k vs 102.6k tokens at T=16); **no matched GRU/`fwp`/`lru` comparison exists on delay or index at a fair equal aid**, and the pointer has no passing baseline |
| more robust | untested beyond the cue-drop curve (monotone, not graceful, and nearly identical across all halting variants) |
| usable in RL | no RL run; the in-loop probe auxiliary is a no-op; interface pretraining does not transfer |

So the supportable statement is: SSP is a competent memory with a **discoverable** training recipe (a mechanism-derived prior) and **one capability result**
(iteration extrapolates to 8 hops from 1). It is not yet shown to be better than, or equal to, what practitioners already use.

**The `fwp` arm is not the incumbent.** `FWPArm` (`arms_learned.py`) is plain linear attention: `M = lam*M + v kT` with one learned scalar `lam`, no delta-rule
correction (`v - M k`), no write strength, no input-dependent gate, and *untied* separate linear `k_head`/`q_head` over the same fixed time features, with
`d = sqrt(budget)` (about 22-23 at 512 floats). The fast-weight family used in practice (DeltaNet, gated DeltaNet and relatives; linear attention is
*equivalent* to a fast-weight programmer, Schlag, Irie and Schmidhuber 2021) has the delta rule and gates. **A delta-rule, tied-key arm must be added before
any "SSP is better" or "SSP ties" statement is made.** (The "recent production models use these" statement was recalled from memory and is not checked.)

## 14.3 Two axes of change

Both axes are judged by the three constraints and by what an **RL agent** needs from a memory: a weak, noisy, delayed learning signal (so a flat region of
the loss is fatal, not merely slow); partial observability (the agent must *decide* what to write and read; there are no given addresses); long, variable
episodes (deployment longer than training); a small fixed state with constant per-step cost; and robustness to a shifting data distribution. The consequence
that drives the plan: **the discovery problem is worse in RL than in supervised learning**, so a prior set by hand per task is not an option.

**Axis 1: make parts learnable** (every hand-set row above is a candidate). In RL more learnable parts means more parameters trained from a weak signal, so a
learnable part is only useful if it gets a strong signal of its own.

- *Self-supervised recall auxiliary* (label-free reconstruction of an earlier observation at a random lag; `rl_memory.SSPMemory.aux_loss` exists): the bridge
  that turns the supervised recipe into an in-loop signal. Caution: the in-training probe auxiliary was a no-op in RUN_011 5.3; the *reason* (does its
  gradient reach the address parameters at all?) must be diagnosed first.
- *Learned kernel width / learned frequencies*: needed when the task's scale is unknown; but the gradient with respect to frequency has the same dead zone, so
  they are supporting pieces after the dead zone is removed, not fixes. (`ls_mode` learned was tried in RUN_008: 0.778, it does not tune itself.)
- *Learned write and forget gates* (input-dependent): RL needs selective storage. Needs a distractor task to show benefit.
- *Learned initialisation across a task family* (a prior from data, not from hand): the bitter-lesson-aligned form of the recipe. Risk: shape-randomised
  pretraining scaled but did not transfer; this differs (it learns the address generator's initial values, not the interface) but may fail alike.
- *Search instead of a prior*: K parallel read heads with random initial offsets; one lands in the healthy region and the gradient routes to it. Trades compute
  for a constant; still needs an offset range, which is tied to the address scale.

**Axis 2: change the architecture.**

1. **Tied encoder with relative addressing (highest-value idea; SSP-specific).** One network maps content (plus time if relevant) to an address, used for the
   item being written and for the query being read. In SSP the *shift is a binding*: the read address can be the current-time address bound with the inverse of a
   lag address, both from the same encoder over the same frequencies. Then there is **no read origin of its own to discover**, so the dead zone found in section 13
   does not exist by construction; what remains to learn is only the lag (a scalar or from content). A discrete-slot or learned-key matrix has no native shift.
   Risks: if the lag must be inferred from content the lag producer may meet its own dead zone (the earlier learned embedding scored 0.362); and it only helps
   when reads are defined relative to writes (content-keyed lookup needs the tied-key form instead). **Untested; nothing here is built.**
2. **A kernel with no dead zone** (heavy-tailed, non-oscillating, or width annealed only in the training loss). One-line change and the cleanest direct fix of the
   cause found; risk: a wide kernel blurs neighbouring items (sinc keeps them orthogonal), and any annealing is a schedule constant.
3. **Delta-rule write for the SSP bundle**: `bundle += beta * (v - read(k)) * k`, so a slot is corrected rather than piled onto. Capacity at fixed floats is limited by
   crosstalk (the T=64 limit at 512 floats looks like it), and the delta rule is what made linear-attention memories usable at length. One extra gate.
4. **Input-dependent write and forget gates**, and **multi-timescale banks** (fast and slow, mixed by a learned gate) so the task's time scale need not be known.
5. **Content-plus-position keys** (key is a function of content and time): the property a deployable memory needs, since real tasks are rarely purely positional;
   the largest and least-evidenced step (variant A was the first attempt and failed).
6. **Iterated reads as a standard option** with learned halting: the one place SSP has a distinctive result (8 hops from 1 hop; no baseline passed). The node limit
   (16 ok, 32 fails, iteration-bound not capacity-bound) is the open research question that decides whether this is usable beyond toy sizes.
7. **A soft, interpolating readout** for continuous address spaces.

## 14.4 Where SSP might be distinctive, and what each would mean for a robotics-style agent (hypotheses; none tested in RL)

| candidate advantage | what it would buy a robot | status |
|---|---|---|
| motion update by binding: translate the whole memory by `phi(-d)` instead of rewriting slots (path integration, egocentric to allocentric) | map-like spatial memory re-expressed in the body frame for one O(D) multiplication | exact in the algebra (RUN_005 T0); the one learning test (variant A, hand-wired path arm) **failed**; that tests the arm, not the idea |
| continuous, smooth addressing: nearby addresses retrieve similar content | "what did I see near here?" at places not exactly visited | untested |
| relative lag recall by binding | latency compensation, delayed feedback | this is the delay task; most directly testable |
| length extrapolation (a phase address works at any length) | deployment episodes longer than training | one weak data point (the recipe is flat in length 16-64); never tested beyond training length |
| iterated reads | route and graph recall over stored places | capability shown with given node identities (16 nodes); real observations are not clean nodes |
| small fixed state, constant per-step cost | onboard compute | holds, but not unique |

Only the first four could be a real advantage over a fast-weight-family memory. Under the three constraints the **gate for pursuing robotics is a spatial recall
task** (14.5, step 3): if SSP cannot beat or match a GRU and a delta-rule fast-weight memory there, the robotics argument is much weaker and the plan changes.

## 14.5 Experiment design (full detail, pass rules and traps are in handoff 11)

Principles for the round: pass rules written **before** running; a **constants scoreboard** next to accuracy for every arm; every baseline must pass its
**trainability gate** on an easy variant before its number is used; equal state floats, parameters, updates, batch and tuning budget; **pod and local runs never
mixed**; `--ep-len` explicit (the RUN_010 error); a fresh `--tag` per protocol; no existing test or threshold edited.

- **Step 0, diagnostics (before any method work).** (a) Baseline audit: add a delta-rule fast-weight arm with tied keys and queries; run each baseline's trainability
  gate. (b) Loss-shape map: retrieval loss against address error per arm and kernel (the gradient-norm probe did not discriminate in RUN_011; the loss shape did).
  (c) Offset grid for **every** arm, including plain and delta-rule `fwp`: is the dead zone SSP-specific, or is address discovery hard for any key-addressed memory?
  (d) Autopsy of the in-loop probe auxiliary (gradient reaching the address parameters? loss falling? correlation with the task metric?). (e) RL reference:
  PPO + GRU on `RepeatPrevious` to convergence over several seeds (the 0.683 at 632k steps is not converged), with random and optimal returns. (f) RL interface
  contract: drop the memory in for the GRU; resets, truncated backpropagation, gradient to the address parameters, per-step cost, a no-op memory behaving like a
  feed-forward policy. (g) Memory-use check in RL: zero the memory at evaluation; a linear probe for the lagged observation. (h) Power: at least 10 seeds
  supervised and at least 5 in RL, with bootstrap intervals.
- **Step 1, supervised track.** Arms: GRU, LRU, plain `fwp`, delta-rule tied-key `fwp`, the current SSP recipe (marked handcrafted), and SSP candidates *one change at
  a time* (tied encoder with relative addressing; heavy-tailed kernel; K random heads; delta-rule write; learned gates). Tasks: index and delay at real lengths and
  several lag ranges; delay with distractors; length extrapolation (train T, test 2T); content-keyed recall (a non-positional family); the offset grid at offsets
  0, 1 and 3. Two columns per arm: a pre-chosen default configuration, and a tuned one at equal tuning budget. **Pass rule (fixed beforehand):** a candidate removes
  the dead zone only if it passes at offset 3 on at least 9 of 10 seeds with no prior; it is competitive only if within a stated margin of the best baseline at
  convergence at equal budget; a speed claim needs a stated ratio over the best *baseline*, not over SSP's old recipe.
- **Step 2, RL track on POPGym.** `RepeatPrevious` at easy, medium and hard (the lag changes with difficulty; it is a delay task, so the supervised recipe applies
  directly), then other memory tasks after verifying the exact list (not re-checked here). Arms: PPO with GRU, plain `fwp`, delta-rule `fwp`, the SSP memory, with
  encoder, PPO hyperparameters and state size matched; each with and without the label-free recall auxiliary. Several million steps and at least 5 seeds; final
  return, steps to a threshold, and learning-curve area, each with intervals. **Gate:** SSP at parity with the GRU at the final return, with any speed claim at a
  stated margin; failure on the task where SSP should be most natural stops the RL track.
- **Step 3, spatial recall task (the gate for robotics).** A purpose-built, partially observed 2-D world: a landmark shown at the start, the agent moves, and the
  model reports the landmark in its current frame. Supervised first, then an RL version. Arms: GRU, delta-rule `fwp`, an attention window, and SSP using relative
  addressing as the motion update. A loss here changes the plan.
- **Ablations:** memory zeroed at evaluation; tied encoder off; auxiliary off; delta rule off; kernel swapped back to sinc.
- **Order and stops:** diagnostics (stop if the baseline audit changes the picture) -> supervised track (stop if no SSP candidate both removes the dead zone and is
  competitive) -> `RepeatPrevious` in RL -> the spatial task (the gate for continuing to robotics).

## 14.6 Decisions (made by the user and the assistant on 2026-10-12; supersedes the open questions)

1. **K random read heads: allowed only as a fallback, with a scale-free default.** Try the tied encoder with relative addressing (14.3 axis 2 item 1) and a
   dead-zone-free kernel first. K heads are acceptable only if one default K (proposed 4-8) works on every task with an offset range that does not depend on the task;
   `K` and the range are counted as constants on the scoreboard and the extra compute is reported. (Assistant's recommendation, accepted by the user's delegation.)
2. **The success rule (user's):** SSP is good enough if it is **as good as the others on the other metrics and better on at least one metric that has real-world
   utility.** Operational form, to be fixed before any run: *parity* = within 5 points absolute of the best matched baseline at convergence, at equal state, tuning
   budget and aid, on every task; *better on one metric* = the metric is chosen **from a list written in advance** (sample efficiency: >= 2x fewer tokens or steps to a
   stated threshold than the best baseline; length extrapolation beyond the trained length; cue-corruption or distractor robustness; state size or per-step cost at equal
   accuracy) and the margin is at least 2 pooled standard errors **and** a meaningful absolute margin (docs/11 4.5(c)). To avoid cherry-picking, report every listed
   metric, not only the one that won, and state how many were tried. If parity holds and no listed metric is won, SSP is a working memory with no demonstrated reason to prefer it.
3. **RL budget (assistant's proposal, to be confirmed against measured throughput):** first *measure steps per second and cost per run* and report an estimate to the
   user before launching anything. Then a **pilot** of 1M steps x 3 seeds x 4 arms on `RepeatPrevious` easy, and only if SSP is at least learning comparably, a
   **confirmatory** run of 5 seeds to at most 5M steps on easy and medium. A hard cap on total pod-hours is set before launching and any run past it is stopped. The numbers
   are starting values, not measured costs.
4. **The spatial recall task is binding for the robotics direction only.** Run the supervised version first (cheap). If SSP with relative addressing is below both the GRU and
   the delta-rule fast-weight arm by more than the parity margin, the *robotics* argument is paused and the user is told; the general memory-module track (rule 2) continues
   on the other tasks, because the success rule can be met elsewhere.
5. **Use POPGym's built-in components for everything RL, when possible, so the comparison is fair (user's rule).** Use POPGym's environments, its PPO configuration and
   training loop through `PPOConfig` with POPGym's own models (RUN_007 section 3: this works with the installed ray), its evaluation and reporting, and its own baseline
   models, including its fast-weight and other memory models if they are present [recalled, unchecked: POPGym ships GRU, LSTM, MLP, positional MLP, fast-weight and
   linear-transformer, LMU, S4D and others; check which fast-weight variant it is and whether it has the delta rule]. The SSP memory is added as a POPGym-compatible model
   (an RLlib `ModelV2`), not by a different loop. **Never compare against results from the in-repo A2C loop** (RUN_006 and RUN_007 showed that loop, not the task, explained
   the gap). The only things built by us: the SSP model class, and tasks POPGym does not have (the spatial recall task), run through the same PPO path. This resolves the
   `ModelV2` versus purpose-built-environment question for the memory-module track; a purpose-built environment is used only where POPGym has no suitable task.
6. The repo question (move the folder?) stays deferred until a learned arm passes the success rule.

**Standing instruction:** nothing is committed; do not run `git add` or `git commit`.


---

## 15. Decision log (appended 2026-10-12, after `runs/RUN_012_practitioner_baseline_and_candidates.md`)

Executes handoff 11 steps 0a-0c and the candidate axis. Full tables, pass rules and traps are in the run record; this
is the judgment. All numbers are pods, 10 seeds, one machine class per table.

- **The `fwp` control was not the incumbent, and the correction changes conclusions (0a).** `FWPArm` was plain linear
  attention: `M += v kᵀ`, one learned scalar decay, *untied* key/query heads, and a delay query keyed on the lag
  alone (which cannot match a key keyed on `t`). The practitioner's form is the delta rule `M += beta (v - M k̂) k̂ᵀ`
  with a tied encoder and a read at the wanted item's own time; it is now built (`fwp_delta`) and gated.
  Under the corrected form: **`delay` is confirmed substitutable** (plain 0.993 / delta 0.998 at 10/10 seeds versus
  the SSP recipe's 1.000 — the old "`fwp` loses on delay" was the read-address convention, not the memory), and
  **on index at equal state the fast-weight family is far behind** (0.726 / 0.744 against the recipe's 0.957 at
  T=32, with 0 of 10 seeds at 0.9 against 10 of 10).
- **The address dead zone is not SSP-specific (0c).** Both fast-weight arms degrade monotonically with a read-address
  offset (0.726/0.744 at offset 0; 0.397/0.321 at offset 1). Discovery-by-address is a property of any key-addressed
  memory, so the RUN_011 mechanism generalises beyond SSP -- and the SSP arm's index advantage cannot be explained by
  "the others were not given a good address", because these were given a tied, absolute one.
- **Tied addressing removes the prior entirely, at a price (step 1, axis 2 item 1).** With the read address produced
  by the *same weights* as the write phase, the arm reaches **0.917 with 9 of 10 seeds at 0.9 from a random start**
  (no `eps`, no offset, no range constant), where the same arm with a separate read head scores 0.622 with 1 of 10.
  The cost is real and must travel with the claim: 0/10 seeds at 0.95 against the prior's 9/10, and ~19x the tokens
  to reach 0.90 (970k against 52k). Tied *with* the prior is the best cell (0.960, 10/10 at 0.9, 8/10 at 0.95,
  51.4k), so prior and tying are complementary, not alternatives. **This is the first configuration in the project
  that works with no task-dependent constant at all** -- the bitter-lesson axis, taken literally.
- **Tying only works where the read address coincides with a write address.** On `delay` the tied arm scores 0.352
  against the recipe's 1.000, because the read is at `t - lag`, an address no write used. The natural repair is
  relative addressing through the shared function (evaluate it at the wanted item's own time); that is measured
  separately (RUN_012 section 5).
- **The dead-zone-free kernel does not pay (axis 2 item 2).** A Cauchy spectrum gives `exp(-|d|/ls)` -- strictly
  positive, monotone, no zero (verified at 4096 dims against the historical kernel's 0.006/0.012 at `d = 1, 2`) --
  and it is *worse*: 0.882 with the prior (4/10 seeds) and 0.732 without, against 0.957 and 0.622. Removing the dead
  zone buys a better start and loses resolution, exactly as the risk note predicted. The RUN_011 diagnosis stands
  (the offset is the obstacle) but "make the kernel smooth" is not the fix.
- **The delta write is initially unoptimisable and needs a near-plain gate init.** At full write strength from step 0
  it does not train on index (0.144 -> 0.085 in 150 steps); with `sigmoid(-4)` it trains (0.142 -> 0.986). Every
  delta-write cell in the pod grids uses the near-plain init; on index at T=32 it then matches the plain bundle
  exactly (0.957), so the capacity axis is only interesting at the capacity limit (T=64 at 512 floats, RUN_012 8).
- **A content-keyed family now exists (`assoc`), because `index` and `delay` are both positional.** On it, the VSA
  bundle with fixed codes and its matched fast-weight arm both solve the task in a local check (1.000 each in 400
  steps) while the untuned GRU/LRU sit at chance -- and the *additive* ablation (bundle the value without the key)
  stays at 0.232 against a 0.250 chance, i.e. the binding is load-bearing. The tuned protocol on pods is in
  RUN_012 section 4; treat the local numbers as a mechanism check only.
- **Scoreboard (task-dependent constants a practitioner must set).** recipe: 4 (offset 0, `eps`, frequencies, ramp).
  tied: 0. tied + prior: 1. cauchy: 3 (no prior, but the kernel width and clip). `fwp_delta`: 0 addressing
  constants (but it learns its encoder, and pays for it on index).
- **Carry into the next round:** (1) the relative-addressing form of tying on delay and on the spatial task;
  (2) the K-head rescue cells; (3) index T=64 at 512 floats for the delta write; (4) the content family at matched
  tuning; (5) the RL track. Items (1)-(4) are in RUN_012; the RL track is `handoffs/11` section 3.2.

  **Addendum (the completed candidate scoreboard, pods, 10 seeds, index T=32, budget 256).** Every candidate in
  section 14.3 was then run to completion; the table is RUN_012 section 5b. The verdicts:
  - **Real improvements: `read_heads` (K) and multi-timescale `banks`.** K=5 spread read heads *from a one-index
    offset error* — a configuration that scores 0.663 with 5 of 10 seeds at K=1 — reach **0.962 with 10/10 seeds at
    0.9, 9/10 at 0.95 and 51.3k tokens**, i.e. exactly the tuned recipe's performance at the same speed, with `K`
    instead of `eps` as the constant; offset 3 (outside the half-index spread) falls to 0.496 with 0/10, so the
    boundary is the spread itself. This is the cleanest way found so far to *replace* the hand-set prior: redundancy
    covering the healthy region rather than a prior pointing at it. `banks=4` gives **0.962 / 10-of-10 at 0.9 /
    9-of-10 at 0.95 in 25.8k tokens — 2.0x fewer tokens to 0.90 than the recipe at equal accuracy and equal state**,
    which is the pre-registered sample-efficiency metric, and `gate="input"` + `banks=2` reaches **1.000 with 10 of 10
    seeds at 0.95 in 25.8k tokens**, the best cell of the round.
  - **Removes the prior without losing much: tied addressing and gates + tied.** Tied with no prior is at parity on
    delay (0.999/0.994, 10/10 at 0.95) and costs ~3 points and ~19x the tokens to 0.90 on index; it also **degrades
    with length** (0.930 / 0.913 / 0.817 at T=32/48/64), so its price scales. `gate="input"` on top of tied is the best
    prior-free index cell (0.977, 9/10 at 0.95) at 4.4x the tokens.
  - **Neutral:** the delta write (0.957 at T=32, and 0.951 against the plain bundle's 0.952 at the T=64 capacity
    limit, so it does not explain the capacity ceiling either), `banks=2` alone, and the interpolating (±0.5) readout.
  - **Negative:** the cauchy kernel (0.882 with a prior, 0.732 without -- it trades the dead zone for resolution),
    `ls_mode="learned"` (0.947, 5/10 at 0.95, 77.3k tokens), and gates *alone* on delay without a prior (0.488 against
    the tied arm's 0.999).
  - **Positive but not a fix:** rung 2 (learned frequencies and codes) reaches 0.999 with 10/10 seeds at 0.95 at the
    recipe's speed, so learning them is harmless and slightly better on accuracy, but the arm still uses the prior.
  - **Axes are asymmetric on this family:** the architecture changes carry the wins; the "make it learnable" changes
    are mostly neutral except K heads (which is search, not learning) and rung 2.
  - **The content-keyed family (`assoc`, local runs, 10 seeds) is the strongest capability separation so far, and it
    does not favour SSP.** With no positional structure at all and equal carried state (256 floats) and equal tuning
    budget, the VSA bundle with fixed codes scores **1.000 with 10 of 10 seeds** while **GRU 0.249 and LRU 0.281 sit at
    chance** (chance 0.250); the additive ablation (same state, same readout, value code bundled *without* the key)
    also sits at chance (0.248), so the binding is load-bearing. But under four added distractor pairs the
    outer-product arm holds at 0.794 against the VSA bundle's 0.589 (0.666 with learned codes): **neither structure
    dominates** -- the SSP form wins where addresses are positional (index, and delay at parity) and loses where they
    are content-keyed, at the same state. The corruption column of that family is vacuous by construction (corrupting
    a *query key* still names a key whose value is a correct answer) and is recorded, not cited.
  - **A trap worth carrying:** splits for a content-keyed family must not change the *alphabet* (a wider value set
    changes `out_dim`, which no arm or readout supports). The first pod run of that family died this way
    (`mat1 and mat2 shapes cannot be multiplied (64x16 and 8x16)`, then an out-of-range codebook index), costing a
    50-minute grid; the distractor form replaces it.
  - **The RL port was run, and it fails on the same dead zone (POPGym, local, 3 seeds).** On `RepeatPreviousEasy`
    (random return -0.512, optimal +1.000) with POPGym's own PPO loop and models: GRU **+0.993** (reaching the 50%
    threshold at a median 361k steps, 3/3 seeds), POPGym's fast-weight model (plain, **no delta rule** -- its source
    says so) -0.396, SSP -0.163 (aux off) / -0.268 (aux on), and at 3M steps SSP is still only -0.057 / -0.097, never
    reaching +0.244. **The learned read-offset bias stayed at |bias| <= 0.011 in all six SSP runs**, which is verbatim
    the supervised failure; hand-setting the offset to the task's lag reaches +0.244 at 230k and +0.710 at 328k, and
    zeroing the memory drops every arm to about -0.50, so the model is capable and the *search* is what fails. Parity
    fails and no listed metric is won (per-step forward 1.25 ms against the GRU's 0.64 ms; state equal). The gate
    therefore holds: **no RL port is licensed yet**, and the next run is not more RL tuning but *moving the supervised
    fix (tied relative addressing) into the RL arm*.
  - **One measured caveat on the older reference:** Ray 2.59 silently drops POPGym's `sgd_minibatch_size`, so the
    0.683-at-632k GRU number from RUN_011 used a 128-step minibatch; the pilot uses `minibatch_size = 8192`.

---

## 16. Current focus and plan (appended 2026-10-13; the next agent starts here)

**Focus, restated by the user.** Find **candidates for architectural and recipe improvements that align with Sutton's
bitter lesson** (remove hand-set constants; prefer learning, search and self-supervision). Once a candidate is good
enough on *most* metrics and possibly **beats the alternatives on sample efficiency**, take it to **POPGym** so the
result rests on a real benchmark rather than our own synthetic tasks.

**Where the evidence stands after RUN_012** (tables in `runs/RUN_012`; §5b is the two-axis scoreboard):

| candidate | index T=32, pod, 10 seeds | verdict |
|---|---|---|
| recipe (incumbent, 4 constants) | 0.957, 9/10 at 0.95, 52.0k tokens | reference |
| **input-dependent gates** | **1.000, 10/10 at 0.95, 51.3k** | win on accuracy and seeds |
| **gates + multi-timescale banks(2)** | **1.000, 10/10, 25.8k** | win: 2x tokens |
| **banks(4)** | 0.962, 9/10, 25.8k | win: 2x tokens |
| **K read heads, offset-1 initialisation** | 0.962, 9/10, 51.3k (offset 3: 0.496, 0/10) | win: repairs a bad start |
| tied addressing, no prior | delay 0.999/0.994 (10/10 at 0.95); index 0.930, degrades with length | win on constants, costs index |
| gates + tied, no prior | index 0.977, 9/10 at 0.95, 231k | prior-free; fails on delay (0.488) |
| rung 2 (learned frequencies and codes) | 0.999, 10/10 | positive |
| delta write / banks(2) / interpolating readout | 0.951-0.961 | neutral |
| cauchy kernel / learned width | 0.882 / 0.947 | negative |
| content-keyed family (local) | VSA 1.000 and outer-product 1.000, 10/10; GRU 0.249, LRU 0.281 | capability separation; the no-bind ablation is at chance |

**The RL decomposition is the most useful new diagnostic** (POPGym `RepeatPreviousEasy`, local, 3 seeds, from the
delegated track; the delegate has not written its report yet, so *treat these as unverified partial numbers*):

| RL arm | final at 1M | steps to the 50% threshold |
|---|---|---|
| POPGym GRU | +0.993 +- 0.005 | 3/3, median 361k |
| SSP, untied learned address (the old form) | -0.163 +- 0.117 | 0/3 |
| SSP, **tied** addressing, no prior | **-0.133 +- 0.136** | **0/3** |
| SSP, tied + aux | -0.202 +- 0.026 | 0/3 |
| SSP, untied, scaled features | -0.116 +- 0.063 | 0/3 |
| SSP, **tied with the read offset set to the task's lag** | **+0.832 +- 0.209** | **3/3 (196k-459k)** |

Read it as a **decomposition of the discovery problem into two parts, which the supervised tasks never separated**:

- **The origin problem** (a read address with its own unknown origin) is what tying removes, and it is the *only* part
  the supervised delay task tested, because there the lag is handed to the arm as an input feature.
- **The lag-selection problem** (which offset to read at, when nothing tells the arm the lag) is separate and tying
  does *not* touch it. In `RepeatPrevious` no lag feature exists, so the arm must find the offset itself; tying alone
  leaves it at -0.133, and *setting* the offset lifts it to +0.832.

The candidate this points at, and the one to try first next: a **bank of read offsets covering the plausible lags
(1..L) with the readout selecting among them** -- selection rather than continuous search, which has no dead zone.
The supervised K-head result (K=5 rescues a one-index error to 0.962 and 9/10 at 0.95) is the same mechanism measured
in the setting where it can be checked cheaply. Full plan, files, pass rules and traps: `handoffs/11-...md`, the
"CURRENT PLAN" block.

---

## 17. Decision log (appended 2026-10-14, after `runs/RUN_013_equal_aid_and_rl_lag_bank.md`)

Executes handoff 11 steps 1-4. Full tables, pass rules and traps are in the run record; this is the judgment. Every
number is a pod row, 10 seeds.

- **The index lead is withdrawn (step 1, pre-registered).** Given the *same* two knobs the SSP arm had, the
  practitioner-form fast-weight arm (`fwp_delta+gate+banks2`) reaches **1.000 with 10 of 10 hold-out seeds at 0.95** in the
  default column and **28 of 28** in the tuned column, against `recipe+gate+banks2`'s 1.000 / 11-of-11 / 28-of-28.
  The handoff's pass rule fired exactly as written, so the claim that SSP is ahead on index is gone. **What produced
  the gap in RUN_012 was the missing aid, not the memory**: `fwp_delta` alone is 0.745 (0/11 at 0.9) and adding the
  input-dependent gate alone takes it to 1.000 (10/10 at 0.95) -- the largest single change measured on that family in
  this project. `index` is now confirmed as a harness check on both axes, not only on the SSP-specificity one.
- **What survives is speed, and it is not yet a win.** At equal state, equal aid and *equal accuracy*, SSP reaches
  0.90 in **2.16x** fewer tokens than the fast-weight arm (30.9k vs 66.7k) but 0.50 in only **1.20x** fewer (25.7k vs
  30.7k), on hold-out rows. So the >= 2x of decision 2 holds **at t90 only**, and on `index` only -- on `delay` the two
  arms tie exactly (1.000 vs 0.998, both 10/10 at 0.95, same 21.6k t50/t90). One task family and one threshold is not
  enough; see the correction log in RUN_013 section 13, which also records that the prior-free configuration is
  **worse at length extrapolation** (conv_extrap_T 0.343 vs the incumbent's 0.457).
- **The hand-set prior is gone, and this is the round's real advance (step 3).** `tied + input gate + multi-timescale
  banks(2)`, started from `init="scrambled"` with **no offset, no `eps`, no schedule and no range rule**, passes on
  **index T=32 (1.000, 10/10 at 0.95), index T=48 (1.000, 10/10), delay 1-4 (0.999, 10/10) and delay 1-8 (0.999,
  10/10)**. RUN_012's prior-free best was tied+gate, which scored **0.488 on delay (0/10)**: the multi-timescale bank
  is what repairs it. Against the incumbent's four hand-set ingredients this configuration has two (fixed frequencies,
  and `banks=2`), and neither is an assumption about where the read address starts. Equal accuracy and equal speed at
  T=32; 1.3x slower to 0.90 at T=48 (101.2k vs 77.7k).
- **The axes interact, and one-change-at-a-time hid it (step 3, crossed cells).** K5 from an offset-3 start scores
  **0.496 (0/10) alone** and **1.000 (10/10) with gates+banks**; gates alone on delay with no prior score **0.488
  (0/10)** and **0.999 (10/10) with banks**. Those are ablation contrasts; the *mechanism* was never diagnosed and the
  earlier "two dead-zone-looking failures were really missing-a-bank failures" reading is **withdrawn** (RUN_013
  section 4). K3 on
  top of the prior-free configuration gives the round's fastest index cell (28.0k tokens).
- **Distractors: direction right, level not there (step 2a).** On `delay_distract` at 800 steps the gate moves every
  memory arm the predicted way (+0.07 SSP, +0.16 tied, +0.14 fast-weight) but nothing passes (best 0.701), and the
  benefit is not SSP-specific. The clean part is the floor: **the GRU and LRU sit exactly at chance (0.127 against
  0.125)** while every key-addressed memory is at 0.48-0.70. A 3000-step re-run of the five informative cells is in
  flight, so the 800-step level is not yet a ceiling.
- **A supervised mirror of the RL condition now exists (step 2b).** `tasks_f.DelayNoLagTask` (`variant="delay_nolag"`)
  is bit-identical to `delay_distract` at `p_dist=0` with the **lag feature removed** (22 features instead of 23), so
  the arm must discover which offset to read at -- the half of the discovery problem that every supervised task so far
  handed to the arm as an input. Three new tests (8/8 in `test_tasks_f`), no leak (untrained MLP 0.123 against chance
  0.125).
- **The lag bank is built and its mechanism is verified (step 4).** `rl.SSPTiedLagBankModel` reads at every lag in a
  fixed bank (default 1..8) and mixes the reads with a learned softmax selector; state unchanged at 256 floats, and
  the selector is exactly uniform at init so no task constant enters. Five new contract tests (**13/13** in
  `test_ssp_tied`), including the claim itself: **from the uniform default, 60 selector-only steps find the right lag**
  (accuracy 0.469 -> 1.000, `lag_eff` 4.50 -> 3.98 against a target of 4). The 1M-step RL pilot is in flight.
- **Code moved rather than added**: `SSPModel.memory_core` was split so the write path lives in `_write_bundle`; the
  extraction is behaviour-preserving (both RL test modules reproduce their counts). All new supervised knobs default
  OFF and the default-path hash guard prints six `ok`.

### 17.1 Addendum (2026-10-14, later): the two axes are now complete, and the RL question is answered in part

- **Axis coverage is closed.** Axis 1 (make parts learnable): gates, banks, K-head search, pretraining, learned
  width/frequencies, the self-supervised aux -- all measured. Axis 2 (change the architecture): tied/relative
  addressing, the dead-zone-free kernel, the delta write, gates+banks, iterated reads, the interpolating readout, and
  now **content-plus-position keys** (`tasks_d.AssocMixedTask` + `arms_assoc.AssocPosSSPArm`, `jobs/p18_assoc_mixed.json`).
  No candidate from section 14.3 remains unbuilt.
- **Content-plus-position keys works, and the ablations fail mode-specifically** (pods, 10 seeds, chance 0.250): the
  full arm scores **1.000, 10/10 at 0.95** on a task where each query asks *either* by content *or* by index and only
  the ``mode`` column says which; `pos=False` scores 0.625 and `bind=False` 0.253, exactly `0.5*1.0 + 0.5*0.25`, i.e.
  each ablation loses precisely the mode it should. The GRU is at chance (0.249) and a content-only SSP arm cannot do
  it (0.634). Costs, stated: **two bindings per pair** and a second fixed codebook. Like the pointer, this is a
  *capability* result -- no matched incumbent passes -- not a win over a working baseline.
- **The distractors ceiling is a capability limit, not a budget limit** (`jobs/p15b`): at 3x the steps the incumbent
  recipe does not move (0.569 -> **0.564**) while the gated cells gain 0.04-0.10 (best 0.763). The GRU stays at exactly
  chance (0.127 against 0.125) across both budgets.
- **Lag selection is hard for *every* architecture, not just SSP** (`jobs/p17b`, the corrected `delay_nolag`): with the
  lag feature genuinely hidden, 10 seeds, 800 steps -- recipe 0.351, tied+gate+banks2 0.400, **K5 read heads 0.418
  (best)**, GRU 0.398, LRU 0.400, and *no arm extrapolates* (all ~0.125 = chance). This is the supervised mirror of
  the RL failure and it removes "the SSP arms are just weak" as the explanation for `RepeatPreviousEasy`.
- **The lag bank clears the dead zone in RL** (`rl.SSPTiedLagBankModel`, local POPGym runs): the selector moves from a
  uniform bank over lags 1..8 to a single sharp lag (`lag_eff` 4.50 -> **2.99**, `sel_entropy` 2.06 -> **0.04**) within
  ~130k steps of PPO, with no lag ever given, and its return tracks `tied_lag4` (the arm *handed* the offset) rather
  than `tied` (lag 0: -0.49). Note the convention: the env is constructed with `k=4` but its `step` compares against
  `deck[-k]` *before* dealing, so the recall lag is `k-1 = 3` (RUN_006 recorded this); the bank found 2.99.
  **It then clears both thresholds in 3 of 3 seeds: T50 at 230k and T90 at 492k**, against the GRU's 361k / 557k
  medians -- **1.57x and 1.13x** -- and finishes at **+0.958 against the GRU's +0.993, inside the 5-point parity
  margin**. That is the first RL run in this project where an SSP memory arm *learns* `RepeatPreviousEasy` (RUN_012:
  -0.133 at 1M; the arm *handed* the offset never reached T90 on one seed), with no lag constant injected. It also
  beats POPGym's own fast-weight model outright, which never learns this task at 2M steps, and its **AULC is 33%
  higher** (+0.582 vs +0.436). It is **not** a decision-2 pass: both speed ratios are short of the >= 2x the rule asks
  for, AULC is not one of the metrics listed in advance, and the per-step cost is *worse* (101 vs 264 steps/s in a
  contended run; 1.25 ms vs 0.64 ms uncontended). So the honest verdict is **parity at convergence, faster but not by
  the pre-registered factor, worse per-step cost** -- a working memory with no demonstrated reason to prefer it, which
  is nevertheless the first working RL memory arm this project has produced.
- **The `fwp`-with-equal-aid result cost the project its index claim but produced the round's main advance**: the
  prior-free `tied+gate+banks2` now passes index T=32, index T=48, delay 1-4 and delay 1-8 at 10/10 hold-out seeds >= 0.95 with
  **two** task-dependent constants where the incumbent needs four. See RUN_013 sections 2-4.

- **The spatial recall task is built and its algebra is verified** (`tasks_e.SpatialRecallTask`,
  `arms_spatial.SpatialSSPArm`, `test_tasks_e.py` 4/4): landmarks written once as current-frame positions, motion
  applied as one group action to the whole bundle, read in the current frame. The decisive check is parameter-free --
  the shifted phase vector reproduces the target to **1.4e-06** while the unshifted one is off by 2.000, so the task is
  SSP-shaped by construction and the shift is load-bearing. A 250-step local run reaches MSE **0.0003** against 0.0068
  (`shift=False`) and 0.0843 (shared roles). **But the matched-baseline comparison REVERSES once the baselines are
  tuned, and the spatial claim is withdrawn.** The first pod run (untuned, 1 config, 1500 steps) had the SSP arm at MSE
  0.096 against the GRU's 1.662 and `fwp_delta`'s 1.643 -- a factor of 17 -- which looked decisive. A trainability gate
  then showed the GRU at 0.630 against a no-memory floor of 1.314 on a *trivial* variant (one landmark, no motion), so
  the baselines were untrained rather than incapable. With 4 configs x 2 search seeds and 4000 steps
  (`jobs/p21_spatial_tuned.json`, 10 hold-out seeds): **tuned GRU 0.021, tuned SSP 0.077** -- the GRU wins by 3.7x, and
  by 13x on the 2x-step-size split (0.035 against 0.457). So **the task does not demonstrate an SSP advantage**, the
  pre-registered withdraw rule fires (tuned baseline <= 0.5), and the robotics argument under decision 4 is **not
  supported**. What survives is internal only: the task is SSP-shaped by construction (phase identity 1.4e-06 shifted
  vs 2.000 unshifted) and the group action is load-bearing *within that arm* (untuned 0.096 full vs 0.303 no-shift).
  The lesson is docs/11 4.2's own rule, learned the hard way: a baseline at the no-memory floor is not a baseline.
