# Handoff 10: make the SSP structure cheap to discover, then prepare the RL port (follows handoff 09)

> **STATUS (appended 2026-10-12): the phases below were executed; read `runs/RUN_011_discoverability_and_halting.md`
> first, and `docs/11` sections 11-13.** The state that changed:
>
> - **The frozen range rule's "4 of 5 unseen variants" is withdrawn**: RUN_010's index jobs for T=24/32/48/64 omitted
>   `--ep-len`, so they all ran at T=16.  At the real lengths the rule decays monotonically (0.984 / 0.871 / 0.833 /
>   0.775 at T=16/24/32/48).  Section 1's `index` is a harness check, not a thesis vehicle (`fwp` ties it with the same
>   aid), and the pointer results below are the live claims.
> - **A new, mechanism-derived recipe exists**: read head = time ramp with a ~3% *scale* error and **offset exactly 0**,
>   fixed frequencies, no schedule, no address-range constant (`recipes.slope_only_cfg`).  It passes every variant inside
>   the memory's capacity at >= 0.95 on >= 9/10 seeds (index T=16 0.996/10, T=24 0.976/10, T=32 0.956/9, T=48 0.973/10,
>   T=64 0.991/10 at 1024 floats, delay 1-4 1.000/10, delay 1-8 0.997/10).  Its only apparent failure, index T=64 at a
>   512-float budget, is the *task's* capacity: the exact rung-0 address scores the same 0.952 there (5/10 seeds) and both
>   reach ~0.99 at 1024 floats.  At T=16 it is 4x cheaper than the schedule (25.7k tokens against 102.6k) and it is the
>   fastest learned arm in the project.
> - **The mechanism is settled**: discoverability is limited by the *offset* of the initial read address, not its size.
>   Offset 0 passes at 9/10 seeds for scale errors of 0/3/10%; every offset >= 1 fails 0/10 and never moves; a dead-zone
>   offset is rescued only when the drift carries the error through zero inside the stream.
> - **Pointer**: rung 3 with the halting reading the cue's hop channel and the start decoder masked from that channel
>   (`mask_hop_in_start`) reaches **1.000 on held-out hops 4-8 on all 10 seeds** with no hop count given and one hop of
>   training; the per-hop aux stays required.  The generic-baseline gate (GRU on the easy variant, 15000 steps) is in
>   flight, and until it passes the pointer result is a capability demonstration, not a win.
> - **RL is still gated**, and by the same two conditions as in RUN_010 (no recipe passing every unseen variant at the
>   time that finding was made, and no amortised interface: the in-training probe auxiliary is a no-op).
> - Pod and local runs are not numerically interchangeable (RUN_011 0b): tabulate one machine class at a time.

Written 2026-10-11 after `runs/RUN_010_recipe_and_controls.md`. **Nothing is committed and the user said not to commit anything;
do not run `git add` / `git commit`.** Read in this order: this file, `runs/RUN_010_recipe_and_controls.md` (sections 2, 4, 6, 7, 9),
`docs/11-ssp-algebra-controller-variants-and-tests.md` (3.5, 3.6, 4.7, 4.8, 10), then `handoffs/09-*.md` sections 15.5-15.7 (operations and traps,
still correct). Result files: `experiments/ssp_algebra/results/` (`sweep_s1_*`, `sweep_s2_*`, `sweep_s3_*`, `sweep_s4_*`, `transfer*.json`, `diag_pointer.json`).

**The user's focus (decided, repeat it back before changing it):** get an SSP structure *plus a training recipe* working really well on the
supervised tasks (faster to learn, more data-efficient, reliable), then port it into RL as a memory component or similar. RL comes after; its
preparation (Phase 5) starts now because it does not depend on the supervised result.

## 1. Where things stand

Supervised, component-level study of an SSP memory (bind/unbind/shift/bundle) against GRU, LRU, transformers and a fast-weight control (`fwp`) at
matched carried state (256 floats unless stated), on `delay`, `index` and `pointer` (variant A `path` is untested). 10 hold-out seeds.

| measured (RUN_010) | value |
|---|---|
| Hand-wired SSP (rung 0) tokens to 0.90, index T=16, floored selection | 24.4k (GRU 196k, fwp 90k, transformer 282k) |
| **Learned** SSP (width schedule) tokens to 0.90 | **132k**: 1.49x faster than the GRU, 2.1x than the transformer, **slower than fwp**, 5.4x slower than rung 0 |
| Low data N=64 / 256, rung 0 | 0.978 / 0.996 (best hardened baseline 0.382 / 0.849) |
| Low data N=64 / 256, **learned** | **0.233** / 0.937 |
| Range-scaling recipe (start 0.5R, schedule 24R updates), unseen variants | pass index T=24/48/64 and delay 1-8 (10/10 seeds); **fail delay 1-4 (7/10)** |
| Probe pretraining of the interface, N=64 | 19/20 seeds aligned, 0.960 (control 0/20, 0.257) |
| Pretrained interface transferred to another shape (frozen, readout only) | **fails**: T=16 to 32 0.441, 8 to 16 items 0.691 (controls 0.182 / 0.198) |
| Pointer, rung 0 / rung 1 + per-hop labels / **rung 3 (no hop count)** | 0.999 / 1.000 / **0.972, held-out hops 0.361** |
| Pointer generic baselines | GRU 0.478 at 6000 updates (the only gate-passing baseline); chain-protocol equal-aid baselines all 0.145-0.183 |

**What the learned arm already is:** faster than every generic arm that converges (GRU, transformer), and better at N=256. The task now is to widen
that gap toward rung 0 and to make it hold at N=64, on unseen shapes, and without hand-built curriculum constants.

**Why the gap exists (measured unless marked):**
- Learned-arm cost is *discovery*, not optimisation (RUN_009): identity-initialised it reaches 0.90 in 50 updates; the scheduled arm needs about 257 updates.
- During discovery, accuracy follows the current kernel width's ceiling (0.83 at width 8, 0.947 at 4, 0.987 at 2, 0.997 at 1, RUN_008), so **a linear width schedule has a speed floor around 100k tokens** (inferred from the ceiling table).
- The basin of attraction is about one index at width 1 and grows roughly in proportion to the width (RUN_008).
- The N=64 collapse is the same address-learning failure with too few examples; the probe pretraining fixes it for the shape it was trained on only.
- The rung-3 extrapolation failure (0.361) is *probably* the halting head: `SSPPointerLearned` uses `nn.Embedding(max_hops, 8)` (`arms_learned.py` around line 424); with training hops 1-2 the rows for hops 3-8 are never trained. **Hypothesis, untested.**
- Batch size is fixed at 64 for every arm and never searched; the schedule is in updates, so tokens-to-0.90 depends on it. **Unmeasured.**

## 2. Targets (pre-registered; do not change after seeing data)

| metric | now (learned arm) | target |
|---|---|---|
| tokens to 0.90, index T=16, no pretraining, floored selection | 132k | <= 50k, clearly under `fwp` (90k) |
| N=64 accuracy, no pretraining | 0.233 | >= 0.9 |
| unseen variants of the frozen rule (>= 0.95 on >= 9/10 seeds each) | 4 of 5 | 5 of 5, including delay lags 1-4 |
| interface pretrained at one shape, frozen, on an unseen shape (>= 9/10 seeds at >= 0.9) | 0.44-0.69 | pass, and >= 3x the no-transfer control |
| pointer held-out hops 4-8, no given hop count | 0.361 | >= 0.9 |
| pointer at 32 and 64 nodes (budget 512) | untested | state the node count at which accuracy falls below 0.9 |

Count and report every condition tried (docs/11 4.7). Select with `--select aulc --min-conv 0.95 --search-seeds 3` for any speed claim.

## 3. What the literature says (searched 2026-10-11; incomplete, several items are snippets)

No paper found studies why gradient descent fails to find an SSP/FHRR read address, builds a validated recipe, or compares at matched state. Related:
- **Coarse-to-fine frequency scheduling is established** in neural fields: BARF (arXiv 2104.06405) and FA-BARF (arXiv 2503.12086, aimed at the slowdown such a schedule causes: our floor). The width schedule is therefore *not* a contribution; the diagnosis, the range rule and the matched evaluation are. (snippets)
- **Multi-scale periodic codes**: residue hyperdimensional computing (Kymn et al., arXiv 2311.04872, snippet) and grid-code literature; kernel-shape control in FPE: arXiv 2608.28334 (read). Precedent exists as a *representation*, not as a fix for discoverability.
- **Capacity limits multi-hop composition** in a holographic memory with 14.5k entities: arXiv 2606.24948 (read, single-author preprint). Our pointer pass is 8 nodes at D=256 and unscaled.
- **Frozen random vectors beat learned ones** when training through HRR operations (Ganesan et al. 2021, secondhand via arXiv 2609.28409): consistent with our learned-codes ablation.
- **A GRU discovers path integration and binding on its own** (arXiv 2602.03490, read): the competitor is capable.
- **Theory**: group-invariant architectures decouple the gradient dynamics and cut sample complexity for Abelian-group tasks (arXiv 2506.21797, read): motivation, not evidence for our setting.
- **RL**: learned Fourier features improve RL sample efficiency (Li & Pathak, arXiv 2112.03257, snippet); VSPG (arXiv 2608.18404) is action-side; no SSP memory as a trainable policy component was found.
Before writing any novelty claim, run a second, narrower pass (SSP in RL, the Frontiers "optimal hyperdimensional representation" paper, "learning encoding phasors with FPE").

## 4. The plan

Order: Phase 0, then Phases 1 and 2 together (both pod-bound), then 3 and 4 in parallel; Phase 5 runs in the background throughout. Every phase: new `--tag`s,
10 hold-out seeds for pass/fail claims, tests written failing-first, a new `runs/RUN_011_*.md` in the RUN_005-010 style (measured vs inferred, bugs, limits, do-not-do).

### Phase 0: cheap diagnostics that choose the design (a few hours of code)

**0.1 Batch size as a speed lever.** Run `ssp_index_learned` (recipe from `recipes.continuous_address_cfg`, index T=16), `gru`, `fwp`, `tf_full` with `--batch` in {8, 16, 32, 64},
floored AULC selection, 3 search seeds, 10 finals. No code change (`--batch` exists). Report tokens to 0.5/0.9 per arm and batch. If the learned arm is update-bound, a smaller batch lowers its token cost;
the baselines get the same sweep, so the comparison stays fair. Whatever batch is chosen for the headline must be the best per arm, not one batch for all. Pass criterion for "update-bound": learned t90 falls at least 2x at some batch with conv >= 0.99.

**0.2 Halting counter.** In `SSPPointerLearned` add `halting="counter"`: replace `hop_emb` by a fixed counter feature `k / max_hops` (and optionally a linear decode of the hop count from the cue), keep `"learned"` as is so RUN_010 stays reproducible.
Tests (`test_pointer.py`): halting weights are a distribution; with training hops (1,2) the halt distribution peaks at the queried hop for hops 4-8 (this must fail on the embedding version first). Run rung 3 + per-hop aux, 10 seeds, with the cue-drop splits (`--robust`).
Success: held-out hops >= 0.9. If still < 0.6, the cause is elsewhere (the cue decoder); do not stack fixes, diagnose with `diag_pointer.py`.

**0.3 Affine-init ceiling (diagnostic, an aid: label it so).** Add `init="affine"` to `SSPIndexLearned`: read head = ramp scaled by `1 + eps*N(0,1)`, write phase as now. Sweep eps in {0, 0.02, 0.05, 0.1, 0.2, 0.4} with a fixed width 1 and with the schedule; plot t90 and success rate against eps. This says how much prior any learned arm needs and bounds what Phase 1 can reach (identity init = 50 updates).

### Phase 1: schedule-free discoverability (the main architectural bet)

Replace the width schedule by a **multi-scale frequency set** so a wide component gives a basin from step 0 and narrow components give resolution. `ssp_ops.make_omega` currently draws `Omega ~ U(-pi, pi) / length_scale` (`kind="uniform"` or `"gauss"`).
Variants (add as `kind` / arm kwargs, keep the old behaviour as the default):
- **A `logmix`**: per-dimension scale from a log-uniform on `[ls_min, ls_max]`, `Omega = U(-pi, pi)/ls`; rule `ls_max = a*R`, `ls_min = 1`, no schedule.
- **B `residue`**: a few incommensurate periods (modular-code style), chosen so the coarse period covers `R`.
- **C band-split**: dimensions split into scale bands with learned per-band gains on the cleanup scores.
Risks to measure, not assume: neighbouring positions get confusable through the wide components (lower ceiling); band splits cut capacity per band (D/4 rule); aliasing for commensurate periods.
Tests: kernel shape (peak at 0, shoulder, no aliasing within `R`), unit modulus, determinism, the **basin diagnostic** (`diag_address.py`: final accuracy against initial address error at 5.3; a schedule-free arm must keep ~0.99 where the single-scale arm fails), gradient reaches the read head at random init.
Protocol: index T=16/32/48/64 (512 floats at 48/64), delay lags 1-4 and 1-8, pools N=64/256, 10 seeds, fixed hyperparameters for every variant, floored AULC selection, compared in the *same* runs against the schedule recipe.
**Pass:** >= 0.95 on >= 9/10 seeds on every variant including delay 1-4; tokens to 0.90 <= 50k at T=16 (at the Phase 0 best batch, for all arms); conv >= 0.99; N=64 >= 0.9 without pretraining.
**Kill / fallback:** if no variant beats the schedule on tokens to 0.90 at conv >= 0.99, try a hybrid (short schedule over a multi-scale set), report the best, and stop there. A pass also removes the schedule constants and settles the "aid or natural" question for the schedule (it is gone).

### Phase 2: make the interface amortised and cheaper

**2.1 Shape-randomised probe pretraining.** Pretrain the address on streams with random length (8-64) and alphabet size (4-32) so the interface cannot fit one shape. The interface parameters (`w_time`, `read_head`, `log_mult`) are shape-independent; the codebook and readout are not, so use a shape-agnostic pretraining arm (max alphabet, variable `T`) or tied interface parameters across per-shape arms. Add `--pretrain-shapes` to `transfer_test.py`; keep the no-transfer control (`--control`).
Pass: >= 9/10 seeds >= 0.9 on unseen shapes (T=32, 48; 16 items; a combined shape) and >= 3x the control.
**2.2 Pretraining budget curve.** N=64 accuracy against pretraining updates {50, 100, 200, 500}; report pretraining tokens separately from task tokens.
**2.3 Probe loss as an in-training auxiliary** (no separate stage): train a throwaway probe jointly and measure whether N=64 and speed improve without pretraining. This is the form RL can use, because the stream's own past observations are free labels.

### Phase 3: fix the pointer side (parallel with Phase 4)

3.1 Full evaluation of the halting counter (from 0.2), 10 seeds, cue-drop curves, aux on and off.
3.2 **1-hop-only supervision**: train hops (1,), per-hop aux on hop 1 only, test hops 2-8 with the hop count given; this checks the "iteration is free" claim.
3.3 **Self-supervised retrieval aux**: synthetic streams where a key is re-queried later and the target is its stored value (no hop labels); measure role gate and query-map alignment, then hop extrapolation by iteration.
3.4 **Scaling**: `--n-nodes` in {8, 16, 32, 64} at budgets 256/512/1024, report the node count where accuracy falls below 0.9 and compare with D/4 (arXiv 2606.24948 reports capacity as the bottleneck).
3.5 **A fair non-SSP control**: gate `fwp`, `lru`, `tf_window` on the easy pointer variant (never done); give the GRU a curriculum (the hop curriculum exists: `harness.make_curriculum("pointer_hops")`) and 12000+ updates, 20 seeds, because its easy-variant solve took about 2.5M tokens after a long plateau and a single-seed search missed it. The pointer pass stays provisional until a gate-passing iterated control is trained at equal aid.

### Phase 4: SSP-specificity (parallel)

`fwp` keys already use multi-scale Fourier features of position (`feats.abs_time_feats`), so it is a strong rival. Give it the same pretrained interface and the same aids; then test where the exact algebra should matter and a key-value memory should not: the redesigned variant A (path integration; handoff 09 section 15.4 step 6). If `fwp` ties SSP everywhere, use the better one as the RL memory and say so.

### Phase 5: RL port preparation (start now; ungated pieces first)

5.1 **Rebuild the POPGym PPO reference** (`experiments/ssp_algebra/ref_ppo/README.md`; its own venv, never install ray into the repo `.venv`) and run `RepeatPreviousEasy` with a GRU to >= 500k steps on a pod, in the background. Reference: +0.42 at 296k steps, still rising. This repairs RUN_006's comparison and is not gated.
5.2 **Module interface** (a design doc plus a stub, no RL runs yet): `write(token, t)`, `read(query)` where the read address comes from the policy's hidden state plus time features (a *new* design question: in the supervised tasks it comes from a given index feature), `aux_loss()` (label-free recall of past observations), `freeze_interface()`. Discrete observations first; continuous observations need an item encoder (out of scope, never claimed).
5.3 **First RL test after the gate below**: `RepeatPreviousEasy`, then `CountRecallEasy`. SSP memory with a frozen pretrained interface + policy, versus a GRU given the *same* auxiliary loss, versus a plain GRU; steps to a fixed return threshold at 100k and 300k, >= 10 seeds. RUN_006 showed a perfect readout did not make RL learn CountRecall in 2000 updates, so credit assignment is a separate bottleneck: expect RepeatPrevious to be the informative one.
Kill: if a frozen, aligned memory plus policy cannot beat the GRU-with-aux on sample efficiency, stay supervised and report it.

## 5. Gate before any RL experiment (docs/11 4.8, restated)

All of: (1) Phase 1 (or its fallback) reaches tokens to 0.90 <= 50k with the recipe passing all unseen variants; (2) the interface is amortised (Phase 2.1) **or** the in-loop auxiliary (2.3) keeps it aligned without a pretraining stage; (3) the pointer scaling test (3.4) is run and reported; (4) speed claims use floored AULC selection. A hand-wired win never licenses RL. If a gate fails, RL results would not be interpretable.

## 6. Operations (unchanged from handoff 09 section 15.5, plus RUN_010 lessons)

- Run from `experiments/ssp_algebra/` as `OMP_NUM_THREADS=1 ../../.venv/bin/python <script>`; repo `.venv` (Python 3.14.6, torch 2.14.0, numpy 2.5.3). A `sys.prefix` RuntimeWarning is harmless.
- Tests, all green at the end of RUN_010: `test_harness` 6, `test_fixes` 2, `test_measurement` 6, `test_learned` 12, `test_analysis` 9, `test_regularisation` 7, `test_select` 10, `test_recipes` 9, `test_pretrain` 7, `test_pointer` 14, `test_trainability`, `selftest`. Never loosen a threshold; add new checks in new functions/files and show them red first.
- Pods: write a job list (`experiments/ssp_algebra/jobs/*.json`) and run `python scripts/ssp_pod_batch.py launch|status|collect <jobs.json>` (use `launch --dry-run` first). One 4-vCPU CPU pod per job, results resumable on R2, pod self-deletes on success. **Declared `rows` must equal `len(arms) * (configs * search_seeds + seeds)`**; a wrong count makes `status` report a finished job as incomplete and `collect` refuse it (RUN_010 7.5b; fetch from R2 directly if that happens). Only pods named `ssp-*` are ours; the account also has `eval-v3-*` pods that must not be touched. Confirm none of ours remain.
- `--arm-cfg` JSON must have no spaces; every protocol gets a fresh `--tag` (the resume key omits budget, pool, steps, robust, ep-len, batch); do not edit `arms*.py`, `tasks_*.py` or `harness.py` while a local sweep runs.
- Sweep flags added in RUN_010: `--search-seeds K`, `--select {last5,aulc}`, `--min-conv`, `--chain`, `--n-nodes`, `--n-query`, `--train-hops`, `--max-hops`, `--train-lags`; plus `--pool`, `--robust`, `--ep-len`, `--batch`, `--dense-eval`, `--label`, `--arm-cfg`.
- Read results with `summarize_runs.py <files>`; for a quick table use the per-row `summary` keys `conv_*`, `aulc_*`, `t50_*`, `t90_*`. Do not pass files that reuse an arm label under a different protocol into `analysis.py`.

## 7. Code map additions from RUN_010 (all under `experiments/ssp_algebra/`)

`recipes.py` (`address_range`, `continuous_address_cfg`, `job_entry`; frozen rule constants), `transfer_test.py` (two-stage transfer, `--control`, `--smoke`), `diag_pointer.py`,
`arms.py` (`LoopedTFArm`, `looped_tf` in `ALL_ARMS`, `dropout` threaded through the baselines), `arms_learned.py` (`SSPIndexLearned.memory_features`/`load_address_params`; `SSPPointerLearned` `cue` and `halting`, rung 3; `FWPArm` iterated pointer read),
`harness.py` (`pretrain_mode` retrieval/probe/full, `weight_decay`, `ans_mask` accuracy), `tasks_b.py` (`chain=True`, `ans_mask`), `run_sweep.py` (flags above, `pick_winners(select, min_conv)`).

## 8. Traps (new ones from RUN_010; the older ones in handoff 09 15.7 still apply)

1. `--select aulc` without `--min-conv` picks configs that give up accuracy (the GRU at 0.857): always floor speed selection.
2. The learned arm and the width schedule count *updates*; compare tokens only at a stated batch size.
3. Chain-protocol numbers are only comparable with each other, and SSP pointer arms cannot run on a chain batch.
4. A single search seed can miss a config that solves a task after a long plateau (the GRU's easy pointer variant): use >= 3 search seeds and long budgets for baselines before saying "cannot learn".
5. `test_pointer.py::test_cue_fix...` pins that the retrieval (not just the output) responds to the cue; do not weaken it.
6. `build_learned` returns `(arm, info)`: two of my tests failed on that.
7. Do not retune the frozen range rule on the delay 1-4 variant; test any new rule on fresh unseen variants.

## 9. Open decisions for the user (ask; do not assume)

1. Is `index` a thesis vehicle or only a harness check? (It is largely substitutable: `fwp`, a position table and learned codes match much of the advantage.) Working assumption: a fast testbed for architecture changes, with claims made on pointer and path integration.
2. If Phase 1 fails and the width schedule stays: is it a costed aid (design cost counted, baselines given an equivalent aid) or ordinary training? Working assumption: costed aid carried by the range rule.
3. Keep the thesis-pass thresholds (5 points, half the tokens, 2 pooled se; 0.95 on 9 of 10 seeds)? Working assumption: yes.
4. RL design (docs handoff 09 section 11): port into POPGym's `ModelV2` first (baseline-faithful) versus a purpose-built pointer environment. Working assumption: `ModelV2` first.
5. Where this folder lives long-term (it is a different thesis from the repo's SSP-as-encoder work). **No commit now.**

## 10. Do not do

- Do not present the coarse-to-fine width schedule as novel, and do not quote the pretraining stage as amortised.
- Do not quote the unfloored speed table, "about 4x", or 5.7x; the claim is 8.0x for rung 0 and 1.49x for the learned arm (floored).
- Do not cite any pointer baseline other than the GRU as competent (`tf_full`, `kernel_attn`, `looped_tf` fail the gate; others were never gated).
- Do not start an RL or larger-model study before section 5 holds, and do not read a hand-wired win as licence for RL.
- Do not change the pre-registered targets in section 2 after seeing data, and do not retune on unseen variants.
