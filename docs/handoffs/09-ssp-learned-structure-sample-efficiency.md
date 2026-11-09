# Handoff 09: SSP learned-structure and sample-efficiency study (supersedes handoff 08)

> **START HERE (latest update 2026-10-11): read section 15 first.** It is the current state, the gating rules and the step-by-step plan for the next agent.
> Sections 0-14 are the history (what was done, in order, with the reasoning at the time); section 15 supersedes any plan in them. Result records: `runs/RUN_008_*.md`
> (learned structure, decision), `runs/RUN_009_recipe_diagnostics.md` (diagnostics, conclusions firm/suggestive/not). Design and decision rules: `docs/11-*.md` (3.5, 4.5, 4.7, 4.8).
> **Superseded for planning (2026-10-11): steps 1-4 of section 15 were executed in `runs/RUN_010_recipe_and_controls.md`; the continuing plan, targets and gate are in `docs/handoffs/10-ssp-discoverability-and-rl-port-prep.md`. Sections 15.5-15.7 (operations and traps) here are still valid.**

Written 2026-10-11 (this part is history; the current state is in section 15). Nothing is committed or pushed. *(Originally this said no RunPod pod had been used. Since then
many CPU pods were used for sweeps and diagnostics; all of ours are deleted; see 15.5.)* Read in this order: this file, then
`runs/RUN_007_ssp_algebra_controller.md` (sections 0, 4.5, 6, 7 first), then
`docs/11-ssp-algebra-controller-variants-and-tests.md` (plan, metrics, decision rules in 3.4 / 4.3 / 4.5 / 4.6), then
`experiments/ssp_algebra/README.md`. Handoff 08 is superseded by this file.

---

## 0. Status update (added after the first version of this handoff; read before section 1)

No coding is in progress and nothing is running. This section records what changed on disk, a new finding that
invalidates part of RUN_007, and the detailed plan to execute. Sections 1-12 below are still correct except where this
section says otherwise.

### 0.1 Code changes already made (uncommitted)

- `harness.py`: added `import zlib` and `stable_seed(*parts)` (crc32-based); `search` now seeds the config draw with it.
- `run_sweep.py`: `build_search_jobs` now seeds with `harness.stable_seed(...)`. This is the fix for the salted-`hash()`
  bug (section 5, item 8). **Not yet re-verified by running the test** after the change.
- New `test_fixes.py` with two checks. On the original code both were **RED**: `seed_stability` (config draws differ
  across `PYTHONHASHSEED` values) and `pointer_oracle_readout` (see 0.2). `seed_stability` should now pass; re-run it.
- `test_harness.py` and `selftest.py` are untouched and must stay untouched and passing (6 gates). New checks go in
  new functions/files, never by loosening an existing one.
- No result file was overwritten. No jobs running. No pods were ever launched.

### 0.2 New finding: the rung-0 pointer arm (`SSPPointerArm`) is broken, so RUN_007 section 4.3 is invalid

- **Oracle test** (`test_fixes.py::test_pointer_oracle_readout`): exact codes, `decay=False`, `dim=256`, readout
  replaced by a pass-through of the cleanup scores. Accuracy by hop count: h1 0.15, h2 0.13, h3 0.20, h4 0.24,
  h5 0.27, h6 0.24, h7 0.25, h8 0.25 (chance 0.125). A correct arm must be >= 0.99 at every hop.
- **Diagnostic** (128 episodes per hop count, same arm, accuracy of the arm's argmax against f^(h+d)(start), computed
  from the batch's `perms`; the task's own target was confirmed equal to f^h in 100% of cases, so the task is right):

  | hop h | vs f^(h-1) | vs f^h (the target) | vs f^(h+1) | vs f^(h+2) |
  |---|---|---|---|---|
  | 1 | 0.77 | 0.14 | 0.39 | 0.22 |
  | 2 | 0.60 | 0.18 | 0.37 | 0.21 |
  | 3 | 0.44 | 0.27 | 0.41 | 0.26 |
  | 4 | 0.30 | 0.22 | 0.42 | 0.23 |
  | 5 | 0.33 | 0.23 | 0.44 | 0.22 |
  | 6 | 0.41 | 0.26 | 0.49 | 0.26 |

  The arm's output is strongly tied to the *neighbours* of the target (h-1 and h+1) and near chance on the target.
- **Hypotheses for the cause (inferred, NOT yet fixed or verified):**
  1. **Commutative binding.** A stored pair is `code(key) * code(value)`, which is symmetric, so unbinding a code from
     the bundle retrieves both its successor and its predecessor with equal strength; every hop is
     direction-ambiguous. Fix to try: asymmetric binding, e.g. store `code(key) * (R * code(value))` with a fixed random
     unit phasor `R` (or separate role vectors), and undo `R` on read.
  2. **One extra unbind after the hop loop.** After `max_hops` iterations the code `c` already represents f^h(s), but
     the code then unbinds once more and scores that, i.e. f^(h+1) (and, by hypothesis 1, f^(h-1)). Fix to try: carry the
     cleanup scores of the last executed hop out of the loop and feed those to the readout.
  Both are consistent with the table. Fix, then re-run the oracle test (needs >= 0.99 at hops 1-8) and the diagnostic;
  if it still fails, the cause is something else and must be diagnosed, not assumed.
- **Consequences:** every SSP-vs-baseline pointer comparison in RUN_007 (SSP leading early at 0.270 vs 0.145, leading
  on held-out hops 0.240 vs 0.151, GRU ahead at convergence 0.359 vs 0.312) came from a broken arm plus a learned
  readout and **must not be cited**. The baseline arms' own pointer numbers are valid. RUN_007 has not yet been
  amended for this: append a note (do not rewrite). The index and delay rung-0 arms were verified (pinned readouts
  1.000), so their upper bounds are real.
- **For the learned rung:** `SSPPointerLearned` must also use asymmetric binding, otherwise it inherits the ambiguity.

### 0.3 Detailed execution plan (do in order; do not start a step before its predecessor's verification passes)

**Ground rules.** Existing tests stay untouched. A new test is shown failing on the broken code, then passing after the
fix. No code edits while a sweep runs (workers re-import modules each task; only `analysis.py` and test files are safe
to edit mid-sweep). Never overwrite old results: every run gets a new `--tag` (`r0fix`, `learn1`, `rob1`). Run from
`experiments/ssp_algebra/` as `OMP_NUM_THREADS=1 ../../.venv/bin/python <script>`. Local only; delete any pod you
create and confirm with `runpodctl pod list`.

**A. Correctness bugs first.**
  - A1. Re-run `test_fixes.py::test_seed_stability` (should now pass).
  - A2. Fix the pointer arm per 0.2 and verify with `test_pointer_oracle_readout` (>= 0.99 at hops 1-8) plus the
    diagnostic table (accuracy against f^h should be ~1, neighbours ~0).
  - Verify: both new tests pass; the 6 existing gates in `test_harness.py` still pass unmodified.

**B. Fair inputs and better measurement.**
  - New `feats.py`: `abs_time_feats(t, periods=(2,4,8,16,32,64))` returning `t/32` plus sin/cos (13 dims), with fixed
    constants independent of `T`.
  - `tasks_c.py`: `DelayTask`/`IndexTask` features append the absolute-time features; query normalisation becomes a fixed
    constant `NORM=16` instead of `/T`; `IndexTask.batch` adds an absolute `index` key; update `in_dim`. (A transformer
    with relative-only RoPE cannot recover absolute index recall, so part of `tf_full`'s 0.37 may be representational,
    not undertraining.)
  - `tasks_b.py`: append absolute-time features (include period 2 so parity is discoverable); add an `inter` key
    (B, nq, max_hops) with the true intermediate nodes (for per-hop supervision); add `obs_noise` that corrupts input
    tokens only (targets stay from clean data); make `cue_drop` selectable by split.
  - `tasks_a.py`: replace `t/(T-1)` with a fixed-constant normalisation.
  - `arms.py`: `MLPArm` clock uses the absolute-time features (no `t/T`); `SSPArm.read` index branch reads at
    `batch["index"]` (clipped to `t`) instead of `query*T`.
  - `harness.py`: `eval_schedule(steps)` (every 25 updates up to 400, then every 100); `train_arm(..., eval_steps=None,
    pool=None)` where `pool=N` pre-generates N sequences (data seed `5000+seed`), samples minibatches from that pool
    with a `_take(batch, idx)` helper, and evaluates on fresh data; `summarize` ADDS `conv_*` (mean of last 3
    checkpoints), `aulc_*`, and `t50_*`/`t90_*` (tokens to 50%/90% of ceiling) while keeping legacy `last5_*` so old
    gates are unmodified.
  - `run_sweep.py`: flags `--dense-eval`, `--pool N`, per-budget `--budget`.
  - Verify (new tests): dense eval yields >= 25 checkpoints; pool mode is deterministic; lag/index features are identical
    for the same `(j, lag)` at `T=16` vs `T=64`; the 6 existing gates pass unmodified.

**C. Baseline trainability gate.** New test: on an easy index task (`ep_len=8`, 800 steps) `gru` and `tf_full` must reach
  >= 0.95 in-distribution. If `tf_full` fails, fix its optimisation in `TFArm` (learning rate, warmup, `d_model`,
  depth), never the threshold; if it still fails, record `tf_full` as untrained and exclude it from every claim.

**D. Re-run rung 0 with the fixes (tag `r0fix`) and apply the stop rule.**
  - Command (background): `run_sweep.py --mode sweep --variants delay index pointer --tag r0fix --budget 256
    --dense-eval --configs 6 --seeds 5 --steps 1500 --workers 10`; then index at `--budget 128` and `--budget 512`
    (light protocol: 4 configs, 3 seeds, stated as such); then low-data index with `--pool` in {64, 256, 1024, 4096}.
  - Verify: JSONL row count equals the number of jobs; every row's `curve` has >= 25 checkpoints; `analysis.py` prints
    converged, speed and AULC separately.
  - **Stop rule (doc 11 4.6):** if hand-wired SSP shows no advantage over *passing* baselines in speed, low-data accuracy
    or fair-encoded extrapolation, stop and report. A learned rung cannot beat its own upper bound. Also check that the
    advantage is not shared by the `fwp` control and the random position table (then it is not SSP-specific).

**E. Learned structure on index (new `arms_learned.py`).**
  - `SSPIndexLearned(task, dim, rung, init)`: rung 1: write phase `Omega * (a_w * t_feat)` and read phase
    `Omega * MLP(idx_feats)` from learned heads; rung 2: `Omega` and codes become `nn.Parameter`s (codes re-normalised
    to unit modulus); `init="identity"` sets the heads to reproduce rung 0 (an aid).
  - `FWPArm`: outer-product memory (state `d*d`, d=16 at 256 floats), write keys/values and read keys from learned heads,
    leak `lam`; for pointer it takes the same hop count and iterates its read. This is the non-SSP control.
  - Harness hooks (training aids): curriculum via an index-range cap `jmax` widened over training; `aux_weight` calling
    `arm.aux_loss(batch)` if present; temperature annealing via `arm.set_temp(step)`.
  - Verify (new tests): determinism; smoke on index/delay; non-zero gradients on `Omega` and codes at rung 2; and
    `rung=1, init="identity"` reaching >= 0.95 on index after a short readout-only training (proves rung 1 can express
    rung 0; it is not a thesis result).

**F. Learned structure on pointer (only after A2 and a competence gate).** First the corrected rung-0 pointer must reach
  >= 0.8 at a longer budget (3000 updates, `n_nodes=6`); if it cannot, defer pointer and report that. Then
  `SSPPointerLearned` rung 1: learned role gate `g_t = sigmoid(MLP(feat_t))` (key vs value), soft key update
  `cur_key = g*code + (1-g)*cur_key`, soft asymmetric bind into the bundle, learned query code via softmax over the
  codebook; hop count still given. Aids: per-hop auxiliary cross-entropy against `inter`, hop curriculum. Verify that an
  oracle-initialised version reproduces rung 0, then run plain vs aided (tag `learn1`).

**G. Robustness and generalisation (tag `rob1`)** on survivors: `extrap_T` at 64 and 128 against the D/4 capacity with
  and without forgetting; pointer `cue_drop` in {0.25, 0.5, 0.75}; `obs_noise` in {0.1, 0.25}; held-out hops. Verify the
  split keys appear in the rows and that corruption alters only inputs, never targets.

**H. Rewrite `analysis.py` per doc 11 4.5 plus `test_analysis.py`** using `conv`, `t50`/`t90`, low-data and fair-encoded
  extrapolation, with fragility checked across rungs and splits and ties going to "buys nothing". Verify with three
  synthetic scenarios (no real data): equal convergence but 3x faster -> `thesis_pass_candidate`; a win only at rung 0 ->
  `structural_prior_only`; equal everywhere -> `algebra_buys_nothing`.

**I. Deferred/optional:** redesign variant A (section 6); re-run the POPGym PPO GRU to >= 500k steps (rebuild
  `/tmp/popgym_ref` from `ref_ppo/README.md`; 4 workers, background); RL; write `runs/RUN_008_*.md` and update docs.

**Risks to watch.** Feature changes touch every arm, so old results are not comparable (hence new tags); if
`test_oracle_delta_readout` or the LRU gate breaks, the code is wrong, do not loosen it. A learned read phase may never
lock on from random init (the sinc kernel is informative only within about half a kernel width): that is the actual
learnability question, so failure is a result to report alongside the aids. Complex autograd through a learnable
`Omega` can produce NaNs: check and clip gradients. GRU at `H=512` costs ~1000 s per run, so the budget curve uses the
light protocol. Hold-out seeds 1000-1004 are never used for selection; evaluation data for seed `s` is
`default_rng(10000+s)`, so arms are paired for free.

**Stop conditions.** Rung 0 shows no advantage after the fixes: report and stop (no learned rungs, no RL). No learned
rung within 5 points of rung 0 even with aids: report "structure present but not discoverable". No RL until a learned
arm passes the thesis rule in section 8.

**Todo state at hand-off:** none of A-I is complete; A is partly done (seeding fix applied, pointer arm diagnosed but
not fixed).

---

## 1. Hypothesis

**SSP gives a learner structure it can interact with**: bind, unbind, shift (a group action on the whole memory),
bundle, and cleanup, all exact (T0 parity 4e-15) and differentiable. The hypothesis is that a learner whose
*controller learns what to feed those operations* (write addresses, read addresses/queries, key/value roles, shifts,
when to stop iterating) will, relative to generic sequence models at the same carried-state budget:

- **H-a, sample efficiency:** reach a given performance with fewer training samples/updates, and do better when only
  a small fixed training set is available.
- **H-b, shorter episodes / extrapolation:** train on short streams, short lags, few hops, small index ranges and
  generalise to longer ones.
- **H-c, robustness:** degrade more gracefully under cue corruption, distractors, observation noise and streams much
  longer than the capacity rule D/4.
- **H-d, competitiveness:** stay within a stated tolerance of the best generic arm *at convergence*.

**Explicit non-claim:** SSP is not expected to beat a transformer or GRU given unlimited samples and unlimited
state; those can model nearly anything. The claim is about the cost of getting there and the failure modes.

**The competing explanation that must be ruled out:** the benefit is merely "an addressable key-value memory with a
given index", which any fast-weight / linear-attention / outer-product memory also has, and is not specific to SSP.
RUN_007 already found the SSP *time parameterisation* is not what wins on index recall (random position table:
0.998 vs 0.999). So the study needs an explicit non-SSP associative-memory control (section 6, arm `fwp`).

## 2. Where we are (the honest state)

RUN_007 ran everything in a **hand-wired** form ("rung 0"): I chose the write address, read address, key/value roles,
hop loop and pose integration, and only a small readout (plus forgetting rate, plus for `delay` a lag-to-offset map)
was learned. So it is an **upper bound / existence proof, not a test of the thesis**.

What rung 0 showed (supervised, 256-float state budget, 5 hold-out seeds, 1500 updates; converged = step 1500):

- **Index-addressed recall:** SSP and GRU tie at convergence (both 0.999). SSP is far faster early (>=0.998 at 300
  updates; GRU 0.786, first >=0.9 at 600). Extrapolation to a 4x longer stream: SSP 0.803 vs ~0.16 for all others,
  but **confounded** (baselines get `j/T`, the SSP arm multiplies by `T`). The SSP *time code* is not load-bearing.
- **Pointer chasing: INVALID, see 0.2.** The rung-0 pointer arm cannot compute f^h(s) even with exact codes (its
  oracle test is at chance), so the earlier SSP numbers (0.270 vs 0.145 early, 0.240 vs 0.151 on held-out hops, GRU
  0.359 vs 0.312 converged) measure a broken arm plus a learned readout and must not be cited. The baseline arms' own
  pointer numbers are valid; none was competent (chance 0.125).
- **Delay:** SSP converges to 1.000 but is slower than GRU/LRU (>=0.9 at 600 vs 300 updates); substitutable as
  pre-registered.
- **Landmark return (variant A proxy):** uninformative (no-memory floor MSE 0.024 vs best 0.013).
- **Baselines undertrained:** `tf_full` reached only 0.37 on index and 0.54 on delay (tasks it can represent
  trivially). There is no valid "upper bound" baseline yet.
- Earlier `proceed_to_rl` verdicts for index and pointer are **withdrawn** (they came from a metric that blended
  speed and convergence).
- **POPGym reference (step 1 of old handoff 08):** POPGym's own PPO + GRU reaches +0.42 on `RepeatPreviousEasy` at
  296k steps (still rising); our in-repo A2C GRU was -0.108 at 490k. So RUN_006's "vsa_delay +0.990 vs GRU -0.108"
  compared against a badly trained GRU. `CountRecallEasy` is flat at chance at ~100k steps under POPGym's loop
  (partial; their budget goes to 15e6).

## 3. What exactly was hand-wired (this is what the new work removes)

Common to every SSP arm: `Omega` fixed (uniform random, a buffer; only `length_scale` is searched), item codes fixed
random unit-modulus (learnable only in one ablation, variant C), and the sequence of operations fixed:
write `M <- lam*M + code * pos(t)`, read by unbinding, cleanup by cosine scores against the codebook.

| variant | hand-wired in RUN_007 (file: `arms.py`) | what was learned |
|---|---|---|
| C delay (`SSPArm`) | lag supplied as input feature; write address `pos(t)` fixed | `delta = softplus(linear(features))` (learned lag-to-offset, corr 0.999), readout MLP, `lam` |
| C index (`SSPArm`, `is_index`) | read address `min(query*T, t)` computed from the supplied `j/T`; write address fixed | readout MLP, `lam` |
| B pointer (`SSPPointerArm`) | key/value roles by step parity (`t % 2`); start key and hop count supplied; loop count from the hop count; query steps = last `n_query` positions | readout MLP, `lam` |
| A path (`SSPPathArm`) | `phi(action)` with fixed `Omega` (exact path integration given); queried landmark chosen by `argmax` of a supplied one-hot | readout MLP, `lam` |

The generic baselines saw the same inputs but had to learn all of the above from scratch, so the comparison so far
is "correct algorithm given" vs "must find the algorithm".

## 4. What this study does: the learnability ladder (details in doc 11 section 3.4)

- **Rung 0:** existing hand-wired arms (reference upper bound, keep them).
- **Rung 1:** the controller emits write address, read address/query, key/value role, shift (action-to-phase map)
  from learned heads. `Omega` and codes still fixed. The product that composes phases and the conjugate that
  unbinds stay fixed in every rung (the claim is the controller learns what to feed them).
- **Rung 2:** also learn the codes and `Omega` (initialised from the SSP prior; ablate against random init).
- **Rung 3:** remove task-visible hints: no hop count (learn to halt), query selection learned rather than given.

Rung 1 replacement table:

| variant | rung 1 |
|---|---|
| C delay | learned write phase scale; offset head from content |
| C index | learned map from a raw absolute-index feature to read phase; learned write phase |
| B pointer | learned role gate from token + position features; learned query code; hop count still given (rung 3 removes it) |
| A path | learned action-to-phase map; learned query code (soft selection over stored landmark codes) |

**Training aids** (apply only where plain training fails; report plain vs aided and count the aid's own sample cost):
curriculum (short lags / 1 hop first); intermediate supervision (per-hop cleanup targets for B, pose targets for A,
auxiliary address losses); near-identity initialisation of address heads; cleanup-temperature annealing; longer
training and larger batches.

**Interpretation:** a rung-1/2 arm close to rung 0 with fewer samples than the generic arms *is* the thesis. A rung
that falls to GRU level means the structure is present but not discoverable by gradient descent, and the question
becomes which aid closes the gap. If none does, that is the finding.

## 5. Measurement fixes that must come first (cheap, local)

These came out of a review of RUN_007 and invalidate several numbers if skipped.

1. **Stop averaging checkpoints and calling it accuracy.** `harness.summarize` reports `last5_*` = mean of the last 5
   of only 6 checkpoints (steps 300-1500), which is a short learning-curve area; it made the GRU look 0.948 on index
   when it converges to 0.999. Report converged score, tokens-to-50%/90%-of-ceiling, and area-under-curve as three
   separately labelled quantities.
2. **Fine evaluation cadence.** Currently every 300 updates, so SSP on index is already at ceiling at the first
   checkpoint (its true time-to-threshold is unmeasured). Evaluate every ~25 updates early, sparser later.
3. **Fair absolute index encoding for every arm.** `IndexTask.batch` sets `query = j/T`; the SSP arm multiplies by
   `T` internally. Give *all* arms the same raw absolute index features (e.g. `j` scaled by a fixed constant not
   tied to `T`, plus sinusoidal features), and keep the SSP rung-0 read from the same feature. Same for `posmlp`'s
   `t/T` clock (it breaks under longer episodes by construction).
4. **Baseline trainability gates** in `test_harness.py`: each baseline must reach >=0.95 of ceiling on an easy,
   trivially representable version before its number is used. `tf_full` currently fails this on index recall; raise
   its budget/steps or fix its optimisation (lr, warmup, d_model) until it passes. Do not claim anything against a
   baseline that failed its gate.
5. **State-budget curve** at 128, 256, 512 floats (the current single 256 point cannot show "the GRU catches up with
   more state"). `build_arm(budget=...)` already supports it.
6. **Low-data regime:** add a mode in `harness.train_arm` that trains on a fixed finite pool of N sequences
   (N = 64, 256, 1024, 4096), cycling epochs, and tests on fresh sequences. This is different from "few updates".
7. **Fix `analysis.py`'s verdict logic:** its margin test used `last5`; its fragility check only compared ablations
   on the `id` split and so could not see hand-wiring. Re-implement per doc 11 section 4.5 (converged tolerance +
   speed + low-data + fair-encoded extrapolation, with rung comparison).
8. **Stable seeding in the sweep runner.** `run_sweep.build_search_jobs` (and `harness.search`) seed the config draw
   with `abs(hash((arm_name, variant, budget)))`, but Python's str hash is **salted per process**, so the config
   drawn for a given `(arm, cfg_idx)` differs between invocations. Resuming a partial search in a new process would
   silently mix configs under one key. Use `zlib.crc32` of a stable string. (Winners in `results/` are safe: the
   config is stored in each row.)

## 6. Code map and what to add/change (all under `experiments/ssp_algebra/`, self-contained, torch + numpy)

Existing (read these first):
- `ssp_ops.py`: `make_omega(dim,d,length_scale,kind,seed,dtype)`, `phi(x,omega)`, `pos(t,omega1)`, `bind`, `unbind`,
  `shift`, `sim`, `leak_write`, `cleanup_scores`, `cleanup(q,codebook,temp)`, `random_codes(n,dim,seed)`.
- `tasks_c.py` (`DelayTask`, `IndexTask`), `tasks_b.py` (`PointerTask`), `tasks_a.py` (`LandmarkTask`). Common batch
  keys: `obs (B,T)`, `t (B,T)`, `target`, `mask`, `query`, `loss` ("ce" or "mse"), `meta`; each task provides
  `features(batch)`, `in_dim`, `out_dim`, `ep_len`. Numpy `Generator` is passed as `rng` and drives data; torch
  seeds drive weights.
- `arms.py`: `MLPArm`, `GRUArm`, `LRUArm`, `TFArm` (full or windowed), `KernelAttnArm`, `SSPArm`, `SSPPointerArm`,
  `SSPPathArm`, `build_arm(name, task, budget, ...)`, `filter_kwargs`, `ALL_ARMS`. Contract: `arm(batch, feat) ->
  logits`, `state_floats(T)`, `n_params()`; seed **before** any submodule is created (determinism gate depends on
  it). State accounting: floats carried between steps (gru `H`, lru `2N`, ssp `2D`, windowed `2*layers*d_model*window`,
  mlp 0); the windowed arms use `d_model=32, layers=1` so the window is not 1-2 tokens.
- `harness.py`: `splits_for`, `make_batch` (threads `lags`, `hops`, `task_kw`), `loss_and_acc`, `evaluate`,
  `train_arm`, `summarize`, `sample_config`, `run_one`, `search`. Eval data for seed `s` is `default_rng(10000+s)`, so
  all arms are evaluated on identical data per seed (pairing is free).
- `run_sweep.py`: `--mode sweep|ablation`, resumable JSONL keyed on `(variant, arm, cfg_idx, seed, phase)`;
  search seeds are the config index 0-7, hold-out seeds 1000-1004; `REFERENCES`/`ABLATIONS` per variant;
  `LR_REF=5e-3` for ablations.
- `analysis.py`, `selftest.py` (T0 parity + hand-wired SSP-time delay readout on POPGym), `test_harness.py` (6 gates:
  LRU delay line, pinned-delta readout, SSP learns delay, state budget within 5%, determinism, all arms x 4 variants).
- `results/`: `sweep_main.jsonl` (501 rows), `ablation_abl.jsonl` (95), `analysis_main.json`, `winners_main.json`.
  `*_pilot.*` are stale pilot files and can be deleted. Each row has the full `curve` (checkpoints with per-split
  scores and token counts), so speed/AUC analysis needs **no re-run** for the numbers already in RUN_007 4.5.
- `ref_ppo/`: how to rebuild the throwaway POPGym PPO reference venv (it lived in `/tmp` and is gone after reboot).

To add:
- `arms_learned.py` (or extend `arms.py`): learned-head SSP arms for rungs 1-3 on index, pointer and path, keeping
  the `arm(batch, feat)` contract, with `Omega` and codes optionally trainable. Keep rung-0 arms untouched.
- **Control arm `fwp`:** a fast-weight / outer-product (linear-attention) memory with learned write/read key heads,
  state `d*d` floats (d=16 at 256 floats), same rung-1 inputs. This is the key non-SSP control for the competing
  explanation in section 1. Also keep the existing random-position-table ablation.
- **Variant A redesign** (`tasks_a.py`): larger steps (`step_scale` ~0.5-0.6), larger arena, queries only about
  landmarks seen long ago, distractor landmarks, longer episodes sized inside capacity (check `D/4`), and report MSE
  relative to the predict-zero floor. The honest competitor is not a GRU but a **pose-integrating arm with K
  stored (landmark, pose) pairs and attention readout** (state `K*(2+L)` floats), so add that arm (`pose_attn`).
- B extras that were promised and never run: cue-corruption axis (the `cue_drop` argument of
  `PointerTask.batch` exists but no split uses it), and rung 3 (no hop count, learned halting; e.g. fixed max hops
  with a learned stop distribution plus per-hop supervision).
- `test_harness.py`: trainability gates (section 5.4) and tests that rung-1 arms are deterministic and run on every
  variant.

## 7. Experiments, in order, with stop points

0. **Free analysis:** nothing to run for RUN_007 4.5; reuse the saved curves for any speed/AUC re-cut.
1. **Measurement fixes** (section 5). Gate: all baselines pass trainability; re-run rung 0 for C/B with fair
   encodings, fine cadence, budget curve; write the converged-vs-speed table. **Stop point:** if rung 0 shows no
   advantage in speed, low-data accuracy or fair-encoded extrapolation, stop and report (doc 11 4.6).
2. **Learned structure on index (C) and pointer (B):** rung 1, then rung 2, then rung 3 for B. Add `fwp` and the
   random-table control at the same rungs. Training aids only if plain training fails. **Stop point:** if no rung >= 1
   reaches the tolerance with the aids, the result is "structure present but not discoverable"; do not proceed to RL.
3. **Variant A:** redesign the task first, rerun rung 0 to confirm it separates arms from the floor, then its rungs.
4. **Robustness / generalisation battery** on surviving arms: cue corruption 0-90%, distractors, observation noise,
   stream length well above D/4 with and without forgetting, held-out hops/lags/entity counts.
5. **Re-run RUN_006's comparison** against a properly trained GRU: POPGym PPO, `RepeatPreviousEasy`, >=500k steps
   (the reference run was killed by a wall-clock cap at 296k, still rising).
6. **RL** only for learned (rung >= 1) arms that pass the thesis rule. Needs a design decision first (below).

## 8. Protocol and decision rules (also in doc 11; thresholds are proposals, fix them before running)

- Matched state in floats (budget curve), equal tuning budget per arm (same random-config count; arm-specific knobs
  such as hidden size or `dim` count toward the budget but state-size knobs are excluded from the search), tune on
  search seeds, report only on hold-out seeds (>=5), never max-over-epochs.
- **Thesis pass (learned rung):** within 5 points absolute of the best matched-state baseline at convergence **and**
  at least one of: (a) reaches 50%/90% of ceiling in <= half the tokens of the best matched-state baseline;
  (b) clearly higher accuracy at the smaller low-data N; (c) >= 2 pooled standard errors plus a meaningful absolute
  margin on a fairly encoded extrapolation or robustness split.
- **Rung rule:** report the highest rung within tolerance of rung 0 and its sample cost.
- A hand-wired win alone never licenses RL. A win only with hand-wired parts is `structural_prior_only`. Matching the
  generic arms without being faster or more robust means the algebra buys nothing for that variant (doc 10's kill
  condition).
- Do not claim a win over `tf_full` unless it passed its trainability gate and the claim is under a stated
  constraint. Never claim anything about continuous-state SSP encoders, or comparability to POPGym's published numbers.

## 9. Environment, running, and traps

- Repo `.venv` is Python 3.14.6, numpy 2.5.3, gymnasium 1.3.0, torch 2.14.0+cu130; `popgym` 1.0.7 installed with
  `--no-deps`. **Never install ray/popgym baselines into it** (ray downgrades gymnasium). The POPGym PPO reference
  uses its own venv; rebuild per `experiments/ssp_algebra/ref_ppo/README.md`.
- Run from `experiments/ssp_algebra/` with `../../.venv/bin/python` and **`OMP_NUM_THREADS=1`** (unpinned workers
  oversubscribe the machine ~20x slower). A `sys.prefix` RuntimeWarning is harmless.
- Hardware: 16 cores, RTX 3050 4 GB (unused). RUN_007's full sweep (501 runs) took ~2 h wall at 10 workers on CPU;
  GRU runs ~180-300 s each, SSP ~50-200 s (Python loop over T).
- **Do not edit task or arm code while a sweep is running:** the pool uses `maxtasksperchild=1`, so each task
  re-imports modules and a mid-run edit changes behaviour between a variant's search and final phases (this
  contaminated one `index` run; it was re-run).
- `env.action_space.sample()` is not seeded by `reset(seed=)`; use your own `default_rng`. POPGym `RepeatPrevious`'s
  `k` counts the card from k back including the current one: observation lag is `k-1`.
- Bugs already fixed and not to be rediscovered: RUN_007 section 7 (14 items). The one that mattered most: for index
  recall the SSP arm read at `t-j` instead of absolute `j`.
- Superseded/stale: RUN_007's `last5` numbers are not "accuracy"; the withdrawn `proceed_to_rl` verdicts.
- `build_arm`'s default `budget` is 512, `run_sweep`'s CLI default is 256; the reported RUN_007 numbers are 256.
- Git: many unrelated files are modified/untracked in this repo; only `experiments/ssp_algebra/`, `docs/`, `runs/`
  belong to this work. Nothing is committed.

## 10. Remote server / pods

*(Superseded by 15.5: pods are now in routine use through `scripts/launch_ssp_pod.py` and `scripts/ssp_pod_batch.py`; the notes below describe the original plan.)*

Nothing so far needed one. A pod becomes worthwhile for step 6 (RL: many seeds x arms x tasks; one PPO reference
run is ~1 h per 300k steps on this machine) or if the sweeps grow past roughly a day locally. If used: CPU pods via
the repo's RunPod conventions (see `docs/handoffs/04`..`06` and `scripts/launch_cell_pod.py`), keep `RUNPOD_API_KEY`
in `.env` (present, untested), create **one pod at a time**, **never touch `cerb-*` pods**, make results resumable
(JSONL synced off the pod), and **delete the pod when done** (verify with `runpodctl pod list`). Docker Hub's
anonymous pull limit kills pods; `RUNPOD_REGISTRY_AUTH_ID` is unset. This folder is not in any existing image, so a
code tarball approach (as `launch_cell_pod.py` does) is needed.

## 11. Open decisions for the user (ask before step 6; steps 1-5 do not need them)

1. **RL design:** (a) port the SSP arm into POPGym's RLlib `ModelV2` interface (the reference harness already runs
   there; apples-to-apples with their GRU on `RepeatPrevious`, but its fixed lag does not exercise a learned address),
   or (b) build a queried-index / pointer environment with a reward design (more relevant to the thesis, but an
   unvalidated loop). Recommendation: (a) first as the baseline-faithful check, (b) only if a learned arm passes.
2. **Thesis-pass thresholds** (5 points, 2x tokens, 2 pooled se): confirm or change before any run.
3. **Scope of variant A:** continue after the redesign, or drop A if B and C settle the question.

## 12. Definition of done for this milestone

A `runs/RUN_008_*.md` (same honesty conventions as RUN_005-007: measured vs inferred, bugs recorded, limitations
listed) containing: the measurement-fixed rung-0 table (converged vs speed vs low-data vs fair-encoded
extrapolation, with the budget curve and passing baselines); the rung 1/2/3 results for index and pointer with the
`fwp` and random-table controls, plain vs aided training; a decision against doc 11 4.5/4.6 stated as one of
"thesis pass", "structural prior only", "structure not discoverable", or "buys nothing"; and a short list of what to
run next. Update `docs/11` (decision log) and amend RUN_007 only by appending.

---

## 13. Status update (appended 2026-10-11, end of the first execution of this handoff)

Steps A-H of section 0.3 are done; results and caveats are in `runs/RUN_008_ssp_learned_structure.md` (read its sections 1, 6 and 7 first). Summary:
rung 0 holds up under fair measurement on index and pointer but is not SSP-specific on index; learned index is discoverable only with a kernel-width
curriculum (basin of attraction ~1 index); learned pointer is discoverable (reliably with per-hop supervision); no RL is licensed. The pointer pass is
provisional. Remote workflow now exists: `scripts/launch_ssp_pod.py` + `experiments/ssp_algebra/pod_job.py` (verified; CPU pods, resumable on R2, self-delete).
Next, in order: (1) a non-SSP iterated-structure control and a comparable aid for the pointer baselines; (2) a pointer arm that reads the cue from the corrupted
features (robustness), then rung 3 (no hop count); (3) reduce the learned pointer's dependence on per-hop supervision; (4) variant A redesign; (5) only then
the RL decision in section 11. Do not re-run the strawman `fwp` (quarantined `results/invalid_buggy_fwp_*`).

---

## 14. Recipe study: framing, correction and pre-registered diagnostics (appended 2026-10-11)

**Framing.** The question after RUN_008 is no longer "can a learner discover the structure" (yes, with a curriculum matched to the failure)
but "is there a *recipe* that does it reliably across a family of problems". `docs/11` section 3.5 defines two families (continuous-address,
discrete-key) and section 4.7 the reliability protocol (tune on two variants, freeze every hyperparameter, require >=0.95 on >=9 of 10 seeds on
unseen variants, report sample cost, no per-variant retuning). Nothing is validated yet: the width schedule's constants (start 8, 750 updates)
were chosen by hand after seeing one task.

**Correction to RUN_008 wording.** "About 2x faster" holds to 50% of ceiling (59k vs 113k tokens) but not to 90%: updates to reach 0.90 / 0.99 on index are
rung 0 75 / 125, fwp 100 / never (plateaus 0.94), rung-1 identity init 225 / 375, **width-scheduled learned arm 350 / 600**, GRU 375 / 800. At 90% the learned
arm is about even with the GRU. `analysis.py` already credited it only with a low-data win; only the prose was loose.

**Diagnostics registered before running (predictions in brackets; all are hypotheses).**
1. *Address-head optimiser.* The identity-init learned arm (already at the right address) is 3x slower to 0.90 than rung 0 (225 vs 75 updates). [If Adam jitter
   on the address parameters is the cause, a 10x lower address learning rate or a frozen write phase brings it toward 75-100 updates.]
2. *Schedule length and adaptivity.* The width-scheduled curve tracks the current width's ceiling (0.88 at update 300, 0.97 at 500, 0.99 at 700).
   [Shorter schedules (250 / 375 / 500) speed it up until discovery breaks; a plateau-driven narrowing should match or beat the best fixed length.]
3. *Learning-rate range.* Only the two largest searched rates (0.0225, 0.0229) worked; the range ends at 0.032. [0.05 is as good or better; 0.1 may destabilise.]
4. *Schedule plus light address supervision.* [Faster than either alone; the aid's cost is reported.]
5. *Label-free interface pretraining.* Learn only the address heads on fresh synthetic streams (retrieval loss on the stream's own tokens, width schedule inside the
   pretraining), freeze them, then train the readout on a pool. [The N=64 collapse (0.279) recovers to >0.8; pretraining tokens are reported separately.]
6. *Pointer.* A temperature schedule on the cleanup; plain training at 10 seeds to estimate the success probability (currently 3/5).
7. *Generality probe.* Index at T=32 with the recipe unchanged (start 8) vs the start-width rule (half the range, 16), against rung 0 and the GRU.
   [If only the rule passes, the start width is a needed, task-derived constant; if both pass, it is not critical; if neither, the recipe is not general.]
All run on RunPod CPU pods (`scripts/launch_ssp_pod.py`), results to `runs/RUN_009_*.md`.

### 14.1 Results of the registered diagnostics (appended 2026-10-11; full detail in `runs/RUN_009_recipe_diagnostics.md`)

28 pod jobs, all clean. Against the predictions above: (1) **refuted as stated**: the identity-init arm is as fast as rung 0 (50 updates to 0.90 vs 75) once its learning rate is sensible;
the earlier 225 was a tie-break artifact (all 6 configs tied at 1.00, the winner had lr 0.00075), and neither a lower address lr nor a frozen write phase matters. The learned arm's cost is discovery, not optimisation.
(2) schedule 375 beats 750 (0.90 at 250 vs 350 updates, same accuracy); 250 starts to lose accuracy; the adaptive rule does not beat the fixed one. (3) lr 0.05 about as good as 0.0316, 0.1 degrades; no gain when combined with 375.
(4) light address supervision was **slower** (prediction wrong). (5) label-free pretraining aligns the address heads on 2 of 5 seeds only (3x longer does not change that); where it works N=64 is 0.91-0.98.
(6) pointer: temperature schedule no effect; plain training 6/10 seeds. (7) **the recipe does not transfer to T=32**: unchanged 0.665, start=T/2 0.849, start=T 0.930, longer schedule 0.938; hand-wired 0.964, GRU 0.942.
**New caveat that touches earlier work:** configs are selected on converged accuracy, which is blind to speed under ties, so every speed number (all arms) is noisy; re-select on AULC before restating the "4x faster" claim.
**Next, in order:** (a) speed-aware config selection and a restated speed comparison; (b) a range-scaling rule for the width recipe, tuned on T=16 and 32, frozen, tested on unseen T and on delay (docs/11 4.7); (c) pretraining through a throwaway readout;
(d) the pointer items of section 13. No RL yet.

---

## 15. Hand-off to the next agent (written 2026-10-11; supersedes the plans above)

### 15.1 Where things stand, in six sentences

Supervised, component-level study of an SSP memory (bind/unbind/shift/bundle) against GRU / LRU / transformers / a fast-weight control at matched carried state, on three synthetic task
families (`delay`, `index`, `pointer`; variant A `path` is untested). **Hand-wired** (rung 0) the memory beats generic arms on index and pointer in speed to 50%, low data (N <= 256) and held-out hops, but the index
advantage is *not SSP-specific* (random position table, learned codes and the `fwp` control match much of it). A **learner can discover** the structure: on index only with a coarse-to-fine kernel-width curriculum
(plain 0.40-0.62, scheduled 0.99 at T=16), on pointer reliably only with per-hop supervision (plain succeeds on about 6 of 10 seeds). The curriculum's **constants do not transfer** (T=32: 0.665 unchanged, 0.93-0.94 with
scaled constants), so no recipe has passed the reliability protocol (docs/11 4.7). **Speed claims are unreliable** because configs are chosen on converged accuracy (RUN_009). Nothing about robustness of a learned arm, RL or larger
models has been tested, and docs/11 4.8 says when it may be.

### 15.2 What the evidence supports (use these exact scopes)

- Firm: learned-arm cost is discovery, not optimisation; width-schedule form carries to T=32 but constants do not; the learned arm's low-data collapse is an address-learning problem (pretrained alignment gave N=64 0.91-0.98 on 2 of 5 seeds);
  per-hop supervision makes the learned pointer reliable (5/5, held-out hops 1.000); a temperature schedule does nothing on pointer.
- Suggestive only: the start width and schedule length scale with the address range (three points per axis); why light address supervision is slower; why retrieval-loss pretraining aligns only 2 of 5 seeds.
- Not concluded: any recipe is reliable; the learned arm is faster than the GRU; robustness; SSP-specificity; amortised pretraining; anything about RL or a larger model.
- Open decisions for the user (ask, do not assume): (1) is index a thesis vehicle or only a harness check (it is largely substitutable)? (2) does the hand-built width schedule count as "natural" learning or as an aid to be costed? (3) keep the thresholds
  (5 points, half the tokens, 2 pooled se; 0.95 on 9 of 10 seeds) as proposed? (4) RL design (section 11: port into POPGym's ModelV2 vs a pointer environment)? (5) commit the work and where it should live (nothing is committed; see 15.6).

### 15.3 Gating (docs/11 4.8)

- Robustness of the *structure* (hand-wired, baselines) may be run any time; pointer cue-corruption first needs an arm that reads the cue from the corruptible features.
- Robustness of a *learned* arm only after its recipe is reliable (>= 9 of 10 seeds).
- RL / larger models only when all hold: a family recipe passes 4.7; its speed claims survive speed-aware selection (step 1); the pointer pass is non-provisional (step 4); some learned-arm robustness evidence; the SSP-specificity caveat is addressed.
- Not gated: re-running POPGym's PPO + GRU on `RepeatPreviousEasy` to >= 500k steps (rebuild the venv from `experiments/ssp_algebra/ref_ppo/README.md`; the earlier run was killed by a wall-clock cap at 296k, so run it in the background with a long timeout or on a pod).

### 15.4 The plan, step by step

Order: step 1 first; steps 2, 3, 4 can run in parallel on pods; then 5, 6, 7. Every step: new tags, 10 hold-out seeds for any pass/fail claim, `summarize_runs.py` for reading results, a new `runs/RUN_0xx` record in the RUN_005-009 style
(measured vs inferred, bugs, limitations, "do not do"), and tests green before and after code changes.

**Step 1 - make speed claims fair (cheap, decides how much of the speed story survives).**
- Problem: `run_sweep.pick_winners` selects the winner by `last5_id` on ONE search seed per config; under ties near ceiling the choice is arbitrary w.r.t. speed (the same arm: 225 vs 50 updates to 0.90).
- Work: add `--select {last5,aulc}` and `--search-seeds K` (default 1 = old behaviour) to `run_sweep.py`; search seeds `ci + 100*k` (cfg_idx unchanged); choose by mean `summary["aulc_id"]` over the K seeds, ties by `conv_id`. Re-run finals (10 seeds, 1000-1009) for rung 0
  `ssp`, `gru`, `lru`, `tf_full`, `tf_window`, `kernel_attn`, `fwp` and the learned width-scheduled arm (375 updates) on index (T=16, 256 floats), and on pointer for rung 0 + baselines. Stable config draws mean existing search configs can be reused.
- Baseline hardening for low data: add `weight_decay` {0, 1e-4, 1e-2} and `dropout` {0, 0.1, 0.3} to `sample_config` for `gru/lru/mlp/tf_*` (add `dropout` to `GRUArm`; weight decay through `harness.make_optimizer` and `TRAIN_KEYS`); same number of random configs for every arm. Changing the search space changes the draws: new tags, not comparable with old files.
- Report: updates and tokens to 0.50 / 0.90 / 0.99, area under the curve, with intervals; low-data N = 64 / 256 / 1024 / 4096 again for the hardened baselines.
- Decision: if the rung-0-vs-GRU gap at 0.90 is < 2x after re-selection, drop the speed claim and keep only data-efficiency and held-out-hops claims; if tuned baselines reach within 5 points of SSP at N = 64 / 256, drop the low-data claim.

**Step 2 - a range-scaling rule for the continuous-address family (docs/11 3.5, 4.7).**
- Hypothesis: start width = a * R and schedule length = b * R, total updates = c * R, with R the address range (T - 2 for index; the maximum lag for delay), learning rate about 0.02 (RUN_009: 0.05 no better, 0.1 worse).
- Code: new `experiments/ssp_algebra/recipes.py` with `continuous_address_cfg(R, a, b, c)` returning the `--arm-cfg` dict (+ steps) and a test; the learned arm already supports `ls_start`, `ls_mode="sched"`, `width_steps` (`set_width` in `arms_learned.py`).
- Variants: index T = 16, 24, 32, 48, 64 (use `--ep-len` and `--budget`: 256 floats up to T=32, 512 at T=48/64 so memory capacity D/4 is not the limit; matched GRUs); delay with lag ranges 1-4 and 1-8 at T = 16 and 32 (verify the `ssp_index_learned` delay path first: `_read_addr` uses `batch["lag"]`, read at `t - s_q`).
- Protocol (4.7): tune (a, b, c) on T=16 and T=32 only, grid a in {0.5, 1, 1.5}, b in {24, 47, 94}, c in {2b}, >= 3 search seeds per point, select by AULC subject to conv >= 0.95; FREEZE; run on unseen T=24, 48, 64 and delay, 10 seeds each.
- Pass: >= 0.95 on >= 9 of 10 seeds on every unseen variant, cost (updates/tokens to 0.90) reported next to a tuned GRU. Kill: >= 2 unseen variants need retuning -> report a variant-specific trick and try a multi-scale frequency bank (several `ls` bands with learned mixing) or a better-initialised learned width instead.
- Count every condition tried while designing the rule and report it.

**Step 3 - make interface pretraining reliable and show that it transfers.**
- Current state: `pretrain_steps` trains only the address parameters on fresh synthetic streams with `retrieval_loss` (cross-entropy of the cleanup scores), freezes them, then trains the readout on a pool; aligns on 2 of 5 seeds, 3x longer does not help.
- Variants to add in `harness.train_arm` (`pretrain_mode`): `probe` (train the address parameters through a throwaway MLP readout on the synthetic task loss, then discard it), `full` (train the whole arm on synthetic streams, freeze the address, re-initialise `arm.mix`), `retrieval` with a smaller tau / normalised scores. Keep the width schedule inside the pretraining.
- Measure: alignment success over >= 20 seeds with a Wilson interval; accuracy at N = 64 and 256; pretraining tokens reported separately from task tokens.
- Transfer test (this is what makes it a recipe component): pretrain the address at T=16 / 8 items, copy only `w_time`, `read_head`, `log_mult` into an arm built for T=32 or 16 items, freeze, train just the readout. Needs a small `load_address_params(src)` helper.
- Adopt if >= 18 of 20 seeds align, N=64 >= 0.9 and transfer holds; otherwise report it as "more training data".

**Step 4 - discrete-key family (pointer): diagnose, control, make the pass real.**
- Diagnose the stuck seeds (about 0.5): instrument `SSPPointerLearned` to log, over training and for >= 20 seeds, role-gate accuracy against parity, whether `qmap` becomes the identity, and per-hop accuracy; run a basin sweep (initialise the role gate at the correct weights plus noise sigma). Then choose fixes from what it shows
  (role-gate init, a schedule on the role gate, best-of-k restarts, label-free key-to-value pretraining, a light per-hop supervision dose); a temperature schedule on the cleanup is already ruled out.
- Controls that make the pass non-provisional: (i) equal-aid baselines: a `chain` mode of `PointerTask` where each query is repeated `h` times with a step marker and a supervised target f^k(s) at step k, for GRU / LRU / transformers / kernel attention (no true intermediate nodes in the input);
  (ii) a looped weight-tied transformer arm (`LoopedTFArm` in `arms.py`, per-iteration outputs, the existing `inter` key for per-hop labels); (iii) an iterated `fwp` read; (iv) a trainability gate for every pointer baseline on an easy variant (4 nodes, 1 hop) and longer training (>= 6000 updates; GRU was 0.45 and rising).
- Rung 3 and the cue fix: make `SSPPointerLearned` decode the start node and hop count from `batch["query"]` (the corruptible features) instead of `batch["start"]` / `batch["hops"]`, with a learned halting distribution (unroll `max_hops`, weight the per-hop outputs by it); this also makes the cue-dropout split valid.
- Decision: the pointer pass stands only if the SSP arm beats the equal-aid baselines on speed or held-out hops with plain or lightly aided training; if a looped transformer with per-hop labels matches it, report "iteration plus supervision", not SSP.

**Step 5 - robustness of learned arms (only after steps 2-4 give a reliable recipe).** Cue dropout 25/50/75% and token noise 10/25% on pointer against the equal-aid baselines; index at 512 / 1024 floats on T = 64 / 128 (does SSP follow the D/4 capacity rule, where do windowed transformers fail);
pre-register what "graceful" means (for example accuracy at 50% corruption >= 0.8 of clean and above every baseline by >= 2 pooled se) before running. `harness.splits_for(..., robust=True)` / `run_sweep --robust` already provide the splits.

**Step 6 - variant A (path integration), where the unique group action matters.** Redesign `tasks_a.py`: step scale 0.5-0.6, a larger arena, queries only about landmarks seen >= 8 steps ago, distractor landmarks, episodes sized inside capacity, MSE reported relative to the predict-zero floor.
Competitors: GRU and a `pose_attn` arm (attention over K stored (landmark, pose) pairs, state K*(2+L)). Compare `SSPPathArm` `shift_memory` vs `read_relative` on state and compute; then its learned rungs (action-to-phase map, learned query code). SO(2) heading afterwards.

**Step 7 - the RL / larger-model decision (only if 15.3 holds).** Two designs from section 11 (a: port the SSP arm into POPGym's `ModelV2`, baseline-faithful; b: a queried-index / pointer environment with a reward). Recommended: (a) first, (b) only if a learned arm passes cleanly. Then embed the SSP memory next to a GRU in a small policy and compare against GRU-plus-`fwp`
memory, many seeds on pods.

### 15.5 Operations

- Environment: repo `.venv` (Python 3.14.6, torch 2.14.0, numpy 2.5.3); ALWAYS `OMP_NUM_THREADS=1`; run from `experiments/ssp_algebra/` as `OMP_NUM_THREADS=1 ../../.venv/bin/python <script>` (a `sys.prefix` RuntimeWarning is harmless). 16 cores locally.
- Tests (all green at the end of RUN_009): `test_harness.py` 6/6, `test_fixes.py` 2/2, `test_measurement.py` 6/6, `test_learned.py` 12/12, `test_analysis.py` 9/9, `test_trainability.py` (gru, tf_full, fwp >= 0.95 on easy index; writes `results/trainability.json`), `selftest.py`. Never loosen a threshold: a failure means the code is wrong. New checks go in new functions/files.
- Local sweep: `run_sweep.py --mode sweep --variants index --arms ssp_index_learned --label <label> --arm-cfg '<compact json>' --tag <unique> --budget 256 --dense-eval --configs 6 --seeds 5 --steps 1500 --workers 12`. Flags: `--pool N` (low data), `--robust`, `--ep-len`, `--label` (one arm only), `--arm-cfg`
  (training keys: `lr, aux_weight, curriculum, curr_steps, anneal_steps, width_steps, addr_lr_scale, pretrain_steps, pretrain_lr`; arm keys: `rung, init, length_scale, ls_start, ls_mode (fixed|sched|learned|adaptive), freeze_write, lam_init, temp`).
- Remote (pods): write a job list (see `experiments/ssp_algebra/jobs/example_jobs.json`) and run `python scripts/ssp_pod_batch.py launch|status|collect <jobs.json>`; each job = one 4-vCPU CPU pod that uploads to R2 and self-deletes; `launch --dry-run` first. Tarball launching is slow when `experiments/ssp_algebra/results` is large (every launch rebuilds and uploads it).
  Only pods whose name starts with `ssp-` are ours; other pods in the account (about 7 `eval-v3-*`) are not and must not be touched. Always confirm with `status` that none of ours are left.
- Reading results: `summarize_runs.py <files>` (final rows only). `analysis.py --tag r0fix --ablation-tag r0fix_abl --extra <files>` applies the docs/11 4.5 rules, but do NOT pass files that reuse an arm label under a different protocol (6000-update baselines, repeat robustness runs): rows are merged by label.
  The command used for RUN_008: `R=results; ALL="$R/sweep_r0fix_b128.jsonl $R/sweep_r0fix_b512.jsonl $R/sweep_r0fix_pool*.jsonl $R/sweep_r0fix_fwp.jsonl $R/sweep_r0fix_fwp_pool*.jsonl $R/sweep_learn1_*.jsonl $R/sweep_learn1long_*.jsonl $R/sweep_learn2_*.jsonl $R/sweep_learn3_*.jsonl $R/sweep_ptr1_*.jsonl $R/sweep_ws8_pool*.jsonl"`.

### 15.6 Result files and code map (all under `experiments/ssp_algebra/`; nothing is committed)

- Rung 0 / baselines: `results/sweep_r0fix.jsonl` (264 rows: delay, index, pointer x 8 arms), `sweep_r0fix_b128|b512` (budget curve), `sweep_r0fix_pool{64,256,1024,4096}` (low data), `sweep_r0fix_fwp*` (valid `fwp` control), `ablation_r0fix_abl.jsonl`, `sweep_rob1_*` + `ablation_rob1_abl.jsonl` (robustness; the pointer cue-drop rows are vacuous for SSP).
- Learned structure: `sweep_learn1_*` (rungs, aids), `sweep_learn1long_*` (6000 updates), `sweep_learn2_*` (aux-only / curriculum-only), `sweep_learn3_*` (width schedules), `sweep_ptr1_*` (learned pointer), `sweep_ws8_*` (low data and long streams for the scheduled arm), `sweep_ptrbase_*` (GRU / tf_full at 6000 updates), `diag1/diag2_lr02/diag3_ls{2,4,8}.json` (basin diagnostics).
- RUN_009: `sweep_dg_*`, `sweep_dg2_*`. **Do not use** `results/invalid_buggy_fwp_*` (a strawman control that could not address by position).
- Code: `arms.py` (rung 0 + baselines), `arms_learned.py` (`SSPIndexLearned`, `SSPPointerLearned`, `FWPArm`, `build_learned`), `feats.py`, `tasks_a|b|c.py`, `harness.py` (`splits_for`, `train_arm` hooks, `make_optimizer`, `build_any`, `run_one`), `run_sweep.py`, `analysis.py`, `diag_address.py`, `pod_job.py`, `summarize_runs.py`; `scripts/launch_ssp_pod.py`, `scripts/ssp_pod_batch.py`.
- Git: many unrelated files in the repo are modified/untracked; only `experiments/ssp_algebra/`, `docs/`, `runs/` and the two `scripts/` files above belong to this work. Do not `git add -A`; `results/` is large (decide with the user whether to commit it, store it elsewhere or ignore it).

### 15.7 Traps (each one cost time or produced a wrong number)

1. Config selection by converged accuracy is blind to speed under ties (step 1). 2. The sweep resume key omits budget, pool, steps, robust, ep-len: always a fresh `--tag` per protocol. 3. Do not edit `arms*.py`, `tasks_*.py` or `harness.py` while a sweep runs (workers re-import per task); on pods the code is a frozen tarball.
4. `--arm-cfg` JSON must have no spaces on pods (the job args are split on whitespace). 5. `mlp` and `posmlp` are the same arm (the clock is in the shared features). 6. The pointer arms read `batch["start"]/["hops"]`, so cue corruption cannot reach them. 7. `temp` annealing is a no-op in the index learned arm.
8. The learned-address RMS error in `diag_address.py` is only meaningful at `length_scale=1` (the write phase is also trained). 9. A learned arm identical to rung 0 at init (`init="identity"`) is a capability check, not a learned result; `analysis.py` treats it as rung 0. 10. Seeds: search seeds are the config index, hold-out seeds 1000-1009; evaluation data for seed `s` is `default_rng(10000+s)`.
11. Report any tuning cost and the number of conditions tried; do not select on hold-out seeds. 12. Reuse of an arm label across protocols silently merges rows in `analysis.py` (see 15.5).
