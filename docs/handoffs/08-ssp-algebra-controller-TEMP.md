# TEMPORARY handoff: SSP algebra controller (code + tests to do)

> **SUPERSEDED (2026-10-11) by `09-ssp-learned-structure-sample-efficiency.md`.** Steps 1-4 of this handoff were
> executed in `runs/RUN_007_ssp_algebra_controller.md` in a hand-wired form, and its verdicts were revised. Keep this
> file only for the original task list; do not start from it.

Written 2026-10-10. **Temporary:** delete or fold into `docs/11-...` once the work lands. Nothing is committed or
pushed. No background jobs are running. Read this first, then `docs/11-ssp-algebra-controller-variants-and-tests.md`
(the plan and decision rules), `runs/RUN_005_hypotesting.md` and `runs/RUN_006_popgym_first_pass.md` (what was
measured).

## 1. Goal in one paragraph

Test whether giving a learner an SSP/VSA memory with algebraic operations (write, shift, read/unbind, cleanup) lets it
learn faster, extrapolate, or work under a fixed state budget, versus GRU, S4D/LRU and transformers. Three variants:
**A** path-integration map, **B** iterated query (hop extrapolation), **C** learned-delta / index-addressed read. The
user wants all three, ablated, supervised probing first and RL only for surviving arms. Expect wins in sample
efficiency, extrapolation and state budget, not "solves what a full-context transformer cannot".

## 2. What exists (do not redo)

`experiments/vsa_memory_stage0/`: encoder-only battery T0/T1/T4/T6/T7, CartPole length-scale diagnostic, H1
interference test (RUN_005). Uses `rl/representations/ssp_legacy.py` and upstream `sspspace`.

`experiments/popgym_vsa/` (RUN_006), self-contained, imports nothing from the repo:
- `tasks.py` task registry, space metadata, scripted oracles. **`spec["lag"] = k-1`**: POPGym's `k` includes the
  current card (hit rate 0.247 at lag=k, 0.991 at k-1).
- `memory.py` `TorchMemory`: FHRR item memory, `code="random"|"block"`, count and delay readouts. Position codes are a
  **table of random vectors, not SSP**.
- `models.py` mlp / posmlp / gru / vsa_count / vsa_delay behind `initial_state / step / reset_state / detach_state`.
- `vec_env.py`, `train.py` (own A2C, truncated BPTT), `headroom.py` (deterministic), `selftest.py`.
- Records: `README.md`, `runs/RUN_006_popgym_first_pass.md`, results under `results/`.

Measured facts you can rely on:
- RepeatPrevious has no feedforward shortcut; CountRecall has a partial one (headroom table in RUN_006 §2).
- Hand-wired delay memory: +0.990 greedy eval on RepeatPreviousEasy at 490k steps, +0.575 at 122k; GRU -0.108. One
  seed, untuned. A structural-prior result only.
- No arm learns CountRecall Easy in up to 2000 updates, even with an oracle-exact readout (block codes). Decode shape
  was ruled out. Cause (RL discovery vs optimisation) is **unmeasured**.
- Random-code count readout error grows with total items bundled (exact only at D>=~4096); block codes are exact.
- Capacity about D/4; chained unbinding depth 1-3 free; recall survives 90% cue loss at D~1000 (RUN_005).

## 3. Environment gotchas

- Repo `.venv` is Python 3.14.6, numpy 2.5.3, gymnasium 1.3.0, torch 2.14.0+cu130. `popgym` 1.0.7 installed with
  `--no-deps`. Do **not** install `popgym[baselines]` there (`ray[rllib]` downgrades gymnasium to 1.2.2).
- **Always `OMP_NUM_THREADS=1`** when running more than one process (unpinned runs were ~20x slower).
- Run scripts from `experiments/popgym_vsa/` (flat imports). A harmless sys.prefix RuntimeWarning appears when using
  `../../.venv/bin/python`.
- `env.action_space.sample()` is not seeded by `reset(seed=)`; use your own `default_rng`.
- Hardware: 16 cores, RTX 3050 4 GB (unused so far), 15 GB RAM.
- `results/` holds superseded files from an unpinned 150-update batch (`{Fam}{Diff}_{model}_s0.json`); the pinned
  runs are `cr_*` and `rp_*`. Either is fine to delete.
- Git: many unrelated tracked files are modified and many untracked; only `experiments/popgym_vsa/`, `docs/`,
  `runs/` are from this work. `experiments/popgym_vsa/.gitignore` ignores `logs/` and `__pycache__/`.

## 4. To code and test, in order

1. **POPGym PPO reference (small, RL).** Throwaway venv outside the repo with `ray[rllib]` + POPGym baselines; one GRU
   on RepeatPrevious Easy and CountRecall Easy. Purpose: is the in-repo A2C loop or the task the problem. Unverified
   whether ray 2.59 runs POPGym's old-style `ray_models`; if it does not, report that and do not infer anything.
2. **Shared supervised harness.** Task-generator interface `(observation sequence, query, target)`; teacher-forced
   training; baselines: mlp, posmlp, GRU, S4D/LRU, full-context transformer, windowed transformer, kernel-attention
   (learned delta), SSP controller. **Matched state size in floats**; decide the accounting rule for the windowed
   transformer (floats per cached token). 5 held-out seeds, mean of the last k, never max-over-epochs. Equal tuning
   budget per arm (random search, same config count; arm-specific knobs like hidden size or `dim` count).
3. **Variant C** (cheapest, validates the harness): learned continuous `delta` read `M * pos(-delta)` with real SSP
   time `pos(t)=exp(i*Omega*t)` replacing the random table; index-addressed recall; POPGym `RepeatPrevious` with
   `num_decks`/`k` constructor args for train-short/test-long. Ablate random table vs SSP time. Expected to be matched
   by S4D / kernel attention; that outcome is informative.
4. **Variant B** (synthetic): pointer-chasing over bound pairs, controller iterates unbind+cleanup, train 1-2 hops,
   test 4-8; cue-corruption axis (0-90%).
5. **Variant A** (needs a small 2-D simulator): pose as SSP `p_t = p_{t-1} * phi(a_t)`, landmarks bundled as
   `landmark * p_t`, relative query by unbinding; SO(2) heading extension. Ablate shift-on-memory vs re-encoding
   (compare state/compute, not accuracy).
6. **Cross-cutting in all variants:** forgetting (leaky decay or gated write; parameterisation not yet chosen),
   stream length >> D/4, fixed vs learned codes, cleanup on/off. One change at a time from a reference config; not a
   full factorial.
7. **RL only for arms that survive** step 2's decision rules (doc 11 §4.5).

## 5. Decision rules (from doc 11, restated)

- Proceed to RL for an arm only if, supervised at matched state with extrapolation, it beats every baseline including
  S4D and the windowed transformer.
- Matches S4D/GRU/kernel attention: the algebra buys nothing in that variant; drop it and report.
- Wins only hand-wired, loses with learned codes: structural-prior result, say so.
- Claim a win over a full-context transformer only under a stated constraint (state budget, extrapolation, stream
  longer than context).
- Never claim anything about continuous-state SSP encoders, or comparability to POPGym's published numbers unless
  step 1 shows the loops agree.

## 6. Open items and traps

- Unverified: ray 2.59 + POPGym baselines; doc 10 §13.1's statements that S4D, LMU, FWP, linear attention lose to GRU
  (taken from the doc, not checked against the paper).
- CountRecall cause is open. Untried levers: gamma=0 (it is a memory-then-answer bandit), no entropy bonus, higher lr,
  more envs, longer budget. Reward scale is 1/ep_len per step, which may starve the gradient.
- `vsa_delay` Hard with dim 256 reaches only ~0.71 policy return (capacity near D/4); raising `dim` is untested.
- Do not cite CountRecall training numbers against VSA (all arms fail), nor the random-code count failure as a
  property of the algebra (codebook cross-talk).
- A full-context transformer is expected to match accuracy at unlimited state; do not frame a non-win there as a
  failure of the harness.
- Whether this belongs in a separate repo is deferred until a variant passes its rule; the POPGym folder is portable.
- Doc 10 line 589+ (MIKASA wiring) is background; MIKASA is out of scope until supervised results justify it.

## 7. Definition of done for the first milestone

Step 1 reported with a clear verdict, and the shared harness plus Variant C run with the full arm set at matched
state size, 5 seeds, written up as `runs/RUN_007_*.md` with the same honesty conventions as RUN_005/006 (measured vs
inferred, bugs recorded, limitations listed).
