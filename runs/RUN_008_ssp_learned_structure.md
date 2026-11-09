# RUN_008 — SSP learned structure and sample efficiency (handoff 09)

**Status: COMPLETE for handoff 09 steps A-H (2026-10-11).** Supervised only, no RL. Code in `experiments/ssp_algebra/`;
the sweeps ran locally (rung-0 main sweep, controls, diagnostics) and on RunPod CPU pods (everything after
`r0fix`'s main sweep). Every number is MEASURED unless marked *inferred*. Protocol: 256 carried floats unless noted, 6 random
configs per arm scored on search seeds, **5 hold-out seeds (1000-1004)**, converged = mean of the last 3 checkpoints, 28+
checkpoints per run (dense evaluation), ± = standard error over seeds. GRU/`tf_full`/`fwp` passed the trainability gate
(`results/trainability.json`: 1.000 / 0.995 / 1.000 on easy index). Pointer baselines had **no** gate until the 6000-update
check in 6.

## 1. Headline

- **Rung 0 (hand-wired) confirms the structural prior on index** and is a real, fair-measurement result: converged tie with the GRU,
  ~4x faster to 50%, far ahead at N<=256 training sequences, moderately ahead on 4x streams. **It is not SSP-specific**: a random
  position table and learned codes match or beat it, and the learned-key `fwp` control is within a few points at convergence.
- **Learned structure, index:** *not discoverable by plain training* (0.40-0.62). The failure is a **narrow basin of attraction**
  (the read address must start within ~1 index of the write address); it is *not* a lack of gradient and *not* weights that cannot
  travel. A **coarse-to-fine kernel-width schedule** (a curriculum on the model, no labels) fixes it: **0.993±0.001** from an
  8-wide kernel annealed to 1 over 750 updates, no address supervision. It keeps a low-data edge at N=256 but **loses** the N=64 and
  long-stream advantages of rung 0, and it does not separate from `fwp` cleanly. Verdict: `structural_prior_only`.
- **Learned structure, pointer:** discoverable. Per-hop supervision gets **1.000 on all 5 seeds, including held-out hops 4-8**; plain
  training finds it on 3 of 5 seeds. GRU reaches 0.450 even at 6000 updates (held-out hops 0.142). Verdict:
  `thesis_pass_candidate`, **provisional** (section 6).
- **Delay:** `algebra_buys_nothing` (pre-registered expectation).
- **No RL is licensed yet**: the pointer pass is provisional, uses an aid, and its robustness (cue-drop) was not testable (section 7, item 2).

## 2. Rung 0 re-run with the fixes (`r0fix`; 264 rows verified, plus budget/pool runs on pods)

Index, converged id / 4x-stream / tokens-to-50%:
- gru 0.999±0.001 / 0.469±0.009 / 113k; **ssp 0.998±0.001 / 0.626±0.020 / 26k**; tf_full 0.972±0.006 / 0.384 / 133k; fwp 0.945±0.005 / 0.409 / 26k
  (tie at the measurement resolution); lru 0.548; tf_window 0.414; kernel_attn 0.406; mlp 0.133; posmlp 0.126.
- Low data (accuracy at N = 64 / 256 / 1024 / 4096): ssp 0.966 / 0.996 / 0.995 / 0.994; fwp 0.354 / 0.852 / 0.921 / 0.937;
  gru 0.195 / 0.320 / 0.988 / 1.000; tf_full 0.162 / 0.314 / 0.983 / 0.971.
- Budget curve 128 / 256 / 512 floats: gru 1.000 / 0.999 / 0.999; ssp 0.961 / 0.998 / 1.000; tf_window 0.225 / 0.414 / 0.716.
  The GRU already solves it at 128 floats, so the SSP edge is speed and data, not state.
- Ablations (fixed reference config, id / 4x / 8x stream): reference 0.998 / 0.655 / 0.402; pos_table 0.996 / 0.726 / 0.451;
  no_cleanup 1.000 / 0.833 / 0.606; learn_codes 1.000 / 0.836 / 0.561; no_decay 0.998 / 0.658 / 0.407; fast_forget 0.998 / 0.634 / 0.370.
  Nothing SSP-specific is load-bearing; forgetting does nothing; cleanup *hurts* extrapolation.
Delay: lru 1.000, gru 0.998, tf_full 0.997, **ssp 0.976±0.013**, kernel_attn 0.800, tf_window 0.777; ssp is among the slowest (207k vs 91-108k).
Pointer: **ssp 0.999 (id and hops 4-8)**; gru 0.223 / 0.156; tf_full 0.152; all others 0.15-0.18 (chance 0.125).
Stop rule (docs/11 4.6) did **not** fire: rung 0 separates on speed, low-data and fair-encoded extrapolation on index and pointer.

## 3. Learned structure on index

Rung 1 learns the write phase (13 weights over fixed time features, initialised at the ramp) and the read phase (linear head on the
raw index, random init); rung 2 also learns `Omega` and the codes. "aux" = squared error between the learned read address and the
true index (a teacher for the address). Conv id at 1500 updates unless noted:
- plain: r1 0.400±0.014, r1-scrambled 0.754±0.090 (unexplained), r2 0.616±0.031.
- aux-only: r1 0.757±0.074, r2 0.857±0.024. curriculum-only (small indices first): r1 0.392±0.015, r2 0.569±0.007. both: r1 0.702±0.006, r2 0.819±0.018.
  **The index-range curriculum does nothing; address supervision accounts for the gain.**
- identity init (rung 0 inside the learned arm): 0.998±0.001 -> the structure is expressible; discovery is the problem.
- 6000 updates (aided): r1 0.829±0.008 (plateau), r2 0.892±0.015 (still rising); r2 plain 0.604±0.010. None within the 0.05 tolerance.
**Basin-width diagnostic** (`diag_address.py`; random init starts ~6 indices off, 7% of queries in the basin): final accuracy is ~0.997 when
the initial RMS address error is <~1, mixed at 1.6 (0.66-1.0), and ~0.25-0.62 from 3-7 (random 0.40-0.48). The read-head gradient is
non-zero (0.1-0.5) and accuracy rises, but the address error never shrinks. Learning rate 0.02 (4x) did **not** help (random 0.408,
sigma=1 0.447, sigma=0.3 *worse* 0.711): the "weights cannot travel far enough" explanation is **refuted**.
**Kernel width widens the basin and costs resolution** (accuracy at init error ~5.3 / ceiling): ls=1 0.455 / 0.997; ls=2 0.456 / 0.987;
ls=4 0.886 / 0.947; ls=8 0.800 / 0.831 (random init: 0.425 / 0.554 / 0.748 / 0.798). Hence a **width schedule** (`ls_mode="sched"`, mult decays
linearly to 1 over 750 updates): from ls=4 0.812±0.016, **from ls=8 0.993±0.001** (all seeds 0.99-1.00, ~59k tokens to 50%, extrap 0.391), learned
width from 4 0.778±0.029. Interpretation of *why* it works is inferred from the basin curves, not separately tested.
Scheduled-width arm vs the rest: low data N = 64 / 256 / 1024 / 4096 = **0.279**±0.021 / **0.898**±0.043 / 0.995 / 0.992 (rung 0: 0.966 / 0.996 / 0.995 / 0.994);
fwp 0.354 / 0.852 / 0.921 / 0.937. 4x / 8x stream: 0.391 / 0.220 (rung 0 0.626 / 0.375; gru 0.491 / 0.294).

## 4. Learned structure on pointer

`SSPPointerLearned` (role gate per token, soft asymmetric bind, learned query code; hop count still given). Conv id / held-out hops 4-8:
plain 0.803±0.108 / 0.658±0.187 (**per seed 1.0, 0.51, 0.51, 1.0, 1.0**); aux-only (per-hop CE on intermediate nodes) **1.000 / 1.000**;
curriculum-only (1 hop first) 0.800±0.109 / 0.659 (same bimodality); aided 1.000 / 1.000 (~54k tokens); identity 1.000 / 1.000.
Baselines: gru 0.223 (1500 updates), **0.450±0.076 at 6000** (id@3000 0.378; held-out hops 0.142; per seed 0.46, 0.45, 0.58, 0.14, 0.62);
tf_full 0.153 at 6000. The GRU is learning, slowly; it is nowhere near competence and does not extrapolate to more hops.

## 5. Robustness (`rob1`, rung 0 vs baselines)

- Index, 4x / 8x stream: ssp 0.626 / 0.375; gru 0.491 / 0.294; fwp 0.409 / 0.280; tf_full 0.368 / 0.248. SSP is highest, everything collapses
  past the D/4 capacity rule; forgetting does not change it.
- Pointer input-token noise 10% / 25%: ssp 0.734 / 0.470; baselines are at chance regardless, so no comparison is possible.
- **Pointer cue-drop (25/50/75%) is NOT a valid test for the SSP arms** (section 7, item 2). Do not cite 0.999.

## 6. Decision (docs/11 4.5, mechanical output of `analysis.py` plus judgment)

- **delay: `algebra_buys_nothing`.**
- **index: `structural_prior_only`.** A learned rung (width-scheduled) is within tolerance and separates from GRU/tf_full at N=256, but the
  `fwp` control is not separated in-distribution (0.993 vs 0.945, 4.8 points), ties at N=256, is ahead at N=64, and the pos_table / no_cleanup /
  no_decay ablations match the full arm.
- **pointer: `thesis_pass_candidate` (provisional).** Within tolerance, separates on speed and on held-out hops (1.000 vs 0.142-0.212).
  Conditions before it can license RL: (a) it needs per-hop supervision to be reliable (plain: 3/5 seeds); (b) the baselines are weak (GRU 0.45 at 4x the
  budget), and no non-SSP iterated-structure control exists (no looped/universal transformer, no fwp on pointer); (c) GRU was not given an
  equivalent aid; (d) robustness to cue corruption is untested.

## 7. Bugs, errors and limitations (recorded so they are not rediscovered)

1. **Rung-0 pointer arm was broken** (commutative bind + one extra unbind). Fixed with a fixed random *permutation* P (a phasor does not work: the
   predecessor term inverts to the same phase) and by carrying the last executed hop's scores out of the loop. RUN_007 section 10 amends 4.3.
2. **Cue-drop bypass:** `SSPPointerArm` and `SSPPointerLearned` read `batch["start"]` / `batch["hops"]` directly, not the (corrupted) query features,
   so cue corruption cannot reach them. Needs an arm that reads the cue from the same features as the baselines (rung 3 territory).
3. **First `fwp` control was a strawman**: write keys came from the item only, so it could not address by position (0.3). Quarantined as
   `results/invalid_buggy_fwp_*`; fixed to key on the fixed time features; passes the gate (1.000) and is re-run. Never cite the old files.
4. **Temperature annealing is a no-op in the index learned arm** (no softmax temperature is used); "aided" on index therefore means address
   supervision plus the small-index curriculum. It does apply on pointer.
5. **Analysis bugs found on real data (synthetic tests missed them):** baseline picked by id then credited on other axes (cherry-picking); pool-trained and
   other-budget rows merged into the main table; speed averaged over only the seeds that reached the threshold; identity-init arm counted as a
   learned rung; table column and JSON-key bugs. All fixed with regression tests (`test_analysis.py`, 9 scenarios).
6. **`test_fixes.py` regression** from new required `run_sweep` args; fixed in the code (optional attributes), not the test.
7. **`mlp` and `posmlp` are now the same arm** (the clock is in the shared features).
8. Learned-address error in `diag_address.py` is only meaningful at length_scale 1 (the write phase is also trained; absolute error vs the index is
   not alignment for wider kernels). Accuracy columns are the reliable ones.
9. **The width schedule is hand-built**: ls_start in {4, 8} and width_steps=750 were chosen after seeing the basin curves; only 3 width conditions were
   tried, with 6 random configs each. The winning config used a learning rate at the top of the searched range.
10. Limits of the setup: LRU is not S4D; kernel_attn is purely positional; windowed arms are tiny; one budget (256) for most arms; one task for the learned-index
    result; thresholds (5 points, half the tokens, 2 pooled se) are the handoff's proposals; component-level supervised comparison, not SSP inside a larger
    model or an RL agent.

## 8. Reproduce

Local: `OMP_NUM_THREADS=1 ../../.venv/bin/python run_sweep.py --mode sweep --variants index --arms ssp_index_learned --label ssp_index_learned_r1_ws8
--arm-cfg '{"rung":1,"init":"random","length_scale":1.0,"ls_start":8.0,"ls_mode":"sched","width_steps":750}' --tag learn3_ws8 --budget 256 --dense-eval
--configs 6 --seeds 5 --steps 1500 --workers 12`. Remote: `python scripts/launch_ssp_pod.py --name <n> --prefix ssp_algebra/<n>/ --sizes 4 --sweep "<run_sweep args>"`
(one 4-vCPU CPU pod per job, results resumable on R2, pod self-deletes on success; arm-cfg JSON must contain no spaces). Tests: `test_harness.py`,
`test_fixes.py`, `test_measurement.py`, `test_learned.py`, `test_trainability.py`, `test_analysis.py`, `selftest.py`, all green at the end of this run.

## 9. Do not do

- Do not cite the first `fwp` numbers, the `rob_cue*` pointer numbers for SSP, or `last5_*` as accuracy.
- Do not read the pointer pass as licence for RL until section 6 (a)-(d) are addressed.
- Do not claim the index advantage is SSP-specific: the position table, learned codes and the `fwp` control say otherwise.
- Do not compare any of this with POPGym's published numbers.
