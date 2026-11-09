# Handoff 06: bound fixes, the re-search, and the hypothesis tests

Written 2026-10-06, immediately after RUN_004. This file carries forward **only section 6 of
[`runs/RUN_004.md`](../../runs/RUN_004.md)** plus the minimum context needed to act on it, so you do not have to
re-read that 900-line results record. Every claim below is derived there: the evidence is RUN_004 sections 3.7 and
3.8, and the conclusions are section 5. Nothing here is a new claim.

## 1. Where things stand

RUN_004 is complete and verified: 27 cells (18 DQN + 9 A2C) each ran a 20-seed policy pass with every policy
verified by exact replay, a 24-condition robustness suite, and deployment including latency; all 20 DQN-settings
variants are done; 32 figures exist. No pods are running.

Headline results (RUN_004 sections 3 and 5):

- **Tile coding >= SSP on the greedy score, under DQN AND under A2C.** Discrete is far behind both.
- **Robustness splits by perturbation type.** Tile coding wins the changed-physics families (`pole_length` in all
  four primary pairs, plus `wide_init` and `cart_mass`). SSP wins the control-imperfection families decisively:
  `action_delay` in all four pairs by 153-342 points of absolute score, and `random_action` in three of four.
- **The DQN settings do not explain the tile-vs-SSP gap.** No variant rescues SSP; slowing the target update
  destroys the best cell (`tile_coding/none/linear` -174.7 at update 100).
- **The referenced ICCM paper's number is reproduced under its own metric** (493.8-495.8 online against its CI
  [496.88, 499.97]) with a linear readout, but **not its architecture** (its 1024-4096 fixed random ReLU layer has
  never been run here). Its comparison set never included tile coding.

## 2. Why a re-search is on the table

Two findings, both from the post-run analysis:

**(a) Both representations sit against search bounds, so neither is at its own optimum.** Winner position in its
searched range, 0% = lower bound and 100% = upper:

```
SSP                                              tile coding
cell              length_scale  n_rotates  lr     cell                lr          num_tilings
none/linear         17%         100%       78%    none/linear          43%         100%
none/deep            1%          94%       14%    none/deep             5%         100%
simple/linear       80%           6%       97%    simple/linear        62%           0%
simple/deep         87%          56%       44%    simple/deep          22%         100%
value/linear         6%         100%       71%    value/linear         53%         100%
value/deep          21%           0%       26%    value/deep            2%           0%
```

The strongest single signal in the whole study is tile coding's `num_tilings`: **32 beats 16 in all six cells**, on
both the median score and the share solved (`none/linear` 480/56% -> 495/75%; `value/deep` 481/56% -> 496/80%),
with a positive rank correlation in all six and no exception. SSP's `length_scale` floor crowds (two-thirds of
trials below 0.14 in two cells) but the effect is modest, about 20-45 points between terciles, and it is flat in the
`simple` cells -- because centering, not the kernel, is what is broken there.

**Ranges:** SSP `length_scale` [0.1, 1.0], `n_rotates` [4, 20], `lr` [1e-4, 0.2]. Tile coding `lr` [1e-4, 1e-1],
`num_tilings` [16, 32], `tiles_per_dim` [10, 12, 16], `iht_size` [32768, 65536, 131072]. The paper used
`length_scale` down to **0.01** and rotations only {4, 5, 6, 7}, i.e. inside our range.

**(b) SSP sees raw state with ONE isotropic length scale, while the other representations rescale per dimension.**
`domain_bounds_` is computed in `experiments/trial_cartpole_dqn.py` but never passed to the SSP encoder, and
`normalize_state` is False in every cell of the grid and every A2C run. Tile coding and discrete rescale each
dimension by its own range (x +-4.8, v +-10, theta +-0.418, omega +-10). Resolving the pole angle (span ~0.4) and
the cart position (span ~5) with a single scalar is a plausible cause of the floor pinning. This is an asymmetry in
the comparison, not a bug, and it is untested.

## 3. Do this first: validity fixes (cheap, no search)

1. **Fix `wall_min`** in `train_summary.json`: it records `time.time() - t0` for the invocation that wrote the
   summary, so a resumed cell reports ~0.05 min for 20 training runs. Use the `cost` block meanwhile.
2. **Add `winner_trial`** to `stage0_cartpole/best_params/*_v2.yaml`. Today no yaml carries it, so the policy pass
   resolves the winner trial only by pulling the refilter summary from R2, and silently falls back to trial 0 if
   that pull fails.
3. **Shift the bounds; do not widen them.** The evidence is directional, and with a log-uniform prior widening
   lowers the sample density everywhere while spending new budget on the side already shown to be worse. Shift so
   the old bound becomes interior, keep the width in log space, and keep the current winner inside the new window so
   a re-search cannot do worse than what exists:

```
parameter                 current            shift to            old winner lands at
SSP length_scale          0.1  - 1.0         0.032 - 0.3         0.10  -> 50%
SSP n_rotates             4    - 20          4     - 32          (integer; extend the top)
SSP lr                    1e-4 - 0.2         3e-5  - 0.06        ~1e-3 -> 80%
tile lr                   1e-4 - 1e-1        3e-5  - 3e-2        1e-4  -> 17%
tile num_tilings          [16, 32]           [16, 32, 64]       32    -> 50%
tile tiles_per_dim        [10,12,16]         [10,12,16,24]      16    -> 67%   (weaker evidence)
SSP n_scales              5                  5 or 8             (paper uses 8; changes kernel shape)
leave alone               tile iht_size, eps, every centering param (interior; no bound pressure)
```

   **Tile coding must be shifted alongside SSP**, or the comparison is biased toward SSP in exactly the way the
   design docs tried to remove. If you do SSP only, label it exploratory and do not use it for the headline.
4. **Test observation normalisation as a factor**, SSP first and then tile coding and discrete, so all
   representations see a comparable domain. DQN already has `normalize_state` and `policy_io` supports it, so no new
   code is needed; the A2C encoder has no such option and needs a small addition.

## 4. The re-search plan (conditional on section 3 moving something)

**Stage 0 -- diagnostic. About $2, hours.** Shifted bounds plus normalisation on/off, ~6 `length_scale` values, 10
seeds, with all other parameters held at each winner's values. Run it on all six SSP DQN cells and the six
tile-coding DQN cells. A clear gain is trustworthy because holding `lr` fixed biases the design *against* the
changed representation; a null result is inconclusive for the same reason.

**Stage 1 -- only if Stage 0 moves something. Joint search + filter + final eval.** Search per cell over
`length_scale`, `lr`, `n_rotates`, `n_scales` (SSP) or `lr`, `num_tilings`, `tiles_per_dim` (tile coding), ~150
trials, then the SAME filter (top-20 x 10 seeds) and final eval (20 fresh seeds) as RUN_003, so the result is
like-for-like. Roughly 6-12 h of search plus ~5 h of filter and final per DQN cell: about 80 pod-hours for the six
SSP cells, a similar amount if tile coding is shifted too, i.e. **~$10-18 and roughly a day of wall time** with
parallel pods. A2C cells are 10-90 s per run, so the arm is a few hours (~$2).

## 5. Hypothesis tests (new knowledge, independent of section 4)

- **C1 -- delay-aware training. Highest practical value.** Randomise (or compensate for) action delay during
  training, then re-test the delay family. This is what the sim-to-real literature actually does about latency, and
  it asks whether SSP's delay advantage can be obtained by any representation once delay is part of training. If it
  can, the conclusion becomes "train for the delay", not "use SSP".
- **C2 -- the gain/delay mechanism. Highest novelty.** Vary the control gain WITHIN one representation (gain-
  penalised tile coding, or an SSP `length_scale` sweep) and test whether delay retention tracks `dQ/dx`. That
  turns RUN_004 section 3.8 from a four-point correlation into a mechanism, and it yields a deployment-useful claim:
  measure `dQ/dx` before shipping instead of measuring delay robustness after.
- **C3 -- the paper's random-ReLU arm, A2C only (~$2).** Fixed random ReLU (1024/4096) -> linear readout, reduced
  100-trial search, for SSP AND tile coding AND a raw-input baseline so the comparison stays fair. Settles whether
  the paper's level needs random features (section 3.7 suggests probably not: we already reach 494-496 without).
- **C4 -- size-matched sweep.** Lower the tile-coding `iht_size` floor below 32768 and the `tiles_per_dim` floor,
  and let SSP `n_rotates` go down to 1-2, so the two representations meet. This is the outstanding half of the
  "30-65x smaller" question, which is currently unresolved in both directions.
- **C5 -- then the existing priority list** in handoff 03 section 6: dimension scaling (stacked history k=2/4/8,
  then Acrobot and LunarLander), centering cleanup on a continuing CartPole, hex-vs-random SSP and
  random-Fourier-feature ablations, then mjlab.

## 6. Do not do

- **No ReLU between the linear cells' layers.** Those cells have no hidden layer by design; adding one deletes the
  linear/deep ablation, which is itself a result (deep helps SSP, does not help tile coding or discrete). The deep
  arm already applies ReLU between its hidden layers -- verified in `dqnBasic.forward_dense`.
- **No re-run of the 27 cells.** Their question is answered, the data is verified, and the replicate matches
  RUN_003 within noise.
- **No search with unshifted bounds**, and no per-representation tuning of the DQN settings (target update is
  load-bearing for tile coding, so tuning it only for SSP biases the comparison the other way).
- **No SB3 deep-MLP baseline** unless a reviewer asks. Its only role is the paper's variance claim; report that
  comparison as unreproduced while reproducing the SSP side of it.

## 7. Tooling that already exists -- do not rebuild it

| script | what it does |
|---|---|
| `scripts/watch_eval_v3.py` | the eval-pod watcher and cost backstop: progress from R2 markers, `--relaunch-missing`, `--dedupe`, `--delete-finished`. Never touches pods not named `eval-v3-*`. |
| `scripts/start_eval_stage.py` | starts `robustness`/`deployment` only for cells whose policy pass is finished, capped per call. Dry-run by default. |
| `scripts/launch_cell_pod.py` | pod launcher. Jobs: `search`, `policy_pass`, `robustness`, `deployment`, `latency`, `dqn_settings`. Honours `RUNPOD_REGISTRY_AUTH_ID` if set in `.env`. |
| `scripts/run_latency_sweep.py` | latency for ALL cells on ONE quiet pod (27 pods would be 27 image pulls). |
| `scripts/run_dqn_settings_sweep.py` | all five DQN-settings variants for one cell on one pod. |
| `scripts/plot_eval_v3.py` | every figure plus sidecar CSVs, in the house plotting style. |
| `scripts/validate_a2c_replay.py` | replays recorded legacy value series through the port. Currently INCONCLUSIVE (it disagrees on `none` centering too, which gates cleanly), so do not use it as evidence without fixing the encoder-domain assumption. |
| `scripts/analyze_robustness.py` | retained scores, bootstrap CIs, Holm correction (`--a2c` for the A2C arm). |

## 8. Ops lessons that cost hours this session

- **Docker Hub's anonymous pull limit kills pods** (`toomanyrequests`). Six of the first eighteen policy-pass pods
  died that way; retrying works because it depends on the node. `RUNPOD_REGISTRY_AUTH_ID` is wired into both pod
  builders and documented in `.env.example`, but it is unset -- set it and this stops.
- **Never infer "which cells are outstanding" from the pod list.** A finished pod self-deletes, so completed cells
  look missing and get relaunched into duplicates. That bug produced dozens of duplicate pods before it was fixed;
  decisions are now made from R2 markers.
- **Pod names encode only (job, cell)**, so DQN-settings variants share a name unless a suffix is added (fixed).
  Dedupe now only deletes pods whose `JOB_ARGS` are byte-identical, so an ambiguous group is left alone.
- **Prefer one pod for many cells or variants** (the latency sweep, the settings sweep). It limits image pulls and
  keeps measurements on a single machine.
- **`cerb-*` pods belong to someone else.** Never stop, restart or delete them.
- Timings are not comparable across cells: the fleet ran on EPYC 7702P / 9354 / 9654 / 9655 / 9655P hosts, and some
  pods were 2 vCPU with 4 workers.

## 9. Open decisions for the next session

1. Shift tile coding alongside SSP, or SSP only (labelled exploratory)?
2. Run Stage 0 across all twelve DQN cells (six SSP + six tile coding), or a subset first?
3. Is the referenced paper's architecture in scope at all (C3), or is the project's question settled as "where is
   SSP better under one fair protocol"?
4. Whose call is the compute for Stage 1 (~$10-18, about a day of wall time)?
