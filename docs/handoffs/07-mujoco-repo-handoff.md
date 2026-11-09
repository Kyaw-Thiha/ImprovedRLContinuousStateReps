# Handoff 07: the new MuJoCo repo — what to port, what is already decided, and the pitfalls

**Update 2026-10-06 (later the same day).** This handoff describes the *spatial-encoding* branch (SSP as an input
representation to a learned policy). It is **not the live direction**. The live direction is now SSP as memory —
content-addressable recall with cue completion — with a benchmark plan in
[`../10-vsa-memory-and-composition-direction.md`](../10-vsa-memory-and-composition-direction.md) sections 12-13 and
decision-log entries 32-33. Read this file only if the spatial-encoding branch is revived; the pitfalls in
sections 5-6 (tile coding on continuous actions, IHT saturation, matched arms, SB3 as the cheap route) still apply
if it is.

Written 2026-10-06. The MuJoCo / continuing-task work moves to a **new repo**. This file is the boundary:
what stays here, what to port, what is already decided, and what would be wrong to assume.

Task selection and duplication map: [`../09-ssp-centering-task-survey.md`](../09-ssp-centering-task-survey.md).
Mechanism and hypotheses: [`../08-ssp-vs-tile-coding-theory-and-hypotheses.md`](../08-ssp-vs-tile-coding-theory-and-hypotheses.md).
Design: [`../02-experimental-design.md`](../02-experimental-design.md) Stage 2a. Decisions: `../07-decision-log.md`
entries 26-29.

---

## 1. What stays in this repo

- **CartPole + DQN + the A2C port.** CartPole is now the **negative control**, not the test. Its value is (a) the
  clean "where a fine grid is affordable, the grid wins" boundary condition, (b) the robustness suite (delay
  retention 0.28-0.54 for tile coding vs 0.92-0.99 for SSP, four primary pairs), and (c) the cheap interaction pilot.
- The 27-cell `eval_v3` record, the policy-verification pattern, and the seed discipline.
- The **legacy A2C provenance problem** (entry 28): unresolved, and it stays unresolved here. No claim about
  "SSP is worse under A2C than reported" until the winners are re-derived from the surviving Optuna DBs.

## 2. What to port

| Thing | Where it lives here | Note |
|---|---|---|
| SSP encoder | `rl/representations/ssp.py` (current `sspspace`) and `rl/representations/ssp_legacy.py` (the A2C-era one: `n_scales` 8, scipy-1.7.3-pinned rotations) | The old scipy rotation draw matters only for reproducing legacy numbers; for new work use the current encoder and fix the rotation draw explicitly |
| Tile coding | `rl/representations/tilecoding.py` (Sutton's `tiles()` + `IHT`) | Built for the DQN sparse-index path and a **discrete** action space. For continuous actions you need the dense/hashed multi-hot features feeding a Gaussian head — see section 5 |
| Centering (two estimators) | `rl/networks/dqnBasic.py` lines 290-293 and 328-330 | `simple` = running average of observed rewards; `value` = TD/Bellman-error update. Both already exist, both need re-deriving under the new formulation (section 4) |
| Centering (A2C) | `rl/rules/td0centering_nengo.py`, `experiments/trial_cartpole_a2c.py` | The NumPy port of the legacy centering rule; useful as a reference for the per-step form |
| Policy save / reload / verify | `rl/policy_io.py`, `scripts/policy_pass_worker.py` | The "a policy is done only when a reloaded policy replays the in-memory episodes exactly" rule is worth keeping verbatim; it caught real bugs |
| Run layout and seeding conventions | `scripts/eval_v3_layout.py`, `docs/02-experimental-design.md` | Seed blocks disjoint per phase (search / filter / final / policy pass); greedy last-k scoring; bootstrap-over-seeds as the unit of analysis |

## 3. What is already decided (do not re-litigate)

1. **The comparison is an interaction, not a ranking.** Centering x encoder, crossed. Main effects are already in
   the literature; the novel quantity is whether centering substitutes for a good encoder or is independent of it.
2. **The estimator matches the data regime, not the algorithm.** On-policy (PPO, A2C) -> the running-average
   estimator is sound and is what Naik et al. used for PPO. Off-policy (SAC, TD3, DQN-with-replay) -> use the
   TD-based form; a moving average over replayed rewards is the unsound variant (Wan et al. state this explicitly).
   The *factor* manipulated in every arm is centering on/off; the estimator is matched to the regime by design.
3. **Four encoder arms, not two:** SSP on the structured dimensions, tile coding on the same dimensions, a raw
   linear readout, **and a raw learned network** — the last because real deployments use learned networks, so a
   claim without it is a claim about linear function approximation.
4. **The axes that make centering bite:** reward offset (add constant `c` per step) and gamma (0.99 vs 0.999).
   Without an offset, a null centering result is uninterpretable — the literature's headline is offset invariance.
5. **Two evaluations no existing testbed forces:** a shifted-distribution evaluation (train on region/layout A,
   evaluate on B) and a control-imperfection axis (action delay, action noise). Both are where this project already
   has evidence.
6. **A true continuing conversion rather than the termination correction**, where possible: on failure, reset the
   state and charge a penalty, with **no terminal flag**. That leaves no terminal transitions, so the centering
   theory needs no modification. De Asis et al.'s termination-corrected differential TD is for tasks that cannot be
   converted — CartPole, and it stays in the old repo.
7. **Penalty-and-reset rather than the true no-reset variants for the first pass**, because PPO (our
   centering-ready algorithm) learns the former and fails on the latter. SAC enters only if a task genuinely needs
   the no-reset regime.

## 4. Do not build on the old centering results

Our CartPole centering numbers (`outputs/eval_v3/*/train_summary.json`, summarised in RUN_004 3.1 and 3.6) were
measured under the **naive episodic application** that decision-log 17 flags and that De Asis et al. (2026) have
since superseded. Two further properties of that evidence:

- Each cell is an **independently tuned search**, so a simple-vs-value difference confounds centering with each
  cell's own tuned hyperparameters. It is not a controlled same-config A/B.
- The pattern is **representation-dependent, not algorithm-dependent**: `simple` is catastrophic where the
  representation is strong under DQN (ssp linear -148, ssp deep -120 relative to `none`) and helps where the
  representation is weak (discrete +16 to +168 under both algorithms), while `value` is roughly neutral everywhere
  (|delta vs `none`| <= 11 in all nine non-discrete cells, against per-seed sds of 7-27).

So: use those numbers as a motivation for the interaction hypothesis, not as evidence for an estimator rule. The
estimator rule in section 3.2 comes from theory and from Naik/Wan, not from CartPole.

## 5. Pitfalls specific to the MuJoCo port

- **Tile coding on continuous actions is not the DQN path.** Our rep returns a sparse index list; for a continuous
  action space you need the dense multi-hot vector (or a hashed equivalent) feeding a Gaussian policy head.
  `map()` already exists for this; `map_indices()` is the DQN-only fast path.
- **The IHT saturates at every setting we ever ran** (`08-...md` section 6: 100% dictionary fill and 81-98% hash
  collisions over the reachable box, at every `iht_size` including the ceiling). Size the hash to the
  active-feature count, or use a coarse grid on the structured dimensions only. For Reacher the relative vector is
  2-D, so a genuine grid there is cheap — which is the fair way to give tile coding its best shot.
- **Match the arms.** Same input information, same active-feature budget, same parameter count for the readout;
  report dimensions and parameter counts per arm. The old Stage-2 plan was "HexSSP on the Cartesian terms only,
  other terms raw", which compares encoder *and information*; both arms must see the same vector.
- **SB3 is already a dependency here** (`requirements.txt:29`, `stable-baselines3>=2.0,<=2.3.2`). The cheap route:
  wrap the env so the observation *is* the encoded feature vector (the encoder is deterministic and cheap), then
  use `policy_kwargs=dict(net_arch=[])` for the linear-readout arm and the default MLP for the learned arm. That
  reuses this repo's encoder code exactly and avoids writing a training loop.
- **PPO centering is not implemented here.** There is no `compute_returns` patch in this repo (checked). For SB3
  PPO the on-policy estimator is a running average of observed rewards subtracted before returns/advantages are
  computed — a few lines.
- **muJoCo itself is not a dependency here** (only `gymnasium`). `gymnasium[mujoco]` in the new repo. Nothing in
  Stage A-C touches mjlab or RSL-RL; mjlab is Stage 3 and is deferred.

## 6. Eval focus: one axis per claim

Every measurement below exists to test a *specific* claim about SSP or centering. Do not run a robustness battery
"because robustness is good"; run the axis that the claim needs, and state the prediction first. Claims and their
provenance: `../08-...md` section 5 and section 2; the referenced paper's own claims: variance and changed-physics
robustness.

| # | Claim under test (source) | Eval axis | Prediction / falsifier |
|---|---|---|---|
| 1 | Compositional continuous space via fractional binding; unbounded/undeclared ranges (SSP literature) | **Distribution shift**: train on region/layout A, evaluate on a shifted B; and evaluate outside the training range | SSP degrades less than the grid. Falsified if they degrade together — then "unbounded" buys nothing here |
| 2 | Relative / relational geometry (binding = translation) | **Train/test geometric translation**; evaluate with the whole configuration translated | The learned value transfers; the grid must relearn. Falsified if transfer is equally good for a grid of matched resolution |
| 3 | Graceful degradation under corruption (HDC's core claim; the paper's robustness result) | **Observation noise and dimension dropout** at several levels | SSP's degradation curve is flatter. Falsified if the curves coincide |
| 4 | Low variance across seeds is one of the paper's *headline* claims (sigma^2 128 vs 20966, Levene p=0.017) | Report **seed variance as a first-class result**, with a pre-declared variance test (Brown-Forsythe/Levene) on the per-seed reward-rate distributions, plus interquartile mean and performance profiles | This is a claim to *test*, not assume: our own CartPole data contradicts it under the greedy metric (SSP sd 21-28 vs tile 10-12 in several cells) |
| 5 | Fixed-size representation of a product space (dimension grows linearly with bound pairs) | **Composition scaling**: reward rate vs number of bound entities N at matched feature budget and matched readout parameters | The grid's cost grows as `bins^(2N)`; the encoder's stays fixed. Falsified if the SSP arm also degrades with N |
| 6 | "30-65x smaller" and cheap inference (project claim, currently unresolved) | **Cost at matched performance**: parameter count and analytic op counts at equal reward rate — *not* wall-clock across two languages | Only meaningful with matched search ranges; the old latency comparison was pure-Python vs NumPy and is not evidence |
| 7 | Latency tolerance is the deployment-relevant axis (our own four-pair result: delay retention 0.28-0.54 vs 0.92-0.99) | **Action delay** at eval, plus the C1 test: randomise delay *during* training | If training with randomised delay gives the grid SSP's delay tolerance, the recommendation becomes "train for the delay", not "use SSP" |
| 8 | Interference, the mechanism proposed for SSP's deficit (Liu et al.) | **Interference diagnostic** (change in squared TD error on held-out probe states), not a headline metric | SSP's interference is materially higher; if so, the mechanism is established rather than inferred |
| 9 | Centering: offset invariance and gamma -> 1 (Naik et al.) | **Reward offset** and **gamma** as crossed factors, with a true continuing conversion | Centering removes offset sensitivity for every arm; the open question is whether the size of its benefit depends on the encoder |

Two reporting requirements that follow from the above: paired comparisons on **common random numbers** (same seed ->
same environment stream across arms), and the unit of analysis is the **seed**, with bootstrap CIs, as in this repo.

Do **not** spend eval budget on: int8/fp16 quantisation (already free for all 27 CartPole cells, and a property of
both representations), wall-clock latency across implementations in different languages, or a size claim without
matched search ranges.

## 7. Does any of this matter for deployment? (honest ledger)

Written because the question keeps coming up: if SSP only wins on generalisation-under-shift, is that enough to
justify it for deployment? Ordered by *deployment* value, not by statistical significance:

| Axis | Status today | Verdict for deployment |
|---|---|---|
| **Action/actuator delay tolerance** | Measured, large, replicated: retention 0.28-0.54 (tile) vs 0.92-0.99 (SSP), significant in all four primary pairs | The strongest candidate. **But it is currently confounded with tuning**: SSP's winners are low-gain (`dQ/dx` 4.3-11) and tile coding's are high-gain (35.8-50.6), because each was tuned for the *score* objective. If C1 (delay-randomised training) or C2 (gain-penalised tile coding) buys tile coding the same tolerance, this is a property of the objective, not of the representation |
| **Variable-length / set-valued state** | Never tested. An SSP can absorb N bound pairs into a fixed-size vector, so one network can ingest a varying number of objects or a changing sensor suite; a grid or MLP cannot (its input dimension is fixed) | The only *structurally unique* capability. If it holds, it is deployment-useful in a way nothing else in this comparison is — variable object counts, sensor dropout, hardware changes |
| **Noise / dropout robustness** | Partially measured: SSP wins `obs_noise` on retained score in 3 of 4 primary pairs, `action_delay` in 4 of 4, `random_action` in 3 of 4 | Real but second-tier; also plausibly subsumed by the gain story above |
| **Generalisation under shift with less training diversity** | Untested | Real but *substitutable*: domain randomisation buys the same invariance (and also covers the non-algebraic changes SSP cannot represent). The defensible form of the claim is quantitative — how many fewer environment variants does SSP need to reach the same transfer level? Training diversity costs money, so a measured ratio is deployment-relevant |
| **Memory** | Loses on the currency that matters: storage favours SSP 30-65x, per-decision compute favours tile coding by 1-3 orders of magnitude, active parameters 32 vs 1001 | Not a deployment advantage. Memory is cheap; latency and energy bind — and the HDC story of cheap bind/bundle assumes VSA-native hardware, not dense matmuls |
| **Task score** | Parity or a small loss when both are tuned | Irrelevant for deployment if both solve the task |
| **Seed variance (the paper's own headline)** | Contradicted under our greedy metric (SSP sd 21-28 vs tile 10-12 in several CartPole cells) | Not an advantage as things stand; worth a pre-declared variance test in the new repo |

Consequences for how the new repo reports results:

1. Report **tail and threshold metrics**, not just means — fraction of runs/episodes below a safety threshold, and a
   CVaR-style expected tail loss. A robustness advantage can exist while the means tie, and worst-case behaviour is
   what a deployment reads.
2. **Run C1/C2 early**, because they decide whether the one large effect we have is a representation property. If it
   is not, the recommendation becomes "randomise your delays and penalise gain" — still a useful, publishable
   finding, just not an SSP finding.
3. The honest bottom line for the write-up: today SSP has **no deployment advantage that has survived scrutiny**
   except possibly the delay axis, and that is unverified. Generalisation alone would not justify it. The new repo's
   job is not to prove SSP is better; it is to determine which of these four axes is real and to report the one that
   is — including the outcome where the answer is "none, and here is the cheaper substitute".

## 8. Stage plan (from `09-...md` section 5, unchanged)

```
Stage A  Harness + centering validation: InvertedPendulum, SB3 PPO, raw/tile/SSP x {none, centered} x 10 seeds,
         with the offset manipulation. Kill: if centering shows nothing here, fix before buying task runs.
Stage B  Reacher, resampled target, penalty-and-reset, PPO. 3 encoders x 2 centering x 2 offsets x 10 seeds
         = 120 runs of ~1M steps + the shifted-target evaluation.
Stage C  Swimmer, raw wrapped angles. 3 arms (clipped grid / grid with manual wrap / periodic encoder)
         x 2 centering x 10 seeds = 60 runs.
Stage D  Only if A-C show the interaction: SAC + TD-based centering on a true no-reset task.
```

Kill criterion for the direction, stated before running: if the SSP arm matches matched tile coding on reward rate
*and* on the shifted evaluation *and* on the delay/noise axis, the practical recommendation collapses to "use a fine
local basis and randomise your delays" — a useful negative result, and one Stage 2a in `02-experimental-design.md`
is written to accommodate.
