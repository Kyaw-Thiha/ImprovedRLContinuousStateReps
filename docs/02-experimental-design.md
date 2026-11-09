# Experimental Design

## The 9-config grid
For each (algorithm, task) pair, the full config set is:
- Representation ∈ {none, tile-coding, SSP}
- Centering ∈ {none, simple, value-based}

## Staged rollout plan

### Stage 0 — Toy-task validation (no GPU, near-zero compute)
- Task: classic CartPole (Gymnasium, 2 discrete actions, 4-dim state)
- Algorithms: DQN-linear, DQN-deep (2 hidden layers). A2C already has legacy results; being re-ported to the same plain-PyTorch harness for a fair comparison.
- Purpose: prove the representation port + both centering implementations work correctly, cheaply, before spending real compute.
- Compute expectation: minutes per run on CPU — no GPU needed or beneficial at this task size.

### Stage 1 — DROPPED
- Was: mjlab CartPole (torque-controlled, continuous) as a cheap pipeline sanity check before Stage 2.
- Decision: skipped. Risk accepted: if Stage 2 misbehaves, harder to isolate whether it's the task, the centering patch, or the representation, since more changes simultaneously. Minimal mitigation: run a short zero-action/random-agent sanity check with centering active before the full Stage 2 sweep, rather than a full separate stage.

### Stage 2 — Real robotics, low risk
- Task: `Mjlab-Lift-Cube-Yam` (arm reaching/lifting, state-based, not vision)
- HexSSP applied specifically to the 3D Cartesian observation terms (`ee_to_cube`, `cube_to_goal`) — direct generalization of the original RatBox spatial-nav result to a real robot arm. Joint-space terms stay raw.
- Algorithms: A2C (custom training loop against mjlab's env), PPO (RSL-RL, with centering patch). SAC optional/deferred — see `04-algorithms-and-centering.md`.

### Stage 2a — SSP x centering on continuing tasks (new, 2026-10-06)

Replaces locomotion-first as the next representation study. Full task survey and duplication map:
[`09-ssp-centering-task-survey.md`](09-ssp-centering-task-survey.md); the mechanism and hypotheses being tested:
[`08-ssp-vs-tile-coding-theory-and-hypotheses.md`](08-ssp-vs-tile-coding-theory-and-hypotheses.md).

- **Criterion:** continuing throughout; the metric is the reward rate (average reward), not episodic return.
- **Tasks:** Tier 1 = continuing Reacher with a resampled target, continuing Swimmer with raw wrapped angles, and
  continuing CartPole as the negative control. Tier 2 = continuing Pusher, then a custom multi-object
  relative-position task. Testbeds for the first three already exist publicly (see `05-tasks-and-benchmarks.md`).
- **Encoder arms:** SSP on the structured dimensions, tile coding on the same dimensions (matched budget), a raw
  linear readout, **and a raw learned-network arm** — the last one because real deployments use learned networks,
  so a claim that omits it is a claim about linear function approximation only.
- **Centering arms:** off vs TD-based (Naik et al. arXiv:2405.09999); gamma in {0.99, 0.999} x reward offset in
  {0, c}. Centering's benefit is largest as gamma -> 1 and under a common offset, so both axes are needed.
- **Two evaluations no existing testbed forces, and both must be added:** (1) a shifted-distribution evaluation
  (train on region/layout A, evaluate on B), where a fixed local basis cannot transfer and a distance-based kernel
  predicts it can; (2) a control-imperfection axis (action delay, action noise, observation noise), where this
  project already has four-pair significant evidence in CartPole.
- **Success criterion is a prediction, not a ranking:** see `09-...md` section 5 for P1-P4 and their falsifiers.
- **What would falsify the direction:** if the SSP arm matches the matched tile-coding arm on reward rate *and* on
  the shift evaluation *and* on the delay/noise axis, then the representation buys nothing beyond a linear
  trend and the write-up becomes a negative result about structured encoders in RL.

### Stage 3 — Advanced / stretch
- Tasks: `Mjlab-Velocity-Flat-Unitree-Go1` (quadruped) first, then `Mjlab-Velocity-Flat-Unitree-G1` (humanoid) — quadruped first because passively more stable.
- Explicitly out of scope: rough terrain variants, vision/depth manipulation tasks (`Mjlab-Lift-Cube-Yam-Rgb/-Depth`), multi-cube segmentation.
- Algorithm: PPO only, primarily. Rationale: mjlab's shipped reward/curriculum configs for these tasks are PPO-tuned; SAC is unproven at this scale in this ecosystem and real GPU-hours-expensive (see `06-related-work.md`, Average-Reward SAC compute note). Tests whether HexSSP's dimensionality advantage holds at much higher state dimensionality than previously validated.
- Requires episodic→continuing conversion (see `05-tasks-and-benchmarks.md`).

## Algorithm roster and task compatibility

| Algorithm | Action space | Where it runs |
|---|---|---|
| DQN (linear) | discrete | Classic CartPole only |
| DQN (2 hidden layers) | discrete | Classic CartPole only |
| A2C | continuous (Gaussian policy) | CartPole (legacy re-port), Stage 2 manipulation |
| PPO (RSL-RL) | continuous | Stage 2 manipulation, Stage 3 locomotion |
| SAC | continuous | Optional, Stage 2 only if added — not RSL-RL's maintained branch, needs separate implementation (e.g. Stable-Baselines3-based) |

DQN is scoped to discrete-action tasks only — mjlab's robot tasks are continuous-action (joint position/effort control), and discretizing joints to force DQN compatibility is not attempted (distorts the action space, not standard practice in the field).

## Hyperparameter tuning methodology

Each of the 9 configs (3 representations × 3 centering modes) is tuned **independently**, per architecture — a full Optuna search per cell, not a reduced/staged procedure. An earlier version of this project tuned representation/network hyperparameters once per representation with centering off, then locked those and only swept the centering step size for the other two centering modes. That staged design was retired: reusing centering-off hyperparameters for centering-on runs assumes centering doesn't shift the optimal learning rate/epsilon/etc., which is exactly the kind of hidden bias the hyperparameter-tuning-fairness literature warns against (see citations below) — a centering variant could look artificially worse (or better) purely because it never got its own search.

The search methodology itself directly follows Antonin Raffin's two-part guide on automatic hyperparameter tuning for RL:
- [Automatic Hyperparameter Tuning — A Visual Guide (Part 1)](https://araffin.github.io/post/hyperparam-tuning/) — samplers/pruners conceptual background, the "n vs. B/n" budget tradeoff.
- [Automatic Hyperparameter Tuning — In Practice (Part 2)](https://araffin.github.io/post/optuna/) — the concrete Optuna recipe this project's search script implements almost verbatim (search space definition, `TrialEvalCallback`-style periodic deterministic evaluation, `TPESampler`+`MedianPruner` configuration, and the "re-evaluate top trials to filter lucky seeds" post-hoc step).

Concretely, per cell:
1. **Single seed per trial during the search** (not averaged over multiple seeds) — multi-seed averaging during active search was Part 2's own explicit anti-pattern ("expensive... and makes it difficult to prune out bad trials early"), and Eimer, Lindauer & Raffin's *"Hyperparameters in Reinforcement Learning and How To Tune Them"* (ICML 2023, [arXiv:2306.01324](https://arxiv.org/abs/2306.01324)) found empirically that using more than ~3 seeds during tuning itself can make the search *harder* without reliably better held-out performance.
2. **A separate, deterministic evaluation environment** (no exploration epsilon) is checked in on periodically during training — not the noisy training-reward trace — matching Part 2's `TrialEvalCallback` pattern.
3. **`TPESampler(n_startup_trials=10, multivariate=True)` + `MedianPruner(n_startup_trials=10, n_warmup_steps=N_EVALUATIONS // 3)`** — no trial can be pruned before 1/3 of its budget, per Part 2's explicit recommendation. Replaces the previous `HyperbandPruner` setup, which had no equivalent warmup floor.
4. **500 trials per cell** (Part 2 cites 500 as rl-baselines3-zoo's own default trial budget).
5. **Post-hoc validation (re-filter)**: candidates are the top 20 completed trials ranked by the mean of their last 5 greedy eval checkpoints (read from the study's intermediate values), each re-run with 10 fresh greedy seeds (1000+) — flat 20 x 10 (the pilot found seed noise ~3x config differences among top candidates, so a staged 3-seed cut was dropped as the default; the staged code path remains behind `--stage1-keep`). Each seed's score is the mean of its last 5 greedy checkpoints; the winner has the best mean (tied -> lower `np.std`). Runs are parallel (one process per vCPU) and resumable via a JSONL on R2. This implements Part 2's "re-evaluate the top trials multiple times to find out which ones were lucky seeds"; see `docs/handoffs/02-dqn-v4-greedy-refilter-plan.md` and decision log entries 14-15. Results are written to new `*_v2` locations; the earlier online-return reeval is superseded.
6. **Final reported result** for the winning config per cell uses ≥20 seeds, per Colas, Sigaud & Oudeyer's *"How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments"* ([arXiv:1806.08295](https://arxiv.org/abs/1806.08295)) — their recommended floor for a statistically reliable standard-deviation estimate, and they recommend going larger still for the actual confirmatory comparison. This directly supersedes this project's earlier ad hoc "5 seeds for final ablation" convention, which sits at the sample size Henderson et al.'s *"Deep Reinforcement Learning That Matters"* ([arXiv:1709.06560](https://arxiv.org/abs/1709.06560)) identified as symptomatic of the field's reproducibility problems.

## Post-search evaluation (eval_v3)

After the search / re-filter / final eval, the winners are retrained on fresh seeds (3000-3019, env seeded) with the final policy saved, and the saved policies are used for robustness (21 test-time perturbation conditions, pre-declared SSP-vs-tile claim on the 8 cells {tile_coding, ssp} x {none, value} x {linear, deep}, Holm-corrected), deployment (size, fp16/int8 quantization, latency) and a DQN-settings test (replay buffer, target update). Protocol, storage layout and result tables: `runs/RUN_004.md`; rationale: decision-log entries 19-24; key layout: `scripts/eval_v3_layout.py`.
