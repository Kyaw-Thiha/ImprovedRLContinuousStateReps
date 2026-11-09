# Project Overview

## What this project is
Extending prior work on Hexagonal Spatial Semantic Pointer (HexSSP) state representations for RL (ICCM 2022 paper, `ImprovedRLContinuousStateReps` repo).

**Direction as of 2026-10-06 (see decision-log entries 26-33).** The original plan — a 3x3 representation x centering grid on CartPole, then mjlab robotics — has been re-scoped by what the evidence supports:

1. **CartPole is now the negative control**, not the test. Its result stands: tile coding >= SSP on the greedy score, with SSP winning the delay/noise robustness families (4/4 primary pairs) and losing on size-for-compute.
2. **No re-search, no legacy recreation** (entries 29-30). The comparison protocol already satisfies the equal-budget standard; the outstanding items are cheap adversarial tuning plus free reanalyses of existing artifacts.
3. **Reward centering stays an axis but is not the protagonist** (entry 31). Its contribution is now the *interaction*: centering's benefit as a function of the approximator's ability to represent the constant function, with a bias feature as the moderator.
4. **The live SSP direction is memory, not state encoding**: content-addressable memory with cue completion (the SPA's declarative-memory module), with structured working memory as an extra task axis (entries 32-33, `10-vsa-memory-and-composition-direction.md`).
5. **Locomotion (`mjlab`) is deferred** (entry 26). Proprioceptive state has no translation invariance and no periodic structure to exploit.

## The two research axes
- **Representation**: none (raw state) / tile-coding / HexSSP
- **Centering**: none / simple (running average reward) / value-based (TD-error-driven, equivalent to Bellman error centering)

These cross to a 9-config grid, tested per algorithm per task (see `02-experimental-design.md`).

## Reading order for a new agent
1. `01-theory-and-background.md` — the two ideas being combined and why
2. `02-experimental-design.md` — what's actually being run, and the staged rollout plan (Stage 2a is the current one)
3. `03-architecture.md` — codebase structure, how the pieces fit together
4. `04-algorithms-and-centering.md` — per-algorithm implementation spec for centering
5. `05-tasks-and-benchmarks.md` — exact task list, in and out of scope
6. `06-related-work.md` — literature this project builds on or cites
7. `07-decision-log.md` — why things are the way they are, so they don't get re-litigated
8. `08-ssp-vs-tile-coding-theory-and-hypotheses.md` — why a dense continuous encoder can lose to a local basis, and the hypotheses that follow
9. `09-ssp-centering-task-survey.md` — task survey, duplication map, and what SSP x centering should be tested on
10. `10-vsa-memory-and-composition-direction.md` — the live direction: memory and composition, the mathematics of SSP, and the benchmark plan (sections 12-13)
11. `handoffs/07-mujoco-repo-handoff.md` — the boundary for a separate MuJoCo repo, if the spatial-encoding branch is revived

12. mjlab reward-centering track (2026-10-31): see `mjlab_centering/`-related docs in [`mjlab_centering/00-plan.md`](mjlab_centering/00-plan.md).
    Current results: [`REPORT_005.md`](REPORT_005.md) (supersedes REPORT_004's pilot numbers).
