# Tasks and Benchmarks

## Direction change (2026-10-06): a continuing-task suite replaces locomotion-first

Rationale and citations: [`08-ssp-vs-tile-coding-theory-and-hypotheses.md`](08-ssp-vs-tile-coding-theory-and-hypotheses.md)
and [`09-ssp-centering-task-survey.md`](09-ssp-centering-task-survey.md). CartPole (Stage 0) is now the **negative
control**, not the test: 4 unstructured scalars, bounded, grid-cheap, cliff-edged at termination. The tests that
can discriminate go on tasks whose state has the structure a spatial encoder assumes — relative, unbounded, periodic,
compositional — and, because centering needs a continuing criterion, on tasks without resets.

| Task | Source | State structure it exercises | Why it is the test | Status |
|---|---|---|---|---|
| Continuing Reacher, target resampled every 50 steps | Wan, Korenkevych & Zhu testbed suite (arXiv:2501.06937; code `facebookresearch/DeepRL-continuing-tasks`) | Relative target vector; the optimal policy is a function of relative geometry only | Translation invariance is directly testable: train on one target distribution, evaluate on a shifted one. A grid cannot transfer | Exists; not yet run here |
| Continuing Swimmer, raw wrapped angles | same suite | Periodic topology; globally unbounded accumulated angle | Wan et al. measured 347%-1608% improvement from *manually* wrapping the angles. The clean test is the reverse: keep the env, change the encoder to a periodic one | Exists; not yet run here |
| Continuing Pusher, object resampled every 100 steps | same suite | Relative object-to-goal and gripper-to-object geometry, with contact | The relative-geometry hypothesis of the old Stage 2, in a cheaper continuing testbed | Exists; not yet run here |
| Continuing CartPole under termination-corrected differential TD | local (`gymnasium`) | None — deliberately | Negative control, and closes the decision-log 17 gap: De Asis et al. (arXiv:2605.04368) give a centering formulation that is valid with termination | Local env; conversion needed |
| Multi-object relative-position navigation (2-4 objects, randomised layouts, held-out shift region) | to build (small 2-D sim; RatBox-style code is the nearest ancestor) | Composition: dimension grows linearly with bound pairs while a grid over the product space explodes | The only available test of the composition claim | To build |

Notes:

- The episodic -> continuing conversion described in the section below is **superseded for these tasks** by the
  published testbeds above; do not reimplement it. It still applies if a local env (CartPole) is converted.
- Stage 2 (`Mjlab-Lift-Cube-Yam`) is replaced in the near term by continuing Pusher/Reacher: the same
  relative-geometry hypothesis, no reset resets, an existing published baseline, and a fraction of the cost.
- Stage 3 (Go1/G1 locomotion) is **deferred**: proprioceptive state has no translation invariance and no periodic
  structure to exploit, and it is the most expensive test in the plan.

## In scope

| Stage | Task | Source | Notes |
|---|---|---|---|
| 0 | Classic CartPole | Gymnasium (`gymnasium==0.26.3` in legacy repo) | 2 discrete actions, 4-dim state. Used for DQN-linear/deep and re-ported A2C. |
| 2 | `Mjlab-Lift-Cube-Yam` | mjlab, i2rt YAM arm | State-based (not vision). HexSSP applied to `ee_to_cube`/`cube_to_goal` observation terms only. |
| 3 | `Mjlab-Velocity-Flat-Unitree-Go1` | mjlab, Unitree Go1 quadruped | Flat terrain only. Run before G1. |
| 3 | `Mjlab-Velocity-Flat-Unitree-G1` | mjlab, Unitree G1 humanoid | Flat terrain only. Harder than Go1 — passively less stable. |

## Explicitly out of scope (for this phase)
- mjlab CartPole (`Mjlab-Cartpole-Balance`/`-Swingup`) — was Stage 1, dropped entirely.
- Rough terrain variants of Go1/G1 velocity tracking — deferred stretch goal.
- Vision/depth manipulation tasks (`Mjlab-Lift-Cube-Yam-Rgb`, `-Depth`) and multi-cube segmentation (`Mjlab-Multi-Cube-Seg-Yam`) — explicitly excluded per user preference (quadruped/humanoid locomotion only for the "advanced" tier, not vision tasks).
- `Mjlab-Tracking-Flat-Unitree-G1` (whole-body motion imitation) — not currently planned; available in mjlab if reconsidered later.
- RatBox (original 2D nav-with-obstacles task) — not being ported to mjlab; may still be used informally for quick DQN-discrete sanity checks if needed, but not part of the formal task list.

## Episodic → continuing conversion (required for centering to be valid)
Per `01-theory-and-background.md`, reward centering requires continuing (non-episodic) tasks. mjlab tasks default to episodic (`TerminationTermCfg(func=mdp.time_out, time_out=True)` plus real failure terminations).
- RSL-RL already bootstraps on `time_outs` (doesn't zero the value at artificial episode-length cutoffs) — this part is already "continuing-like," no change needed.
- Still needed: convert genuine failure terminations (fall, illegal contact) from hard episode-enders into reset-with-penalty-and-continue, matching the conversion the original reward-centering paper used for its MuJoCo/PPO experiments (Appendix: large truncation horizon + penalty-and-reset instead of episode end).

## Robot roster
- **In use**: Unitree Go1 (quadruped), Unitree G1 (humanoid) — both via mjlab's `asset_zoo`.
- **Available in mjlab but not currently used**: i2rt YAM (used for manipulation, not locomotion — already covered above).
