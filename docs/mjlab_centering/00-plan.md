# mjlab reward-centering study: plan

Scope: reward centering (and, as a deferred tier, SSP input encoding) for PPO on a **continuing** Go1 flat-terrain
locomotion task, run GPU-parallel in mjlab 1.6.0 with rsl-rl-lib 5.5.1. Everything for this track lives in
`docs/mjlab_centering/` and `mjlab_centering/`. R2 prefix: `mjlab_ssp_centering/`. Origin: `docs/REPORT_001.md`
(options 5/6) and the design discussion of 2026-10-31.

## Task conversion (continuing, penalty-and-reset)
- `time_out` termination removed, `episode_length_s = 1e6`; terrain curriculum off; velocity curriculum kept.
- `ContinuingVecEnv` returns `dones = 0` always. A fall (the env's own `fell_over` termination) auto-resets the robot,
  costs `P = 10` reward, and carries no terminal flag. Predefined-reset continuing setting of Wan et al. (ICLR 2025).
- Reward offset `c = 1.0` per step on offset cells (base reward is dt-scaled, about 0.1 per step).
- Optional free random reset `p_reset` (default 0), auto-enabled if >5% of envs exceed 2000 steps without a reset.

## Centering (CenteredPPO)
- Centre in `process_env_step` BEFORE rsl-rl adds the `gamma * V(s)` time-out bootstrap (the doc-04 sketch is wrong for 5.5.1).
- value: `r_bar += eta * mean(delta)` once per iteration. simple: `r_bar += beta * (r_rate - r_bar)`.
- Critic-fix rivals (uncentered): value clipping off; return normalisation (running mean+std, PopArt-lite);
  reward scaling (std only).

## Matrix (seeds 2000-2004, paired across arms; 500 iterations unless the plateau gate says otherwise)
| Block | Cells | Runs |
|---|---|---|
| eta sweep | value, g=0.999, off=c, eta in 3 values x 2 seeds | 6 |
| A | {none,value} x g{0.99,0.999} x off{0,c} x 5 seeds | 40 |
| critic-fix | clip-off 5, ret-norm(mean+std) 5, reward-scale 3 (g=0.999, off=c) | 13 |
| E shock | {none,value} x {offset shift, action_rate x5} x 5 seeds, 700 it, shock at it 400 | 20 |
| lr grid | 3 lr x {none,value} x 2 seeds (g=0.999, off=c) | 12 |
| penalty check | reference arm, P x0.3 and x3, 2 seeds each | 4 |
| deferred | SSP-low 2 cells x 3 seeds; then seeds 2005-2009 for the four headline g=0.999 cells | 6 + 20 |

Reference arm `J_ref`: raw, none, g=0.99, offset 0, **through the continuing wrapper**.
Plateau gate: reference arm 800 iterations x 2 seeds; plateau = first iteration whose next-100-iteration `J_task`
changes by <3%. If later than iteration 400, every run's length becomes that point (seeds cut from low-priority blocks).

## Budget and order
Soft stop $20 (finish the current tier), hard kill $25. Community-cloud GPUs (4090 $0.34/h, 3090 $0.22/h).
Order: benchmark+plateau gate -> eta sweep -> A -> critic-fix -> E -> penalty check -> lr grid -> SSP-low -> extra seeds.
Cut order when throughput is low: SSP-low, lr grid, reward scaling, penalty check to one multiplier. Never cut:
A, clip-off, ret-norm, E, eta sweep. Below 20k steps/s keep only A, E, clip-off, ret-norm.

## Logs and evals
- `iter.jsonl` per iteration: reward rates (seen/task/per-term), fall and reset counts, `steps_since_reset` max/p95,
  r_bar and gap, mean delta, V/return stats, explained variance, critic error mean/MSE/`const_frac`, losses, lr,
  entropy, action std, grad/weight norms.
- `eval.jsonl` every 25 iterations: 512-env deterministic eval, 500 steps: `J_task`, fall rate, tracking errors.
- `final.json`: 5k-step nominal deployment eval, command-range x1.5 and one-step action-delay conditions.
- Final checkpoint always stored; pre-shock checkpoint for E.
- Derived offline (`analysis/derive.py`): T50/T90/T95 vs `J_ref` and self-normalised, last-5-eval final, AULC, collapse
  rate, shock dip/recovery, `const_frac`, bootstrap CIs over seeds with paired differences.
