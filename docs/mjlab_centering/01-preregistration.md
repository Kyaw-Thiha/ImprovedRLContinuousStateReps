# Pre-registration (frozen before any campaign run; do not edit after the first launch)

Written 2026-10-31. Changes after launch go in `02-decisions.md` as dated entries, never here.

## Metric
`J_task` = mean per-step task reward (no penalty, no offset) of the deterministic eval policy, mean of the last 5
evals per seed. Falls per 1k steps and reset count are reported as a second axis, never used as a normaliser.
Unit of analysis: seed. Bootstrap 95% CI (5000 resamples); paired differences on shared seeds 2000-2004.

## Predictions
- P1 (gamma=0.99, off=0): centered vs uncentered final `J_task` within +-3%.
- P2 (gamma=0.999, off=c): uncentered `J_task` is below 0.90x the centered value; centered loses less from 0.99 to 0.999.
- P3 (offset): at off=c the uncentered arms degrade versus off=0; centered arms are unchanged within +-3%.
- P4 (critic-fix rivals, gamma=0.999, off=c): the question is whether centering is better than, equal to, or worse than
  clipping off and mean+std return normalisation. Pass for "centering is a distinct fix": centered `J_task` exceeds both
  rivals by a paired CI that excludes 0. Equal within +-3% means "another way to fix the critic" and is reported as such.
- P5 (shock): after the offset shift at it 400, the centered arms recover to 95% of their pre-shock `J_task` faster than
  the uncentered arms (paired CI on recovery iterations excludes 0). Dip depth is reported with CIs.
- P6 (mechanism): uncentered critics show larger early `const_frac` and larger |V| than centered ones at gamma=0.999.
- P7 (simple vs value): not tested in this round except as noted in 00-plan (deferred).

## Rules
- Plateau gate decides run length (see 00-plan). Length is common to all arms in a block.
- Observation normalisation is ON for all arms (rsl-rl empirical normaliser), identical across arms.
- Fall penalty P=10 and offset c=1.0 are frozen; sensitivity check is x0.3 and x3 on the reference arm only.
- Hyperparameters: mjlab Go1 PPO defaults; only eta (3 values) and learning rate (3 values) are varied, as declared in
  the matrix. eta chosen on the sweep cell (seeds 2000-2001), then frozen for every centered cell.
- Claims use seeds 2000+ only; the eta sweep seeds are reported separately and not pooled.
- No test and no threshold above is edited to make a check pass.
