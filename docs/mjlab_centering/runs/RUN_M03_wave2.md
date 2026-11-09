# RUN_M03 — wave 2: the SSP x centering factorial on Go1, with critic-fix rivals and the shock block

Started 2026-11-01. Design: decisions 35-67 in `../02-decisions.md`. Data:
`mjlab_centering/results/merged_r2`, rebuilt by `analysis/restore_runs.py` from the per-run R2 mirrors (authoritative;
every run that uploaded `final.json`/`DONE` is recoverable even after its pod is deleted).

## 0. What this run answers

At gamma 0.999 with a constant reward offset c = 1.0 on a continuing Go1 velocity task, 800 iterations x 2048 envs,
10 claim seeds (2000-2009), paired across arms:

1. Does reward centering (eta 0.03, r_bar from the first batch) beat no centering?
2. Does centering beat the cheaper critic fixes (value-clipping off, return normalisation) - i.e. is it a distinct fix
   or just "make the critic easier to fit"?
3. Does an SSP input encoder (command + projected gravity, `ssp_low`) change either effect (the interaction)?
4. When the reward function is changed mid-training, does centering recover faster?

Primary metric: `J_task` (per-step task reward of the deterministic eval policy, mean of the last 5 evals). Secondary:
`track_lin_frac` (fraction of the best possible tracking reward). Equivalence margin: 3% of the best cell mean, frozen
before analysis (decision 67); the CI half-width is reported next to it so no claim is tighter than the design allows.

## 1. Final factorial (Go1, offset c, 800 iterations x 2048 envs, claim seeds)

All four cells at gamma 0.999 have 10 paired seeds; the SSP cells at gamma 0.99 have 10; the raw cells at gamma 0.99
have 3 (cut, see decision 67).

| cell | gamma | n | J_task | track_lin |
|---|---|---|---|---|
| raw, no centering | .999 | 10 | 0.0296 | 0.306 |
| raw, centered | .999 | 10 | 0.0850 | 0.710 |
| ssp_low, no centering | .999 | 10 | 0.0347 | 0.306 |
| ssp_low, centered | .999 | 10 | 0.0681 | 0.384 |
| raw, no centering | .99 | 3 | 0.0602 | 0.313 |
| raw, centered | .99 | 3 | 0.0977 | 0.856 |
| ssp_low, no centering | .99 | 10 | 0.0576 | 0.312 |
| ssp_low, centered | .99 | 10 | 0.0806 | 0.545 |

Paired contrasts (shared seeds, bootstrap 5000, unit = seed):

| gamma | contrast | mean | 95% CI | half-width | margin |
|---|---|---|---|---|---|
| .999 | centering, raw | **+0.0554** | [+0.0507, +0.0603] | 0.0048 | 0.0025 |
| .999 | centering, SSP | **+0.0334** | [+0.0315, +0.0356] | 0.0021 | 0.0025 |
| .999 | SSP main (uncentered) | +0.0051 | [+0.0008, +0.0094] | 0.0043 | 0.0025 |
| .999 | **interaction** | **-0.0220** | [-0.0276, -0.0169] | 0.0054 | 0.0025 |
| .99 | centering, raw | +0.0375 | [+0.0338, +0.0395] | 0.0029 | 0.0029 |
| .99 | centering, SSP | +0.0230 | [+0.0198, +0.0259] | 0.0031 | 0.0029 |
| .99 | SSP main (uncentered) | -0.0021 | [-0.0027, -0.0009] | 0.0009 | 0.0029 |
| .99 | interaction | -0.0180 | [-0.0203, -0.0143] | 0.0030 | 0.0029 |

1. **Centering is the main effect**: +0.055 J_task at gamma .999 (+187% over uncentered) and tracking 0.31 -> 0.71.
   Every arm without it fails to learn to walk (tracking 0.30-0.31 in 30 of 30 runs, i.e. the standing level). On J_task
   they do not sit at the standing policy's 0.077: they decay from 0.069 at iteration 50 to about 0.03, i.e. the offset makes
   them *worse than standing*, and their action std grows from 1.0 to about 1.3. This is the Wan et al. prediction reproduced in a GPU-parallel robot pipeline.
2. **SSP alone is neutral** (+0.005 [+0.001, +0.009] at gamma .999, -0.002 at .99). The encoder is not a free win.
3. **The interaction is negative and well resolved** (-0.022 at .999, -0.018 at .99, both CIs well clear of 0): centering
   buys about 40% less when the input is 22x wider. The uncentered arms are equally broken with and without SSP
   (0.035 vs 0.030), so this is not the encoder absorbing the offset - it is the centred arm converting a cleaner target
   into less reward at that width, consistent with wave 9's width penalty and with the KL-adaptive schedule taking
   smaller steps on a wider input.
4. **Precision**: the interaction CI half-width (0.0054) is about twice the 3% margin, so the supported claim is "the
   interaction is negative and at least 0.017 in magnitude", not a tight equivalence statement.

**Convergence check (added after review).** Slope of the eval metric over the last 250 iterations, per 100 iterations, mean
over seeds with a 95% bootstrap CI: raw centered tracking +0.042 [+0.017, +0.074] (10/10 seeds rising), J_task +0.0033
[+0.0019, +0.0047]; SSP centered tracking +0.015 [+0.007, +0.024] (9/10 rising), J_task -0.0001 [-0.0004, +0.0002] (4/10);
raw uncentered tracking -0.008 [-0.015, -0.002] (2/10). The centered arms had not converged at 800 iterations, so the SSP and
interaction rows above are statements about learning speed at a fixed budget (decision 70d, plan in `../03-ssp-plan.md`).

## 2. The critic-fix rivals: centering is not "just fixing the critic" (n=10 each)

gamma .999, offset c, all uncentered except the last row:

| arm | n | J_task | track_lin |
|---|---|---|---|
| value clipping off (`value_clip=False`) | 10 | 0.0679 | 0.308 |
| return normalisation (batch mean+std of targets) | 10 | 0.0236 | 0.299 |
| no fix (plain uncentered) | 10 | 0.0296 | 0.306 |
| **value centering** | 10 | **0.0850** | **0.710** |

Turning off rsl-rl's +-0.2 value clip recovers most of the J_task gap (0.030 -> 0.068) - so value clipping *is* part of
why high-discount runs struggle here, and a practitioner who only looks at J_task could conclude that clipping off is
nearly as good. The tracking column says otherwise: clip-off stays at the standing floor (0.308) and never learns to
walk, while centering reaches 0.710. Return normalisation makes it worse at this budget (0.024, 0.299).

## 3. Shock block: a +4 offset applied at iteration 600 (branched from the mother's checkpoint, n=10 per arm)

| arm | pre-shock J_task | min after shock | J_task at 800 | notes |
|---|---|---|---|---|
| centred, resumed, no shock (control) | 0.082 | - | 0.088 | continues to rise |
| centred, +4 offset | 0.082 | 0.049 (it 700) | 0.057 | partly recovered, not back to its control |
| uncentred, resumed, no shock (control) | 0.033 | 0.025 (it 650) | 0.030 | already failed |
| uncentred, +4 offset | 0.033 | ~0.001 (it 800) | ~0.001 | falls monotonically, no recovery |

**Branch validation: PASS** (pre-registered requirement): 20 resumed-unshocked controls vs their unbranched mothers, mean
difference +0.0012, 95% CI [-0.0020, +0.0045], between-seed SD 0.031. Checkpoint branching does not change a run.

What can and cannot be concluded:
- A centred arm is not immune to a large reward shift: J_task drops 0.033 (about 40%) and is not back to its control after 200
  iterations.
- **The earlier centred-vs-uncentred "recovery time" comparison (64 vs 79 iterations to recover half of the dip) is withdrawn.**
  It is an artefact: the half-dip level is derived from the minimum, which for the uncentred arm occurs at the end of the run,
  so the first post-shock evaluation already counts as "recovered". The uncentred arm does not recover.
- The uncentred arm had never learned (J_task already below the standing policy), so a shock on top of it is not a meaningful
  comparison. Pre-registered P5 is **inconclusive**. A valid test shocks a *working* policy (trained without the offset),
  with a moderate shift and a recovery window of at least 400 iterations.

## 4. Pending when this file was opened

Nothing: every planned run finished. The gamma .99 raw cells (3 seeds) are the documented cut, and the action-rate shock
arm was dropped by decision 57 before it ran. All 158 mirrored run ids are recovered in `mjlab_centering/results/merged_r2`
(`analysis/restore_runs.py`), which is the authoritative dataset for these tables.
