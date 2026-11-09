# Summary table (corrected protocol, claim seeds 3000-3019; last-5-eval means; 'walks' = last-5 tracking >= 0.5)

CIs: 'boot' = 95% bootstrap over seeds (5000 resamples, seed 0, `plot_eval_v3.bootstrap_ci`); 't' = the 95% t-interval used in REPORT_005.

## 1. Go1 offset dose-response, gamma 0.999, 2000 iterations: centered - uncentered, paired by seed

| offset c | uncentered walks | centered walks | uncentered tracking | centered tracking | paired dtrack (boot 95% CI) | paired dtrack (t 95% CI) | seeds positive | n pairs |
|---|---|---|---|---|---|---|---|---|
| 0 | 19/20 | 20/20 | 0.854 | 0.868 | +0.0143 [-0.0009, +0.0428] | +0.0143 [-0.0145, +0.0430] | 10/20 | 20 |
| 0.1 | 12/20 | 20/20 | 0.645 | 0.884 | +0.2396 [+0.1228, +0.3582] | +0.2396 [+0.1060, +0.3732] | 19/20 | 20 |
| 0.3 | 2/20 | 18/20 | 0.362 | 0.806 | +0.4440 [+0.3326, +0.5457] | +0.4440 [+0.3305, +0.5574] | 18/20 | 20 |
| 1 | 0/20 | 19/20 | 0.307 | 0.854 | +0.5472 [+0.4874, +0.5796] | +0.5472 [+0.4860, +0.6084] | 19/20 | 20 |

## 2. Bias-init (critic output bias = c/(1-gamma), uncentered), Go1, gamma 0.999

| offset c | arm | walks | tracking | paired dtrack vs uncentered (boot) | (t) | paired dtrack vs centered (boot) | n pairs |
|---|---|---|---|---|---|---|---|
| 0.1 | bias-init | 20/20 | 0.883 | +0.2382 [+0.1210, +0.3570] | +0.2382 [+0.1043, +0.3721] | -0.0014 [-0.0031, +0.0002] | 20 |
| 1 | bias-init | 18/20 | 0.821 | +0.5144 [+0.4275, +0.5757] | +0.5144 [+0.4309, +0.5979] | -0.0328 [-0.1255, +0.0581] | 20 |

## 3. Shock test (Go1, offset-0 mothers, shock at iteration 1200, pooled seeds 3000-3019; t 95% CIs, `shock_tables.py`)

**(A) all 20 seeds, as pre-registered**

| shock | uncentered shocked - control | centered shocked - control | arm contrast (centered - uncentered) | seeds positive | n |
|---|---|---|---|---|---|
| off+0.1 | -0.0042 [-0.0175, +0.0091] (n=20, end within 95%: 20/20) | -0.0014 [-0.0034, +0.0006] (n=20, end within 95%: 20/20) | +0.0028 [-0.0107, +0.0163] | 16/20 | 20 |
| off+1.0 | -0.5124 [-0.5938, -0.4311] (n=20, end within 95%: 2/20) | -0.0029 [-0.0046, -0.0011] (n=20, end within 95%: 20/20) | +0.5095 [+0.4286, +0.5905] | 20/20 | 20 |
| actx5 | -0.0552 [-0.0750, -0.0353] (n=20, end within 95%: 13/20) | -0.0451 [-0.0474, -0.0428] (n=20, end within 95%: 18/20) | +0.0100 [-0.0096, +0.0297] | 13/20 | 20 |

**(B) walking mothers only (mother's last-3-eval tracking >= 0.8); uncentered n=16, centered n=20**

| shock | uncentered shocked - control | centered shocked - control | arm contrast (centered - uncentered) | seeds positive | n |
|---|---|---|---|---|---|
| off+0.1 | -0.0126 [-0.0160, -0.0091] (n=16, end within 95%: 16/16) | -0.0014 [-0.0034, +0.0006] (n=20, end within 95%: 20/20) | +0.0114 [+0.0067, +0.0162] | 15/16 | 16 |
| off+1.0 | -0.5760 [-0.5843, -0.5678] (n=16, end within 95%: 0/16) | -0.0029 [-0.0046, -0.0011] (n=20, end within 95%: 20/20) | +0.5726 [+0.5636, +0.5816] | 16/16 | 16 |
| actx5 | -0.0479 [-0.0501, -0.0458] (n=16, end within 95%: 9/16) | -0.0451 [-0.0474, -0.0428] (n=20, end within 95%: 18/20) | +0.0034 [-0.0002, +0.0069] | 11/16 | 16 |

## 4. G1, gamma 0.99, offset +1, fall penalty 5.7, 2500 iterations (seeds with DONE markers; fewer than 20 per arm = pilot)

| arm | n | walks | tracking (last-5 mean) | J_task |
|---|---|---|---|---|
| uncentered | 20 | 0/20 | 0.309 | 0.0610 |
| centered | 20 | 20/20 | 0.816 | 0.0941 |
| bias-init (uncentered) | 20 | 20/20 | 0.820 | 0.0949 |

| paired contrast | n pairs | dtrack (boot 95% CI) | (t 95% CI) | seeds positive | dJ_task (boot 95% CI) |
|---|---|---|---|---|---|
| centered - uncentered | 20 | +0.5063 [+0.4903, +0.5179] | +0.5063 [+0.4909, +0.5217] | 20/20 | +0.03313 [+0.03180, +0.03441] |
| bias-init - uncentered | 20 | +0.5102 [+0.4969, +0.5201] | +0.5102 [+0.4974, +0.5231] | 20/20 | +0.03385 [+0.03284, +0.03502] |
| bias-init - centered | 20 | +0.0040 [-0.0005, +0.0096] | +0.0040 [-0.0016, +0.0095] | 10/20 | +0.00072 [-0.00012, +0.00177] |

## 5. CartPole, linear DQN, no centering: final score (mean of the last 5 of 15 greedy checkpoints), 20 seeds 3000-3019

| arm | n | mean | sd (population, as REPORT_001/006) | sd (ddof=1) | solved (>= 475) | chosen length scale |
|---|---|---|---|---|---|---|
| tile coding, new search (tile_h2) | 20 | 495.76 | 7.23 | 7.42 | 100% | - |
| SSP, new search (ssp_h2) | 20 | 486.10 | 24.31 | 24.95 | 80% | 0.0220 |
| SSP, new search, no input normalisation (ssp_h2raw) | 20 | 469.75 | 19.83 | 20.35 | 40% | 0.0152 |
| tile coding, old search (RUN_004 replicate) | 20 | 499.64 | 0.99 | 1.02 | 100% | - |
| SSP, old search (RUN_004 replicate) | 20 | 461.65 | 35.88 | 36.81 | 45% | 0.1477 |
| discrete, old search pipeline (RUN_004 replicate) | 20 | 251.25 | 48.13 | 49.38 | 0% | - |

| paired contrast (by seed) | mean | bootstrap 95% CI (`h2_report.boot_ci`) |
|---|---|---|
| ssp_h2 - tile_h2 | -9.66 | [-22.07, +0.68] |
| ssp_h2raw - ssp_h2 | -16.35 | [-31.19, +0.52] |
| ssp_h2 (new) - ssp old | +24.46 | [+4.07, +43.85] |
