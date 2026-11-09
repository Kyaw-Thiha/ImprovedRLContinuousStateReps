## Cell table (generated)

| cell | robot | γ | c | n (unc / cen) | r̄ (it 0-100) | L | S mid (± SD over seeds) | S late (± SD) | κ mid | κ late | walks k/n [Wilson 95%] | mean tracking (unc) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Go1 g.999 c=0 | go1 | 0.999 | 0 | 20 / 20 | -0.0040 | 3.98 | 2.48 ± 0.073 | 1.26 ± 0.23 | 1.61 | 3.15 | 19/20 [0.76, 0.99] | 0.854 |
| Go1 g.999 c=0.1 | go1 | 0.999 | 0.1 | 20 / 20 | +0.0860 | 86.0 | 2.5 ± 0.05 | 1.22 ± 0.02 | 34.5 | 70.4 | 12/20 [0.39, 0.78] | 0.645 |
| Go1 g.999 c=0.3 | go1 | 0.999 | 0.3 | 20 / 20 | +0.2645 | 264 | 2.5 ± 0.072 | 1.45 ± 0.56 | 106 | 183 | 2/20 [0.03, 0.30] | 0.362 |
| Go1 g.999 c=1 | go1 | 0.999 | 1 | 20 / 20 | +0.9182 | 918 | 2.51 ± 0.071 | 1.3 ± 0.36 | 366 | 704 | 0/20 [0.00, 0.16] | 0.307 |
| G1 g.99 c=1 | g1 | 0.99 | 1 | 20 / 20 | +0.7451 | 74.5 | 1.15 ± 0.037 | 0.403 ± 0.03 | 65.0 | 185 | 0/20 [0.00, 0.16] | 0.309 |
| Go1 g.99 c=0 (pilot) | go1 | 0.99 | 0 | 3 / 3 | +0.0087 | 0.869 | 1.06 ± 0.037 | 0.457 ± 0.01 | 0.821 | 1.9 | 3/3 [0.44, 1.00] | 0.888 |
| G1 g.999 c=0 (pilot) | g1 | 0.999 | 0 | 3 / 3 | -0.1967 | 197 | 1.76 ± 0.026 | 2.09 ± 0.013 | 112 | 94.1 | 2/3 [0.21, 0.94] | 0.527 |
| G1 g.99 c=0 (pilot) | g1 | 0.99 | 0 | 3 / 0 | -0.0558 | 5.58 | 0.352 ± 0.0037 † | 0.407 ± 0.0042 † | 15.8 | 13.7 | 3/3 [0.44, 1.00] | 0.732 |

† S taken from the uncentered arm (no centered counterpart exists).

## Robustness inputs: L with the fall penalty added back (generated)

| cell | r_task + c | L (no penalty) | fall rate /1k (it 0-100) | κ mid, L no-penalty | κ late, L no-penalty |
|---|---|---|---|---|---|
| Go1 g.999 c=0 | +0.0069 | 6.94 | 1.09 | 2.8 | 5.49 |
| Go1 g.999 c=0.1 | +0.0956 | 95.6 | 0.96 | 38.3 | 78.3 |
| Go1 g.999 c=0.3 | +0.2753 | 275 | 1.08 | 110 | 190 |
| Go1 g.999 c=1 | +0.9506 | 951 | 3.24 | 379 | 729 |
| G1 g.99 c=1 | +0.8324 | 83.2 | 15.32 | 72.6 | 207 |
| Go1 g.99 c=0 (pilot) | +0.0199 | 1.99 | 1.12 | 1.87 | 4.35 |
| G1 g.999 c=0 (pilot) | -0.1151 | 115 | 14.31 | 65.5 | 55.0 |
| G1 g.99 c=0 (pilot) | -0.0558 | 5.58 | 14.71 | 15.8 | 13.7 |


## Bias-init reference arms (generated)

| cell | n | critic starts at | r̄ (bias arm, it 0-100) | L_eff = abs(r̄ − c)/(1−γ) | κ_eff mid | κ_eff late | walks k/n | mean tracking |
|---|---|---|---|---|---|---|---|---|
| Go1 g.999 c=0.1 bias-init | 20 | c/(1−γ) = 100 | +0.0961 | 3.86 | 1.54 | 3.16 | 20/20 | 0.883 |
| Go1 g.999 c=1 bias-init | 20 | c/(1−γ) = 1000 | +0.9958 | 4.17 | 1.66 | 3.2 | 18/20 | 0.821 |
| G1 g.99 c=1 bias-init | 20 | c/(1−γ) = 100 | +0.8224 | 17.8 | 15.5 | 44.1 | 20/20 | 0.820 |


## Logistic regressions, claim cells only: 5 cells, 100 seeds, 33 walk (generated)

| x | model | slope on x (per decade) | SE | robot (G1) coef | log-lik | AIC | McFadden R² | seed-level AUC | one threshold serves all 5 cells? |
|---|---|---|---|---|---|---|---|---|---|
| log10 L | x alone | -2.65 | 0.56 | – | -36.5 | 77.0 | 0.42 | 0.83 | no |
| | x + robot | -3.05 | 0.75 | -3.78 (SE 1.52) | -26.6 | 59.2 | 0.58 | | |
| log10 kappa (S mid, it 100-300) | x alone | -3.37 | 0.83 | – | -31.0 | 66.0 | 0.51 | 0.93 | yes |
| | x + robot | -3.06 | 0.75 | -2.75 (SE 1.51) | -26.6 | 59.2 | 0.58 | | |
| log10 kappa (S late, last 250) | x alone | -3.70 | 0.96 | – | -29.4 | 62.8 | 0.54 | 0.95 | yes |
| | x + robot | -3.09 | 0.78 | -2.22 (SE 1.52) | -27.0 | 60.0 | 0.57 | | |
| log10 L, no fall penalty | x alone | -2.99 | 0.63 | – | -36.5 | 77.1 | 0.42 | 0.83 | no |
| | x + robot | -3.42 | 0.82 | -3.85 (SE 1.52) | -26.2 | 58.4 | 0.59 | | |
| log10 kappa, L no fall penalty, S mid | x alone | -3.89 | 0.95 | – | -30.3 | 64.6 | 0.52 | 0.93 | yes |
| | x + robot | -3.43 | 0.82 | -2.69 (SE 1.51) | -26.2 | 58.4 | 0.59 | | |
| log10 kappa, L no fall penalty, S late | x alone | -4.22 | 1.07 | – | -28.6 | 61.3 | 0.55 | 0.95 | yes |
| | x + robot | -3.48 | 0.86 | -2.10 (SE 1.52) | -26.6 | 59.2 | 0.58 | | |


## Within-cell check, mixed-outcome cells only (generated)

| cell | walks | mean L, walking seeds | mean L, failing seeds | AUC (fail has higher L) | AUC, L without penalty |
|---|---|---|---|---|---|
| Go1 g.999 c=0 | 19/20 | 4.42 | 6.03 | 0.63 | 0.21 |
| Go1 g.999 c=0.1 | 12/20 | 91.8 | 77.4 | 0.12 | 0.09 |
| Go1 g.999 c=0.3 | 2/20 | 264 | 265 | 0.53 | 0.39 |
| G1 g.999 c=0 (pilot) | 2/3 | 197 | 196 | 0.50 | 0.50 |

