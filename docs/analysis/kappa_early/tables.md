## Cell table (generated)

| cell | n | L | S_cen (kappa.md) | kappa (S_cen mid) | S_ret ± SD | kappa-hat ret | S_v100 ± SD | kappa-hat v100 | S_rew | walks k/n [Wilson 95%] | mean tracking (unc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Go1 g.999 c=0 | 20 | 3.98 | 2.48 | 1.61 | 1.52 ± 0.1 | 2.62 | 1.84 ± 0.061 | 2.16 | not logged | 19/20 [0.76, 0.99] | 0.854 |
| Go1 g.999 c=0.1 | 20 | 86.0 | 2.5 | 34.5 | 1.22 ± 0.79 | 70.7 | 0.791 ± 1.3 | 109 | not logged | 12/20 [0.39, 0.78] | 0.645 |
| Go1 g.999 c=0.3 | 20 | 264 | 2.5 | 106 | 1.61 ± 0.09 | 164 | 0.946 ± 0.19 | 280 | not logged | 2/20 [0.03, 0.30] | 0.362 |
| Go1 g.999 c=1 | 20 | 918 | 2.51 | 366 | 4.23 ± 0.14 | 217 | 1.67 ± 0.44 | 550 | not logged | 0/20 [0.00, 0.16] | 0.307 |
| G1 g.99 c=1 | 20 | 74.5 | 1.15 | 65.0 | 3.32 ± 0.067 | 22.4 | 2.25 ± 0.16 | 33.0 | not logged | 0/20 [0.00, 0.16] | 0.309 |
| Go1 g.99 c=0 (pilot) | 3 | 0.869 | 1.06 | 0.821 | 1.2 ± 0.011 | 0.725 | 0.883 ± 0.012 | 0.985 | not logged | 3/3 [0.44, 1.00] | 0.888 |
| G1 g.999 c=0 (pilot) | 3 | 197 | 1.76 | 112 | 2.25 ± 0.015 | 87.6 | 1.91 ± 0.012 | 103 | not logged | 2/3 [0.21, 0.94] | 0.527 |
| G1 g.99 c=0 (pilot) | 3 | 5.58 | 0.352 † | 15.8 | 0.296 ± 0.00085 | 18.9 | 0.179 ± 0.006 | 31.1 | not logged | 3/3 [0.44, 1.00] | 0.732 |

† S_cen taken from the uncentered arm (no centered counterpart).

## Regressions on the 5 claim cells (generated)

| x | model | slope per decade | SE | robot (G1) coef | log-lik | AIC | seed-level AUC | AIC gain from robot | one threshold, 5 cells? | strictly monotone rates? |
|---|---|---|---|---|---|---|---|---|---|---|
| L alone | x alone | -2.65 | 0.56 | – | -36.5 | 77.0 | 0.826 | | no | no |
| | x + robot | -3.05 | 0.75 | -3.78 (SE 1.52) | -26.6 | 59.2 | | **17.8** | | |
| centered κ (S mid) | x alone | -3.37 | 0.83 | – | -31.0 | 66.0 | 0.934 | | yes | no |
| | x + robot | -3.06 | 0.75 | -2.75 (SE 1.51) | -26.6 | 59.2 | | **6.8** | | |
| S_ret | x alone | -2.00 | 0.40 | – | -45.3 | 94.7 | 0.826 | | no | no |
| | x + robot | -3.25 | 0.92 | -4.88 (SE 1.62) | -29.5 | 65.1 | | **29.6** | | |
| S_v100 | x alone | -1.76 | 0.35 | – | -43.9 | 91.8 | 0.826 | | no | no |
| | x + robot | -2.73 | 0.76 | -4.72 (SE 1.60) | -28.9 | 63.9 | | **27.9** | | |


## Diagnostic only: spread of the uncentered S, and the same windows on the centered arm (generated; not a candidate)

| cell | S_ret unc (median [min, max] over seeds) | S_ret, same window, centered arm | S_v100 unc (median [min, max]) | S_v100, same window, centered arm |
|---|---|---|---|---|
| Go1 g.999 c=0 | 1.53 [1.36, 1.71] | 1.22 | 1.86 [1.69, 1.91] | 1.70 |
| Go1 g.999 c=0.1 | 0.89 [0.83, 3.18] | 1.22 | 0.33 [0.08, 3.96] | 1.66 |
| Go1 g.999 c=0.3 | 1.58 [1.49, 1.82] | 1.22 | 0.98 [0.58, 1.25] | 1.69 |
| Go1 g.999 c=1 | 4.19 [4.07, 4.66] | 1.22 | 1.58 [0.84, 2.41] | 1.67 |
| G1 g.99 c=1 | 3.32 [3.20, 3.42] | 1.98 | 2.29 [1.77, 2.43] | 1.94 |
| Go1 g.99 c=0 (pilot) | 1.20 [1.19, 1.21] | 1.11 | 0.88 [0.87, 0.89] | 0.82 |
| G1 g.999 c=0 (pilot) | 2.25 [2.23, 2.26] | 2.11 | 1.91 [1.90, 1.92] | 1.97 |
| G1 g.99 c=0 (pilot) | 0.30 [0.29, 0.30] | n/a | 0.18 [0.17, 0.19] | n/a |


## Verdicts (generated)

- S_ret: (a) False  (b) False  (c) False  => FAIL
- S_v100: (a) False  (b) False  (c) False  => FAIL

