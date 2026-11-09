# SSP follow-up plan: convergence first, then a staged (not full-grid) hyperparameter search

Written 2026-11-01 after review. Pre-registration for these runs lives here; results go in a new `runs/RUN_M04_*.md`.
Costs below are priced from measured wall time (run records RUN_M02/M03), not estimated.

## 1. What we know and do not know about SSP

Known (run data):
- Uncentered, SSP is neutral (+0.005 [+0.001, +0.009]); centered, it gains less than raw (+0.033 vs +0.055).
- At equal 1068-wide input, HexSSP (0.082 J_task / 0.54 tracking) beats random Fourier (0.072 / 0.44) and trails raw
  (0.097 / 0.83) - 3 seeds, one length scale.
- SSP-centered runs have a smaller mean learning rate than raw-centered (1.3e-4 vs 2.9e-4), a higher final entropy (14.7 vs 10.8)
  and a higher action std (0.83 vs 0.61): the policy commits more slowly.
- Centered arms were still improving at iteration 800 (raw tracking +0.042 per 100 iterations, SSP +0.015).

Not known:
- Whether the gap is convergence speed or a real ceiling (no run is long enough).
- Whether the width of the code, the code itself, the length scale, or the optimiser schedule causes it. Nothing was tuned:
  one hexagonal 8x8 grid (513 features per 3-d term, actor input 48 -> 1068), length scale 0.3, the KL-adaptive learning-rate
  schedule with `desired_kl` 0.01, mjlab's network size. The only sweep (length scale) was run in a setting where every arm
  fails and carries no information.

## 2. Hypotheses and the experiment that separates each

| id | hypothesis | decisive comparison |
|---|---|---|
| H1 | not converged: the gap closes with more iterations | S0: raw vs SSP, centered, 2000 iterations |
| H2 | width: 1068 inputs shrink the KL-adaptive step | S1: SSP at 2x2 / 4x4 / 8x8 against random Fourier of the same width |
| H3 | length scale is mis-set for the command / gravity ranges | S2: length scale sweep, **centered** (informative) |
| H4 | optimiser schedule, not the code | S3: `desired_kl` 0.03 and a fixed learning rate, applied to raw, RFF and SSP alike |
| H5 | there is a real structured-code advantage at fixed width | S1 and S4: SSP vs matched RFF |

Fairness rule (tune every baseline): any optimiser setting applied to SSP is applied identically to raw and to matched RFF,
otherwise a "win" could be a tuning win.

## 3. Staged design (a full grid is not worth it)

All cells: centered, eta 0.03, first-batch r_bar, offset c, gamma 0.999, Go1. Tuning uses search seeds 1000-1002; only the final
confirmation uses claim seeds. Per-iteration cost measured: raw $0.0000997, SSP $0.0001175 per run (6-way on an A40, $0.59/h).
Actor input widths: 2x2 -> 108, 3x3 -> 188, 4x4 -> 300, 8x8 -> 1068 (raw 48).

- **S0 convergence** (claim seeds 2000-2003): raw and SSP 8x8, 2000 iterations. 8 runs, **$1.74**.
- **S1 width / code** (search seeds, 1500 iterations): raw; SSP 2x2, 4x4, 8x8; random Fourier 4x4, 8x8; 3 seeds. 18 runs, **$3.09**.
- **S2 length scale** (search seeds, 1500 iterations): best S1 SSP at length scale 0.15, 0.6, 1.2; 3 seeds. 9 runs, **$1.59**.
- **S3 optimiser** (search seeds, 1500 iterations): `desired_kl` 0.03 and fixed lr 3e-4, each on raw, matched RFF and best SSP; 3 seeds.
  18 runs, **$3.01**.
- **S4 confirmation** (claim seeds 2000-2005, 2000 iterations): best SSP vs raw vs matched RFF, the single config chosen on search
  seeds. 18 runs, **$4.02** (10 seeds: 30 runs, $6.69).
- Side control, not SSP: **G1 `--episodic` reference**. Done in RUN_M04 (decision 75): one seed walks at
  `track_lin_frac` 0.813 and the time-outs-kept variant walks at 0.811, so the wrapper's missing time-outs are the G1
  cause. No further paid control is needed; a G1 factorial at the fixed wrapper is a separate decision (entry 76).

## 4. Gates (decided now, before any result)

- **G0 after S0.** If SSP-centered reaches at least 95% of raw-centered J_task *and* tracking within 0.05 at 2000 iterations on the
  4 seeds, the 800-iteration gap was convergence speed: stop the search, rerun the factorial at 2000 iterations. Spend: $1.74.
- **G1 after S1.** A narrower SSP (2x2 or 4x4) within 5% of raw while 8x8 is not -> width/optimiser story, go to S3. All SSP widths
  below raw by more than the margin and matched RFF equal to SSP -> the code adds nothing at any width: stop SSP tuning, write
  that up.
- **G2 after S2/S3.** Take the single best SSP config (by mean J_task over the 3 search seeds, ties broken by tracking); do not
  carry more than one config to S4.
- **Claim rule for S4.** SSP is "fixed" only if (a) its paired difference to raw has a 95% CI inside the equivalence margin
  (3% of raw's mean J_task), and (b) it beats matched RFF with a CI excluding 0. Otherwise the conclusion is "SSP does not
  match raw under centering at this budget, even tuned".

## 5. Code to add before launching (about 30 minutes plus tests)

- `RunSpec.desired_kl` and `RunSpec.schedule` ("adaptive" or "fixed"), plumbed through `build_agent_cfg` (it already takes
  `learning_rate`).
- `waves.ssp_s0() .. ssp_s4()` builders and manifests, with count / seed-disjointness tests like the existing wave tests.
- No change to the encoder: `ssp_n_rotates`, `ssp_n_scales`, `ssp_length_scale` already pass through `RunSpec.extra`, and `rff_low`
  already matches the SSP width.

## 6. Budget and time

Balance $8.03 (no pods running).

- S0 + G0 only: $1.74 (about $2.0 with overhead). Decides whether any tuning is needed.
- S0 + S1 + S2: $6.41 (about $7.4 with a 15% overhead for bootstraps and under-filled tails). Fits the balance, leaves about $0.6.
- Full S0-S4 at 6 confirmation seeds: $13.44 (about $15.5 with overhead): **needs a top-up of about $7.5**.
- G1 diagnosis (decision 73): offset 0 uncentered + centered at offset 1, 1 seed x 2000 iterations, about $1, on top of
  any of these. The `--episodic` control ($0.52) is conditional on the offset-0 run failing.

Wall clock: a 1500-iteration SSP run takes about 1.8 h at 6-way concurrency, a 2000-iteration run about 2.4 h. With three pods
S0 + S1 finish in about 3-4 hours, S2 and S3 add about 2 hours each.
