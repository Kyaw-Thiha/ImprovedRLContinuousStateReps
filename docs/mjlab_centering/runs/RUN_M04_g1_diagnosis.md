# RUN_M04 — G1 re-diagnosis under the continuing wrapper

Started 2026-11-01. Design: decision 73 in `../02-decisions.md`. Data: campaign `g1diag-7c41e2` on R2
(`mjlab_ssp_centering/g1diag-7c41e2/runs/<job_id>/`), pulled with `analysis/pull_r2.py`.

## 0. Why this run exists

Step 0 (free) read every G1 job spec and `final.json` from the previous waves: **all of them ran at offset 1.0**. The
2500/5000-iteration reference probe was `center=none, gamma=.99, offset=1.0`, i.e. the same cell that also fails on
Go1, and the centered eta sweep ran only 800 iterations (too short for G1, decision 50). REPORT_002 4.1, decision 59
and decision 70a blamed the continuing conversion or the pinned versions; they are corrected in place and superseded
by decision 73. Reading the pinned mjlab config (8bb2bc9e) found no curriculum that our wrapper freezes: the command
curriculum is step-count based, the terrain curriculum is off on flat, and only start-state resets are lost.

## 1. Runs (1 search seed, 2000 iterations, gamma .99, 2048 envs, parallel pods)

| job | change against the failed reference | question |
|---|---|---|
| `g1diag_raw_none_g0.99_o0_s1000` | offset 1.0 -> 0.0 | does the offset alone explain the G1 failure? |
| `g1diag_raw_value_g0.99_o1_s1000` | centering on (value, eta .03, r_bar first-batch) | does centering rescue G1 as it rescues Go1? |
| `g1diag_episodic_none_g0.99_o0_s1000` | our wrapper off: time-outs and fall termination kept, no fall penalty | added after run A: is the conversion itself the cause? |

Pre-registered before launch: "walks" = mean `track_lin_frac` of the last 5 evals >= 0.5 (standing floor 0.267, the
failed reference 0.29-0.35). One seed is a probe, not a claim.

## 2. Launch

| pod | id | gpu | price |
|---|---|---|---|
| `mjc-g1diag-7c41e2-0` | `x95cqxvz83riv8` | RTX 4090 SECURE | $0.89/h |
| `mjc-g1diag-7c41e2-1` | `xb5wytq9ca1g5v` | RTX 4090 SECURE | $0.89/h |

The cheaper GPUs (4090 COMMUNITY $0.34, A6000 $0.33, A40 $0.59) had no stock at launch, so the fallback was the
$0.89/h SECURE 4090. Measured bootstrap was ~17 min (first mirrored row), not the ~95 min the bootstrap comment warns
about, and a 2000-iteration G1 job took ~22 min, so a 4-job campaign is ~2.2 pod-hours. Both workers finished with the
queue drained but did not self-delete within ~10 min, so the pods were deleted through the API after the per-run R2
mirrors were verified (16.5 min of idle billing in total). Expected: bootstrap plus about 1.1 h of training per job (measured 1.95 s/iteration for G1 at
5-6 concurrent processes on an A40).

## 3. Results

Run A (offset 0, no centering), 2000 iterations, DONE:

| iter | track_lin_frac | J_task | falls / 1k |
|---|---|---|---|
| 49 | 0.243 | 0.0409 | 10.2 |
| 549 | 0.319 | 0.0440 | 0.0 |
| 1049 | 0.320 | 0.0445 | 0.0 |
| 1549 | 0.313 | 0.0442 | 0.0 |
| 1999 | 0.289 | 0.0427 | 0.0 |

Last-5 mean: `track_lin_frac` 0.307 (standing 0.267), J_task 0.0472. Action std 1.0 -> 0.573. **Does not walk**, so the
offset is not the cause (decision 74). Go1 at the same cell walks by iteration ~300 (`plateau_ref_*`).

Run B (centered, offset 1), 2000 iterations, DONE: last-5 mean `track_lin_frac` 0.309, J_task 0.0458, action std 0.552,
falls 0.02/1k, `r_bar` 0.879 and still climbing. **Does not walk.**

## 4. The decisive pair

| arm | time-outs | falls terminal | centering | offset | track_lin | J_task | walks |
|---|---|---|---|---|---|---|---|
| wrapped (A) | no | no | none | 0 | 0.307 | 0.0472 | no |
| wrapped (B) | no | no | value | 1 | 0.309 | 0.0458 | no |
| episodic | yes | yes | none | 0 | **0.813** | 0.0926 | **yes** |
| time-outs kept, falls non-terminal | yes | no | none | 0 | **0.811** | 0.0924 | **yes** |

Episodic curve: 0.067 (iter 49, 121 falls/1k) -> 0.307 (799) -> 0.359 (1049) -> 0.681 (1299) -> 0.788 (1549) -> 0.810
(1799). The time-outs-kept arm follows the same curve (0.814 at 1799). Both take off around iteration 1100; the wrapped
arms are flat from iteration 300 to 2000 and their action std shrinks to 0.55-0.57, against ~0.38 in the reset arms.

Reading: removing the time-outs is what stops G1 learning. Keeping them is sufficient, and the fall termination is not
needed. Centering does not substitute for the resets. Recorded as decisions 75-76; spend on the campaign was about $4
(two RTX 4090 SECURE pods at $0.89/h, roughly 2.2 pod-hours total).
