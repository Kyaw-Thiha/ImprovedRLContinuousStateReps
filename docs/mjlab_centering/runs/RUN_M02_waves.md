# RUN_M02 — gate, wave 1 (tuning), the linear slice, and the ship-blocking defects found

Started 2026-10-31, continues from RUN_M01 (whose pod-failure narrative is superseded; see decisions 35-47).
Everything below is measured, not estimated, unless marked as an estimate. Numbers come from the per-run JSONL on the
pods, pulled to `mjlab_centering/results/`.

## 1. Gate: does the plain reference arm learn, and does the continuing wrapper change that?

`analysis/gate_ref.py`, two search seeds per arm, 1500 iterations x 2048 envs, one A40. Both arms were evaluated on the
same continuing task (eval env always through the wrapper, offset 0, penalty 10).

| arm | J_task final | track_lin_frac final | falls / 1k | plateau |
|---|---|---|---|---|
| wrapped (continuing) seed 1000 | 0.1028 | 0.892 | 0.010 | ~600 |
| wrapped seed 1001 | 0.1012 | 0.885 | 0.021 | ~600 |
| episodic (mjlab default) seed 1000 | 0.1022 | 0.888 | 0.016 | - |
| episodic seed 1001 | 0.1024 | 0.890 | 0.016 | - |
| standing policy (zero actions, measured) | 0.0771 | 0.283 | 0.31 | - |

Reading: the continuing wrapper, the 10-point fall penalty and the eval env **do not change learning** (wrapped 1.00x
episodic on both metrics). The reference arm tracks velocity at 0.89 of the maximum per-step term, i.e. 3.2x the standing
baseline, and J_task is +32% over standing, so both metrics discriminate once scaled correctly.
Cross-check against the stock pipeline: `mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-Go1` at 2048 envs reported
track_linear_velocity f ~ 0.25-0.32 at its iterations 75-84 (Mean reward ~40); ours is 0.26 at iteration 99.

The script's pre-set `final >= 3x first eval` criterion misses by 0.4% on one episodic seed (2.987). Recorded as
FAIL-by-0.4% on a badly chosen relative criterion, PASS on substance; the criterion for future gates is now
`final >= 2x standing`. Stated openly as post hoc.

## 2. Two harness defects that would have invalidated the study

1. **The SSP observation arm had never run.** `install_encoder` was called from nowhere, so every "ssp" job in the old
   manifests trained on raw observations. The draft also recursed (it read the original term back through the manager,
   which after replacement *is* the SSP term), and none of the three implementations moved the encoder to the GPU. Fixed,
   wired into training and eval envs, and now verified inside a real env: Go1 actor 48 -> 1068, critic 72 -> 1092;
   G1 actor 99 -> 1119.
2. **The eval log divided reward terms by `num_envs` twice**, so every `term/*` and `track_*_frac` in the first gate
   logs was 128x too small. Two decisions (43, 44) were written on the mis-scaled numbers and are corrected by 45-46.
   A regression test (`tests/test_eval_terms.py`) now pins the convention against a fake env: `term_rewards` is
   weight x f x dt, so `f = term / (weight x dt)`.

Both are the class of defect the plan's "gate trainability before anything else" rule exists to catch. Neither was
catchable from the run records alone, because the records had no per-term or tracking columns before this run.

## 3. Wave 1 (tuning, search seeds 1000-1001 so far, 800 iterations x 2048 envs)

### 3a. eta for value centering (Go1, gamma 0.999, offset c)

| eta | r_bar init | n | J_task | track_lin | note |
|---|---|---|---|---|---|
| 0.01 | zero | 2 | 0.0571 | 0.317 | stalls: indistinguishable from the SSP arm's floor |
| 0.03 | zero | 3 | 0.0710 (0.0791, 0.0791, 0.0547) | 0.486 | **fails on seed 1002**: the transient makes it unreliable |
| 0.1 | zero | 3 | 0.0884 (0.0871, 0.0871, 0.0908) | 0.800 | |
| 0.3 | zero | 3 | 0.0915 (0.0913, 0.0913, 0.0920) | 0.861 | best mean, but a 10x step |
| 0.03 | first_batch | 3 | 0.0908 (0.0891, 0.0891, 0.0942) | 0.860 | **frozen for wave 2**: best and tightest spread |

Two findings, both new to this project:

- **eta is not a nuisance parameter.** At 0.01 the centering arm fails to learn (J_task 0.057, tracking 0.317 - the
  standing-policy floor), while at 0.03-0.3 it reaches 0.089-0.091 / 0.84-0.86. The r_bar estimate starts at 0 and
  must climb to about 1.1 before the critic sees a stationary target; a small eta means the centre chases the policy
  for most of the run.
- **Initialising r_bar from the first batch is a substitute for a large eta**: eta 0.03 + first-batch init (J 0.0891,
  track 0.840) matches eta 0.3 with zero init (0.0913 / 0.862). That is the cheapest form of the fix and the one to
  carry into the factorial if it holds on the third seed (running).

Third search seed (1002) is decisive for the mechanism: eta 0.03 with zero init collapses on that seed (J 0.0547,
tracking 0.301 - the same failure mode as eta 0.01), while eta 0.03 with first-batch init gives 0.0942/0.868, the best
single number in the sweep. Frozen for wave 2: **eta 0.03 with r_bar initialised from the first batch** - the same
performance as eta 0.3 at a 10x smaller step, and the only small-eta cell with seed-stable behaviour.

### 3b. SSP length-scale sweep (Go1, uncentered, gamma 0.99, offset c) - UNINFORMATIVE, see correction below

| arm | length scale | n | J_task | track_lin |
|---|---|---|---|---|
| ssp_low (`command` + `projected_gravity`) | 0.15 | 3 | 0.0563 | 0.311 |
| ssp_low | 0.30 | 3 | 0.0569 | 0.311 |
| ssp_low | 0.60 | 3 | 0.0580 | 0.319 |
| raw comparator: NOT like-for-like (the gate arm had no offset; with the offset and no centering raw also fails, J_task ~0.03-0.06) | - | 2 | ~0.09 | 0.86 |

> **Correction (decision 70b).** This sweep ran with no centering at offset c, where every arm fails (the raw comparator
> at the same setting is also at the standing floor). It therefore could not discriminate length scales, and the statements
> below that read it as 'length scale makes no difference' or 'SSP does not train' are withdrawn. The width control (wave 9)
> replaces them.

The SSP arm is flat for all 800 iterations at every length scale: tracking 0.31 = the standing floor, J_task 0.057.
Length scale makes no difference (0.15/0.30/0.60 within 0.014 J_task), which argues against "the kernel bandwidth is
wrong" and for something structural.

Training diagnostics point at the optimiser, not the encoder's information content: the SSP run's adaptive learning
rate collapses to 1e-5..7.6e-5 (raw keeps 1e-4..4e-4), its entropy stays at 7.7 against raw's 11-12, and its value
function fits well (explained variance 0.95), i.e. it is a stable, badly-stepping policy. With a 22x wider input the
same `desired_kl=0.01` clips every update, so the KL-adaptive schedule hands the SSP arm much smaller effective steps.

**Consequence (decision 48):** the raw-48 vs ssp-1068 comparison is confounded with input width. An equal-width control
is now implemented (`encoder="rff_low"`: random Fourier features on the same two terms, same 510-wide output per term,
verified 1068 actor / 1092 critic) and wave 5 runs raw / rff_low / ssp_low at 3 seeds. Only after that can a
representation claim be made - as of now the honest statement is "SSP-low at 22x width does not train in 800 iterations
at the mjlab KL settings".

### 3c. G1 does not learn at the Go1 run length

| arm | n | J_task | track_lin | falls / 1k |
|---|---|---|---|---|
| G1 standing (zero actions, measured) | - | 0.0436 | 0.267 | 10.3 |
| G1 value centering, eta 0.01 | 3 | 0.0258 | 0.311 | - |
| G1 value centering, eta 0.03 | 3 | 0.0404 | 0.303 | - |
| G1 value centering, eta 0.1 | 3 | 0.0410 | 0.306 | - |

Every G1 arm sits at or below its own standing baseline after 800 iterations, so a G1 centering/SSP contrast at this
length measures nothing but the floor. Wave 6 runs a G1 raw reference at 2500 iterations to set G1's own run length
before wave 3 is generated.

## 4. Linear-critic slice (NumPy, exact truth on continuing Catch)

Fixed-policy TD(0) against the exactly computed value of the chase policy (225 states); 20 seeds; error under the
stationary distribution, **split into a constant part and a state-dependent part**. The split matters: with centering the
critic fits r - r_bar_hat, so a wrong r_bar leaves a pure constant k/(1-gamma) that cannot change any action (Naik eq. 6).
Each arm is scored against its own correct target (uncentered: true v; centered: v - r_bar_hat/(1-gamma) with the learned r_bar).

Offset c, gamma 0.99 (MSE / state-part / constant part):

| features | no centering | centered |
|---|---|---|
| raw (no bias, cannot express a constant) | 8252 / **96.8** / 8155 | 4.44 / **0.031** / 4.41 |
| raw + bias | 0.306 / **0.027** / 0.279 | 2.69 / **0.024** / 2.67 |
| tile coding | 186 / **1.93** / 184 | 0.449 / **0.007** / 0.443 |
| hexssp | 0.274 / **0.073** / 0.202 | 0.285 / **0.052** / 0.233 |
| rff | 1.70 / **0.81** / 0.89 | 0.210 / **0.133** / 0.077 |

At gamma 0.999 the same pattern is amplified (raw+bias: 1.24 / 0.033 uncentered vs 220.8 / **0.025** centered; tile:
3156 / 0.123 vs 43.5 / **0.007**).

**Corrected claim (decision 54).** An earlier draft read the 220.8 as "centering is worse with a bias unit". It is not:
98.6% of that number is the leftover constant, and the decision-relevant state-dependent error is *smaller* with
centering (0.025 vs 0.033). What the table shows:

- Where the basis cannot express a constant (raw), an offset destroys learning without centering (state-part 96.8 against
  0.03) - three orders of magnitude, and the learned |v| goes to 16.8 instead of ~1070.
- A bias unit removes the offset problem at the *value* level without centering (state-part 0.027), exactly as predicted.
- Dense codes (hexssp, rff) also express a near-constant, so centering helps them least.

**Linear one-step actor-critic (600 runs, 10 seeds), offset c, task return, r_bar pre-rolled on the initial policy:**

| features | gamma | no centering | centered |
|---|---|---|---|
| raw | .9 / .99 / .999 | -0.0665 / -0.0665 / -0.0665 | 0.0314 / 0.0303 / 0.0321 |
| raw + bias | .9 / .99 / .999 | -0.0665 / -0.0665 / -0.0665 | 0.0137 / 0.0098 / 0.0115 |
| tile | .9 / .99 / .999 | 0.106 / 0.017 / -0.014 | 0.105 / 0.106 / 0.106 |
| hexssp | .9 / .99 / .999 | 0.0527 / -0.0666 / -0.0664 | 0.0502 / 0.0547 / 0.0621 |
| rff | .9 / .99 / .999 | 0.103 / -0.062 / -0.066 | 0.103 / 0.105 / -0.062 * |

(-0.0665 and -0.0664 are the same collapsed policy: task return -1 every step minus the offset; the * entry is a seed
or two that also collapse.)

**The control result contradicts the value result, and that is the finding.** A bias unit absorbs the offset for the
*critic* but not for the *actor*: with raw+bias at every gamma, uncentered control collapses while centered control
learns. So "the critic can represent a constant, therefore centering is unnecessary" is false at the controller level in
this setting - what matters is whether the *TD errors* carry the offset early enough to destroy the policy. This is the
mechanism the Go1 experiment tests with a deep critic, and it is why the critic-fix rivals (clip-off, return-norm) are
the deciding arms there.

The first AC grid (r_bar starting at 0) collapsed in most offset cells for *both* arms; the pre-roll is what makes the
comparison informative, and it is the same fix as the Go1 eta result (RUN_M02 s3a).

## 5. Cost and throughput (measured)

| item | value |
|---|---|
| pod bootstrap (venv + mjlab git pin + JIT warm-up) | ~2 min (not ~95) |
| concurrent training processes per A40 | 5-6 at ~1.9-2.1 s/iteration (Go1), ~2.9-5 s (G1) |
| 800-iteration Go1 run at that load | ~$0.07-0.10 |
| spend to date | balance 28.17 -> ~24 (3 A40s at $0.59/h) |

## 6. What is running when this entry was written

- pod 1: wave 5 equal-width diagnostic (raw / rff_low / ssp_low x 3 seeds, 800 iterations).
- pod 2: last 6 jobs of wave 1 (G1 eta seeds 1001/1002).
- pod 3: wave 6 (G1 length probe at 2500 iterations, 2 seeds; eta 0.1/0.3 third-seed confirmations).

## 7. G1: our reference does not learn, and that is NOT an expected property of the task

| iterations | 400 | 1100 | 1800 | 2500 | 3200 | 3900 | 4600 | 4949 |
|---|---|---|---|---|---|---|---|---|
| tracking (f of max) | 0.291 | 0.338 | 0.296 | 0.307 | 0.335 | 0.283 | 0.306 | 0.309 |
| J_task | 0.033 | 0.037 | 0.038 | 0.039 | 0.042 | 0.040 | 0.042 | 0.042 |
| falls / 1k | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Standing baseline: tracking 0.267, J_task 0.0436. Our G1 reference (continuing wrapper, raw, no centering, gamma .99) learns
to stop falling and never walks, at 6x the Go1 length.

**This does not mean G1 cannot learn at this budget.** An independent run of the stock `Mjlab-Velocity-Flat-Unitree-G1` task
(2048 envs, mjlab 1.3.0, rsl-rl 5.0.1; github.com/chaoticcuriosity-io/g1-humanoid-rl) reports mean reward 3.3 at iteration 500,
21.3 at 900, 31.5 at 1400 and 50.5 at 2050 with 995/1000-step episodes: a sharp climb between iterations 500 and 1000. The
earlier statement here that stock G1 was "worse at the same budget" came from one reading at iteration 75 and is withdrawn.
Our flat curve is the uncentered-at-offset-c cell (offset 1.0, no centering), which also fails on Go1, so it does not by itself
implicate our setup: neither the continuing conversion (no terminal signal on a robot that falls far more often than Go1) nor
the pinned versions (mjlab 8bb2bc9e, rsl-rl 5.5.1). Controls still to run: the same G1 reference at offset 0, centering at
offset 1, and `--episodic` (decision 73). Until then G1 is reported as
**unresolved**, not as a budget limitation. (Resolved afterwards in RUN_M04 / decision 75: the wrapper's missing time-outs were
the cause, not the offset.) The no-G1-factorial rule of decision 56 stands only because the budget is
spent, not because G1 was shown to be unlearnable.

## 8. Operations (decisions 60-61)

- Pods are driven by `keeper.sh`: flock-guarded, restarts `podrun` if it dies, counts pending jobs from the manifest, and on
  drain pushes `_FINAL_results.tgz` to R2 and deletes the pod via the API, so an unattended pod cannot bill indefinitely.
- Every pod snapshots to R2 every 5 minutes, and every queue passes `--r2` so each job also mirrors per run.
- Two losses were self-inflicted and are documented in entry 60: `pkill -f` matching the invoking shell, and a queue started
  from an ssh one-liner that did not survive session teardown.

## 9. Wave 9: the equal-width control that rescues (and reframes) the SSP arm

Centered (eta 0.03, first-batch init), Go1, gamma 0.99, offset c, 800 iterations, search seeds. Every encoded arm has the
same 1068-wide actor input, so width is held constant and only the code differs.

| arm | J_task (s1000 / s1001) | track_lin (s1000 / s1001) |
|---|---|---|
| raw (48-wide) | 0.100 / 0.098 | 0.876 / 0.858 |
| rff_low (equal width) | 0.072 / 0.070 | 0.457 / 0.382 |
| ssp_low (equal width) | 0.085 / 0.082 | 0.596 / 0.543 |

Learning curves (seed 1000, tracking): raw 0.29 -> 0.88; ssp_low 0.33 -> 0.67 and still rising; rff_low 0.33 -> 0.47 and flat.

Reading:
- The 22x wider input costs about a quarter of the task reward by itself (raw 0.099 vs rff 0.071). Wave 1's "SSP is flat at
  the standing floor" was measuring that width penalty, not the SSP code.
- At fixed width the hexagonal code is worth +0.013 J_task and +0.15 tracking over random Fourier features, on both seeds,
  with a visibly different slope. That is the first SSP advantage in this project that survives an equal-aid control.
- It is not a win over raw at this budget: raw is ~0.015 J_task ahead. The defensible claim is "SSP beats an equal-width random
  feature map and narrows the gap to raw input, without closing it".

