# RUN_007 — SSP algebra controller: supervised probing of three variants

**Status: COMPLETE for handoff steps 1-4 plus the cross-cutting ablations (2026-10-11). CONCLUSIONS REVISED after review: read section 0 first, it overrides the verdicts and "win" wording further down.** Local only, no GPU, no
pod: 16 cores, repo `.venv` (Python 3.14.6, torch 2.14.0, numpy 2.5.3). Every number below is MEASURED in this
session unless marked otherwise. Written with the RUN_005/006 conventions: measured vs inferred is labelled, bugs
are recorded, limitations are stated rather than implied.

Code is a new self-contained folder, `experiments/ssp_algebra/` (see its `README.md`). Nothing else in the repo was
modified. Artifacts: `results/sweep_main.jsonl` (501 runs), `results/ablation_abl.jsonl` (95 runs),
`results/analysis_main.json` (the verdicts), `selftest.json`.

**Step 7 (RL for surviving arms) was not run**: it is the next milestone, and section 6 states which arms it applies
to and what the gate is.

---

## 0. Revised conclusions (read this first)

**What this run actually tested.** Every SSP arm here is *hand-wired*: I chose the write address, the read address, the
key/value roles, the hop loop and the pose integration, and only a small readout (and the forgetting rate, and for
`delay` a lag-to-offset map) was learned. So this is **rung 0** of the learned-structure ladder in
`docs/handoffs/09-ssp-learned-structure-sample-efficiency.md`: it shows what the algebra can do *when operated
correctly*, an upper bound on what a learner could discover. It does **not** test the thesis (that a learner can
discover and use SSP structure with fewer samples, shorter episodes, and better robustness than generic networks).
Generic baselines had to learn all of that from scratch, so the comparison is "correct algorithm given" against
"must find the algorithm".

**Corrections to the sections below.**

1. **The headline metric was a short learning-curve area, not a converged score.** `last5_*` averages 5 of only 6
   checkpoints (steps 300-1500, evaluated every 300 updates), so it blends speed and convergence. Converged scores
   (step 1500) are in section 4.5. They change several readings.
2. **`index`: converged parity, not a win.** At step 1500 the GRU and SSP are both **0.999**. The in-distribution
   advantage is *speed only*: SSP is at 0.998 at the first checkpoint (300 updates) while the GRU first passes 0.9 at
   600 updates. The checkpoint spacing is too coarse to say how much faster. The extrapolation gap (0.803 vs 0.159 at a
   4x longer stream) is real in the data but **confounded**: baselines receive the query as `j/T` while the SSP arm
   multiplies by `T` internally, so it is handed the de-normalisation.
3. **`pointer`: the GRU is better at convergence in-distribution** (0.359 vs 0.312) and still rising. SSP leads early
   (0.270 vs 0.145 at 300 updates) and on held-out hop counts (0.240 vs 0.151 at step 1500), but no arm is near
   competence (chance 0.125).
3b. **`delay`: SSP converges to 1.000**, tied with LRU/GRU, but is slower (first >=0.9 at 600 updates vs 300).
4. **The mechanical verdicts `proceed_to_rl` for `index` and `pointer` are withdrawn.** By docs/11 section 4.5 both are
   `structural_prior_only` (they win, if at all, only with hand-wired operations) and neither licenses RL. The
   `analysis.py` margin test used the `last5` metric and its fragility check only inspected the in-distribution
   split; both need fixing before it is used again.
5. **The baselines are undertrained, so `tf_full` is not an upper bound here.** It reaches only 0.37 on `index` and
   0.54 on `delay`, tasks it can represent trivially. Any claim against it needs a trainability gate first.
6. **The ablations show the SSP *time parameterisation* is not what wins on `index`** (`pos_table` 0.998 vs 0.999):
   an addressable key-value memory is doing the work, which is not specific to SSP.

**What the data does support (at rung 0, hand-wired, one budget of 256 floats, 5 seeds):** the *shape* the thesis
predicts, namely fast early learning on `index` and `pointer`, a lead on held-out hop counts, and parity with the
best generic arm at convergence. Whether that survives removing the hand-wiring is the open question.

**The step-1 finding (section 3) stands and matters on its own:** POPGym's own PPO trains a GRU on
`RepeatPreviousEasy` to +0.42 at 296k steps where our A2C GRU sat at -0.108 after 490k. RUN_006's headline
comparison (hand-wired delay memory +0.990 vs GRU -0.108) was therefore largely a badly trained baseline; treat that
result as structural-prior-only until it is re-run against a properly trained GRU.

**Next:** `docs/handoffs/09-ssp-learned-structure-sample-efficiency.md` (measurement fixes, then learnable-structure
rungs, then robustness). Step 7 (RL) was not run and is not licensed by these results.

---

## 1. What was built

| File | What it is |
|---|---|
| `ssp_ops.py` | FHRR/SSP primitives: `phi(x)=exp(i.Ωx)`, `pos(t)` (time as an SSP), bind/unbind/shift, leaky bundle, cleanup |
| `tasks_c.py` | variant C: learned-delta read (per-episode lag as an input feature) and index-addressed recall |
| `tasks_b.py` | variant B: pointer chasing over bound pairs, per-query hop count |
| `tasks_a.py` | variant A: 2-D landmark return (path integration + relative-coordinate query); supervised proxy for the simulator |
| `arms.py` | 8 arms, explicit state-float accounting, budget-matched construction, variant-specific SSP arms |
| `harness.py` | teacher-forced training, checkpointed eval on in-distribution and extrapolation splits, random search |
| `run_sweep.py` | parallel resumable sweep and ablation runner |
| `analysis.py` | the docs/11 section 4.5 decision rules, applied mechanically |
| `selftest.py`, `test_harness.py` | the two gates |

**State matching** counts only floats carried *between steps*: `gru: H`; `lru: 2N`; `ssp: 2D`;
`tf_window`/`kernel_attn: 2·layers·d_model·window`; `mlp`/`posmlp: 0`. Reference budget **256 floats**
(`gru H=256`, `lru N=128`, `ssp dim=128` complex, windowed arms `window=4` at `d_model=32, layers=1`,
`tf_full` unlimited state = 2048-6144). State-size knobs are *excluded* from the random search, since tuning them
would break the matching the claim rests on; every arm sees the same inputs.

## 2. Gates (both pass)

`selftest.py`:
- **T0 algebra parity** (float64): bind `4.3e-15`, unbind `4.4e-15`, kappa-depends-only-on-difference `2.2e-16`.
  In float32 the same test reads `2.4e-6`, which is *not* an algebra error but rounding set by the phase magnitude
  `|Ω·x| ≈ 12`; parity is therefore gated in float64, as RUN_005 did.
- **Hand-wired SSP-time delay readout as a policy** on POPGym `RepeatPreviousEasy` (`k=4`, `lag=k-1` re-confirmed):
  policy return `dim=128 +0.913`, `dim=256 +0.990`, `dim=512 +0.998`, against oracle `+1.000` and random `-0.522`.
  RUN_006's random position *table* reached `+0.982` at dim 256 with a trained readout, so real SSP time is
  equivalent-or-better at equal dimension.

`test_harness.py` (6/6): the LRU recurrence with a trained readout reaches **1.000** on a fixed-lag delay (the
substitute really contains a delay line); with delta pinned to the true lag the cleanup argmax **is** the answer;
the SSP arm reaches 0.948 with `corr(learned delta, true lag) = +0.999`; every memory arm is within 5% of the
budget at 256/512/1024; identical seeds give identical curves; all 8 arms train on all 4 variants.

## 3. Step 1 — POPGym's own PPO reference (the loop-vs-task question)

POPGym's `baselines/ppo.py` **does not run on ray 2.59 unmodified**; the failures are API drift, not protocol:
`local_dir` is now a hard error (renamed `storage_path`); a single-element `ray.tune.grid_search` resolves to a bare
value so its `trial_name_creator` raises; `num_workers`/`num_envs_per_worker` were renamed
`num_env_runners`/`num_envs_per_env_runner` and `horizon` was removed; and the baseline extra declares **no**
dependencies (`dnc`, `einops`, `opt_einsum`, `scipy`, `pillow`, `matplotlib`, `imageio` all had to be installed).

The reference therefore drives `PPOConfig` directly with POPGym's own `ray_models` classes and its own config dict,
with the old API stack requested explicitly so the shipped `ModelV2` models apply. Throwaway venv outside the repo
(Python 3.14, ray 2.59.0, torch 2.14.1+cpu, popgym 1.0.7). Deviations: the entry point and that explicit flag.

| env | steps reached | episode return | reading |
|---|---|---|---|
| `RepeatPreviousEasy` | 296k | `-0.502 → +0.418`, still rising | **their loop learns it** |
| `CountRecallEasy` | 98k (wall-clock cap) | `-0.929 → -0.910`, flat | **no learning at ~100k steps** |

- For `RepeatPreviousEasy` our in-repo A2C GRU was at `-0.108` after 490k steps (RUN_006 §4) while POPGym's own
  loop is at `+0.42` after 296k and rising: **the loop, not the task, explains that gap.**
- For `CountRecallEasy` POPGym's own loop is *also* at chance at ~100k steps, consistent with RUN_006's shared
  failure. This is **partial**: their published budget is up to 15e6 steps, so it does not show their loop cannot
  learn the task, and no comparison to published numbers is claimed (different budget, different machine).

## 4. The sweep

Equal random-search budget per arm (8 configs, scored on search seeds only), winner re-run on 5 **hold-out** seeds
(1000-1004). **Numbers in 4.1-4.4 are `last5` = the mean of the last 5 of 6 checkpoints, which is a short
learning-curve area (it includes step 300); converged step-1500 scores are in 4.5 and are what to quote for
"accuracy".** Never max-over-epochs. Reference budget 256 floats.
± is the standard error over the 5 hold-out seeds. For `path` the metric is `-MSE` (higher is better; the MSE itself
is the negative of the number shown).

### 4.1 `delay` — variant C, learned continuous delta

| arm | state | params | id | extrap_lag | extrap_T |
|---|---|---|---|---|---|
| lru | 256 | 10,280 | **+0.998±0.001** | +0.122±0.002 | +0.367±0.013 |
| gru | 256 | 222,024 | +0.995±0.000 | +0.199±0.005 | +0.354±0.014 |
| **ssp** | 256 | **1,683** | +0.933±0.003 | +0.185±0.005 | +0.369±0.013 |
| kernel_attn | 256 | 3,820 | +0.787±0.005 | +0.128±0.003 | +0.368±0.013 |
| tf_full | 4096 | 103,080 | +0.493±0.004 | **+0.322±0.010** | +0.276±0.002 |
| tf_window | 256 | 15,720 | +0.388±0.003 | +0.126±0.002 | +0.366±0.003 |
| mlp / posmlp | 0 | 584/1288 | +0.127 / +0.126 | +0.128 / +0.126 | +0.126 / +0.125 |

**Verdict: `algebra_buys_nothing`** — not separated from the GRU and the LRU in-distribution (it is *below* both).
Two honest sub-results: (a) on the held-out **lag** axis the LRU collapses to chance (`0.122`, floor `0.127`) and
the kernel/windowed arms do too, while the SSP arm holds `0.185` — the continuous delta transfers to lags it never
saw, which the diagonal-recurrence readout does not; but (b) the GRU (`0.199`) and the full-context transformer
(`0.322`) hold up at least as well. The learned delta itself is learned perfectly
(`corr(learned delta, requested lag) = +0.999`), so this is not a mechanism failure — it is the pre-registered
expectation that variant C is substitutable (docs/11 section 1.1).

*Revision:* at step 1500 SSP is the highest in-distribution arm (1.000) but tied with LRU/GRU (0.999/0.998), and it
learns more *slowly* (first checkpoint at >=0.9: 600 updates vs 300). On held-out lags (final): SSP 0.224, GRU 0.201,
tf_full 0.361, LRU 0.122 (chance 0.127). The expectation that this variant is substitutable stands.

### 4.2 `index` — variant C, index-addressed recall

| arm | state | params | id | extrap_T (4× longer stream) |
|---|---|---|---|---|
| **ssp** | 256 | **1,673** | **+0.999±0.001** | **+0.793±0.008** |
| gru | 256 | 222,024 | +0.948±0.008 | +0.165±0.008 |
| lru | 256 | 18,760 | +0.443±0.007 | +0.169±0.003 |
| tf_full | 2048 | 28,424 | +0.348±0.014 | +0.189±0.005 |
| kernel_attn | 256 | 5,132 | +0.247±0.002 | +0.155±0.002 |
| tf_window | 256 | 14,408 | +0.245±0.003 | +0.153±0.002 |
| posmlp / mlp | 0 | 2,568/584 | +0.128 / +0.124 | +0.120 / +0.122 |

**Mechanical verdict: `proceed_to_rl` -- WITHDRAWN (see section 0).** The `last5` numbers above blend speed and
convergence. At step 1500 the GRU and SSP are tied at 0.999 (section 4.5); the SSP arm is *faster* (>=0.998 at 300
updates vs the GRU's 0.786) and holds 0.803 on a 4x longer stream where every other arm falls to 0.15-0.20 (the
no-memory floor is ~0.12). It does this with 1,673 parameters against the GRU's 222,024. The extrapolation split is
confounded by input encoding (baselines get `j/T`, the SSP arm multiplies by `T`), so treat it as a hand-wired
result that has not been shown against fairly encoded baselines.

What it isolates: the ablations (section 5) show index recall is insensitive to *which* position code is used
(`pos_table` = 0.998), so what is load-bearing is the addressable memory (bundle + unbind + cleanup at a fixed
address), not the smooth SSP time parameterisation. Revised classification: `structural_prior_only`.

### 4.3 `pointer` — variant B, iterated unbind + cleanup

| arm | state | params | id | extrap_lag (hops 4-8) |
|---|---|---|---|---|
| **ssp** | 256 | **1,865** | **+0.299±0.002** | **+0.245±0.007** |
| gru | 256 | 224,328 | +0.236±0.014 | +0.168±0.007 |
| lru | 256 | 19,144 | +0.171±0.003 | +0.170±0.001 |
| kernel_attn | 256 | 5,228 | +0.166±0.002 | +0.166±0.002 |
| tf_full | 3072 | 28,520 | +0.160±0.005 | +0.169±0.008 |
| mlp | 0 | 2,696 | +0.147±0.002 | +0.179±0.004 |
| tf_window | 256 | 14,504 | +0.145±0.004 | +0.164±0.012 |
| posmlp | 0 | 1,480 | +0.144±0.002 | +0.176±0.003 |

**Mechanical verdict: `proceed_to_rl` -- WITHDRAWN (see section 0).** On the `last5` metric SSP is best in
distribution (`0.299` vs GRU `0.236`), but at convergence the GRU is ahead (`0.359` vs `0.312`, section 4.5) and still
rising, while SSP plateaus. SSP does lead early (0.270 vs 0.145 at 300 updates) and on held-out hop counts
(`0.240` vs `0.151` at step 1500; trained on 1-2 hops only). **No arm is near competence**: chance is `0.125` and
solving is `1.0`. The result is a structural-prior one, because the hop iteration is hand-wired from the
task-visible hop count (docs/11 section 3.2), and the ablations do not remove that hand-wiring (they only remove
forgetting and the soft cleanup, both of which change nothing here). Revised classification:
`structural_prior_only`.

### 4.4 `path` — variant A, 2-D landmark return (MSE, lower is better)

| arm | state | params | id (MSE) | extrap_T (4× longer path) | extrap_step (2× steps, 2× arena) |
|---|---|---|---|---|---|
| gru | 256 | 227,010 | **0.013**±0.000 | **0.117±0.005** | **0.065±0.002** |
| tf_full | 4096 | 105,474 | 0.014±0.000 | 0.166±0.008 | 0.140±0.007 |
| lru | 256 | 10,978 | 0.015±0.000 | 0.175±0.008 | 0.099±0.005 |
| ssp | 256 | 17,603 | 0.016±0.000 | 0.181±0.009 | 0.133±0.006 |
| kernel_attn | 256 | 3,846 | 0.019±0.001 | 0.213±0.010 | 0.168±0.008 |
| tf_window | 256 | 14,434 | 0.019±0.001 | 0.213±0.010 | 0.182±0.009 |
| posmlp / mlp | 0 | 1,346/1,218 | 0.024±0.001 | 0.234±0.011 | 0.230±0.010 |

**Verdict: `algebra_buys_nothing`** — and here it is not close: the GRU integrates the path best on both
extrapolation axes, and the SSP arm is mid-pack, only ~0.008 MSE better than the no-memory floor on the weakly
discriminative in-distribution split. The diagnosis is recorded rather than hidden: with 16 steps of ≤0.25 the
*remembered displacements are small*, so predicting ≈0 already scores 0.024 MSE and the task separates arms only on
the extrapolation axes. A future version needs larger steps/older queries (see 6).

### 4.5 Converged scores and learning curves (added after review)

Same hold-out runs as 4.1-4.4 (5 seeds, winner config per arm), read at **step 1500** instead of averaged over
checkpoints. ± is the standard error over the 5 seeds.

`index` (accuracy; `extrap_T` = 4x longer stream):

| arm | id | extrap_T |
|---|---|---|
| gru | +0.999±0.000 | +0.159±0.011 |
| **ssp** | +0.999±0.001 | **+0.803±0.008** |
| lru | +0.569±0.008 | +0.161±0.005 |
| tf_full | +0.368±0.014 | +0.195±0.015 |
| tf_window | +0.248±0.003 | +0.153±0.004 |
| kernel_attn | +0.246±0.006 | +0.154±0.004 |
| posmlp / mlp | +0.140 / +0.139 | +0.109 / +0.105 |

`delay` (accuracy):

| arm | id | extrap_lag | extrap_T |
|---|---|---|---|
| ssp | +1.000±0.000 | +0.224±0.009 | +0.369±0.013 |
| lru | +0.999±0.001 | +0.122±0.003 | +0.367±0.013 |
| gru | +0.998±0.000 | +0.201±0.008 | +0.348±0.020 |
| kernel_attn | +0.803±0.005 | +0.126±0.005 | +0.368±0.014 |
| tf_full | +0.543±0.005 | +0.361±0.028 | +0.262±0.003 |
| tf_window | +0.391±0.001 | +0.126±0.005 | +0.368±0.004 |
| posmlp / mlp | +0.124 / +0.122 | +0.127 / +0.127 | +0.125 / +0.124 |

(`delay` `extrap_T` does not discriminate: every memory arm sits at ~0.36.)

`pointer` (accuracy; `extrap_lag` = held-out hops 4-8):

| arm | id | extrap_lag |
|---|---|---|
| gru | **+0.359±0.032** | +0.151±0.009 |
| ssp | +0.312±0.004 | **+0.240±0.007** |
| kernel_attn / lru | +0.189 / +0.184 | +0.162 / +0.162 |
| tf_full / tf_window | +0.181 / +0.156 | +0.147 / +0.150 |
| mlp / posmlp | +0.154 / +0.151 | +0.174 / +0.163 |

`path` (-MSE) is unchanged in ordering (GRU best, SSP mid-pack); see 4.4.

**Learning curves, `id` split, mean over 5 hold-out seeds** (updates; one update is ~513 tokens for `index`, ~864 for
`delay`):

| variant / arm | 0 | 300 | 600 | 900 | 1200 | 1500 | first checkpoint >=0.9 |
|---|---|---|---|---|---|---|---|
| index / ssp | .125 | **.998** | .999 | .999 | .999 | .999 | 300 (5/5 seeds) |
| index / gru | .113 | .786 | .952 | 1.000 | 1.000 | .999 | 600 (5/5) |
| index / lru | .106 | .313 | .346 | .457 | .532 | .569 | never |
| delay / ssp | .124 | .668 | .999 | 1.000 | 1.000 | 1.000 | 600 (5/5) |
| delay / lru | .129 | .998 | .999 | .997 | .998 | .999 | 300 (5/5) |
| delay / gru | .131 | .983 | .997 | .998 | .998 | .998 | 300 (5/5) |
| pointer / ssp | .127 | .270 | .296 | .304 | .310 | .312 | never |
| pointer / gru | .122 | .145 | .161 | .211 | .306 | .359 | never |

Limits of this table: the first evaluation after step 0 is at 300 updates, so any arm already at ceiling there (SSP on
`index`) has an unmeasured, possibly much shorter, time-to-threshold. The first fix in handoff 09 is evaluating every
~25 updates early on.

## 5. Ablations (one change at a time from a per-variant reference, 5 hold-out seeds)

| variant | reference (id) | change | result | reading |
|---|---|---|---|---|
| delay | 0.608±0.014 | random position table instead of SSP time | **0.371±0.009** | the SSP time parameterisation *is* load-bearing here (−0.24) |
| delay | 0.608 | cleanup off | **0.923±0.001** | cleanup is **not** load-bearing — the raw retrieved vector with a linear readout is better |
| delay | 0.608 | learned codes | 0.752±0.009 | learned codes help |
| delay | 0.608 | no decay / gated write / slow forget (λ init 0.99) | 0.572 / 0.621 / 0.609 | forgetting is not decisive at T=16 |
| index | 0.999±0.001 | pos_table / learn_codes / no_cleanup / no_decay / fast_forget (λ init 0.9) | 0.998 / 1.000 / 1.000 / 0.999 / 0.998 | **nothing is load-bearing**: the addressable memory is doing the work, not the mechanism details |
| pointer | 0.283±0.005 | hard cleanup / no decay | 0.282 / 0.281 | the same — and the hand-wired hop loop is untouched by these |
| path | 0.018±0.000 | shift-on-memory vs re-encode | 0.018 (identical) | same answer, as docs/11 section 4.4 expects; the comparison is compute |
| path | 0.018 | no decay | 0.019 | no effect |

**Compute on the one axis where it is the deliverable** (shift the whole map with one multiplication vs re-encoding
each stored item), 5 runs each, same job pool, same 1,440,000 tokens per run: `shift_memory` **46.6 s** vs reference
**55.1 s** (`-15%`) at identical state (2·dim floats) and identical accuracy. Contended single-machine timings, so
directional only.

## 6. Decision-rule outcomes and what follows

Mechanical outputs of `analysis.py` and the revised readings:

| variant | `analysis.py` verdict | revised reading |
|---|---|---|
| C `index` | `proceed_to_rl` | **withdrawn** -> `structural_prior_only`: converged parity, faster early learning, extrapolation lead confounded by input encoding |
| B `pointer` | `proceed_to_rl` | **withdrawn** -> `structural_prior_only`: GRU better at convergence in-distribution, SSP ahead early and on held-out hops, nobody competent |
| C `delay` | `algebra_buys_nothing` | stands (substitutable, as pre-registered); SSP converges to 1.000 but is slower |
| A `path` | `algebra_buys_nothing` | stands as "untested": the proxy was too easy to separate arms (no-memory floor 0.024 vs best 0.013) |

No RL is licensed by this run. Next steps are in `docs/handoffs/09-ssp-learned-structure-sample-efficiency.md`:
1. Fix the measurement (fine evaluation cadence, fair input encoding, baseline trainability gates, converged vs
   area-under-curve reported separately, state-budget curve).
2. Replace hand-wiring with learned structure in rungs (addresses, key/value roles, codes and frequencies, halting),
   with training aids (curriculum, intermediate supervision, near-identity initialisation, temperature annealing).
3. Redesign the variant-A proxy (larger steps, old queries, continuous targets; the competitor is attention over
   stored (pose, landmark) pairs).
4. Robustness and generalisation battery on survivors; RL only afterwards.
5. MIKASA and the continuous-state encoder thread remain out of scope, unchanged by this run.

## 7. Bugs found and fixed (recorded so they are not rediscovered)

1. **Tune/RLlib API drift** (§3) — the reference would not start at all.
2. **The arm readout was built before the seed was set**: `ArmBase.__init__` used the global torch RNG, so equal
   seeds with different RNG history gave different weights and the determinism gate failed.
3. **The SSP arm's readout input dimension was wrong** (`out_dim + in_dim` against an `n_items + in_dim` feature
   vector), so the arm could not run.
4. **Kernel attention mixed head and sequence axes** in its einsum, and its Q/K projections were dead parameters;
   replaced with a value-only projection (purely positional attention, as designed).
5. **The delay task's query tensor was mis-shaped** (`(B,1)` reshaped to `(B,T)`).
6. **The capability ceiling was superposition capacity, not the algebra**: at `dim=256` with 32 live writes the SSP
   arm plateaued at **0.629** while the pinned-delta readout was exact and the delta head learned the lag perfectly
   (`corr=+0.999`) — exactly RUN_005's `m ≈ D/4` rule. Fixed by sizing the reference task inside capacity (T=16)
   and learning λ; the arm then reached 0.948 with λ→0.877.
7. **The forgetting horizon is not a constant.** λ initialised at 0.9 improved `delay` but collapsed `index` to
   0.55, *below* the then-current no-memory floor, because a short horizon down-weights the old writes that index
   recall asks about. Fixed by making `lam_init ∈ {0.9, 0.99}` a **searched** knob plus forgetting ablations.
8. **The index task leaked its own answer**: `j = min(uniform(0,T), t)` asked for the *current* observation whenever
   the sampled index exceeded `t`, so a memoryless MLP scored **0.597**. Fixed to a row-constant `j ∈ [0, T-2]` with
   `valid = t > j`; the no-memory floor is now 0.165 (chance 0.125 + repeated symbols).
9. **The SSP arm read the index task at the wrong place — the bug that mattered most.** For index-addressed recall
   the read must be at the *absolute* position `j`; the code read at `t - j` (the delay convention), i.e. it
   answered "the item j steps ago" while being scored on "the item at index j". It sat at 0.184 with the
   pinned-index readout perfect. Fixed; the arm went from 0.184 to **0.996** in-distribution and 0.841 on a 4×
   longer stream. Two runs of the `index` variant were discarded because of this.
10. **`--workers` with `maxtasksperchild=1` re-imports modules per task**, so a mid-run code edit changed task
    behaviour between one variant's search and final phases. The affected rows were identified and re-run; the
    lesson is to edit task code only between sweep runs.
11. **The learning-rate search range was too low**: the gate's working rate (1e-2) sat at the very top of
    log-uniform(-4,-2); widened to log-uniform(-3.5,-1.5), after which the SSP arm's search found 0.940.
12. **The ablation reference config had no `lr`** (KeyError), and `analysis.py` crashed when a file held only the
    search phase (and its extrapolation guard used dict-`in` rather than a key-prefix test).
13. **The reported metric mixed speed and convergence.** `last5_*` is the mean of the last 5 of 6 checkpoints (steps
    300-1500), so it is a short learning-curve area. It made the GRU look 0.948 on `index` when it converges to 0.999,
    and made SSP look better than the GRU on `pointer` in-distribution when the GRU is ahead at step 1500. Found
    during review; converged scores added in 4.5.
14. **`analysis.py` verdict logic was too generous.** Its margin test used the same `last5` metric, and its
    "hand-wiring" fragility check only compared ablations on the in-distribution split, so it could not detect that
    the winning arms are hand-wired. Do not reuse the verdicts until it is fixed.

## 8. Limitations

- **5 hold-out seeds**, margin tests are differences of means against pooled standard errors, and the raw numbers are
  reported so the thresholds can be disagreed with.
- **A supervised budget of 1500 updates** is small, and the transformers suffer most: the full-context transformer
  reaches only 0.493 on `delay`. Those are budget statements, not statements about transformers — so the "upper
  bound" role of `tf_full` is weak here, and the `delay`/`pointer`/`path` verdicts should be read with that in mind.
  (For `index` it does not matter: the SSP arm is at 0.999 while `tf_full` is at 0.348.)
- **The ablation references are fixed configs at lr 5e-3**, not each variant's searched winner, so the reference
  values in section 5 (e.g. `delay` 0.608) are lower than the swept numbers (0.933). Comparisons *within* section 5
  are like-for-like; comparisons to section 4 are not.
- **Variant B's iteration is hand-wired** from the task-visible hop count, and **variant A is a supervised proxy**
  (no policy, translations only, SO(2) deferred).
- **Timings are contended**: measurement runs shared the machine with the reference PPO jobs, so the compute
  comparison in section 5 is directional.
- **`path`'s in-distribution split is weakly discriminative** (floor 0.024 vs best 0.013), so its verdict rests
  mostly on the extrapolation axes and on the task redesign in section 6.
- **Everything here is rung 0 (hand-wired).** See section 0 and handoff 09 for the exact list of what was
  hand-wired. These results cannot say whether a learner can *discover* the structure.
- **Extrapolation on `index` is confounded by input encoding** (baselines get `j/T`, SSP multiplies by `T`).
- **Only one state budget (256 floats)** and one supervised budget (1500 updates); no sample-efficiency regime with a
  fixed small dataset was run; evaluation is every 300 updates, too coarse to resolve early learning.

## 9. Do not do

- Do not cite the `delay` numbers as evidence for the algebra: SSP only ties the GRU/LRU at convergence and learns
  more slowly, and that variant was pre-registered as substitutable.
- Do not cite the `pointer` numbers as "solves pointer chasing": every arm is near chance in absolute terms.
- Do not cite the fix-9 numbers from the earlier `index` runs (0.184/0.597): they came from a mis-addressed read and
  a leaked task, both recorded above.
- Do not compare these numbers with POPGym's published table (different loop, different budget, different machine).
- Do not use the `path` ablations to argue the algebra is inert until the proxy separates the memory arms from the
  floor.
- Do not quote `last5` numbers as "accuracy"; quote converged (step 1500) scores from 4.5 and report speed separately.
- Do not cite `proceed_to_rl` for `index` or `pointer`: withdrawn (section 0, 6).
- Do not read the `index` extrapolation result as "SSP extrapolates, baselines cannot" until baselines are given a
  fair absolute-index encoding.
- Do not treat `tf_full` as an upper bound in this run (it is undertrained: 0.37 on `index`).

---

## 10. Amendment (appended 2026-10-11, after handoff 09 A1/A2) — the rung-0 pointer arm was broken

Section 4.3 and the `pointer` row of 4.5 **must not be cited**. The hand-wired `SSPPointerArm` could not compute
`f^h(s)` even with exact codes: its oracle test (`test_fixes.py::test_pointer_oracle_readout`, pass-through readout,
`decay=False`, `dim=256`, 512 floats) was at chance for every hop (h1 0.15 … h8 0.25 against chance 0.125), and a
diagnostic showed its answer tracked the *neighbours* of the target (f^(h-1) 0.77 and f^(h+1) 0.39 at h=1) rather
than f^h (0.14). So every SSP-vs-baseline pointer comparison in 4.3 and 4.5 (0.270 vs 0.145 early, 0.240 vs 0.151
held-out, GRU ahead 0.359 vs 0.312 at convergence) is a property of a broken arm plus a learned readout. The
*baseline* arms' own pointer numbers are untouched by this and remain as reported; no arm was competent.

Two causes, both now fixed in `arms.py::SSPPointerArm`:

1. **Commutative binding.** A stored pair was `code(key) * code(value)`, which is symmetric, so unbinding a code
   from the bundle returned its successor and predecessor with equal strength and every hop was
   direction-ambiguous. Fix: bind asymmetrically as `code(key) * P(code(value))` with a fixed random permutation
   `P` (a seeded `torch.randperm` buffer) and unbind with `P^{-1}`. **Correction to handoff 09 section 0.2
   hypothesis 1:** a fixed random *phasor* `R` does **not** work, because the predecessor term `conj(R)*conj(R)`
   inverts to `R^{1-2h}` and for every hop the retrieved value is `conj(R^{2h})*v` for *both* the successor and the
   predecessor — the ambiguity survives. A non-multiplicative (permutation-like) operator is required.
2. **One extra unbind after the hop loop.** The loop ended with `c = f^h(s)` and then unbound once more, scoring
   `f^(h+1)(s)`. Fix: carry the cleanup scores of the last *executed* hop out of the loop and feed those to the
   readout (queries with h=0 score the start code directly).

Verification after the fix (same commit, local, `OMP_NUM_THREADS=1`):

- `test_pointer_oracle_readout`: h1..h8 all **1.00** (>= 0.99 required).
- Diagnostic (128 episodes per hop, `n_nodes=8`): the f^h column is **1.00** at h=1..6; f^(h-1) and f^(h+1) are at
  chance (0.10-0.15); f^(h+2) reads 0.21-0.29, which is a permutation-cycle artefact, not arm error (for a random
  permutation on 8 nodes, `f^(h+2)(s) = f^h(s)` exactly when `s` lies in a 1- or 2-cycle, probability ~0.25).
- A short in-distribution training run (`n_nodes=6`, `dim=256`, 512 floats, seed 1000, 1500 updates) now reaches
  **1.000** on `id` *and* **1.000** on held-out hops 4-8, at the first checkpoint (100 updates / 51,200 tokens).
  This is a large change from the broken arm's 0.15 and it is an upper-bound/oracle-corrected result for a
  *rung-0* arm, so it is still a structural-prior number, not evidence that a learner discovers the algebra.

Consequences: pointer is **re-opened** for rung 0 re-run (handoff 09 step D) and, if competence holds at a longer
budget with `n_nodes=6`, for the learned rungs (step F). Nothing in sections 1-9 above is otherwise changed.
