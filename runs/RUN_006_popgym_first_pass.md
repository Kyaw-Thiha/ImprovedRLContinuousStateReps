# RUN_006 — POPGym first pass: headroom, readout sufficiency, and a 490k-step training pass

**Status: COMPLETE (2026-10-10). CPU only, no GPU.** Everything below is MEASURED in this session on the local
machine (RTX 3050 Laptop 4 GB unused, 16 cores, repo `.venv` Python 3.14.6, numpy 2.5.3, torch 2.14.0+cu130,
`popgym` 1.0.7 installed with `--no-deps`). Nothing is committed. The new code is one folder:
`experiments/popgym_vsa/` (see its `README.md`).

This run executes Stage 1 of `docs/10-vsa-memory-and-composition-direction.md` section 13.4 — POPGym, the cheap
harness for the "content-addressable recall from a cue" rank-1 problem — and answers the three questions that had to
be settled before any learning curve could be interpreted: *is the task solvable without memory*, *can the readout
answer the task at all*, and *does the answer survive contact with a learning loop*.

**Reproduce:**

```bash
cd experiments/popgym_vsa
../../.venv/bin/python headroom.py     #  8 s
../../.venv/bin/python selftest.py     # 17 s
OMP_NUM_THREADS=1 ../../.venv/bin/python train.py --fam RepeatPrevious --diff Easy --model vsa_delay --updates 600
OMP_NUM_THREADS=1 ../../.venv/bin/python train.py --fam CountRecall --diff Easy --model vsa_count --code block --cleanup --updates 600
```

---

## 1. Environment and harness facts (measured, because they set the plan)

- `popgym` 1.0.7 imports on Python 3.14 with `numpy` and `gymnasium` only; `torch` and `ray` are **not** core deps.
- Its RLlib baselines pin `ray[rllib]`, which downgrades `gymnasium` 1.4.0 -> 1.2.2 and uses the deprecated
  old-style model API. The training loop here is therefore our own (~140 lines, A2C with truncated BPTT).
- Observation spaces are **discrete** (suits, colours, card ids), so this folder tests the *algebra* — bundle as
  storage, unbind as retrieval — not the continuous SSP encoder. That is deliberately the same object as RUN_005
  T4/T6/T7, now with a learning loop attached.
- Throughput: 85k-290k raw env steps/s on CPU. Episodes are fixed length: RepeatPrevious 51/103/155 and CountRecall
  51/103/207 steps for Easy/Medium/Hard.
- `OMP_NUM_THREADS=1` is mandatory in practice: 10 unpinned runs (16 threads each) oversubscribed the machine and
  ran ~20x slower; pinned runs complete 600 updates in 50-80 s.

## 2. Headroom — RepeatPrevious has no shortcut, CountRecall partially does

100 eval episodes; the feedforward ceiling is a tabular majority vote of the oracle action on (step index, obs),
fitted on 800 episodes, so it lower-bounds a trained positional MLP.

| env | random | FF(step, obs) | FF(obs) | oracle |
|---|---|---|---|---|
| RepeatPreviousEasy | -0.513±0.121 | -0.511 | -0.498 | +1.000 |
| RepeatPreviousMedium | -0.505±0.093 | -0.498 | -0.490 | +1.000 |
| RepeatPreviousHard | -0.496±0.093 | -0.502 | -0.494 | +1.000 |
| CountRecallEasy | -0.931±0.054 | **-0.436** | -0.921 | +1.000 |
| CountRecallMedium | -0.922±0.038 | **-0.523** | -0.924 | +1.000 |
| CountRecallHard | -0.886±0.037 | **-0.443** | -0.884 | +1.000 |

These are the final values and they are bit-identical across repeated runs of `headroom.py` (verified over three
runs) after the seeding fix in section 5.4; before it the random arm wobbled by ~0.02, which is the same order as
the effects this harness is meant to detect, so it had to be closed.

RepeatPrevious is a clean memory test (a feedforward policy with a clock does no better than random). CountRecall
has a partial shortcut worth about half the gap from random, so the positional MLP is a mandatory baseline there —
the same warning POPGym's authors record for masked stateless CartPole.

## 3. Readout sufficiency — the delay needs lag k-1, and the count needs the right codebook

VSA readouts used directly as an argmax policy, 50 episodes, `dim=256`:

| code | task | vsa policy | agreement | oracle |
|---|---|---|---|---|
| random | RepeatPrevious Easy/Medium/Hard | +0.982 / +0.856 / +0.705 | .991 / .928 / .852 | +1.000 |
| block | RepeatPrevious Easy/Medium/Hard | +0.981 / +0.878 / +0.713 | .990 / .939 / .856 | +1.000 |
| random | CountRecall Easy/Medium/Hard | **+0.016 / +0.915 / -0.261** | .508 / .957 / .369 | +1.000 |
| block | CountRecall Easy/Medium/Hard | **+1.000 / +1.000 / +1.000** | 1.000 / 1.000 / 1.000 | +1.000 |

**Finding 1 (a bug, now fixed and documented in `tasks.spec_from_env`):** POPGym's RepeatPrevious `k` counts the
card from k back *including the current one*, so the observation-index lag is `k-1`. Measured hit rate for k=4:
**0.247 at lag=k (chance), 0.991 at lag=k-1**. Before the fix the whole delay arm looked like a refutation of the
memory it was meant to test.

**Finding 2 (a real limitation of the representation):** the count readout's noise is the *code cross-talk*,
`sum_{i != q} counts_i * rho_iq` with `rho_ij ~ N(0, 1/(2D))`. It grows with the total number of items ever
bundled, not with the queried count, so it is a property of the codebook rather than of the count. Measured
exactness of the rounding (13 items, 30 streamed values, env-free):

| D | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| random, exact | 0.480 | 0.677 | 0.991 | 1.000 |
| random, mean abs err | 0.583 | 0.395 | 0.158 | 0.108 |
| block, exact | 1.000 | 1.000 | 1.000 | 1.000 |

At `dim=256` the error is 0.6 for CountRecallEasy (counts ~26) — above the 0.5 rounding threshold — which is why
random codes fail Easy and Hard while passing Medium (the four Medium code overlaps happen to cancel: mean |rho|
0.0264 against the 0.044 theoretical). A **block-orthogonal codebook** has `rho = 0` exactly and solves all three
difficulties with no learning at all. The honest claim is therefore *"the algebra solves CountRecall given a
codebook whose cross-talk is below the rounding threshold"*, not *"the algebra solves CountRecall"* — and whether
choosing the codebook that way is legitimate is a modelling decision, not an algebraic fact.

## 4. Training pass — one arm solves its task, and CountRecall is not learned by anything

600 updates x 16 envs x 51 steps ≈ **490k env steps**, `dim=256`, single seed, threads pinned. Score is the mean of
the last 10 update-returns, plus a 20-episode greedy evaluation.

**RepeatPreviousEasy** (oracle +1.000):

| arm | train last-10 | greedy eval |
|---|---|---|
| **vsa_delay** | **+0.889** | **+0.990±0.029** |
| gru | -0.104 | -0.108±0.094 |
| posmlp | -0.479 | -0.481±0.128 |

The VSA delay memory solves the task within the budget while the GRU (POPGym's strongest reported architecture) is
nowhere near it. That is the positive result of this run.

A second, earlier batch (150 updates, about 122k env steps, same code but launched unpinned and so ~2,400 s per run)
gives a learning-curve point for the same arms on RepeatPreviousEasy: **vsa_delay greedy eval +0.575±0.086**
(train last-10 -0.341), against mlp -0.458 and posmlp -0.477, i.e. the delay memory is already above the
feedforward arms at a quarter of the budget and reaches +0.990 by 600 updates. The same batch's CountRecall runs
(mlp -0.933, posmlp -0.927, vsa_count/block -0.918 Easy and -0.780 Hard) are all at or near chance, consistent with
section 4. Its GRU runs crashed on the state-mask bug in section 5.2 and are superseded by the pinned reruns; the
files `results/{CountRecall,RepeatPrevious}{Easy,Hard}_*_s0.json` are from that batch. One seed each.

**CountRecallEasy** (oracle +1.000):

| arm | 600 updates | 2000 updates |
|---|---|---|
| gru | -0.910 | -0.771 |
| mlp | -0.918 | — |
| posmlp | -0.729 | — |
| vsa_count (block) | -0.888 | -0.718 |
| vsa_count (block, cleanup decode) | -0.835 | — |
| random policy | -0.928 | — |

**Nothing learns CountRecall.** The VSA arm does not beat the positional MLP, and the positional MLP does not beat
random by much, so at this budget the task is not being solved by *any* arm — it is not a discriminating
comparison, it is a shared failure.

**The decode hypothesis was tested and refuted.** Given that the block-code readout *is* the oracle action at every
step (agreement 1.000, section 3), the obvious suspect was the decode: one scalar estimate has to be turned into
one of 27 counts, a rounding staircase. Adding the standard VSA nearest-neighbour decode
(`exp(-(est-k)^2/2)` over candidate counts, verified to widen the feature from 5 to 32 dims) moved the result the
wrong way: -0.888 -> **-0.835**. So the bottleneck is not the shape of the decode. Combined with 2000 updates
moving every arm only to about -0.72, the evidence points at RL *discovery* (budget and/or optimisation), not at
the representation — but that is an inference from a failure, not a measurement of one.

## 5. Bugs found and fixed (recorded so they are not rediscovered)

1. **`lag = k-1`, not `k`,** for RepeatPrevious (§3, Finding 1).
2. **The GRU state mask broadcast wrongly.** `reset_state` used a `(B,1,1)` keep-mask while the GRU keeps its state
   as `(1,B,H)`, silently producing `(B,B,H)` and crashing with `Expected hidden size (1, 16, 64), got [16, 16, 64]`.
   Fixed per model; verified that all five arms preserve state shape and zero the terminated row.
3. **Count cross-talk is not negligible at `dim=256`** for large counts (§3, Finding 2). Not a code bug, but it
   invalidates "the VSA memory is exact for counting" as a claim.
4. **`env.action_space.sample()` is not reproducible across runs** even with `env.reset(seed=...)`, because the
   *space* carries its own RNG. This made the random and feedforward baselines wobble by ~0.02 — the same order as
   the effects the harness exists to detect. Fixed by giving both arms their own seeded `default_rng`;
   `headroom.py` is now bit-identical across three consecutive runs (oracle was always exactly +1.000, which is
   what made the wobble attributable to the sampler rather than to the environments).

## 6. Limitations

1. **One seed, one budget.** Section 4 is a first pass, not a result. No error bars on any training number.
2. **The loop is our own A2C, not POPGym's PPO.** Its hyperparameters (lr 3e-4, gamma 0.99, lambda 0.95, entropy
   0.01) were not tuned, and no published POPGym number is directly comparable to it. The GRU number here is a
   GRU trained by *this* loop, not POPGym's reported GRU.
3. **CountRecall is unsolved by every arm at these budgets**, so the stop rule in `experiments/popgym_vsa/README.md`
   fires only vacuously (the VSA arm does not beat the positional MLP, but neither arm is near the oracle). The rule
   cannot discriminate until some arm actually solves the task; treat "some arm solves CountRecall first" as the
   prerequisite, not as a result.
4. **RepeatPrevious self-test returns are not oracle-perfect** (0.705 at Hard) because k=64 holds ~64 live positions
   against `dim=256` — near RUN_005 T4's `m ~ Dc/4` capacity rule. Raising `dim` is the untested fix.

## 7. Next steps, in cost order

1. **Get one arm to solve CountRecall Easy, then re-run the stop rule.** Cheapest levers, in order: raise the
   budget (2000 updates only reached -0.72), `gamma=0` (the task is a memory-then-answer bandit, so bootstrapping
   over 51 steps may be pure variance), drop the entropy bonus, raise `lr`, more envs, and check the *reference*
   number by running POPGym's own PPO on that env once. Without this, section 4's CountRecall half says nothing.
2. **Prediction to test if (1) lands:** with the same loop, a memoryless MLP should fail while *both* the GRU and
   `vsa_count` solve it; the interesting quantity is then steps-to-competence, not final return.
3. **One seed -> 5 seeds, and report mean-of-last-k** (already the metric) rather than MMER.
4. **Raise `dim` for RepeatPrevious Hard** (256 -> 1024) to test the capacity explanation in §6.4.
5. **Only then MIKASA** (`RememberColor`/`ShellGame` in RGB) — docs/10 section 14.4 predicts those are a *weak*
   structural fit for SSP, and section 14.1's "cheap joints mode" correction means they cost a vision stack.

## 8. Do not do

- Do not cite the CountRecall numbers in section 4 as evidence against VSA memory. No arm solves the task; the
  comparison is between two failing policies.
- Do not cite the `vsa_count` random-code failure as a property of the algebra. It is a property of the *codebook's*
  cross-talk at `dim=256` (§3) and is exactly repaired by block codes.
- Do not compare these training numbers against POPGym's published table. Different loop, different budget.
- Do not run training without `OMP_NUM_THREADS=1` when launching more than one process.
