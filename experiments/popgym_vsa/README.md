# POPGym x VSA memory

Stage-1 harness from `docs/10-vsa-memory-and-composition-direction.md` sections 13.1 and 13.4: put the VSA
bundle/unbind memory next to the field's own baselines on POPGym's memory-diagnostic tasks.

The encoder thread (continuous SSP) is **not** what this folder tests. POPGym's target observations are discrete
symbols, so what is under test is the algebra -- bundle as storage, unbind as retrieval -- over random or
block-orthogonal item codes. That is deliberately the same object RUN_005 measured in T4/T6/T7, now with a learning
loop attached.

## Layout

```
tasks.py      single source of truth: env construction, space metadata, scripted oracles
memory.py     FHRR item memory (random and block codebooks), count and delay readouts
models.py     mlp / posmlp / gru / vsa_count / vsa_delay behind one recurrent interface
vec_env.py    N envs in lockstep (these tasks have fixed-length episodes, so no truncation logic)
headroom.py   random / oracle / best-feedforward returns (is the task solvable without memory?)
selftest.py   the VSA readouts run *as policies* vs the oracle; codebook and dimension sweeps
train.py      small advantage actor-critic with truncated BPTT (no RLlib)
```

## Reproduce

```bash
cd experiments/popgym_vsa
../../.venv/bin/python headroom.py                    #  8 s
../../.venv/bin/python selftest.py                    # 17 s
../../.venv/bin/python train.py --fam CountRecall --diff Easy --model vsa_count --code block --updates 150
```

`popgym` is installed into the repo `.venv` with `--no-deps` (it needs only numpy and gymnasium, both present).
**Do not** install its `baselines` extra there: `ray[rllib]` drags `gymnasium` down to 1.2.2 and pins the deprecated
old-style model API (RUN_005 section 6 records ray 2.51.2/torch 2.8.0/155 packages; on Python 3.14 it resolves to
ray 2.59.0/torch 2.14.1/60 packages). The training loop here is therefore our own.

## Headroom (`headroom.json`)

100 eval episodes, feedforward ceiling fitted on 800 episodes.

| env | random | FF(step, obs) | FF(obs) | oracle |
|---|---|---|---|---|
| RepeatPreviousEasy | -0.513 | -0.511 | -0.498 | +1.000 |
| RepeatPreviousMedium | -0.505 | -0.498 | -0.490 | +1.000 |
| RepeatPreviousHard | -0.496 | -0.502 | -0.494 | +1.000 |
| CountRecallEasy | -0.931 | -0.436 | -0.921 | +1.000 |
| CountRecallMedium | -0.922 | -0.523 | -0.924 | +1.000 |
| CountRecallHard | -0.886 | -0.443 | -0.884 | +1.000 |

`headroom.py` is now bit-identical across repeated runs (verified over three runs): the random arm previously drew
from the *space's* RNG, which `env.reset(seed=...)` does not pin, and wobbled by ~0.02.

* **RepeatPrevious has no feedforward shortcut.** Step-index-and-observation does no better than random, so this is a
  clean memory test and POPGym's own PPO-saturation warning does not apply.
* **CountRecall has a partial shortcut** (the step index is worth roughly half the gap from random), so a positional
  MLP is a mandatory baseline here, exactly as POPGym's authors found for masked stateless CartPole.
* The feedforward ceiling is a tabular majority vote, so it *lower-bounds* what a trained MLP reaches.

## Sufficiency (`selftest.json`)

VSA readouts plugged straight into an argmax policy, 50 episodes, `dim=256`, versus the oracle:

| code | task | vsa policy | agreement | oracle |
|---|---|---|---|---|
| random | RepeatPreviousEasy/Medium/Hard | +0.982 / +0.856 / +0.705 | .991 / .928 / .852 | +1.000 |
| block | RepeatPreviousEasy/Medium/Hard | +0.981 / +0.878 / +0.713 | .990 / .939 / .856 | +1.000 |
| **random** | CountRecallEasy/Medium/Hard | **+0.016 / +0.915 / -0.261** | **.508 / .957 / .369** | +1.000 |
| **block** | CountRecallEasy/Medium/Hard | **+1.000 / +1.000 / +1.000** | **1.000 / 1.000 / 1.000** | +1.000 |

Two things this establishes, both of which change the plan:

1. **The delay readout needs lag `k-1`, not `k`.** POPGym's `k` counts the card from k back *including the current
   one*, so the observation-index lag is `k-1`. Measured hit rate for k=4: **0.247 at lag=k (chance), 0.991 at
   lag=k-1**. Recorded in `tasks.spec_from_env` so it cannot be rediscovered as a result.
2. **The count readout is only as exact as the codebook.** Random phasors overlap with `rho_ij ~ N(0, 1/(2D))`, so
   the estimate carries error `sum_{i != q} counts_i * rho_iq` -- it grows with the *total* number of items ever
   bundled, not with the queried count. At `dim=256` that is 0.6 for CountRecallEasy (counts ~26) and it fails; at
   4096 it is 0.108 and exactness is 1.000. A block-orthogonal codebook has `rho = 0` exactly, so **the bundle is
   the count vector** and all three difficulties are solved perfectly by the raw readout.

Exactness of the count readout vs dimension (13 items, 30 streamed values, env-free):

| D | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|
| random, exact | 0.480 | 0.677 | 0.991 | 1.000 |
| random, mean abs err | 0.583 | 0.395 | 0.158 | 0.108 |
| block, exact | 1.000 | 1.000 | 1.000 | 1.000 |

So the honest statement is not "the algebra solves CountRecall" but "the algebra solves CountRecall **given a
codebook whose cross-talk is below the rounding threshold**, and plain random phasors need `D >= ~4000` for these
count magnitudes". Whether the codebook may be chosen that way is a modelling decision, not an algebraic fact.

## First training pass (490k env steps, 600 updates, one seed)

Recorded in full in `runs/RUN_006_popgym_first_pass.md`. Greedy evaluation, 20 episodes, oracle is +1.000:

| task | arm | greedy eval |
|---|---|---|
| RepeatPreviousEasy | **vsa_delay** | **+0.990±0.029** |
| RepeatPreviousEasy | gru | -0.108±0.094 |
| RepeatPreviousEasy | posmlp | -0.481±0.128 |
| CountRecallEasy | gru / mlp / posmlp | -0.910 / -0.918 / -0.729 |
| CountRecallEasy | vsa_count (block) | -0.888 |
| CountRecallEasy | vsa_count (block, cleanup) | -0.835 |

The delay memory solves its task inside the budget; **nothing learns CountRecall**, so that half of the comparison
is between two failing policies and carries no evidence either way. The decode-shape hypothesis for CountRecall was
tested and refuted (adding the nearest-neighbour decode made it slightly worse), which localises the problem to RL
discovery rather than the representation.

## Stop rule

If a fixed-memory VSA arm does not beat the positional MLP on CountRecallEasy at matched budget, drop the
direction. If it does not match the GRU on RepeatPreviousEasy, treat that as a capacity or design issue rather than
a kill. Both thresholds come from `headroom.py`, not from POPGym's selection-inflated MMER.

**Prerequisite, learned the hard way in the first pass:** the rule only discriminates once *some* arm actually
solves the task. Both arms sitting near random is a shared failure (budget or optimisation), not a verdict on the
representation, so "make one arm solve CountRecallEasy" comes before re-applying the rule.

## Cost

Encoder-only work (headroom, selftest) is seconds to tens of seconds on CPU. Training runs are CPU-only:
`--updates 150` at `--envs 16` is roughly 120k env steps and finishes in about a minute per run at 16 cores.
POPGym's own claim is under two hours per baseline on one consumer GPU; that is the scale for a full comparison.
