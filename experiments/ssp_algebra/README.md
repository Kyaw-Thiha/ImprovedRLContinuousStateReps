# experiments/ssp_algebra — SSP algebra controller: three variants, supervised probing

Implements the plan in `docs/11-ssp-algebra-controller-variants-and-tests.md` (see also
`docs/handoffs/08-ssp-algebra-controller-TEMP.md`). Tests whether giving a learner an SSP/VSA memory with algebraic
operations (write, shift, read/unbind, cleanup) lets it learn faster, extrapolate, or work under a fixed state
budget, against GRU, S4D/LRU, transformers and a learned-delta kernel-attention control.

**Status (2026-10-11):** everything here is the *hand-wired* ("rung 0") version of the SSP arms: the write/read
addresses, key/value roles, hop loop and pose integration are set in code, and only a small readout is learned. It is an
upper bound, not a test of whether a learner can discover the structure. Known issues (the `last5` metric blends speed
and convergence, `index` extrapolation is confounded by input encoding, `tf_full` is undertrained, `hash()`-seeded
config draws are not stable across processes) and the plan to fix them are in
`docs/handoffs/09-ssp-learned-structure-sample-efficiency.md`. Results and corrected conclusions:
`runs/RUN_007_ssp_algebra_controller.md` (read its section 0 first).

Nothing here imports the repo's `rl/` package: the folder is self-contained (torch + numpy, plus `popgym` only for
the step-1 reference task and `experiments/popgym_vsa/tasks.py` for the real env's metadata).

## Files

| File | What it is |
|---|---|
| `ssp_ops.py` | FHRR/SSP primitives: `phi`, `pos` (time as SSP), bind/unbind/shift, leaky bundle, cleanup |
| `selftest.py` | the gate: T0 algebra parity, and a hand-wired SSP-time delay readout on POPGym `RepeatPrevious` |
| `tasks_c.py` | variant C: learned-delta read (per-episode lag given as input) and index-addressed recall |
| `tasks_b.py` | variant B: pointer chasing over bound pairs, with a per-query hop count |
| `tasks_a.py` | variant A: 2-D landmark return (path integration), supervised proxy for the simulator |
| `arms.py` | the eight arms + state-float accounting + budget-matched construction |
| `harness.py` | teacher-forced training, checkpointed eval on in-distribution and extrapolation splits, random search |
| `test_harness.py` | the harness gate (see below) |
| `run_sweep.py` | parallel, resumable sweep and ablation runner |
| `analysis.py` | the docs/11 section 4.5 decision rules, applied mechanically |

## Protocol (fixed in code, so it cannot drift)

- **State size matched in floats**, only floats carried *between steps* count:
  `gru: H`; `lru: 2N`; `ssp: 2D`; `tf_window`/`kernel_attn: 2 * layers * d_model * window`; `mlp`/`posmlp: 0`.
  State-size knobs are excluded from the random search — tuning them would break the matching the claim rests on.
- **Every arm sees the same inputs**: `onehot(obs) || normalised query || t/T`, plus rotary relative position for the
  transformers. The windowed arms are therefore handicapped *only* by the state budget, which is the claim.
- **Equal tuning budget per arm**: the same number of random configs, scored on training-side seeds only.
- **Hold-out seeds only** for reported numbers, mean of the last `k` checkpoints, never max-over-epochs.
- **Extrapolation** is always a held-out regime: longer lag (C), more hops (B), longer path / larger steps (A).

## Running

```bash
OMP_NUM_THREADS=1 ../../.venv/bin/python selftest.py          # primitives gate
OMP_NUM_THREADS=1 ../../.venv/bin/python test_harness.py      # harness gate
OMP_NUM_THREADS=1 ../../.venv/bin/python run_sweep.py --mode sweep --tag main
OMP_NUM_THREADS=1 ../../.venv/bin/python run_sweep.py --mode ablation --tag abl
../../.venv/bin/python analysis.py --tag main
```

`OMP_NUM_THREADS=1` is mandatory (16 unpinned workers oversubscribe the machine ~20x slower; the same lesson as
`experiments/popgym_vsa`).

## Known limits (stated, not discovered later)

- **Variant B's iteration count is hand-wired** from the task-visible hop count, as docs/11 section 3.2 specifies.
  A win there is a structural-prior result, and `analysis.py` downgrades the verdict if the ablations show the
  hand-wired parts are load-bearing.
- **Variant A is a supervised proxy**, not the 2-D simulator with a policy. SO(2) heading is deferred.
- 5 hold-out seeds: the margin test is a difference of means against a pooled standard error, reported with the raw
  numbers so the thresholds can be disagreed with.
