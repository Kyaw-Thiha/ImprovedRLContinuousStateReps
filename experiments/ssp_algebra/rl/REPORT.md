# POPGym-native RL pilot: SSP memory vs POPGym's GRU and fast-weight models on RepeatPreviousEasy

Everything here runs POPGym's own PPO config (`ppo.py`'s config dict, ported to ray 2.59 by `ref_ppo/run_ref.py`), its own env
(`popgym-RepeatPreviousEasy-v0` with POPGym's `PreviousAction` + `Antialias` wrappers) and its own `ray_models`
(`GRU`, `FastWeightProgrammer`). The SSP arm is `rl/ssp_model.py`, an RLlib `ModelV2` subclassing POPGym's `BaseModel`. The in-repo A2C loop is not used
or compared against. Reference venv: `/tmp/popgym_ref` (python 3.14.6, ray 2.59.0, torch 2.14.1+cpu, popgym 1.0.7, gymnasium 1.2.2), rebuilt from
`ref_ppo/requirements-frozen.txt` exactly as `ref_ppo/README.md` says. **Pods used: 0** (all runs local, 1 CPU thread per run).

## Verdict

- **SSP did not learn RepeatPreviousEasy within 3M steps under its one documented default; the GRU solved it in under 1M.**
  GRU: final return +0.993 ± 0.005 (3 seeds, last 10% of 1M), 50%-threshold reached at 361k steps (median).
  SSP (aux off): -0.057 ± 0.113 after 3M steps (last 10% of 3M); it never reached the 50% threshold (+0.244) in any seed.
  The SSP arm's learned read offset (the address-head bias) stayed at 0 (|bias| <= 0.011) in all 6 SSP runs (final 3M checkpoints): the model never learned to read 4 steps back.
  This is the supervised study's *dead zone* (an offset >= 1 never starts moving; handoff 11 section 1) reappearing in RL. It is the finding; it was not tuned away.
- **Plumbing is not the cause.** A diagnostic arm that is identical except the address bias is *hand-set* to the task's lag (-4) learns fast
  (see "Diagnostic"), the memory-zeroed ablation shows the memory carries all of the return, and the vectorised core equals the stepwise `rl_memory.SSPMemory`
  (12 contract tests pass). The hand-set arm uses the task's own constant, so it is **not a claim**, only a check that the wrapper can learn.
- **Against the success rule (handoff 11 decision 2):** parity fails (-0.057 vs +0.993 for the best baseline). No listed advantage metric is won:
  sample efficiency (SSP never reaches the threshold; GRU does at 361k), state size (equal, 256 floats for every arm), per-step cost (SSP
  1.25 ms vs GRU 0.64 ms per rollout step at B=8, see below; SSP *training* throughput is higher than GRU's but lower than the FWP's). Length extrapolation
  and robustness were not run (RepeatPreviousMedium/Hard were out of the pilot's scope). The POPGym FWP also failed to learn (-0.262 ± 0.048 at 2M),
  so on this task SSP is better than POPGym's FWP and worse than its GRU.

## Setup (identical for every arm; only `custom_model` differs)

PPO via `PPOConfig().update_from_dict(popgym config)` with the old API stack (POPGym's models are `ModelV2`): `train_batch_size` 32 768 steps/iteration
(= `bptt_cutoff 1024 x 1 worker x 32 envs`), `rollout_fragment_length` 1024, `batch_mode complete_episodes`, `max_seq_len` 1024, `minibatch_size` 8192
(= `8 x bptt_cutoff`, 4 minibatches), `num_epochs` 30, `gamma` 0.99, `vf_loss_coeff` 1.0, lr 5e-5, clip 0.3, entropy 0, `kl_coeff` 0.2 (RLlib defaults for
the rest). Preprocessor / actor / critic heads, `hidden_size` 256 and the 128-wide `feature_map` are POPGym's own (`base_model.BaseModel`). Arms:

| arm | model | state floats | total params |
|---|---|---|---|
| `gru` | POPGym `GRU` | 256 | 413 830 |
| `fwp` | POPGym `FastWeightProgrammer` (plain linear attention, **no delta rule**, see SURVEY.md) | 256 (16x16) | 261 814 |
| `ssp` | `rl/ssp_model.py::SSPModel`, aux off | 256 (128 complex) | 155 618 |
| `ssp_aux` | same, label-free recall aux on (`aux_coeff` 0.1) | 256 | 155 618 |
| `ssp_oracle4` (diagnostic only) | `ssp` with the address bias hand-set to -4 | 256 | 155 618 |

Env: random return -0.512, optimal +1.000 (`results/rl_env_returns.json`, 300 episodes each). Thresholds: 50% = random + 0.5 (optimal - random) = **+0.244**;
90% = **+0.849**. Seeds 0, 1, 2 for every arm (RLlib `seed` and, for SSP, the codebook/frequency seed). The reported return is RLlib's `episode_reward_mean`
(mean over the last 100 episodes; POPGym's `ppo.py` has no separate evaluation, its reported number is this training return, the MMER of its paper).
Curves are in `results/rl_pilot_RepeatPreviousEasy_<arm>_s<seed>.jsonl` (one row per PPO iteration, written incrementally; `results/rl_pilot_RepeatPreviousEasy_summary.json`
is the aggregate; `python rl/report.py --horizon 1000000 --horizon 2000000 --horizon 3000000` regenerates the tables).

## Results

`final` = mean over the last 10% of the first H steps (rows with 0.9 H < steps <= H), then mean ± sample std (ddof 1) over 3 seeds. `AULC` = area under the
return-vs-steps curve over [0, H] divided by H (the mean return along the curve; random = -0.51, a perfect-from-step-0 curve would be 1.0). `steps to`
= first iteration whose return reaches the threshold, "k/3 seeds (median)". Wall time is the mean per run **while 6-7 runs shared the machine** (contended;
solo numbers are in the next section).

**H = 1M steps (the contracted pilot; all 4 arms, 3 seeds)**

| arm | seeds | final return (last 10% of 1M) | steps to +0.244 (50%) | steps to +0.849 (90%) | AULC | wall to 1M (h, contended) |
|---|---|---|---|---|---|---|
| gru | 3 | **+0.993 ± 0.005** | **3/3 (median 361k)** | **3/3 (median 557k)** | **+0.436 ± 0.024** | 1.05 |
| fwp | 3 | -0.396 ± 0.073 | 0/3 | 0/3 | -0.455 ± 0.046 | 0.67 |
| ssp | 3 | -0.163 ± 0.117 | 0/3 | 0/3 | -0.338 ± 0.061 | 0.94 |
| ssp_aux | 3 | -0.268 ± 0.107 | 0/3 | 0/3 | -0.406 ± 0.049 | 0.92 |

Per seed (final, 1M): gru 0.992 / 0.989 / 0.999; fwp -0.317 / -0.460 / -0.411; ssp -0.052 / -0.286 / -0.150; ssp_aux -0.167 / -0.381 / -0.258.
Steps to +0.244 per GRU seed: 361k / 394k / 361k.

**Extension (same runs, resumed from their RLlib checkpoints; the GRU is converged and was not extended).** `fwp` to 2M, `ssp` and `ssp_aux` to 3M:

| arm | H | seeds | final (last 10% of H) | steps to +0.244 | AULC |
|---|---|---|---|---|---|
| fwp | 2M | 3 | -0.262 ± 0.048 | 0/3 | -0.379 ± 0.054 |
| ssp | 2M | 3 | -0.088 ± 0.125 | 0/3 | -0.225 ± 0.093 |
| ssp_aux | 2M | 3 | -0.145 ± 0.117 | 0/3 | -0.297 ± 0.083 |
| ssp | 3M | 3 | -0.057 ± 0.113 | 0/3 | -0.172 ± 0.101 |
| ssp_aux | 3M | 3 | -0.097 ± 0.108 | 0/3 | -0.235 ± 0.093 |

Per seed (final, 3M): ssp +0.055 / -0.171 / -0.055; ssp_aux +0.012 / -0.205 / -0.097. SSP is still rising slowly at 3M (last-10% mean +0.03 from the 2M window to the 3M window) and has not
converged; whether it would eventually reach parity was not tested (the cost to 10M would be about 9 h per run contended). The aux loss (label-free recall of the *current*
token) did not help: it pulls the head towards "read now", the opposite of the lag the policy needs, and the SSP arm without it is higher at every horizon (differences are inside
~1 std of 3 seeds, so this is a non-result, not a measured harm).

**Memory-zeroed ablation** (`rl/eval_ckpt.py`, final checkpoints, 100 episodes each, actions sampled from the policy; `results/rl_ablation_memory_zeroed.json`). Zeroing the
memory state before every step drops *every* arm to chance (about -0.50), so what each policy earns comes from its memory:

| arm (seed 0 / 1 / 2) | with memory | memory zeroed each step |
|---|---|---|
| gru | +0.998 / +0.671 / +0.999 | -0.498 / -0.514 / -0.489 |
| fwp | -0.246 / -0.279 / -0.267 | -0.502 / -0.490 / -0.502 |
| ssp | +0.055 / -0.208 / -0.047 | -0.485 / -0.487 / -0.520 |
| ssp_aux | +0.039 / -0.199 / -0.083 | -0.493 / -0.485 / -0.518 |
| ssp_oracle4 (diagnostic, 328k) | +0.770 | -0.494 |

The with-memory numbers match the last training-curve rows (ssp s0 +0.055 vs +0.055), except GRU seed 1: its checkpoint is the weights after the final PPO update, which
spiked entropy (0.067 -> 0.144) and lost return (training curve 0.997, checkpoint 0.671); this is an update instability in the last iteration and does not change the
converged-at-1M statement (its last-10% mean is 0.989).

**Why SSP stays near chance (mechanism checked on the final 3M checkpoints, not just stated).** The learned address-head bias is -0.000 / +0.011 / -0.005 (ssp seeds 0-2)
and -0.002 / +0.009 / +0.003 (ssp_aux seeds 0-2): none of the six runs moved the read from "now" to "4 steps back"; all the address learned is a small drift of the ramp slope
(ramp weight 32.64 / 30.62 / 31.56 against the write ramp 32.12 / 31.79 / 31.91 for ssp seeds 0-2). The hand-set -4 arm keeps its bias at -3.99 and learns.
That is consistent with the supervised finding that the read offset, not its size, decides whether training starts (RUN_011: offset >= 1 never moves; offset 0 passes);
it was reproduced here without any tuning. (How the policy earns +0.05 at best from a read that stays at "now" is not analysed; the ablation shows it is memory-driven, the
sinc side-lobes of a slope-drifted read are the likely route, untested.)

**Diagnostic arm (not in the comparison): hand-set read offset -4 (= the task's lag).** One seed, 328k steps: -0.501 at 33k, +0.242 at 197k, +0.424 at 230k, +0.637 at 295k,
+0.710 at 328k. It first reaches the +0.244 threshold at 230k steps vs the GRU's median 361k (1.6x fewer, one seed, and it uses the task constant, so no claim; the
pre-registered win needs >= 2x over the best baseline with a learned-from-default arm). It shows that the model class can learn this task quickly through POPGym's PPO once the address is in
the right place; the open problem is discovering that address, as in the supervised track.

## Throughput and cost (item 3; measured on this 16-thread machine, Python 3.14 CPU torch)

POPGym PPO at the config above (32 768 steps/iteration, 120 SGD steps/iteration, local sampler with 32 envs, `OMP_NUM_THREADS=1`).

**Solo** (one run alone, 3 iterations each, `results/rl_throughput_solo_<arm>.jsonl`):

| arm | steps/s | sample s/iter | learn s/iter | minutes per 1M steps |
|---|---|---|---|---|
| gru | 510 | 14.8 | 49.4 | 32.7 |
| fwp | 805 | 14.7 | 25.8 | 20.7 |
| ssp | 645 | 15.4 | 35.5 | 25.8 |
| ssp_aux | 644 | 15.2 | 35.6 | 25.9 |

**Contended** (6-7 runs at once plus another agent's jobs): about 260-310 steps/s for gru / ssp, about 410 for fwp, i.e. 1.0 h per 1M for gru, 0.9 h for ssp, 0.7 h for fwp (table above).
The whole study: 3 x GRU 1M, 3 x FWP 2M, 3 x SSP 3M, 3 x SSP+aux 3M, 1 x diagnostic 328k about 24 single-thread run-hours, all local; 0 pod-hours.
`test_per_step_cost_is_constant_and_bounded` (batch 8, T=1 forward only): SSP 1.25 ms at t=0 and 1.14 ms at t=5000 (constant in the episode position), GRU 0.64 ms.
Memory per concurrent run was about 1.1 GB (a remote rollout worker costs another 0.65 GB, so runs use the learner process as sampler: 0 remote workers x 32 envs, which keeps
POPGym's `train_batch_size = bptt_cutoff x workers x envs` at 32 768).

## What was built (all new files under `experiments/ssp_algebra/rl/`; nothing existing edited, nothing committed)

- `ssp_model.py`: `SSPModel(BaseModel)`, an RLlib `ModelV2` wrapping `rl_memory.SSPMemory` (codebook, `w_time`, `omega`, `addr_head`, `lam`). Discrete token = `argmax(obs_flat[:, :n])`;
  state = the 256-float bundle plus POPGym's own time counter (zero at every episode start through RLlib's initial-state handling); the write/read over a `[B, T]` chunk is one
  causal decay-matrix product (equal to the step loop); read address = `addr_head([hid_proj(z_t), abs_time_feats(t)])`; read-out = cleanup scores / 0.1 -> `Linear` -> POPGym's actor/critic;
  `aux_coeff` > 0 adds `SSPMemory.aux_loss` in `custom_loss`; `ablate_memory` zeroes the read-out.
- `test_ssp_model.py`: 12 contract tests (forward shapes, state size = GRU's 256, init/reset equals fresh, chunked == whole sequence, padding, vectorised == stepwise `SSPMemory`,
  aux loss == `SSPMemory.aux_loss`, gradient reaches `addr_head`/`w_time`/`lam`, aux flag on/off through `custom_loss`, ablated == feed-forward policy, lagged recall at the right address, constant per-step cost,
  state does not grow). Run: `cd experiments/ssp_algebra/rl && OMP_NUM_THREADS=1 /tmp/popgym_ref/bin/python test_ssp_model.py` -> 12/12 pass.
- `common.py` (config), `run_pilot.py` (one run, incremental JSONL, checkpoint save/restore), `launch_pilot.sh`, `launch_extend.sh`, `measure_throughput.sh`, `env_returns.py`, `report.py`,
  `eval_ckpt.py`, `SURVEY.md`.
- Results: `results/rl_env_returns.json`, `results/rl_pilot_RepeatPreviousEasy_*_s*.jsonl`, `results/rl_pilot_RepeatPreviousEasy_summary.json`, `results/rl_ablation_memory_zeroed.json`, `results/rl_throughput_solo_*.jsonl`.

## Decisions and deviations (stated, not hidden)

1. **Reproduction of `run_ref.py` and an API drift it contained.** `run_ref.py --env RepeatPreviousEasy --model GRU` ran in the rebuilt venv (4 workers x 8 envs, 65.7k steps in 473 s: -0.502 -> -0.440, same
   trend as the committed smoke). While measuring I found that **ray 2.59's `PPOConfig.update_from_dict` silently ignores the old key `sgd_minibatch_size`**: `minibatch_size` stayed at RLlib's default 128 timesteps
   (7 680 SGD steps per iteration instead of POPGym's intended 120 at 8 x 1024 = 8192). `ref_gru_600k.json` and the earlier reference numbers therefore used 128. `rl/common.py` sets `minibatch_size = 8192` (POPGym's
   `ppo.py` value) and `run_pilot.py` records the effective value in each file's meta row. With it, the GRU reaches 90% of optimal at a median 557k steps (the 128-minibatch reference had 0.683 at 632k). `run_ref.py` itself was not edited.
2. **Local sampler (0 remote workers x 32 envs)** instead of POPGym's 4 workers x 16 envs: 12 simultaneous remote workers used 13.9 of 15.4 GB. Batch layout per iteration is the same formula (`train_batch_size` 32 768 vs POPGym's 65 536); same for every arm.
3. **SSP default (one, not tuned):** address head initialised to "now with a slope error" (the `slope_only_cfg` recipe: ramp weight = write ramp x (1 + 0.03 N(0,1)), bias 0) in place of `SSPMemory`'s constant-0 address (which would read absolute index 0
   and cannot even express the lag); `dim` 128 complex (256 floats = GRU `hidden_size`), `hidden` 32, `lam_init` 0.99, `tau` 0.1, aux coefficient 0.1. The "policy hidden state" is the feed-forward feature `hid_proj(z_t)` (no recurrent policy state exists in this wrapper). No sweep was run on any of these.
4. **No `fwp_delta` arm: POPGym ships no delta-rule fast-weight model** (SURVEY.md quotes the update). The RL comparison is against its plain linear-attention `FastWeightProgrammer`.
5. **Extension by checkpoint restore** (fwp to 2M, ssp / ssp_aux to 3M): hyperparameters unchanged; RNG streams are not continuous across the join (a `restored` row marks it). The 1M table uses only rows up to 1M and is what the contract asked for.
6. Only RepeatPreviousEasy was run (the contracted pilot); Medium (k = 32) / Hard (k = 64) were not, so length extrapolation and robustness metrics are untested.
7. Seeds: 3 per arm (pilot size). The success rule asks for at least 5 for a confirmatory RL run; this report is the pilot and says nothing finer than "SSP fails here".

## Unresolved

- Whether SSP reaches parity with more steps (still rising at 3M) or only with a learned/discovered read offset. Candidate next steps (each a new, pre-registered design, not done here because the plan forbids per-arm tuning): the
  tied-encoder / relative-addressing and dead-zone-free kernel items of handoff 11 section 3.1 ported into `SSPModel`, or K read heads with an offset range (decision 1: only as a fallback).
- The aux loss as specified (recall of the current token) is mis-aligned with a lagged read; a lag-aware label-free target would be a design change.
- `RepeatPreviousMedium/Hard`, the other POPGym memory tasks, and 5-seed confirmation were not run.
