# RUN_009 — Recipe diagnostics for learned SSP structure (handoff 09 section 14)

**Status: COMPLETE (2026-10-11).** 28 RunPod CPU pod jobs (21 + 7), each 6 random configs scored on search seeds and **5 hold-out seeds**
(1000-1004; 10 for one pointer run), 1500 updates unless noted, 256 carried floats, dense evaluation (28+ checkpoints, integrity-checked), converged = mean of
the last 3 checkpoints, ± = standard error. "upd->0.9 / 0.99" = first checkpoint at which the 5-seed mean accuracy reaches the threshold. Everything is MEASURED
unless marked *inferred*. Predictions were written into the handoff **before** running (handoff 09 section 14). Code: `arms_learned.py`
(`freeze_write`, `ls_mode="adaptive"`, `address_params`, `retrieval_loss`), `harness.py` (`addr_lr_scale`, `pretrain_steps`, `make_optimizer`), tests in `test_learned.py` (12/12).

## 1. Headline

1. **The learned arm is not slow once it is at the right address.** Identity-initialised, with a sensible learning rate (0.0225) it reaches 0.90 in **50 updates and 0.99 in 50**,
   about as fast as hand-wired rung 0 (75 / 125; the two come from differently selected configs, so "as fast as", not "faster"). RUN_008's "3x slower than rung 0 even at the right
   address" (225 / 375) was an **artifact of config selection**: all 6 configs tied at 1.00 on the search seeds and the tie-break picked lr = 0.00075. The hypothesised cause (Adam jitter on
   the address parameters) is **refuted**: a 10x lower address learning rate and a frozen write phase change nothing (50 / 75 and 50 / 50). So the whole cost of the learned arm is
   **discovery**, not optimisation.
2. **Discovery can be sped up modestly.** A shorter width schedule (375 updates) gets 0.90 at 250 and 0.99 at 400 updates with no loss of accuracy (0.992), against 350 / 600 for the
   750-update schedule. Not better: an adaptive (plateau-driven) schedule, a larger learning rate on top, light address supervision.
3. **Label-free interface pretraining is a double-edged result.** When it aligns the address heads, the N=64 collapse disappears (**0.91-0.98 at N=64**, vs 0.279 without it; hand-wired 0.966).
   But it aligns them on only **2 of 5 seeds**, and 3x more pretraining does not change that.
4. **The recipe does not transfer to a longer stream.** At T=32 the unchanged recipe gets 0.665 and the "start = half the range" rule 0.849 (hand-wired 0.964, GRU 0.942). Both the start width and the schedule
   length have to grow with the address range (see 3). **No recipe has passed the reliability protocol (docs/11 4.7).**
5. **Pointer:** a temperature schedule does nothing; plain training succeeds on 6 of 10 seeds.

## 2. Diagnostics against the pre-registered predictions

**H1 address-head optimiser** (identity init; prediction: lower address lr or frozen write brings 225 toward 75-100 updates): conv id / upd->0.9 / upd->0.99
- original reference (selected lr 0.00075) 0.998 / 225 / 375
- **control: lr 0.0225, nothing else 0.995 / 50 / 50**
- + address lr x0.1: 0.995 / 50 / 75; + frozen write: 0.995 / 50 / 50; + both: 0.996 / 50 / 50
- ws8 discovery arm + frozen write: 0.997 / 700 / 800 (reference ws8 0.993 / 350 / 600): freezing the write phase *slows* discovery.
- Verdict: the effect is real but the stated cause is wrong; it is the learning rate that the tie-break chose.

**H2 schedule length / adaptivity** (prediction: shorter is faster until discovery breaks; adaptive matches or beats the best fixed): ls 8 -> 1
- 750: 0.993 / 350 / 600; 500: 0.993 / 275 / 500; **375: 0.992 / 250 / 400**; 250: 0.989 / 200 / 1000; adaptive (cap 1000): 0.993 / 500 / 900.
- Shorter helps down to ~375; 250 starts to cost accuracy. **Adaptive did not match the best fixed schedule.** (Partly as predicted.)

**H3 learning-rate range** (prediction: 0.05 as good or better, 0.1 may destabilise): lr 0.0229: 0.993 / 350 / 600; 0.0316: 0.993 / 300 / 600; 0.05: 0.990 / 250 / 600; 0.1: 0.981 / 250 / never.
Combined schedule 375 + lr 0.05: 0.987 / 225 / 1200 (worse than 375 alone). (Confirmed, small effect; the combination does not add.)

**H4 schedule + light address supervision** (prediction: faster): aux 0.1 and 0.5 both 0.997 / 700 / 800 vs 0.993 / 350 / 600. **Wrong:** slightly higher accuracy but slower. *Inferred:* the squared address error
(tens of index^2 early) dominates the gradient and the clip at 1.0 starves the readout; not tested.

**H5 label-free interface pretraining** (500 updates on fresh synthetic streams, retrieval loss on the memory's own cleanup scores, address heads only, then frozen; prediction: N=64 recovers to >0.8):
- no pool: 0.556±0.153, seeds 0.28, 0.96, 0.26, 0.29, 0.99; pool N=64: 0.484±0.166 (0.19, 0.16, 0.19, 0.91, 0.97); pool N=256: 0.525±0.161 (0.25, 0.18, 0.94, 0.27, 0.99).
- pretraining 1500 instead of 500, N=64: 0.484±0.178 (0.12, 0.19, 0.17, 0.98, 0.97). Pretraining tokens 256k / 769k, **not** counted in the task tokens.
- **Mechanism supported, reliability not:** where it works N=64 is rung-0-like; it works on 2/5 seeds regardless of pretraining length. The task-loss schedule with the same width schedule succeeds on 5/5, so the retrieval loss is the weaker signal.
  *Inferred (untested):* at wide kernels the cleanup scores are nearly uniform, so the loss through them gives a weak gradient that the nonlinear readout otherwise amplifies.

**H6 pointer** (prediction: success improves): plain at 10 seeds 0.805±0.075, **6/10 reach >=0.95** (stuck seeds sit at 0.51-0.53); plain + temperature anneal 0.804±0.107, 3/5 (same pattern). No effect.

## 3. Generality probe: index at T=32 (hand-wired rung 0 0.964±0.005, 125 updates to 0.9; GRU 0.942±0.025, 600)

Learned arm, width schedule from `ls_start` to 1 (conv id; upd->0.9):
- unchanged (start 8, 750 updates): 0.665±0.024; rule start 16 (T/2), 750: 0.849±0.013; **start 32 (T), 750: 0.930±0.004 (800)**
- start 16, schedule 375: 0.738±0.009; start 16, lr 0.05: 0.893±0.024; **start 16, schedule 1500 with 3000 updates: 0.938±0.003 (1400)**
Both a wider start and a longer schedule help, and shortening the schedule hurts here while it helped at T=16. *Inferred* from three points per axis: the required start width and schedule length scale with the address range.
Best T=32 results (0.93-0.94) are within 5 points of the best baseline (GRU 0.942) but below 0.95 and slower than the 50-125 updates of the T=16 learned arm.

## 4. Errors, corrections and limitations

1. **Config selection by converged accuracy is blind to speed.** When several configs tie near ceiling the winner is arbitrary with respect to speed (here 225 vs 50 updates for the same arm). Every speed number
   in RUN_008 and here (t50/t90, updates-to-threshold) comes from configs chosen on convergence, for *all* arms including the baselines. Speed comparisons are therefore noisy and, for any single arm, conservative.
   A speed-aware selection (area under the learning curve on the search seeds) has not been run.
2. The RUN_008 sentence "3x slower than rung 0 at the right address" and the hypothesis behind diagnostic 1 are retracted (1.1).
3. `ls_mode="adaptive"` is a hand-built plateau rule (5% over 25 updates, 0.75x); it was one setting, not tuned.
4. T=32 sits at the rung-0 capacity limit for 256 floats (D/4 = 32 items), so the probe is also a stress test of the memory; a bigger budget was not run. Only one longer stream length was probed.
5. Everything is the index task; the discrete-key family got only the temperature and seed-count checks. Pointer restarts and a label-free pointer pretraining were not run.
6. No result here is a reliability pass (docs/11 4.7): that needs frozen constants on unseen variants.
7. The rung-0 speed references used here (75 / 125 updates; 125 updates to 0.9 at T=32) come from the same convergence-selected configs, so they carry the caveat in item 1 as well.
8. Pointer success rates are small-sample: 6 of 10 has a rough 95% Wilson interval of 31%-83%, so 3/5 and 6/10 are not distinguishable and "about 40% of seeds get stuck" is an estimate with wide error.

## 5. What can and cannot be concluded

**Firm (measured, with a control, 5 seeds):**
- The learned index arm's cost is discovery, not optimisation (1.1); the Adam-jitter explanation is refuted.
- During discovery accuracy follows the current kernel width's ceiling, so the narrowing speed is the limit; a 375-update schedule is better than 750 at T=16.
- The T=16 constants do not transfer to T=32, but coarse-to-fine still works there with scaled constants (0.93-0.94): the *form* carries, the *constants* do not.
- The low-data collapse of the learned arm is an address-learning problem: where interface pretraining aligned the heads, N=64 was 0.91-0.98.
- The two-family split is supported: the width schedule has no analogue on pointer, and a temperature schedule did nothing there.

**Suggestive (inferred, untested):** why light address supervision is slower (gradient domination plus clipping); why retrieval-loss pretraining aligns only 2 of 5 seeds (weak gradient through
near-uniform cleanup scores at wide kernels); that the start width and schedule length scale with the address range (three points per axis); that about 40% of pointer seeds get stuck.

**Not concluded:** that any recipe is reliable (none has passed docs/11 4.7); that the learned arm is faster than the GRU (speed numbers are from convergence-selected configs); anything about
robustness, RL or larger models; that the advantage is SSP-specific (the `fwp` control and the position table still match much of the index advantage); that pretraining is amortised
(transfer is untested and the 256k pretraining tokens are not task samples only if the interface is reused).

## 6. Claim-to-step map and next steps (detail: handoff 09 section 15)

| claim to test | step | confirmed if | refuted if |
|---|---|---|---|
| speed claims are fair | 1: speed-aware config selection, baseline hardening for low data | the SSP-vs-GRU gap at 0.90 survives (at least 2x) and tuned baselines stay >5 points behind at N=64/256 | the gap vanishes or tuned baselines close the low-data gap: drop that claim |
| "family recipe, constants follow a rule" | 2: range-scaling rule tuned on T=16/32, frozen, tested on unseen T and on delay | at least 0.95 on 9 of 10 seeds on every unseen variant, no retuning | two or more unseen variants need retuning: variant-specific trick; try a multi-scale bank or a better learned width |
| "staging (interface first) restores data efficiency" | 3: reliable and transferable interface pretraining | at least 18 of 20 seeds align, N=64 at least 0.9, and the interface transfers to another length/alphabet | stays unreliable or does not transfer: it is just more training data |
| pointer pass is real | 4: stuck-seed diagnosis, equal-aid baselines, non-SSP iterated control, rung 3, cue fix | SSP still beats equal-aid baselines on speed or held-out hops with plain or lightly aided training | a looped transformer with per-hop labels matches it: the finding is "iteration plus supervision" |
| robustness (H-c) | 5: after the cue fix and a reliable recipe | pre-registered graceful-degradation thresholds met against equal-aid baselines | no separation |
| unique SSP operation matters | 6: variant A redesign | shift-on-memory beats attention over stored (pose, landmark) pairs on accuracy, state or compute | matched by the attention arm |
| RL / larger models | 7: only after docs/11 4.8 holds | all gate conditions hold | any fails: stay supervised |

Order: step 1 first (cheap, decides how much of the speed story survives); steps 2, 3, 4 can run in parallel on pods; then 5, 6, 7.

## 7. Do not do

- Do not cite "3x slower at the right address" (RUN_008 wording) or Adam jitter as a cause.
- Do not claim label-free pretraining solves low data: it aligns on 2 of 5 seeds, and amortisation is unproven.
- Do not quote the T=16 recipe as general; it fails at T=32 without retuning, and per-problem tuned constants are a technique, not a recipe.
- Do not compare speeds across arms without noting that configs were chosen on convergence, not speed.
- Do not start RL, a larger-model study or the learned-arm robustness study before docs/11 4.8 holds.
