# Handoff 11: turn the SSP memory into a deployable, learnable, simple module, then test it where it must work (follows handoff 10)

> ## STATUS (appended 2026-10-14; read `runs/RUN_013_equal_aid_and_rl_lag_bank.md` first)
>
> **The two axes of change are now fully covered.** `docs/11` section 14.3 lists 5 learning-component candidates and 7
> architectural ones; every one has a measured cell, including the last unbuilt one (content-plus-position keys).
> Sections 1-11 of RUN_013 hold the tables; `docs/11` section 17 is the judgment. Headlines:
>
> - **The index lead is withdrawn** (step 1's pass rule fired). Given the same two knobs, `fwp_delta+gate+banks2`
>   reaches **1.000 with 10/10 hold-out seeds at 0.95**, matching `recipe+gate+banks2`. The input-dependent gate
>   alone takes the fast-weight arm from 0.745 (0/11 at 0.9) to 1.000 -- the gap RUN_012 saw was the missing aid.
>   What survives is **speed at t90 only** (SSP 2.16x fewer tokens to 0.90, but just 1.20x to 0.50, on hold-out rows),
>   and it shows on `index` only, and
>   `delay` ties exactly, so it is **not yet** a sample-efficiency win under decision 2.
> - **The hand-set prior is gone.** `tied + input gate + multi-timescale banks(2)`, from `init="scrambled"` with no
>   offset, no `eps`, no schedule and no range rule, passes **index T=32 (1.000, 10/10), index T=48 (1.000, 10/10),
>   delay 1-4 (0.999, 10/10) and delay 1-8 (0.999, 10/10)** at >= 0.95 on hold-out rows -- **but it is worse at length
>   extrapolation** than the incumbent (conv_extrap_T 0.343 vs 0.457), so the prior bought generalisation past the
>   trained range. See RUN_013 section 13. Two task-dependent constants against the
>   incumbent's four, and neither is an assumption about where the read address starts. This is the round's main
>   advance on the bitter-lesson axis; it also repairs RUN_012's `tied+gate` delay failure (0.488 -> 0.999), whose
>   cause was the missing bank.
> - **The axes interact.** K5 from an offset-3 start is 0.496 (0/10) alone and **1.000 (9/9) with gates+banks**;
>   gates alone on delay with no prior is 0.488 (0/10) and 0.999 (10/10) with banks. One-change-at-a-time hid both.
> - **Content-plus-position keys works** (`assoc_mixed`, `assoc_pos_ssp`): 1.000, 10/10 at 0.95 on a task where each
>   query asks by content *or* by index, with the ablations failing **mode-specifically** (`pos=False` 0.625 = content
>   solved + index at chance; `bind=False` 0.253; GRU 0.249). A capability result, not a win over an incumbent, and it
>   costs two bindings per pair.
> - **Distractors: direction right, level short.** At 3000 steps the incumbent recipe is flat (0.569 -> 0.564, a
>   capability ceiling) while gated cells reach 0.73-0.76; the GRU stays at exactly chance (0.127 vs 0.125).
> - **Lag selection is hard for every architecture.** The new `delay_nolag` variant (lag feature genuinely hidden)
>   leaves *all* arms at 0.35-0.42 with extrapolation at chance -- recipe 0.351, GRU 0.398, LRU 0.400, best K5 0.418.
>   So the RL failure on `RepeatPreviousEasy` is not "the SSP arms are weak"; it is a hard problem for everything here.
> - **The RL lag bank clears the dead zone, and then learns** (`rl.SSPTiedLagBankModel`): from a uniform bank of lags
>   1..8 the learned selector converges to a single sharp lag (`lag_eff` 4.50 -> **2.99**, `sel_entropy` 2.06 ->
>   **0.04**) in ~130k PPO steps with no lag given, and it then clears **both** thresholds in **3 of 3 seeds** (T50 at
>   230k, T90 at 492k, against the GRU's 361k / 557k medians = **1.57x and 1.13x**) and finishes at **+0.958 against
>   the GRU's +0.993 -- inside the 5-point parity margin**. **No lag constant was injected.** First RL run in the
>   project where an SSP memory arm learns `RepeatPreviousEasy` (RUN_012: -0.133 at 1M; POPGym's own fast-weight model
>   never learns it at 2M). AULC is 33% higher (+0.582 vs +0.436). It is **not** a decision-2 pass: both ratios are
>   short of the >= 2x the rule asks for and per-step cost is worse (101 vs 264 steps/s contended). The aux arm never
>   learns. All six runs completed 1M steps; see RUN_013 section 10.1.
>
> **Table denominators were wrong and are corrected.** Every table in this handoff's earlier STATUS blocks and in
> `runs/RUN_013` originally counted 11 rows per cell (10 hold-out seeds + 1 search run) and the *tuned* columns pooled
> search configs with final runs. `experiments/ssp_algebra/report_holdout.py` now does the reading (hold-out rows only,
> `n` printed); RUN_013 section 13 is the correction log, and it supersedes any earlier figure in this file.
> One measurement is **missing and cannot be recovered from the saved state**: the memory-zeroed ablation of the RL
> `tied_bank` arm, because a machine reboot wiped `/tmp` (checkpoints and logs). The durable RL result files survived.
>
> **Three traps cost real time this round, all now guarded** (RUN_013 section 11): `--arm-cfg` JSON must be
> single-quoted in a shell (bash brace-expands it otherwise); `ssp_pod_batch.py launch` resumes on **R2**, not on
> `results/`, so re-launching re-creates pods for cells already collected locally (`scripts/ssp_pods.py kill --prefix`
> is the durable reaper); and **a variant that hides an input must be tested at the consumption site, not at the
> feature builder** -- `DelayNoLagTask` zeroed the feature column while the arms read `batch["lag"]` directly, and
> every arm scored ~1.000 on a task it was not solving (RUN_013 section 8; the test now asks the arms themselves).
>
> - **The spatial recall task -- the robotics gate -- is built and verified, and the gate is FAILED.**
>   (`tasks_e.SpatialRecallTask`, `arms_spatial.SpatialSSPArm`, `test_tasks_e.py` 5/5.) Landmarks are written once as
>   current-frame positions, motion is applied as **one group action to the whole bundle**, and reads are in the
>   current frame; the mechanism is verified before training (shifted phase identity 1.4e-06 against 2.000 unshifted).
>   The first untuned pod run looked like a 17x win (SSP 0.097 against the GRU's 1.659) -- **but a trainability gate
>   showed the GRU could not even carry one 2-vector on a trivial variant (0.630 against a 1.314 floor), i.e. the
>   baselines were untrained.** Tuned (4 configs x 2 search seeds, 4000 steps, 10 hold-out seeds,
>   `jobs/p21_spatial_tuned.json`): **tuned GRU 0.021 against the SSP arm's 0.077 -- the GRU wins by 3.7x**, and by
>   13x on the 2x-step-size split. **So the robotics argument under decision 4 is NOT supported and the 17x is
>   withdrawn.** What survives is internal: the task is SSP-shaped by construction and the group action is
>   load-bearing within the arm (0.096 vs 0.303 no-shift, untuned). Do not quote p19's numbers as a comparison.
>
> **Next steps, in order.**
> 1. **Finish the RL lag-bank pilot** (`rl/launch_bank.sh`, 3 seeds x 1M steps, `tied_bank` and `tied_bank_aux`) and
>    apply rule 4: >= 2/3 seeds at +0.244 ("learns"), parity with the GRU at the final return, and a >= 2x speed claim
>    against the GRU's 361k T50. Report the `lag_eff`/`sel_entropy` trajectory as the mechanism check. If it passes,
>    extend to 3M steps and then re-run the **same task with the equal-aid fast-weight arm** so the comparison is not
>    SSP-vs-GRU only.
> 2. **The lag-bank idea in the supervised harness** is still untested: `delay_nolag` shows that neither K read heads
>    (0.418) nor anything else solves lag selection, so the bank needs to be built on `SSPIndexLearned` (a bank of
>    read offsets selected by a softmax over the policy/feature state) and run on `jobs/p17b`'s task at 10 seeds. That
>    is the cheap discriminator the RL port lacks.
> 3. **Settle the speed claim**: the 2.16x t90 index separation (1.20x at t50) needs a second task family to count
>    (delay ties) and a second threshold to hold, and the
>    `fwp_delta+gate+banks2` arm should be given the K-heads/banks treatment too so the comparison stays equal-aid.
> 4. **Only then** the confirmatory RL run (5 seeds, up to 5M, easy and medium) with a hard pod-hour cap.
>
> **Machine discipline unchanged:** pod and local rows are never tabulated together; `--ep-len` explicit; a fresh
> `--tag` per protocol; `test_default_path_hash` six `ok` after every edit to `arms_learned.py` / `ssp_ops.py` /
> `harness.py` / `run_sweep.py`; never edit an existing test or threshold; nothing is committed.
>
> Local test state at hand-off: **all 23 modules pass** with unchanged counts (harness 6/6, fixes 2/2, measurement 6/6,
> learned 12/12, analysis 9/9, regularisation 7/7, select 10/10, recipes 10/10, pretrain 7/7, pointer 14/14, halting,
> multiscale, affine, pretrain_shapes, trainability, selftest, rl_memory, candidates, fwp_delta, fwp_gates) plus the
> new `test_tasks_f` (**9/9**), `test_tasks_d` (**5/5**), and in the reference venv `test_ssp_model` 12/12 and
> `test_ssp_tied` **13/13**.


> ## CURRENT PLAN (appended 2026-10-13; **start here** -- the STATUS block below is history)
>
> **Focus (the user's words, 2026-10-13).** Find **architectural and recipe candidates that align with Sutton's bitter
> lesson** (no task-dependent constants; prefer learning, search, self-supervision). Once one is good enough on most
> metrics and *possibly beats the alternatives on sample efficiency*, take it to **POPGym** so the result rests on a real
> benchmark instead of our own synthetic tasks.
>
> **Where the work stands.** RUN_012 measured every candidate from `docs/11` section 14.3 on pods (10 seeds). The
> scoreboard is `runs/RUN_012` section 5b; the judgment is `docs/11` sections 15-16. Wins so far, all at index T=32,
> budget 256, 10 seeds, one configuration each (no tuned column yet):
>
> | candidate | conv | seeds >= 0.95 | tokens to 0.90 | status |
> |---|---|---|---|---|
> | recipe (4 task-dependent constants) | 0.957 | 9/10 | 52.0k | reference |
> | input-dependent gates | **1.000** | **10/10** | 51.3k | best accuracy |
> | gates + multi-timescale banks(2) | **1.000** | **10/10** | **25.8k** | best overall cell |
> | multi-timescale banks(4) | 0.962 | 9/10 | **25.8k** | 2x fewer tokens |
> | K=5 read heads from a one-index offset error | 0.962 | 9/10 | 51.3k | repairs a bad start (0.663 at K=1) |
> | tied relative addressing, **no prior** | delay **0.999/0.994**; index 0.930 (T=48 0.913, T=64 0.817) | 10/10 on delay | 43.2k delay / 970k index | removes all constants; costs index |
> | gates + tied, no prior | index 0.977 | 9/10 | 231k | best prior-free index cell; **fails on delay (0.488)** |
> | rung 2 (learned frequencies + codes) | 0.999 | 10/10 | 51.4k | positive |
> | delta write / banks(2) / interpolating readout | 0.951-0.961 | 8-9/10 | 51-52k | neutral |
> | cauchy kernel / learned width | 0.882 / 0.947 | 0/10 / 5/10 | 205k / 77k | negative |
> | content-keyed family (`assoc`, **local**) | VSA 1.000, outer-product 1.000, 10/10; **GRU 0.249, LRU 0.281**; no-bind ablation 0.248 | 10/10 | 153.6k | strongest capability separation; under distractors the outer product wins (0.794 vs 0.589) |
>
> **The single most useful new finding (RL, unverified partial numbers from the delegated track).** On
> `RepeatPreviousEasy`, tying alone does **not** fix RL: SSP untied -0.163, **tied -0.133**, tied+aux -0.202, GRU +0.993;
> but **tied with the read offset initialised to the task's lag reaches +0.832 with 3/3 seeds at the 50% threshold**.
> That separates the discovery problem into **origin** (what tying removes) and **lag selection** (what nothing in the
> supervised delay task tested, because there the lag is given as an input feature). `RepeatPrevious` gives no lag
> feature, so the arm must find the offset itself.
>
> **THE PLAN (do these in order; each step states its own pass rule and stop).**
>
> 1. **Equal aid, equal tuning (fixes the biggest threat to the index result).** Give `FWPArm` the same two knobs the
>    SSP arm has: `gate="input"` (an MLP over `task.in_dim` emitting write strength and decay; identity init -- output
>    weight zero, bias set so sigmoid ~= 1 and ~= `lam_init`) and `banks=K` (K square matrices of
>    `d_b = round(sqrt(budget/K))`, outputs concatenated, carried state unchanged within rounding and reported exactly).
>    Then run `jobs/p14_equal_aid.json`: arms `fwp_delta`, `fwp_delta+gate`, `fwp_delta+gate+banks2`, the SSP recipe,
>    `recipe+gate`, `recipe+gate+banks2`, at index T=32 and delay 1-4, **two columns** (`configs=1` and
>    `configs=6 --search-seeds 3 --select aulc --min-conv 0.0`). New test file `test_fwp_gates.py` (identity-gate
>    equivalence to 1e-6, state split within 4%, 120-step training on index, gradient reaches the gate, `ValueError` on
>    bad `banks`). *Pass rule (write it before running):* if `fwp_delta+gate+banks2` lands within 5 points and 10/10 of
>    `recipe+gate+banks2`, **the index lead is withdrawn**.
> 2. **The two missing tasks.** (a) `tasks_f.py` `DelayDistractTask`: lags with marked distractor symbols that are
>    never targets, splits `id` / `extrap_T` (2x T) / `dense` (p=0.5 against 0.25 trained) -- this is where gates
>    should matter and where no candidate has been tested. (b) the **lag-bank** split for the RL parity test in step 4.
>    New `test_tasks_f.py`: determinism, no target on a distractor step, alphabet unchanged across splits, no-memory
>    floor near chance.
> 3. **Crossed combinations (the axis interactions are untested).** `jobs/p16_cross.json`: tied+gate+banks2 at index
>    T=32/48 and delay 1-4/1-8; `recipe+gate+banks2+K3`; `gate+banks2+K5` from offset-1 and offset-3 starts; K5 on delay
>    and on T=48. *Pass rule:* the prior-free target is **>= 9/10 seeds at 0.95 with no prior, on both index and delay**.
>    Tied+gates currently fails that on delay (0.488) -- explain it or record it.
> 4. **RL with the candidates, on POPGym (the user's priority: only once step 1 has not killed the index lead).** The
>    delegated track left `rl/ssp_tied_model.py`, `rl/test_ssp_tied.py`, `rl/run_pilot.py`, `rl/report_tied.py`,
>    `results/rl_tied_*` and **no `REPORT_TIED.md`** (it stopped before writing it; the summary JSON exists). First
>    re-run `rl/test_ssp_tied.py` in the rebuilt venv and treat every number as unverified until then; `/tmp/popgym_ref`
>    was wiped, so rebuild it from `ref_ppo/requirements-frozen.txt` per `ref_ppo/README.md`. Then, in the same PPO loop
>    and with the same thresholds (+0.244 at 50%, +0.849 at 90%), test the candidates the supervised round favours:
>    **a lag bank of read offsets 1..L with the output selecting** (the lag-set diagnostic shows the arm can use a
>    correct offset: +0.832) and **input-dependent gates**. 3 seeds, ~1M steps, report the offset trajectory per arm as
>    the mechanism check, extend a winner to 3M, and cap the local compute (the GRU costs ~1 h per arm per 1M steps).
> 5. **Spatial recall task** (`tasks_e.py`, supervised first): a landmark shown at the start, the agent moves, and the
>    model reports the landmark in its current frame; arms GRU, `fwp_delta`, `pose_attn`, and an SSP arm that shifts the
>    *bundle* by `phi(-d)` (relative addressing as the motion update). Per decision 4 this gates the robotics direction
>    only.
> 6. **Content-plus-position keys and the pointer limit** (lower priority): `tasks_d` mixed mode (query by content or by
>    index) with `AssocPosSSPArm`; one 32-node pointer cell at 3x steps, to see whether the 0.205 was a step budget.
> 7. **Full regression + records.** All modules must keep their counts (harness 6/6, fixes 2/2, measurement 6/6, learned
>    12/12, analysis 9/9, regularisation 7/7, select 10/10, recipes 10/10, pretrain 7/7, pointer 14/14, plus the ok/PASS
>    modules and `test_fwp_delta`, `test_candidates`, `test_default_path_hash`). Write RUN_013 and update docs/11.
>
> **The guard to run after every edit to `arms_learned.py` / `ssp_ops.py` / `harness.py` / `run_sweep.py`:**
> `OMP_NUM_THREADS=1 ../../.venv/bin/python test_default_path_hash.py` -- it must print six `ok` lines. It holds the
> pre-edit hashes of the default `fwp` and rung-1 SSP paths (the `/tmp` script that checked this is gone; the test is
> the durable form, and its expected values must never be changed to make it pass). It caught three silently dropped
> lines this round.
>
> **New traps from RUN_012 (each cost real time):** (a) smoke the *exact* `arm_cfg` locally at 30 steps before any pod
> launch -- the stale-code cells and the dead `p9` grid were both this; (b) **a split must never change the alphabet**
> (the assoc family crashed twice that way); (c) the delta write needs `beta_init=-4.0` (near-plain) or it does not
> train; (d) a pod cell at 0 rows after 20 minutes is dead -- kill it (`DELETE /v2/pods/<id>` with the bearer key from
> `.env`) and debug locally; (e) `/tmp` is not durable -- put assets in the repo; (f) Ray silently drops POPGym's
> `sgd_minibatch_size`, so the RL pilot sets `minibatch_size = 8192`; (g) pod and local results are never tabulated
> together.

> ## RUNNING STATE (2026-10-13, later; written after the work was **stopped on the user's instruction** -- all pods
> were deleted, verified through the API, and no experiment is running)
>
> **Code and tests built in this stretch (all verified; every new knob defaults OFF):**
>
> - **`FWPArm` now has `gate="input"` and `banks=K`** -- the same two knobs the SSP arm has, so the fast-weight
>   baseline can be given *equal aid* (plan step 1). The gate *modulates* the decay and the write strength rather than
>   replacing them, so at its identity init a gated arm equals its ungated self. Verified by the new
>   **`test_fwp_gates.py`** (5 checks, all pass): `banks=1` is **bit-identical** to the plain arm; the identity gate
>   reproduces the ungated memory to **0.00e+00**; `banks=2` carries 242 floats and `banks=4` 256, i.e. the state is
>   *split* and never grown; in 120 steps on index, gate 0.163 -> 0.994, banks=2 0.150 -> 0.912, gate+banks=2
>   0.106 -> 1.000.
> - **`tasks_f.py`: `DelayDistractTask`** (variant `delay_distract`, wired into `run_sweep.make_task` and
>   `harness.splits_for`): delay recall with marked distractors that are never answers, so the memory must choose
>   what to store. Splits keep the alphabet fixed -- `id`, `dense` (p=0.5 against 0.25 trained), `extrap_T` (4x T),
>   `extrap_lag` (2x/3x the trained lags). Verified by the new **`test_tasks_f.py`** (5 checks, all pass):
>   deterministic; an oracle reading the lag-th *content* step answers **every** masked step; no masked target sits on
>   a distractor; alphabet and feature width identical in all five splits; constant predictor 0.142 and untrained MLP
>   0.125 against chance 0.125, so the task does not leak.
> - **`test_default_path_hash.py`**: the durable form of the bit-exactness guard. `/tmp` was wiped, so the six
>   pre-edit hashes of the default `fwp` and rung-1 SSP paths now live in the repo. Run it after every edit to
>   `arms_learned.py` / `ssp_ops.py` / `harness.py` / `run_sweep.py`; it must print six `ok`, and **its expected
>   values must never be changed to make it pass**. Currently six `ok`, with `test_harness` 6/6, `test_learned` 12/12,
>   and `test_candidates`, `test_fwp_delta`, `test_tasks_f`, `test_fwp_gates` all passing.
>
> **Pods stopped mid-run (nothing running now):**
>
> - **`jobs/p14_equal_aid.json`** (24 cells, 468 rows): `fwp_delta`, `+gate`, `+gate+banks2` against the recipe,
>   `+gate`, `+gate+banks2`, at index T=32 and delay 1-4, each with a **default** column (`configs=1`) and a **tuned**
>   column (`configs=6 --search-seeds 3 --select aulc --min-conv 0.0`). **The delay column finished and was
>   collected** (`results/sweep_p14_delay_*`: 12 cells, 11 rows per default cell and 28 per tuned cell). The **index
>   column did not**: only `index_recipe_def` (11 rows) and `index_fwp_delta_tun` (28 rows) came back; the rest were
>   deleted with the pods.
> - **`jobs/p15_distract.json`** (9 cells, 99 rows): `delay_distract` with recipe, +gate, +gate+banks2, tied,
>   tied+gate, `fwp_delta`, +gate, GRU, LRU. **No rows arrived** -- stopped within a minute of launch.
> - **To continue: just re-run both job lists** (`launch --dry-run` first). Neither is broken; they were interrupted.
>   Their existing rows are in `results/` and the resume key will skip them, so a fresh `--tag` is needed only if a
>   config changes (the tag *is* the resume key).
>
> **Two more traps found here (both cost a cycle):**
>
> - **`harness.make_batch` threads `lags` and `cue_drop` itself**, so a split must pass them at the **top level** of
>   its spec, never inside `task_kw`, or the call dies with "got multiple values for keyword argument". Second time a
>   split has tripped on this (the first was `cue_drop`).
> - **Smoke tags are resume keys**: re-using `--tag smoke_x` with a different `arm_cfg` makes `run_sweep` report "0 to
>   run" and silently measure nothing. Use a fresh tag per smoke and check the log for "N already done".
>
> **The plan above is unchanged and still the next agent's route.** The highest-value next run remains the RL
> lag-bank experiment (tying alone left RL at -0.133 while setting the offset reached +0.832), and step 1's pass rule
> stands: if `fwp_delta+gate+banks2` lands within 5 points and 10/10 of `recipe+gate+banks2`, the index lead is
> withdrawn. Note that the partial p14 **delay** rows are already on disk for that reading.

> **STATUS (appended 2026-10-12, after `runs/RUN_012_practitioner_baseline_and_candidates.md`; read that first for
> tables, `docs/11` section 15 for the judgment).** Steps 0a-0c and the candidate axis were run. What changed:
>
> - **The `fwp` control was not the incumbent, and this round built the real one.** `fwp` was plain linear attention
>   (`M += v kᵀ`, untied heads, delay query keyed on the lag alone). `fwp_delta` (delta rule + tied encoder + read at
>   the wanted item's own time) now exists and passes its gate. With it: **`delay` is substitutable** (0.993/0.998 at
>   10/10 seeds against the SSP recipe's 1.000 — the old loss was the address convention), and **on index at equal
>   state the fast-weight family is far behind** (0.726/0.744 against 0.957 at T=32; 0/10 seeds at 0.9 against 10/10).
> - **The address dead zone is not SSP-specific**: both fast-weight arms degrade monotonically with a read offset, so
>   RUN_011's mechanism generalises. The SSP index advantage survives the "the others had a worse address" objection,
>   because these were given a tied, absolute one.
> - **Tied addressing removes the prior entirely, at a price** (`addr_mode="tied"`): 0.917 with 9/10 seeds at 0.9 from
>   a random start, versus 0.622/1-of-10 with a separate head. The price (0/10 at 0.95 against 9/10, and ~19x the
>   tokens) travels with the claim. Tied + prior is the best cell (0.960). Tying works only where the read address
>   coincides with a write address, which is why `delay` needed the relative form.
> - **The dead-zone-free kernel does not pay** (cauchy 0.882 with a prior against 0.957): it trades the dead zone for
>   resolution, as the risk note predicted.
> - **A content-keyed family exists** (`tasks_d.AssocTask`, `arms_assoc`): the VSA bundle and its matched fast-weight
>   arm both solve it locally (1.000 in 400 steps) while the untuned GRU/LRU sit at chance, and the additive ablation
>   (no key) stays at chance — the binding is load-bearing. Pod numbers at matched tuning are in RUN_012.
> - **The RL track ran (POPGym, local, 3 seeds) and it fails on the same dead zone.** `RepeatPreviousEasy`: GRU +0.993
>   (50% threshold at a median 361k steps, 3/3 seeds), POPGym's fast-weight model (plain, no delta rule) -0.396, SSP
>   -0.163 (aux off) / -0.268 (aux on); at 3M steps SSP is -0.057 / -0.097 and never reaches +0.244. **The learned
>   read-offset bias stayed at |bias| <= 0.011 in all six runs**; hand-setting it reaches +0.710, and zeroing the
>   memory drops every arm to -0.50. So parity fails, no metric is won, and **the follow-up is to move the supervised
>   fix (tied relative addressing) into the RL arm** -- that run is delegated and in progress.
> - **Still not done:** the spatial recall task (handoff 11 step 3 / `tasks_e.py`) was not built, and the interface
>   pretraining cell was killed at 0 rows (its answer stands from RUN_011 section 5).
> - **Every new knob defaults OFF** and the plain arms are bit-for-bit unchanged (a fixed-seed forward-pass hash was
>   checked after each edit); all existing test modules keep their counts (6/6, 2/2, 6/6, 12/12, 9/9, 7/7, 10/10,
>   10/10, 7/7, 14/14, plus the two new modules).

Written 2026-10-12 after `runs/RUN_011_discoverability_and_halting.md` (read its section 0 and section 12 first). **Nothing is committed and the user said
not to commit: never run `git add` or `git commit`.** All of `docs/`, `experiments/ssp_algebra/` and `runs/` is untracked. Read in this order: this file;
`runs/RUN_011` sections 0b, 0, 12, 11; `docs/11-ssp-algebra-controller-variants-and-tests.md` sections 12, 13, 14; then `handoffs/10-*.md` section 5 (the RL gate).
Confidence labels: **[measured]**, **[inferred]**, **[recalled, unchecked]**.

## 0. What the project is for now (the user's words, restated; repeat them back before changing the plan)

The experiments are **not just to find out how SSP behaves.** They exist to find a **capable solution that makes SSP usable as a memory module** inside larger
models solving real problems (RL agents on robotics-like tasks first). The user expects SSP to **match other modules, be more sample-efficient, and "maybe" more
robust**; any further advantage is a bonus, and none has been found so far. Three constraints govern every decision:

1. **Sutton's bitter lesson.** Do not over-handcraft. Prefer learning, search and self-supervision to hand-set structure. The user is **open to making parts of the
   model learnable** and to **changing the architecture**. Count the task-dependent constants of every method; fewer is better even at a small accuracy cost.
2. **Real-world utility.** A memory unit deployable in models doing actual problem solving: one default configuration, works from a default initialisation, plugs
   into a policy, small fixed state, constant per-step cost, no per-task tuning. Passing only on tasks hand-matched to SSP does not count.
3. **Match generic memories, be more sample-efficient, maybe more robust**, against *matched, trainable, incumbent* baselines at equal state, tuning budget and aid.

## 1. Where things stand [measured, all in RUN_011]

- **Discovery mechanism (settled).** The read address's **offset**, not its size, decides whether training starts: at index T=32 every offset >= 1 fails 0/10 and never
  moves (flat 0.25-0.34 for 1500 steps); every offset-0 cell passes (9/10 at >= 0.95) for scale errors 0 / 3% / 10%; a dead-zone offset is rescued exactly when the drift
  carries the error through zero inside the stream (3 pre-registered sign cells: 0.957 and 0.956 at 9/10, 0.637 at 4/10 for a late crossing). Cause: `kappa = sinc(pi*err)`
  at `length_scale = 1` is zero at every integer error, so a constant offset leaves every position equally uninformative.
- **Recipe `recipes.slope_only_cfg()`** (read head = ramp x (1 + 0.03 N(0,1)), offset exactly 0, fixed frequencies, no schedule, no range constant; `{"rung":1,"init":"affine",
  "affine_eps":0.03,"affine_offset_zero":true,"length_scale":1.0,"ls_mode":"fixed","lr":0.02,"lam_init":0.99,"temp":0.1}`): passes every variant inside the memory's capacity, 10 seeds, pods:

| variant | conv | seeds >= 0.95 | tokens to 0.90 | the schedule rule |
|---|---|---|---|---|
| index T=16 | 0.996 | 10/10 | 25.7k | 0.984, 10/10, 102.6k |
| index T=24 | 0.976 | 10/10 | 38.2k | 0.871, 0/10 |
| index T=32 | 0.956 | 9/10 | 52.0k | 0.833, 0/10 |
| index T=48 | 0.973 | 10/10 | 77.7k | 0.775, 0/10 |
| index T=64, 1024 floats | 0.991 | 10/10 | -- | 0.690 (512 floats) |
| index T=64, 512 floats | 0.952 | 6/10 | 152.7k | -- |
| delay 1-4 | 1.000 | 10/10 | 21.6k | 0.929, 7/10 |
| delay 1-8 | 0.997 | 10/10 | 18.4k | 0.999, 10/10, 128.4k |

  T=64 at 512 floats is the task's capacity (`D/4 ~ 64` against address range 62): the exact rung-0 address scores 0.952 (5/10) there and 0.992 (10/10) at 1024 floats. Token counts are
  `updates x batch x 8` for index T=16, at batch 64.
- **Pointer.** Rung 3 trained on **one hop**, `halting="cue_hop"` + `mask_hop_in_start=True` + per-hop aux 0.5 (`jobs/p4_combined_fixed.json`): **1.000 on held-out hops 4-8, 10/10
  seeds**, no hop count given; 0.513 without the aux. Iteration limit: 16 nodes pass, 32 fail (0.205 at 512 floats, 0.304 at 1024, 0/10) -- iteration-bound, not capacity-bound; 64-node
  cells were stopped on cost. **No generic baseline passes its own gate** (GRU 0.247 over 10 seeds at 15000 steps; `fwp`/`lru`/`tf_window` 0/30), so the pointer is a *capability
  demonstration*, not a matched-baseline win.
- **What does not work.** Batch size as a speed lever; multi-scale frequencies (0.35-0.89, discovery failures); relative and absolute affine priors; shape-randomised interface pretraining
  (scales with budget, fails transfer); the in-training probe auxiliary (a no-op, 0.274/0.275 vs 0.269); learned halting (the decode compresses its slope); variant A with a hand-wired path arm
  (MSE 0.147 vs GRU 0.049, `pose_attn` 0.101, LRU 0.135).
- **Honest standing of the claims.** `fwp` ties SSP on index with equal aid, so index is a harness check. There is **no matched-baseline sample-efficiency win** anywhere (the only speed
  ratio is against SSP's own earlier schedule). The recipe keeps four hand-set ingredients (offset 0, `eps = 0.03`, fixed frequencies, a time ramp). RL was **gated** at the time of writing; it has since been run -- see `runs/RUN_012` section 6 and the CURRENT PLAN block at the top of this file.
- **POPGym reference:** PPO + GRU on `RepeatPreviousEasy`: 0.683 at 632k steps, **not converged** (`experiments/ssp_algebra/ref_ppo/ref_gru_600k.json`; the venv at `/tmp/popgym_ref` is outside the repo and
  may be gone).

## 2. The key fact about `fwp` that changes the comparison [measured by reading the code, `arms_learned.py` `FWPArm`]

The `fwp` control is **plain linear attention**: `M = lam*M + v kT`, one learned scalar `lam`, no delta-rule correction (`v - M k`), no write strength, no input-dependent gate; `k_head` and `q_head` are
**separate untied** linear layers over the same fixed time features; `d = round(sqrt(budget))` (about 22-23 at 512 floats, so about `d` well-separated items). The fast-weight family used in practice
(linear attention is mathematically a fast-weight programmer: Schlag, Irie and Schmidhuber 2021; DeltaNet and gated variants add the delta rule and gates) is **stronger**. Descendants in production
models [recalled, unchecked: Qwen3-Next, Kimi Linear]. Consequences: (1) the "`fwp` ties SSP" result is about a weak control; (2) because `fwp`'s keys are untied it has the *same read-origin problem*, so its
tie says little; (3) **no claim of the form "SSP is better than / equal to the fast-weight family" can be made until a delta-rule, tied-key arm exists and passes its trainability gate.**

## 3. The plan for this round (details, pass rules and stops; rationale in docs/11 section 14)

General rules (all from this session's mistakes): write every pass rule **before** running; report a **constants scoreboard** (task-dependent constants, parameter count, state floats) next to accuracy;
every baseline passes its **trainability gate** on an easy variant before its number is used; equal state floats, parameters, updates, batch and tuning budget; **pod and local runs never in one table**;
set `--ep-len` explicitly and check tokens per update against the expected figure (index T=16: 511-514 per update at batch 64; T=32 about 1025; T=64 about 2049); a fresh `--tag` per protocol; never edit
an existing test or threshold; use 10 seeds for supervised pass/fail and at least 5 for RL, with bootstrap intervals.

## 3.0 Step 0: diagnostics before any method work

| id | question | do |
|---|---|---|
| 0a | Is the baseline the real incumbent? | Add `fwp_delta`: delta-rule write `M += beta (v - M k) kT`, **tied** k/q encoder (one network for write and read), `beta` and decay learnable (input-dependent gate as a flag); same `d*d` state floats. Run its trainability gate on index, delay and pointer easy variants. |
| 0b | Where is each arm's dead zone? | Retrieval loss against address error per arm and kernel (not gradient norm: that probe did not discriminate in RUN_011; the loss shape did). |
| 0c | Is the dead zone SSP-specific? | The offset grid (0, 0.5, 1, 3) at index T=32 for plain `fwp`, `fwp_delta` and the SSP arm. If untied `fwp` stalls too, discovery is a general property of key-addressed memories. |
| 0d | Why was the in-loop probe a no-op? | Does the aux gradient reach `w_time` / the read head? Does its loss fall? Does it correlate with task accuracy? (`harness.probe_aux_weight`, RUN_011 5.3.) |
| 0e | RL reference | PPO + GRU on `RepeatPrevious` (easy, medium, hard) to convergence, several seeds, with the random and optimal return recorded. |
| 0f | RL interface contract | Replace the GRU with `rl_memory.SSPMemory`: episode resets, truncated backpropagation, gradient to address parameters, per-step cost, a no-op memory behaves like a feed-forward policy. Extend `test_rl_memory.py` (new functions only). |
| 0g | Is the memory used in RL? | Zero the memory at evaluation; a linear probe for the lagged observation from the policy state. |

Stop rule: if 0a or 0c changes the picture (for example a delta-rule arm clearly beats SSP on index and delay at equal budget, or plain `fwp` shows the same dead zone), report it and ask the user before continuing.

## 3.1 Step 1: supervised track, one change at a time

Baselines: GRU, LRU, plain `fwp`, `fwp_delta`, the current recipe (mark it **handcrafted: 4 constants**). SSP candidates, each built **alone** on the current harness arms (`SSPIndexLearned` / `SSPPointerLearned`
in `arms_learned.py`; new knobs default OFF so earlier results reproduce):

1. **Tied encoder with relative addressing** (the idea to bet on). One encoder maps content (+ time) to an address, used for the written item and the query. Relative read = current-time address
   bound with the inverse of a lag address (SSP's shift is a binding, `ssp_ops.py`), so there is no read origin of its own. What is learned: the lag (scalar from a task feature, or from content). Test it on the
   **offset grid**; the pass rule below needs it to pass at offset 3 with no prior. Risk: the lag producer may meet its own dead zone (the earlier learned embedding scored 0.362); gains only when reads are
   relative to writes.
2. **A kernel with no dead zone** (replace `sinc(pi*err)`; heavy-tailed or non-oscillating; or widen only in the training loss). First read the history of `ls_mode="learned"` (RUN_008: 0.778, does not tune itself). Risk: blurs neighbours; any annealing is a constant.
3. **K parallel read heads** with random offsets (compute for a constant; still needs an offset range). Report the combination's behaviour: does it learn to ignore the missing heads?
4. **Delta-rule write for the SSP bundle** (capacity at fixed floats; the T=64-at-512-floats limit looks like crosstalk) and **learned write/forget gates**; **multi-timescale banks** so `lam_init` need not be known.
5. **Learned initialisation across a task family** (hold out lengths and delays). Risk: shape-randomised pretraining did not transfer.
6. **Learned frequencies / kernel width**, only after 1 or 2 works (their gradient has the same dead zone).

Tasks: index and delay at real lengths and lag ranges; **delay with distractors** (selective writing); **length extrapolation** (train T, test 2T; never tested); **content-keyed recall** (a non-positional family,
so the design is not hand-matched to position); the offset grid at offsets 0, 1, 3. Report two columns per arm: a **pre-chosen default configuration** and a **tuned** one at equal tuning budget.
Speed claims keep `--select aulc --min-conv 0.95 --search-seeds 3`, quoting tokens with their batch.

**Pass rules (fixed here, before running):**
- *Removes the dead zone:* >= 9 of 10 seeds at >= 0.95 at **offset 3 with no prior**, at index T=32. Passing only at offset 0 repeats what is already known.
- *Competitive:* within 5 points absolute of the best baseline at convergence, at equal state, tuning budget and aid (the 4.5 tolerance in docs/11).
- *Speed claim:* a ratio over the best **baseline** (never over SSP's old schedule), with the batch stated.
- *Simplicity:* fewer task-dependent constants than the current recipe (4) to count as progress on the bitter-lesson axis.
- *Stop:* if no SSP candidate both removes the dead zone and is competitive, stop and report.

## 3.2 Step 2: RL track on POPGym

`RepeatPrevious` easy, medium, hard (a delay task: the supervised recipe applies directly; difficulty changes the lag), then other memory tasks after verifying the exact POPGym list [recalled, unchecked: RepeatFirst,
Autoencode, Concentration, CountRecall]. Arms (encoder, PPO hyperparameters and state size matched): PPO + GRU, PPO + `fwp`, PPO + `fwp_delta`, PPO + SSP memory; each **with and without** the label-free recall
auxiliary (`SSPMemory.aux_loss`, trained on the stream's own tokens: zero extra samples). Several million steps (the GRU's 0.683 at 632k is not converged), at least 5 seeds; report final return, steps to a stated
threshold, and learning-curve area, each with intervals, and the constants scoreboard. **Gate:** SSP at parity with the GRU at the final return, any speed claim at a stated margin; failure on the task where SSP should be
most natural stops the RL track. Cap steps and seeds with the user before launching (the budget is the largest cost).

**POPGym-native rule (decision 5, section 4):** build this track on POPGym's own PPO configuration, models, environments and reporting. First task for the agent here: read the installed POPGym
and `ref_ppo/README.md`, list which memory models and environments ship, check whether its fast-weight model uses the delta rule, and record it. The SSP arm is an RLlib `ModelV2` plugged into
that loop; our `rl_memory.SSPMemory` is the component it wraps. Use the budget proposal in section 4, decision 3, and ask the user to confirm it against measured throughput.

## 3.3 Step 3: the spatial recall task (the gate for robotics)

Build a small, partially observed 2-D world: a landmark is shown at the start, the agent moves, and the model must report the landmark in its **current** frame (supervised first, then an RL version). Arms: GRU, `fwp_delta`, an
attention window, and SSP using **relative addressing as the motion update** (translate the bundle by `phi(-d)`: a binding, O(D), no rewriting). This tests the claim most likely to be unique to SSP, and the one earlier test
(variant A, hand-wired path arm) failed; `pose_attn` already exists as a competitor in `arms.py`. **A loss here changes the plan** (the robotics argument weakens); say so to the user rather than reinterpreting.

## 3.4 Ablations to attach to any positive result

Memory zeroed at evaluation; tied encoder off; auxiliary off; delta rule off; kernel swapped back to sinc.

## 4. Decisions (made by the user and the assistant on 2026-10-12; they replace the open questions)

1. **K random heads:** allowed only as a fallback. Try the tied encoder with relative addressing and a dead-zone-free kernel first. K heads are acceptable only with one default
   K (proposed 4-8) that works on every task with an offset range that does not depend on the task; count K and the range as constants and report the extra compute.
2. **The success rule (the user's words: as good as the others on the other metrics, better on at least one metric with real-world utility).** Fix before any run: *parity* =
   within 5 points absolute of the best matched baseline at convergence, at equal state, tuning budget and aid, on every task. *Better on one metric* = a metric from a list
   written in advance (sample efficiency: >= 2x fewer tokens or steps to a stated threshold than the best baseline; length extrapolation past the trained length; cue-corruption
   or distractor robustness; state size or per-step cost at equal accuracy), with a margin of at least 2 pooled standard errors **and** a meaningful absolute margin. Report every
   listed metric (not only the winner) and say how many were tried. Parity without a won metric means "a working memory with no demonstrated reason to prefer it".
3. **RL budget (proposal, confirm against measured throughput).** First measure steps per second and cost per run, and report an estimate to the user before launching anything.
   Then a pilot of 1M steps x 3 seeds x 4 arms on `RepeatPrevious` easy; only if SSP is learning comparably, a confirmatory run of 5 seeds to at most 5M steps on easy and medium.
   Set a hard cap on total pod-hours first and stop any run past it. The numbers are starting values, not measured costs.
4. **The spatial recall task is binding for the robotics direction only.** Supervised version first. If SSP with relative addressing is below both the GRU and `fwp_delta` by more
   than the parity margin, pause the *robotics* argument and tell the user; the general memory-module track continues on other tasks, since decision 2 can be met elsewhere.
5. **Use POPGym's built-in components for everything RL when possible (the user's rule, for fairness).** POPGym's environments, its PPO configuration and loop through `PPOConfig`
   with POPGym's own models (RUN_007 section 3: works with the installed ray), its evaluation and reporting, and its own baseline models, including its fast-weight and other memory
   models if present [recalled, unchecked: POPGym ships GRU, LSTM, MLP, positional MLP, fast-weight and linear-transformer, LMU, S4D and others; **check which fast-weight variant it is
   and whether it has the delta rule**; if it does, it may serve as `fwp_delta` for RL]. Add SSP as a POPGym-compatible RLlib `ModelV2`, not through a different loop. **Never
   compare against the in-repo A2C loop.** We build only the SSP model class and tasks POPGym lacks (the spatial task), run through the same PPO path. A purpose-built environment is used
   only where POPGym has no suitable task. This closes the `ModelV2` versus purpose-built question for the memory-module track.
6. The repo question (move the folder?) stays deferred until a learned arm passes decision 2.

## 5. Operations and traps (all learned the hard way)

- **Pods.** Only touch pods named `ssp-*`; never `eval-v3-*`. Launch with `.venv/bin/python scripts/ssp_pod_batch.py launch|status|collect <jobs.json>`; run `launch --dry-run` first. Declared `rows` must equal
  `len(arms)*(configs*search_seeds+seeds)`. Report with `.venv/bin/python experiments/ssp_algebra/report_jobs.py <jobs.json...>`. Use the repo `.venv` python (system python lacks boto3/torch). `/tmp/reap_pods.py` (deletes `ssp-*` by prefix)
  is outside the repo and may be gone. Do not leave pods idle; stop any that run too long. A 64-node pointer cell was too slow and was deleted unfinished.
- **Local runs:** from `experiments/ssp_algebra/`, `OMP_NUM_THREADS=1 ../../.venv/bin/python <script>` (mandatory: unpinned workers oversubscribe ~20x slower). 16-core machine.
- **Pod and local runs are not numerically interchangeable** (a four-row replay is bit-identical on a pod and differs locally by up to 0.43, `replay_check.py`). Tabulate one machine class at a time and record where a tag ran.
- **`--ep-len` must be explicit** (the default is 16; the RUN_010 index "T=24..64" jobs omitted it and ran at T=16). Check the stored `tokens` per update.
- `--arm-cfg` JSON must contain no spaces. Every protocol gets a fresh `--tag`.
- **Never edit or loosen an existing test, threshold or the pre-registered targets;** add new test functions or files. Do not add tests with blind global string replacements (RUN_011 section 11): after touching a test file,
  run the full module and check the **test count**, not only the pass line. All 17 modules passed at the end of RUN_011: `test_harness`, `test_fixes`, `test_measurement`, `test_learned`, `test_analysis`, `test_regularisation`,
  `test_select`, `test_recipes` (10), `test_pretrain`, `test_pointer` (14), `test_trainability`, `selftest`, `test_halting`, `test_multiscale`, `test_affine` (9), `test_pretrain_shapes`, `test_rl_memory`.
- **A probe that does not discriminate is not evidence:** the gradient-norm basin probe did not separate the hypotheses; the retrieval-loss-against-address-error shape did. Design the discriminating cell first, write the
  predictions down, then run.
- **Do not state the recipe's record as "passes every variant"**; the accurate form is "passes every variant run inside the memory's capacity" (T=64 needs 1024 floats).
- **Do not claim a speed win against SSP's own earlier recipe as a result**; the claim needs a baseline.

## 6. Where the code is

`experiments/ssp_algebra/` (README.md lists files): `arms.py` (mlp, posmlp, gru, lru, tf_full, tf_window, kernel_attn, looped_tf, ssp, `pose_attn`), `arms_learned.py` (`SSPIndexLearned` with `affine_eps/abs/slope_err/offset/offset_zero`,
`omega_kind`, `ls_hi`, `fine_frac`; `SSPPointerLearned` with `halting` in {`learned`,`counter`,`cue_hop`} and `mask_hop_in_start`; `FWPArm` with interface hooks; `LEARNED_ARMS = ("ssp_index_learned","fwp","ssp_pointer_learned","fwp_delta","assoc_ssp","assoc_fwp")`),
`harness.py` (arm construction at ~line 373, `pretrain_shapes`, `probe_aux_weight`), `recipes.py` (`slope_only_cfg`, `DEFAULT_SLOPE_EPS=0.03`, `continuous_address_cfg`), `ssp_ops.py` (`phi`, `pos`, bind/unbind/shift, `make_omega`),
`tasks_a/b/c.py`, `rl_memory.py` (the policy-memory interface stub: `reset`, `write`, `read`, `read_from`, `aux_loss`, `freeze_interface`), `run_sweep.py` (flags: `--variants --arms --budget --configs --search-seeds --select
--min-conv --seeds --steps --batch --ep-len --n-nodes --train-lags --train-hops --max-hops --tag --arm-cfg --label --robust --pool`), `report_jobs.py`, `replay_check.py`, `diag_address.py`, `diag_pointer.py`. Job lists:
`experiments/ssp_algebra/jobs/` (`p5f_slopeonly003.json` is the recipe validation; `p5_basin*.json` the offset grid and sign cells; `p4_combined_fixed.json` the pointer; `p6_gate.json` and `p5cap_nodes.json` the gates and node limit).
Results: `experiments/ssp_algebra/results/` (`sweep_p5f_*`, `sweep_p5_rule_*`, `sweep_p5g_*`, `sweep_p5cap_*`, `sweep_p4com_fixed.jsonl`, `sweep_p6_gru_gate.jsonl`, `sweep_p5b_*`/`p5c_*`/`p5d_*` grid and sign cells).
**What exists as of RUN_012 (do not rebuild these):** the `fwp_delta` arm (`rule="delta"`, tied encoder, absolute read time); the tied / relative-addressing mode (`addr_mode="tied"`, `read_residual`, `read_feat_scale`, `read_feat_shift`); the dead-zone-free kernel (`omega_kind="cauchy"` in `ssp_ops.make_omega`); K read heads (`read_heads=K`, half-index spread); the delta write for the SSP bundle (`write_rule="delta"`, needs `beta_init=-4.0`); input-dependent gates (`gate="input"`) and multi-timescale banks (`banks=K`); the **content-keyed family** (`tasks_d.AssocTask`, `arms_assoc.AssocSSPArm` / `AssocFWPArm`, variant `"assoc"` in `run_sweep.make_task` and a branch in `harness.splits_for`); the RL model (`rl/ssp_model.py`, `rl/ssp_tied_model.py` and their tests). Job lists: `p7_grid_fwp`, `p8_candidates`, `p10_tied`, `p11_axes`, `p12_banks`, `p13_gate_clean`, `p9_assoc_local`. **Still to be written:** `tasks_f.py` (delay with distractors), `tasks_e.py` (spatial recall) and `arms_spatial.py`, `tasks_d` mixed mode plus the content-position arm, the `FWPArm` gate/banks knobs, the lag-bank RL arm, and a `REPORT_TIED.md` from the delegated track.

## 7. First actions

1. Read the documents in the order at the top; repeat the goal and the three constraints (section 0) back to the user and confirm the open decisions in section 4 that block you (the RL budget first).
2. Run the existing test modules once to confirm the baseline state, locally, before changing anything.
3. Step 0a (`fwp_delta` and its gate), then 0c (the offset grid for every arm). These are cheap and decide whether the rest is well posed.
4. Write the pass rules for each candidate into the job list or a RUN_012 file **before** launching, keep a fresh tag per protocol, and report with the constants scoreboard.
