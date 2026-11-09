# Handoff 03: preliminary results, conclusions, eval plan, SSP strengths, next experiments

Written 2026-10-05 (evening). Audience: the next agent, who takes over once the DQN search/filter/final-eval
pipeline has finished. Read after `docs/handoffs/02-dqn-v4-greedy-refilter-plan.md` (which also contains the
build notes for the chained per-cell pipeline) and `docs/07-decision-log.md` (entries 14-18).

The user's standing preferences, stated in this conversation:
- Do not waste compute. Every proposed metric or run must answer a question. If you cannot say what result
  would change a decision, do not run it.
- No raw-input ("none" representation) baseline. The user explicitly declined it.
- Search and filtering are being wrapped up. The remaining work is on the EVALUATION side, and (secondarily)
  on experiments that show where SSP is actually good.
- The user is open to re-running evaluation for A2C too, via a PyTorch port if needed.
- Report honestly. Several earlier hypotheses in this conversation were wrong or weakened (see "Retractions").

Nothing in this repo is committed to git. `scripts/.s3cfg/` contains R2 credentials and is now gitignored.

---

# 1. State of the pipeline when this was written

All 18 DQN cells (3 representations x 3 centering modes x linear/deep) run one RunPod CPU pod each
(4 vCPU, $0.12/hr, names `dqn-v2-<rep>-<centering>-<arch>`). Each pod does: top-up search -> greedy filter
(top 20 by last-5 greedy mean x 10 fresh seeds) -> final eval (winner, 20 fresh seeds) -> pushes to R2 -> deletes
itself (watcher deletes it after a 12-minute grace if it does not).

UPDATE (2026-10-06): ALL 18 cells are complete. The last cell (`tile_coding/value/deep`) wrote its summary to R2 at
10:55 UTC; the watcher deleted its pod at about 11:24 UTC (it did not self-delete). No `dqn-v2-*` pods remain. The full
18-cell results, extra metrics, and R2 locations are in `runs/RUN_003.md`; the table in 2.2 below is final.
(The text below was written when 16 of 18 cells were done; ignore its "pending" remarks.)

HOW TO VERIFY THE END STATE (do this first):
- Per-cell summary exists: `stage0_cartpole/refilter_v2/<study>/final_eval_summary.json` for all 18 studies
  (`study = dqn_<rep>_<centering>[_deep]_v4_hparam_search`; list in `scripts/launch_cell_pod.py::ALL_CELLS`).
- `runpodctl pod list` (with `RUNPOD_API_KEY` exported from `.env`) shows NO `dqn-v2-*` pods. Delete any that
  remain (`DELETE https://api.runpod.io/v2/pods/<id>`; see `scripts/watch_cells.py`). Pods named `cerb-*` are
  NOT ours and must not be touched.
- The watchers (`scripts/watch_cells.py`) are local background processes. A laptop reboot kills them (this
  happened once; the pods kept running and results kept landing on R2). If cells are still running and no
  watcher is alive, restart one: `python scripts/watch_cells.py --cells <rep/centering/arch,...> --max-hours N`.

R2 layout (bucket in `.env`; never copy credentials into docs):
- `stage0_cartpole/refilter_v2/<study>/{reeval_v2_runs.jsonl, reeval_v2_results.json, final_eval_runs.jsonl,
  final_eval_summary.json}`: the new filter and final-eval outputs. `final_eval_summary.json` holds the
  per-seed scores, `eval_history` (15 greedy checkpoints per seed), timing, size, pod info, and the winner's
  search params.
- `stage0_cartpole/best_params/dqn_<rep>_<centering>[_deep]_v2.yaml`: the winners.
- `stage0_cartpole/optuna_dbs/optuna_studies_<study>.db`: the search DBs (intermediate values per trial are the
  15 greedy checkpoints). Backups of all 18 as of 2026-10-05 morning: `stage0_cartpole/backup_2026-10-05/optuna_dbs/`.
- A2C legacy winners: `stage0_cartpole/best_params/{discrete,ssp,tile_coding}_{none,simple,value}.yaml`.

Known imperfections: the CSV tree for `discrete/none/linear`'s final eval is incomplete on R2 (52 of ~60 files;
a watcher deleted the pod before its last sync). The JSONL and summary for that cell are complete. The pod
self-delete via the RunPod API was not confirmed to work; the watcher is the backstop.

---

# 2. Preliminary results

## 2.1 Protocol (what the numbers mean)

- Task: CartPole-v1, 1000 training episodes per run, max 500 steps. Greedy (argmax, no epsilon) evaluation at
  15 checkpoints (every 66 episodes, 10 episodes each).
- Search objective: last greedy checkpoint, single seed per trial (500 trials per cell), TPE + MedianPruner.
- Filter: candidates = top 20 completed trials by mean of the last 5 greedy checkpoints; each re-run on 10 fresh
  seeds (1000+); per-seed score = mean of its last 5 greedy checkpoints; winner = best mean, ties to lower np.std.
- Final eval: the winner on 20 fresh seeds (2000+), same score. Reported as mean +/- np.std across seeds,
  "solved" = share of seeds whose score is >= 475. Initial weights follow the run seed in filter/final (the
  search used the legacy fixed init). The environment reset is unseeded, so re-runs are NOT bit-reproducible.
- DQN settings that were FIXED for every cell and never tuned: replay buffer 5000, batch 64, target update every
  10 gradient updates, learning_starts 1000, train_freq 4, gamma 0.99, Adam, grad-clip 10.
  (`sample_dqn_params` exists in `scripts/run_hparam_search_dqn.py` but is not called by the objective.)

## 2.2 Final eval: 20 fresh seeds, greedy, last-5 mean (filter winner's own 10-seed mean in brackets)

| Cell | Final eval | Solved | Linear/deep dims | Params |
|---|---|---|---|---|
| tile_coding/none/linear | 498.6 +/- 3.6 (499.6) | 100% | 32,768 | 65,538 |
| tile_coding/simple/linear | 496.3 +/- 7.1 (498.6) | 100% | 131,072 | 262,146 |
| tile_coding/value/linear | 495.4 +/- 6.6 (494.9) | 100% | 65,536 | 131,074 |
| tile_coding/none/deep | 489.7 +/- 20.1 (493.9) | 85% | 32,768 | 2,101,506 |
| ssp/none/deep | 486.4 +/- 21.4 (496.2) | 80% | 951 | 65,218 |
| tile_coding/simple/deep | 479.5 +/- 18.4 (461.7) | 60% | 65,536 | 4,198,658 |
| ssp/value/deep | 474.6 +/- 26.7 (491.7) | 60% | 201 | 17,218 |
| ssp/value/linear | 468.7 +/- 34.6 (492.5) | 60% | 1,001 | 2,004 |
| ssp/simple/deep | 363.6 +/- 67.3 (388.4) | 0% | 651 | 46,018 |
| ssp/simple/linear | 311.5 +/- 101.6 (317.0) | 5% | 251 | 504 |
| discrete/value/linear | 265.8 +/- 62.2 (299.4) | 0% | 14,641 | 29,284 |
| discrete/none/linear | 260.4 +/- 68.0 (299.5) | 0% | 50,625 | 101,252 |
| discrete/simple/deep | 259.4 +/- 57.0 (294.6) | 0% | 50,625 | 3,244,354 |
| discrete/simple/linear | 255.1 +/- 50.5 (263.6) | 0% | 50,625 | 101,252 |
| discrete/none/deep | 232.9 +/- 53.8 (280.5) | 0% | 50,625 | 3,244,354 |
| discrete/value/deep | 220.8 +/- 45.1 (273.9) | 0% | 130,321 | 8,344,898 |
| ssp/none/linear | 469.0 +/- 30.7 (478.1) | 55% | 1,001 | 2,004 |
| tile_coding/value/deep | 498.0 +/- 2.8 (499.7) | 100% | 65,536 | 4,198,658 |

Winner's-curse gap (filter mean minus final mean) is real and large for SSP (e.g. ssp/value/linear 492.5 -> 468.7)
and discrete (about 40), small for tile coding. Cell values should be read from the FINAL eval, not the filter.

## 2.3 Other measurements already made (from data on R2; scripts were ad hoc, re-derive as needed)

- Sample efficiency (median over seeds; checkpoints are 66 episodes apart): first greedy checkpoint >= 475 /
  episode from which every later checkpoint stays >= 475: tile linear 198-396 / about 462 (all three centerings);
  tile none/deep 264 / 528; tile simple/deep 198 / 825; ssp/none/deep 396 / 561; ssp/value/linear 396 / 726;
  ssp/value/deep 330 / 726; ssp/simple/deep 264 / 924; ssp/simple/linear 660 / 990; discrete 594-825 / 990.
- Ease of tuning (share of the first 50 search trials with last-5 greedy mean >= 450): tile none/value 38-46%,
  tile simple/linear 28%, ssp none/value 16-22%, ssp simple 0-2%, discrete 0-2%.
- "Learn-then-collapse" rate under DQN: among trials whose greedy eval EVER reached >= 475, the share ending
  with a last-5 mean < 400. Tile linear: none 3%, value 4%, simple 12%. Tile deep: 23% / 9% / 75%. SSP: none
  34% (linear and deep), value 18% (linear) / 34% (deep), simple 91% / 88%. Discrete: 83-100% (but few trials
  ever reach 475).
- SSP dimension vs score (search data, completed trials): Spearman(dim, score) is weak (|rho| <= 0.11) in five
  of six SSP cells; ssp/value/linear is the exception (+0.34: 0% of trials with < 500 dims reach 450, 46-59% of
  those above). Below about 500 dimensions the median score is much worse in ssp/none linear (19 vs 422-433)
  and ssp/none/deep (257 vs 377-459). The ssp/value/deep winner has only 201 dims and scores 474.6. These are
  confounded with the other hyperparameters and with TPE's sampling.
- Winners' centering parameters: all six `simple` winners have beta at or near the lower search bound
  (1.3e-5 to 9.2e-5; range 1e-5 to 1e-1). The `value` winners' eta are interior (0.001-0.058; range 1e-4..1).
- Search ranges that matter for size claims: SSP `n_rotates` 4-20 with `n_scales` fixed at 5, so
  ssp_dim = 2*(n_rotates*n_scales)*(d+1)+1 = 50*n_rotates+1, i.e. 201-1001 dims; tile coding `iht_size`
  in {32768, 65536, 131072}, `num_tilings` {16, 32}, `tiles_per_dim` {10, 12, 16}. The two size ranges do NOT overlap
  (tile coding could never go below 65k linear params; SSP ranged from about 400 to 2,000 linear params).
- Timing: per-run wall times were measured with 4 runs sharing one pod and different winning configs; they are
  not clean speed comparisons. Time per environment step is roughly comparable across representations
  (about 0.3-0.9 ms/step, deep nets and tile/simple/deep higher).

## 2.4 Pilot findings that still apply (details in handoff 02)

In ssp/simple/deep the search rank did not predict fresh-seed performance (Spearman 0.05), about 90% of score
variance was seed noise (ICC 0.10), and 3-seed cuts are essentially coin flips among top candidates. That is why
the filter is flat 20 x 10 and why the final eval (20 fresh seeds) is the number to report. Varying initial
weights with the seed did not noticeably change mean or spread. A good ssp/none/deep config (#154) with simple
centering switched on degraded dose-dependently (beta 1e-5: 469; 1e-4: 338; 1e-2: 271) from 497 without it.

## 2.5 Legacy A2C results (Nengo path, NOT comparable to the DQN numbers)

Best search values from the old A2C search: online rolling training return (last 100 episodes), TPE with a
multi-seed protocol and no greedy eval, produced before any of the 2026-10 methodology changes.

| | none | simple | value |
|---|---|---|---|
| SSP | 498.4 | 488.6 | 499.4 |
| Tile coding | 495.0 | 490.9 | 495.1 |
| Discrete | 118.5 | 378.4 | 267.6 |

IMPORTANT FACTS about the A2C results: the A2C search config sets `state_neurons: null` and the script never
samples a neuron count, so these winners are a plain linear readout on the representation (no Nengo neuron
layer), same as our `linear` DQN. Their yaml files hold only `eps`, `lr` (0.4-0.8, plain SGD TD(0)),
`length_scale`, `n_rotates` (SSP), tile settings, and the centering parameter; fixed values (trials=1000,
act_dis=0.9, state_dis=0.99, dynamic_epsilon=True) are in `BASE_PARAMS` of `scripts/run_hparam_search_a2c.py`.
RUN_001 recorded CVs across seeds of about 0.1% (SSP) and 0.2% (tile coding), i.e. both essentially zero.

---

# 3. Conclusions we reached in this conversation

Confidence is noted. "DQN setup" = the fixed settings in 2.1.

1. **Tile coding >= SSP on score under DQN on CartPole** (high confidence for the completed pairs): none/deep
   tie (489.7 vs 486.4); simple/linear 496 vs 312 and simple/deep 480 vs 364 (large); value/linear 495 vs 469
   (about 27 points, larger than the standard error of about 8). The pairs `none/linear` (SSP pending) and
   `value/deep` (tile pending) are open. SSP is a solid representation (80% solved for ssp/none/deep) but not
   better than tile coding here. Discrete is far worse than both, and deep networks do not rescue it
   (232-259 vs 255-266 linear).
2. **The "SSP is 30-65x smaller" claim is NOT yet demonstrated.** The winners differ by 30-65x in parameters
   (e.g. ssp/value/linear 2,004 params at 468.7 vs tile value linear 131,074 at 495.4), but tile coding was
   never allowed below its 32,768-entry hash-table floor, so the comparison is an artifact of the search ranges.
   At equal parameter count tile coding already looks as good (tile none linear: 65,538 params, 498.6, vs
   ssp/none/deep: 65,218, 486.4). A size-matched sweep is required before any size claim.
3. **SSP dimension has a threshold effect, not "no effect"** (medium confidence, confounded): see 2.3. The
   repo's CogSci-2022 MiniGrid finding (64-532 dims, little effect) does not transfer cleanly to our CartPole
   DQN data. Dims below 201 were never tested.
4. **SSP is harder to tune than tile coding** (about 2x fewer good trials in the first 50) but far easier than
   discrete.
5. **SSP is less stable than tile coding under DQN** (collapse rates 5-10x higher on linear). Hypothesis (not
   tested here): global dense features interfere (updates on success-only replay data overwrite values of failure
   states) and the fixed DQN settings (replay 5000 holds about 10 successful episodes; target update every 10 updates)
   make it worse. Literature support by analogy (neural-network features, not SSP): Liu et al. 2019 (sparse /
   local features avoid interference and keep bootstrap targets stable; tile coding is their baseline), Liu et al.
   2023 (an interference measure correlates with control instability; CartPole chosen because agents forget there;
   target networks and smaller layers reduce interference). Direct test: see Section 5 (DQN-settings test).
6. **The A2C-vs-DQN SSP difference is algorithm/protocol, not a neuron layer.** Under legacy A2C, SSP >= tile
   coding in all three centerings. Both are linear readouts. Differences: on-policy TD(0) with plain SGD, no replay,
   no target network, online-return metric. (Retraction: an earlier hypothesis that a fixed random neuron layer
   explained the gap was wrong for these results; see Retractions.)
7. **Centering.**
   - Code matches Naik et al. 2024 (eq. 4 simple, Algorithm 3 value): no bug found.
   - The paper states centering is not applicable to episodic tasks and converts MuJoCo to continuing tasks.
     CartPole pays +1 every step, so simple centering drives the centered reward to zero and removes the failure
     signal; shown dose-dependently on a good config (2.4).
   - All six `simple` winners pin beta near the lower bound, so the simple cells measure how well a representation
     tolerates nearly-disabled centering, not centering. Tile coding works at beta about 2.6e-5.
   - Under legacy A2C simple centering is benign (SSP 488.6, tile 490.9, discrete 378 > none 118). Hypothesis:
     in A2C the policy is a separate parameter set that freezes when the centered TD error goes to zero, while in
     DQN the greedy policy is the argmax of values that decay toward zero. If true, decision-log entry 17
     ("centering hurts on episodic CartPole") applies to DQN, not to all algorithms. The decision log has NOT
     been updated with this caveat; do so when A2C is re-evaluated.
   - Value-centering cells behaved well (ssp 469-475, tile 495, discrete 266).
8. **Prior literature** (alphaXiv + web; not exhaustive): the user's ICCM 2022 paper compared HexSSP only to
   tabular discretization and a deep A2C MLP (not tile coding); CartPole HexSSP CI [496.9, 500.0] over 10 runs,
   150x lower terminal variance than deep A2C (sigma^2 127.8 vs 20,966; 20 seeds), and robustness when RatBox
   speed was halved while tabular failed. Their shallow network was SSP -> fixed random ReLU layer (1,024-4,096
   neurons) -> learned readout, trained with A2C; ours is a linear readout. Related work found: Komer & Eliasmith
   2020 (SSP beat six other encodings for supervised navigation), Konidaris et al. 2011 (Fourier basis beats RBF
   and polynomial; full basis grows as (n+1)^d, mitigated by uncoupled/limited-coupling bases), Chouinard 2026
   (arXiv 2605.08458: the default HexSSP "hypergeometric" kernel flattens as input dimension grows at fixed
   length-scale; the Gaussian-kernel variant does not), Furlong et al. 2026 (Nature Comms; SSP embeddings keep
   fixed size under binding, constant-size sample selection, 60-200x less compute and 30-188x less energy per
   sample on neuromorphic hardware, for Bayesian optimization, not RL), Krausse et al. 2025 (grid-cell VSA with
   binding; no RL), Mir et al. 2024 (hexagonal vs square grids perform the same in a memory-recall model),
   Damberger et al. 2025 (SSP place-cell representations for adaptive control of a 3-link arm). Hyperdimensional
   RL for edge exists (HDPG, edge-friendly HDRL; abstracts only). No paper found testing SSP vs tile coding at
   matched size in RL, or SSP dimension scaling in RL (weak evidence of absence: alphaXiv covers arXiv only).
9. **Seed-to-seed reliability is NOT a place SSP shows strength here** (high confidence): DQN tile std 3.6-7 vs
   SSP 21-35 (up to about 100 for simple); A2C CVs are about 0 for both. Do not run it as a separate experiment;
   just report what the final eval already measured.
10. **Not worth running** (user decisions or low value): a raw-input "none" representation baseline; a 500-trial
    retune of anything; a full multi-objective size search (a sweep is cheaper and answers the same question).

## Retractions / corrections made during the conversation

- The A2C neuron-layer hypothesis (Section 3, item 6) was wrong for the grid's A2C numbers.
- "SSP dimension does not have huge effects" was imported from the user's MiniGrid repo and is only partly
  supported by our CartPole data (item 3).
- The size advantage (item 2) looked like a result but is a search-range artifact.
- An early cost estimate of "a few hours" for a multi-objective search was too low for the code as built
  (one pod runs trials serially: 5-24 hours per 500-trial search).

---

# 4. Evaluation plan (for after the search/filter finishes)

Goal: find out where SSP genuinely helps, with metrics tied to specific questions, without disturbing the
existing headline numbers. Every item below states the question it answers and what result would change what we
believe or do. Cost figures are estimates from the earlier final-eval pods (about 13 minutes wall per cell for
20 seeds on discrete, tens of minutes for the heavier cells at $0.12/hr per pod) and are UNVERIFIED for the new
suite; time one cell first.

Keep unchanged: greedy, last-5 mean, 20 fresh seeds, mean +/- np.std, share solved. Add, never replace.

## 4.0 Free analyses (no new runs; do these first)

From `final_eval_runs.jsonl` / `final_eval_summary.json` for all 18 cells:
- Stability after learning: the episode from which every checkpoint stays >= 475, the share of checkpoints >= 475
  after first reaching it, and the largest drop from a seed's peak.
  Question: is a lower mean caused by never learning or by learning then degrading?
- Per-seed distributions (dots), share solved, a bootstrap CI on each cell's mean, and paired (by seed index)
  bootstrap comparisons for the key pairs (tile vs SSP at each centering/architecture).
  Question: are the tile-vs-SSP differences larger than noise? (With 20 seeds the SE is about 1-5 for strong
  cells and about 15-20 for weak ones.)
- Time-to-solve and sample efficiency curves from `eval_history`.

## 4.1 One retraining pass that SAVES POLICIES (the enabler for everything below)

The final-eval policies were not saved. Add to the final-eval step (`final_evaluate` in
`scripts/run_hparam_search_dqn.py`, worker `scripts/parallel_jobs.py::eval_worker`):
- Save, per seed, the trained network `state_dict`, the exact parameter dict, the seed, AND the
  representation state, not just the weights:
  - Tile coding: the IHT (index hash table) dictionary. Tile indices are assigned on first encounter, so the
    policy is meaningless without it.
  - SSP: the phase/encoding matrix (the rotations are sampled; confirm whether they are seeded).
  - Discrete: only the bin edges/params.
- Verify the checkpoint round-trips BEFORE relying on it: reload and re-evaluate a few seeds; the greedy score must
  match the original within noise. If it does not, fix the save before running the suite.
- Store the per-episode returns of the 100-episode final-policy eval (today only the mean is kept).
  Question: what is the worst-episode / worst-10% behavior (a policy with mean 480 can hide rare failures)?
- For centering cells, log R-bar and the mean max-Q on a fixed probe set at each checkpoint.
  Question: does simple centering really drive R-bar to about 1 and the values toward 0 as hypothesized
  (conclusion 7)? This is the cheapest direct test of that explanation.
- Push the weights to a NEW R2 prefix (suggest `stage0_cartpole/eval_v3/<study>/policies/`); do not overwrite
  `refilter_v2`.
- Since re-training re-samples (unseeded env reset), treat these 20 seeds as a fresh sample and report the
  difference from the first final eval as a free replicate (it also measures run-to-run reproducibility).

## 4.2 Robustness suite (test-time only, on the saved policies; greedy)

This is the user's main interest and the cheap place SSP's smooth kernel could show. It mirrors the user's
RatBox protocol: tune at the original dynamics, change them at test time with no retraining.
- Physical shifts (CartPole attributes `gravity`, `masscart`, `masspole`, `length`, `force_mag`; derived
  `total_mass` and `polemass_length` must be recomputed when masscart/masspole/length change): pole length
  x0.5, x1.5, x2; pole mass and cart mass x0.5, x2; gravity x0.8, x1.2; force magnitude x0.7, x1.3.
- Observation noise: Gaussian with sigma = {0.01, 0.05, 0.1} x the per-dimension std of states along a clean
  rollout (scale-aware; the four CartPole dimensions have very different scales).
- Control imperfections: action delay of 1 and 2 steps; 5% random actions (also the planned optional probe).
- Wider initial states: reset bounds widened (e.g. +/-0.1 and +/-0.2; confirm the installed Gymnasium version
  supports reset options for CartPole before relying on this).
- 20 episodes per condition per seed; include the unperturbed condition as the reference.
- Metrics: score retained (% of the unperturbed score), the area under the degradation curve per perturbation
  family, and the worst condition. Report per cell and per (representation, centering, architecture).
- Pre-declared interpretation (so we do not rationalize afterwards): SSP's robustness is "real" only if its
  retained score beats TILE CODING's by more than the paired-bootstrap noise at the same centering/architecture;
  beating discrete is expected and is not evidence. If tile coding retains as well, the user's robustness claim is
  generic to fixed features, and that is a result worth reporting.
- Cost: 18 cells x 20 seeds x about 25 conditions x 20 episodes x up to 500 steps is about 4.5M environment
  steps per cell-pass; SSP encoding is the slow part. Time one cell on the pod before launching all.

## 4.3 Deployment metrics (on the saved policies, quiet single-process benchmark)

- Model bytes at float32/float16/int8 INCLUDING the encoder constants (SSP phase matrix; tile hash table).
- Post-training quantization retention (score at fp16 and int8 vs fp32).
- Latency per action on a CPU (include SSP encode: matmul + complex exponential + inverse FFT; tile coding is
  16-32 lookups). Time on a quiet pod, one process.
  Question: SSP should win on memory and may lose on compute; which wins depends on the target device, and we
  have never measured it.
- Report these WITH the Pareto caveat from conclusion 2 (not size-matched), until the size sweep exists.

## 4.4 DQN-settings confound test (optional but needed before concluding SSP is "unstable")

For the SSP and tile-coding winners (same variants for both, linear and deep), 10 seeds each: replay buffer
5,000 vs 50,000, and target-update every 10 vs 100 vs 500 updates. Metric: collapse rate and final score.
Question: is the SSP collapse an artifact of the fixed DQN settings? If a larger buffer or slower target removes
it, SSP's instability and its gap to tile coding are partly a DQN-setup effect and the conclusions in Section 3
(items 1 and 5) must be softened. Cost is small (a few dollars at most; estimate unverified).

## 4.5 A2C: port + same evaluation

The old A2C code is Nengo-based (now renamed `experiments/trial_cartpole_nengo.py`, `rl/networks/acBasic_nengo.py`,
`rl/rules/td0_nengo.py`, `rl/rules/td0centering_nengo.py`, `scripts/run_hparam_search_a2c_nengo.py`; logic unchanged);
the Stage-0 pod image has no Nengo and the docs pin the legacy stack to Python 3.9. So A2C re-evaluation needs a
Nengo-free port (planned in `docs/02-experimental-design.md`, `docs/handoffs/01-stage0-repo-rebuild.md`). It is a
NumPy float64 port, not PyTorch: there is no autograd or optimizer, and per-step NumPy is as fast or faster:
`rl/networks/acLinear.py` (rule), `experiments/trial_cartpole_a2c.py` (loop), `rl/representations/ssp_legacy.py`
(encoder), tests in `tests/test_a2c_port.py`, gate in `scripts/gate_a2c_port.py`.
- NO new hyperparameter search is needed, provided the port is faithful. Reuse the nine A2C winners
  (`outputs/hparam_search/best/{rep}_{centering}.yaml`, read only) plus the constants in `A2C_BASE_PARAMS`
  (copied from `BASE_PARAMS` of the legacy search script).
- What the legacy rule actually is (the port keeps all of it; an earlier version of this note said "softmax
  policy over linear heads, plain SGD", which was wrong):
  - one weight matrix, row 0 = state value, other rows = linear action PREFERENCES; the behaviour policy is
    epsilon-greedy over the preferences (`force_rho_one=True`, so the softmax target policy never matters), and
    the greedy policy is `argmax(preferences)`;
  - every update is divided by `||phi(s)||^2` (normalised step), so lr 0.4-0.8 is a fraction of an NLMS step; the
    taken action's preference row moves by `act_dis * td * phi`, the value row by `lr * td * phi` (no policy-gradient
    term); the TD target bootstraps from a STALE V(s) and the preferences used to act are computed from the
    weights BEFORE the step's update; termination is an extra `n_done=1` step on the same state with reward 0;
  - dynamic epsilon (`-0.001` when the last 10 episodes beat the previous 10 by one std), act_dis 0.9 / state_dis
    0.99, 1000 episodes, weights start at zero, the three centering rules (value: R-bar += eta * lr * delta, with
    the old R-bar in the TD error);
  - SSP encoder = the OLD one: n_scales 8, scales linspace(0.5, 2), principal-log phase recovery, and the SO(4)
    rotations that scipy 1.7.3 (the legacy pin) draws for `random_state=1`. Current scipy (1.18) draws DIFFERENT
    rotations for the same seed, which changes the kernel enough that the first SSP gate run failed (262 vs 498);
    `legacy_special_ortho_rvs` reimplements the 1.7.3 algorithm. This is neither the DQN `SSPRep` (n_scales 5, current
    scipy rotations) nor upstream `sspspace`; `LegacySSPRep` reproduces the old one (golden values generated by the
    real vendored code under Python 3.9 / scipy 1.7.3, tests/data/legacy_ssp_golden.npz). A2C SSP dimension is
    80*n_rotates+1, DQN's is 50*n_rotates+1. Tile coding and discrete reuse the existing reps (sparse index path,
    tested equal to the dense one). No neuron layer (the grid's A2C had none).
- VALIDATION GATE (`scripts/gate_a2c_port.py`, local, ~3 min for 4 cells x 10 seeds on 16 cores): run the port with the
  old parameters, scored by the OLD online metric (mean of the last 100 training episodes). Pass rule fixed in
  advance: the old value is inside the bootstrap 95% CI of the port's 10-seed mean, or within 2 SE. RESULT
  (2026-10-06, seeds 1000-1009): ssp/none 490.6 (old 498.4), tile_coding/none 492.4 (495.0), discrete/none 112.4
  (118.5), tile_coding/value 496.4 (495.1): all PASS. The centering math of `simple`/`value` is pinned by exact
  tests against the real legacy rule code, so the other five cells were not re-run for the gate. Independent
  check: replaying the states/rewards of a real old SSP run (`ssp/none`, 80 episodes) through the port reproduces
  the recorded value series to 2e-6 and agrees with the inferred actions at the expected rate. No retune needed.
- Then evaluate with the SAME suite as DQN: greedy = argmax of the policy (epsilon off), 20 fresh seeds, saved
  policies, robustness suite, stability metrics, centering diagnostics. Expect fresh-seed numbers to be a bit
  lower than the old search-winner values.
- Why it matters: it is the algorithm in the user's paper, SSP >= tile there, and it tests the DQN-specific
  explanations in Section 3 (items 5-7) under identical metrics.

## 4.6 Optional instrumentation (only if 4.4 leaves the collapse mechanism unclear)

The interference measure of Liu et al. 2023 (change in squared TD error on a held-out batch after each update),
logged during training for a few seeds per representation. Skip otherwise.

## 4.7 Reporting conventions

mean +/- np.std over seeds (unchanged), per-seed dots, share solved, bootstrap CIs, paired (by seed) comparisons
for pre-declared pairs only. Primary table from the final eval, not the filter. State the DQN fixed settings and
that results are CartPole-only.

## 4.8 Do NOT run

Raw-input baseline; seed-reliability as a separate experiment; any 500-trial search; a full multi-objective
size search; mjlab runs before the Stage-0 questions above are answered.

---

# 5. Strengths of SSP (as established from the literature and docs) and how our data bears on each

1. **Smooth, tunable kernel generalization.** phi(x) = F^-1{exp(i Theta x)}; the dot product is a shift-invariant,
   radial-like kernel (HexSSP: "hypergeometric" kernel) with a length-scale; no bin edges. Our data: far better
   than discrete; not better than tile coding on CartPole under DQN. The robustness suite (4.2) tests this.
2. **Representation size decoupled from resolution and linear in input dimension:**
   ssp_dim = 2*(n_rotates*n_scales)*(d+1)+1 (tile grids grow roughly as tiles^d; hashing hides it with collisions;
   the full Fourier basis grows as (n+1)^d). NOT yet tested beyond d = 4. Caveat from Chouinard 2026: the default
   kernel flattens with d at a fixed length-scale, so length-scale must be re-tuned per dimension.
3. **Compact footprint.** 2,004 params at 468.7 (ssp/value/linear). Not yet size-matched against tile coding
   (conclusion 2).
4. **Compositional algebra.** Binding and shifting combine multiple variables/objects into one fixed-size vector
   (RFF compositions concatenate or take tensor products, so their dimension grows; VSA binding does not).
   Demonstrated for Bayesian optimization by Furlong et al. 2026; no RL demonstration found. Not testable on
   CartPole.
5. **Spatial/metric variables.** Grid- and place-cell structure suits positions and displacements (the
   Lift-Cube `ee_to_cube` / `cube_to_goal` plan; Komer & Eliasmith 2020 for navigation).
6. **The user's paper's claims:** low seed variance (vs deep nets) and robustness to changed dynamics (RatBox
   speed halved). Variance: our DQN data does not support an advantage over tile coding; A2C shows both near
   zero. Robustness: the one claim still open and cheap to test (4.2).
7. **Neuromorphic and biological compatibility** (grid cells; Loihi 2 / SpiNNaker energy results for SSP-BO). The
   lab's evidence is for Bayesian optimization, not RL, and we cannot measure it here.

---

# 6. Experiments to do next (priority order), after Section 4

1. **Robustness suite** (4.2) for DQN and A2C. Already specified; nearly free once policies are saved.
2. **Size sweep** (the answer to "is SSP more size-efficient?"). Take today's `none`/linear winners for tile
   coding and SSP; vary only the size knobs; 10 seeds per size; add a light retune (about 20 trials per size,
   learning rate / epsilon / length-scale) only if small sizes look penalized by the old values.
   Needed code changes: lower the tile-coding floors (`iht_size` far below 32,768, smaller `tiles_per_dim`) and let SSP
   `n_rotates` go from 1 to 20 (about 51-1001 dims). Prior evidence that small SSPs may be fine: the user's MiniGrid
   repo found 64-532 dims barely mattered; our data shows a threshold near 500 in three cells.
   Success = SSP keeps its score at sizes where tile coding degrades. Failure (tile coding matching SSP down to a few
   thousand params) is also a legitimate result. Run both DQN and A2C only if the first looks informative.
3. **Dimension scaling** (user considers this very important). Design:
   - Controlled: CartPole with a stacked history of the last k states (k = 2, 4, 8 -> 8, 16, 32 dims). The extra
     dims carry real but redundant information. (Distractor noise dims are less fair: any kernel's similarity is
     corrupted by irrelevant coordinates.)
   - Natural: Acrobot (6 obs dims) and LunarLander (8), discrete actions.
   - Comparators: SSP vs tile coding (the incumbent; use standard high-dimension practice: hashing and tilings over
     low-order subsets of dimensions, not one full grid) vs discrete as a reference (it breaks as bins^d).
   - SSP: re-tune length-scale per dimension. A Gaussian-kernel SSP is a one-off DIAGNOSTIC at the largest d if
     SSP struggles (tests the kernel-flattening prediction); it is not a baseline. The Fourier-basis baseline is a
     second wave, only needed for a publication-grade "SSP is special" claim.
   - Small searches (about 40-60 trials per task per representation: lr, eps, length-scale, size knob), not 500.
     LunarLander is slower than CartPole; cost not yet estimated.
   - Success = SSP reaches a fixed fraction of the maximum score at a size that grows more slowly with d than tile
     coding's.
4. **DQN-settings test** (4.4) if not already done.
5. **Centering cleanup** (docs call for it): a continuing CartPole variant (reset with a penalty, long horizon),
   a widened beta range (about 1e-8..1e-1, so "centering nearly off" is an explicit option), and the A2C-vs-DQN
   explanation of simple centering (log R-bar, 4.1). Update decision log entry 17 with the A2C caveat.
6. **Later:** the planned A2C->PyTorch results into the 9-config tables; Hex-vs-random SSP and random-Fourier-feature
   ablations (Mir et al. suggests the hexagonal lattice may not matter); a rotation-robustness test (SSP's kernel is
   roughly radial, tile coding axis-aligned; CartPole's different variable scales complicate it); a compositional
   multi-object task (the only place binding can show; custom environment needed).
7. **mjlab (Stage 2/3):** VERIFIED facts: mjlab uses MuJoCo Warp (GPU), runs thousands of parallel environments,
   captures the simulation step as a CUDA graph, supports multi-GPU, and the README says an NVIDIA GPU is required
   for training (macOS evaluation only). Its observation manager supports noise, delay and clipping; the event
   manager provides domain randomization and external pushes (so robustness tests are nearly free there). Shipped
   tasks: Go1/G1 velocity tracking, G1 motion tracking, YAM cube lifting. The paper pages read gave no throughput
   numbers. The user's laptop has an RTX 3050 Laptop GPU with 4 GB (small environment counts only, untested);
   RunPod CPU pods cannot run it, so GPU pods are needed. Alignment with SSP's strengths: Lift-Cube (SSP on the 3D
   `ee_to_cube`/`cube_to_goal` terms) covers the spatial strength; Go1/G1 test dimension loosely (not a controlled
   sweep); robustness is built in; compositional is not covered (multi-cube is out of scope); PPO feeds an MLP, so
   "SSP pairs well with shallow approximators" is not tested there. Do the cheap Stage-0 versions first.

---

# 7. Operational notes for the next agent

- Code: `scripts/run_hparam_search_dqn.py` (search -> filter -> final eval chain; flags `--workers`,
  `--final-eval`, `--idle-after`, `--self-delete`), `scripts/parallel_jobs.py` (parallel, resumable, R2-backed
  runner; worker `eval_worker`; `_silence_tqdm`), `scripts/launch_cell_pod.py` (one pod per cell via a
  content-hashed code tarball on R2, `--backup-dbs`), `scripts/watch_cells.py` (progress + cost guard, 12-minute
  grace), `scripts/smoke_test_refilter.py` (local end-to-end test; run it after any change to the chain).
  `run_best_ablation.py` remains the local serial final-eval tool. Tests: `pytest -q` (11 passing).
- Pods fetch code from an R2 tarball at start (no Docker push; the image has a 6 GB dependency layer). Tarballs
  exclude secrets, PNGs and outputs. New eval code must be delivered the same way.
- `dqnBasic.py` now tolerates `torch.set_num_interop_threads` being called after parallel work started (a test
  import-order failure).
- Search DBs on R2 were rewritten by the pods' syncs (they only gained rows/attributes: `reeval_v2_*`,
  `reeval_v2_best_trial`, `stale_running_failed`, FAIL rows for stale trials). Originals are in the backup prefix.
- Stale RUNNING trials were converted to FAIL and the target raised so every cell has 500 finished trials.
- The Gymnasium/torch versions on the pods differ from the dev machine (torch 2.14 local); compare ranks, not
  absolute numbers, between local and pod runs.
- Documents updated in this session: decision log 14-18, `02-experimental-design.md` item 5, `RUN_002.md`
  methodology section, handoff 02 addenda. NOT yet updated: the centering caveat on decision log 17, and the
  project docs with the final 18-cell table.

# 8. Open decisions for the user

1. Run the DQN-settings test (4.4) before the size sweep, or skip it?
2. A2C: port to PyTorch (recommended) vs run the legacy Nengo code in a separate Python 3.9 environment.
3. Which cells get the robustness suite first (all 18 vs a representative subset to time it).
4. Whether to extend the size sweep and the dimension scaling to A2C or keep them DQN-only initially.
