# Handoff: DQN v4 — greedy-eval filtering, last-checkpoint ranking, per-cell next steps

Written 2026-10-04. Status: CODE IMPLEMENTED (same day, see "Implementation status" at the end of
this file); pilot running/analyzed there too. Nothing on R2 has been written or modified.
Supersedes the "Open methodology issue" section of `runs/RUN_002.md`
(that section's diagnosis was incomplete; see "Why" below). Read after
`docs/HANDOFF_dqn_v4_search_2026-09-29.md` and `runs/RUN_002.md`.

## Decisions already made (with the user)

- Keep Raffin's design: ONE seed per search trial, then re-evaluate top trials with fresh seeds.
- Use **greedy (deterministic, no-epsilon) eval** for BOTH the search checkpoints and the
  post-hoc filtering step. Greedy eval is mainstream (SB3 `evaluate_policy` defaults to
  `deterministic=True`; Raffin's `TrialEvalCallback`; Henderson et al.'s "policy optimization view").
- Rank candidates for filtering using the **last several greedy checkpoints**, not the single
  last one (proposed K=5; to be validated by the pilot below).
- **Do not change the std computation** (`np.std` in `reevaluate_top_trials`, mean ± std across
  seeds). Anything added next to it (per-seed dots, share of seeds solved) is additive only.
- Record extra run metadata (time, size, steps) — see "Metadata to record".
- Do NOT change the search objective value (still the last greedy checkpoint, `terminal_eval_reward`),
  the TPE/MedianPruner config, or the search space. Resumed trials must stay comparable with the
  trials already in each study.

## Why (evidence, all from data already on R2)

1. **Search and filtering measured different things.** Search scores `terminal_eval_reward`
   (greedy eval, 10 episodes, at episode ~990, checkpoint 15 of 15; `eval_freq = 1000 // 15 = 66`).
   `reevaluate_top_trials()` records `metadata["terminal_reward"]` = mean of the last 100 TRAINING
   episodes played epsilon-greedy, and never sets `eval_freq` (defaults to 0) so no greedy eval is
   even computed. Sampled `eps` reached 0.33–0.48 in `ssp/simple/deep`, so the online number mostly
   measures exploration noise. (Machado et al. 2017 do recommend an online last-100 metric — it is a
   legitimate protocol — but it is not what the search selected on, and not with eps this high.)
2. **A single greedy checkpoint reading of 500 is a snapshot lottery, within the SAME metric.**
   DQN's greedy policy swings between ~500 and ~0 between checkpoints. Example (ssp/simple/deep
   trial 125): last 5 checkpoints = 500, 184, 14, 500, 500. Across the 130 trials tied at 500 in
   that cell, the median of last-5 mean is 357 and only 12 have last-5 mean >= 450. In
   ssp/none/deep the same measure has median 494 (81 of 114 >= 450) — those configs are stable.
3. The 15 greedy checkpoints per trial are already in each Optuna DB
   (`trial_intermediate_values`, step = 1..15, present for every COMPLETE trial checked), so a
   last-K ranking costs NO compute and NO R2 scraping.
4. Checked on the 7 ssp/simple/deep trials with known 10-seed reeval means: those means were
   ONLINE-return means, not greedy, so they do not validate the last-5 ranking. Spearman between
   single-run online return and reeval mean was 0.54 (n=7, p~0.2). The pilot below is what actually
   tests the proposed ranking against multi-seed greedy results.

## Part 1 — Rank by last-K greedy checkpoints (candidate selection)

Replace `_select_reeval_candidates` (+ `_lookup_episodes_to_learn`, which scrapes R2 .txt files and
is ~70% NaN) in `scripts/run_hparam_search_dqn.py` (approx. lines 392–440):

- Pool = all COMPLETE trials with exactly 15 intermediate values (pruned trials are excluded).
  This is NOT limited to trials tied at 500 — that restriction was the winner's-curse source.
- Sort key: (mean of last 5 checkpoints desc, mean of last 8 desc, "stable-from" checkpoint asc,
  then seeded random). "Stable-from" = earliest checkpoint index from which every later checkpoint
  is >= 475 (smaller = faster and stable).
- A second tie-break level IS needed. Trials with last-5 mean exactly 500: tile_coding/none 83,
  tile_coding/none_deep 67, ssp/none_deep 45, tile_coding/value_deep 42, ssp/value_deep 30,
  tile_coding/simple 20, tile_coding/value 12, ssp/value 9, ssp/simple/deep 5, ssp/none 5. Even
  last-8 leaves 45 tied in tile_coding/none and 29 in tile_coding/none_deep. If more than N tie on
  every key, take a seeded random sample and PRINT/LOG the tie size.
- Pool = top ~10% of the ranked completed trials, minimum 20, cap ~40 (ties at the cutoff resolved
  by the sort keys above, then seeded random). Measured top-10% sizes/cutoffs (last-5 mean) on R2
  data: discrete cells 16–34 trials, cutoffs ~275–405 (almost all have last-1 < 500, so a
  "tied at 500" rule would have excluded them); ssp/simple/deep 44 (cutoff 407), ssp/simple 40
  (395), ssp/none 27 (488), tile_coding/simple/deep 28 (448); saturated cells (ssp/none/deep,
  ssp/value/deep, tile_coding/none, none/deep, value/deep) have cutoff exactly 500 and 30–83
  trials at it, so the quantile cannot rank inside them.
- STAGED FILTER (successive-halving style) instead of flat N x 10 seeds: stage 1 = 3 greedy seeds
  for every pool candidate; keep the better HALF by 3-seed mean (conservative — std ~50 makes 3
  seeds noisy, so do not cut harder); stage 2 = 7 more seeds on survivors (10 total); winner = best
  10-seed mean, same tie rule as before (lower std wins a tie, `np.std` unchanged). Cost: pool 30
  -> 90 + 15x7 = 195 runs (same as flat 20 x 10 = 200); pool 40 -> 120 + 20x7 = 260 runs.
- Ranking is cheap but evaluating candidates is not; the staged filter is what keeps a bigger pool
  affordable. Winner's-curse note: the max of many noisy means is biased upward, which is why the
  final eval uses 20 fresh seeds.
- Discrete cells have almost no trials near 500 (last-5 >= 450: zero in all six). Ranking by last-5
  over all COMPLETE trials gives them a sensible pool without needing ties at all.

## Part 2 — Switch filtering (reeval) and final eval to greedy

Code changes (approx. line numbers, verify before editing):

- `experiments/trial_cartpole_dqn.py` (~277–284, return dict ~323–334): accumulate every checkpoint
  into a list `eval_history` and return it (plus `eval_time`, see metadata). Keep
  `terminal_eval_reward` = last checkpoint unchanged.
- `scripts/run_hparam_search_dqn.py::reevaluate_top_trials` (~443–540): set `eval_freq` and
  `n_eval_episodes` to the SAME values the search used (`eval_freq = 1000 // n_evaluations`,
  `n_eval_episodes` from args — not the defaults of 0/10). Per-seed score = mean of last K greedy
  checkpoints (same K as the ranking). Also store the last-1 value and the full history. Winner rule
  unchanged: best mean (rounded) then lower std. `np.std` untouched.
- `reeval_results.json` should additionally store, per candidate, the per-seed scores and
  `eval_history`s — not just mean/std.
- `make_objective` (~281–345): store `trial.set_user_attr(...)` for `eval_history`,
  `terminal_reward`, and timing fields. Do NOT change the returned objective value.
- `scripts/run_best_ablation.py` (final eval; 20 seeds default is already right): it records
  `terminal_reward`, never sets `eval_freq`, loads `dqn_{rep}_{centering}.yaml` with no `_deep`
  handling, and its output dir is keyed on (model_type, rep, centering) only — linear and deep
  would collide. Needs the same greedy fix plus architecture handling.
- Update docs: `docs/02-experimental-design.md` item 5 and `docs/07-decision-log.md` entry 11
  (both still say "top 10 by search score, 10 seeds"), and replace RUN_002's "Open methodology
  issue" section with a pointer to this doc.

Final eval (not part of the re-filter, listed so it is not forgotten): 20 fresh seeds per winning
config (use seeds >= 2000; search used 0..499, reeval 1000+), report mean ± std exactly as now,
per-seed dots and share-of-seeds-solved alongside. Optional secondary robustness probe: greedy
policy with 5% action noise. Not used in search or filtering.

## Metadata to record (agreed)

Per run, as Optuna `user_attrs` (search) / run metadata (reeval, final):

- training time and eval time SEPARATELY (the new greedy eval adds overhead; `total_time` and
  `build_time` already exist in the pytry metadata and are already on R2 in each trial's .txt)
- representation size (`dimensionality`) and active features per step; learnable parameter count
- env steps and gradient updates actually performed
- episodes/steps to first reach greedy >= 475 from the checkpoints (sample efficiency; NOT the same
  as the existing `episodes_to_learn`, which uses the noisy online return)
- pod CPU type and the parallelism setting, one line

Caveat: wall time is only comparable across cells at the WINNING configs and on a quiet machine
(per-cell median full-run minutes seen so far: discrete 0.4–1.9, ssp 0.8–3.6, tile_coding 2.4–6.1,
driven partly by sampled `iht_size`/`n_rotates`). With parallel seeds on one pod, timings are
contended; time a few final-eval runs single-process. Do not put time in the objective or pruner.

## Pilot (do this before the full re-filter)

7 ssp/simple/deep trials (125, 306, 285, 457, 42, 56, 78 — last-5 greedy means 339, 213, 438, 406,
332, 223, 299) plus 2–3 ssp/none/deep trials (e.g. 154, 344), 10 fresh seeds each, greedy eval
with search settings. ~70–100 runs, about an hour on one 4-vCPU pod if seeds run in parallel.
Question: does the last-5 mean from the single search run predict the 10-seed greedy mean? Sanity
expectations: ssp/none/deep trial 344 (500 on all last 5) should be ~500; ssp/simple/deep
candidates should land near their last-5 values. If the ranking does not predict, increase K or the
pool size before spending on all cells.

Rank-band calibration (added 2026-10-04): in ONE cell (ssp/simple/deep, widest spread) also sample
~5 trials from each rank band (top 1%, 5–10%, 10–25%, around the median by last-5 mean), 5 greedy
seeds each (~100 extra runs). Question: how fast does the true multi-seed greedy mean fall with
rank? Falls fast -> flat top-20 is enough. Roughly flat -> the deeper pool + staged filter is
worth it. This is the only way to tell whether trials below the top ranks can beat the top ones,
since all search data is single-seed.

## Part 3 — Per-cell state (R2, audited 2026-10-04) and what each needs

"finished" = COMPLETE + PRUNED trials in the v4 study on R2. All studies have `target_n_trials=500`.
EVERY cell needs re-filtering (all earlier reeval used the online metric), but only some can start now.

A. Search complete on R2 — can re-filter immediately (11 cells):
- discrete: none (239C/260P), none/deep (186/314), simple (337/163), simple/deep (266/234),
  value (304/196), value/deep (157/343)
- ssp: none (270/230), none/deep (192/308), simple (394/106), simple/deep (440/60), value (151/349)
- Notes: `discrete/none` (the old "test pod") now shows 500 trials and `reeval_done=true` on R2;
  RUN_002 had called its state unknown — VERIFY, then treat as done-but-needs-refilter like the rest.
  `ssp/none` has `reeval_done=false` already (cleared earlier). `ssp/value` has no reeval flag and a
  pool of 61 tied / 48 with last-5 >= 450.

B. Needs a few more search trials (~7–9), then re-filter (5 cells):
- ssp/value/deep: 492 finished + 2 interrupted-RUNNING = 494
- tile_coding: none, none/deep, simple, simple/deep: 491 finished + 2 interrupted-RUNNING = 493 each
- Mark the leftover RUNNING trials as FAIL before resuming (e.g. `study.tell(number,
  state=optuna.trial.TrialState.FAIL)`), and check whether the resume logic
  (`remaining = target - current_total`) counts RUNNING trials; off by at most 2 either way.

C. Needs a large search top-up (2 cells):
- tile_coding/value (linear): 301 finished + 2 = 303 on R2 -> ~200 trials to re-run.
- tile_coding/value/deep: R2 has only 211 finished + 2 = 213, but RUN_002 recorded 500/500 seen in
  live logs. ~287 trials are missing from R2 and the pods were terminated — treat as lost. R2 is the
  truth; do not assume the log count.
- Both are the slowest cells (tile_coding/value/deep median full run ~6 min). Launch these pods
  first and with SSH enabled (`startSsh`) so a hang (Bug #2) can be diagnosed and the local DB
  pulled before termination.

Strong-candidate counts (trials tied at 500 with last-5 greedy mean >= 450, from current R2 data;
B and C cells will change after more trials): ssp/none 47, ssp/none/deep 81, ssp/simple 6,
ssp/simple/deep 12, ssp/value 48, ssp/value/deep 76, tile_coding/none 188, none/deep 168, simple 113,
simple/deep 22, value 86, value/deep 103, all discrete 0. Cells with few strong candidates
(ssp/simple, ssp/simple/deep, tile_coding/simple/deep, all discrete) are the ones where low final
scores are probably genuine, not artifacts.

## Recommended order of work

1. Pilot (above). It needs only the Part 2 code change in `reevaluate_top_trials` + `eval_history`.
2. Implement Parts 1 and 2 and the metadata; dry-run against a downloaded DB with `--no-sync`.
3. Re-filter group A (11 cells) with the staged filter — ~195–260 runs/cell (pool 30–40); roughly
   4 h/cell if seeds run 4-way parallel (estimate from median run times; parallel speedup not yet
   measured), ~$0.5–1/cell.
4. Resume search for groups B and C, then re-filter them.
5. Final eval (20 fresh seeds x 18 winners), then graphs.

Recommended engineering, NOT yet approved by the user: run (candidate, seed) jobs in 4 processes
(pods have 4 vCPU; reeval is serial and torch is single-threaded, handoff showed cpu.util 44–144%),
and append each (candidate, seed) result to a JSONL that is synced to R2 immediately, so a hung
pod or a restart resumes instead of redoing everything. This is what would have prevented the
`tilecoding/value/linear` loss.

## Safety rules for R2

- Writing new reeval/best_params results over the existing keys is destructive. Prefer new
  prefixes (e.g. `stage0_cartpole/best_params_v2/`, `.../reeval_v2/`) and keep the old files.
- Do not clear `reeval_done` on existing studies without the user's explicit OK; store the new
  reeval outcome under new user_attr keys or the new prefix instead.
- Never copy R2 credentials into docs (the old handoff embeds them inline; avoid repeating that).

## Open questions for the user

1. K=5 (proposed) vs another window — decide after the pilot.
2. Pool = top ~10% by last-5 (min 20, cap ~40) with the staged filter (proposed, see Part 1) vs
   the flat top-20 x 10 seeds. Decide after the rank-band calibration in the pilot.
3. Reeval/final per-seed score: last-K mean (proposed, matches the ranking) — confirm, and keep the
   last-1 value alongside.
4. In saturated cells where many candidates reeval at ~500 ± small std, break ties by earlier
   "stable-from" checkpoint (sample efficiency)? Proposed yes.
5. Archive prior results under new R2 prefixes (proposed) vs overwrite.

## Audit snippet (read-only, reproduces the Part 3 numbers)

```python
import sqlite3, numpy as np
# download stage0_cartpole/optuna_dbs/optuna_studies_<study>.db first; study names look like
#   dqn_ssp_none_deep_v4_hparam_search, dqn_tile_coding_value_v4_hparam_search
con = sqlite3.connect(path)
sid = con.execute("select study_id from studies where study_name=?", (study,)).fetchone()[0]
states = dict(con.execute("select state,count(*) from trials where study_id=? group by state", (sid,)))
inter = {}
for n, v in con.execute(
    "select t.number,i.intermediate_value from trial_intermediate_values i "
    "join trials t on t.trial_id=i.trial_id where t.study_id=? and t.state='COMPLETE' "
    "order by t.number,i.step", (sid,)):
    inter.setdefault(n, []).append(v)
last5 = {n: float(np.mean(v[-5:])) for n, v in inter.items() if len(v) == 15}
# Always resolve study_id first: DB files are shared across studies and trial numbers repeat.
```

## Implementation status (2026-10-04, same day)

Implemented and smoke-tested (tiny-budget end-to-end runs; NOT yet exercised on a pod or on a
full-length study). Nothing is committed — the working tree already had many uncommitted changes
from before this session; review `git diff` on just these files:

- `experiments/trial_cartpole_dqn.py`: returns `eval_history`, `eval_time`, `train_time`,
  `env_steps`, `n_updates`, `n_params`, `n_active`, `final_eval_reward`; new param
  `final_eval_episodes` (0 = off). `terminal_eval_reward` unchanged (= `eval_history[-1]`).
- `scripts/run_hparam_search_dqn.py`:
  - `make_objective` also stores those fields as Optuna user_attrs; the returned value is unchanged.
  - Old `_select_reeval_candidates` / `_lookup_episodes_to_learn` / `TIE_EXPANDED_POOL_SIZE` removed.
    New: `last_k_mean`, `stable_from_index`, `rank_trials_by_checkpoints`, `select_reeval_pool`
    (top `--pool-quantile` 0.10, clamped to `--pool-min` 20 .. `--pool-max` 40), and a staged
    `reevaluate_top_trials` (`--stage1-seeds` 3, `--stage1-keep` 0.5, then up to `--reeval-seeds` 10;
    `--last-k` 5). Each reeval run now sets `eval_freq`/`n_eval_episodes` like the search.
  - Safety: results go to study user_attrs `reeval_v2_*` (old `reeval_*` untouched), best-params file
    `dqn_<rep>_<centering>[_deep]_v2.yaml` (new `variant` arg on `_best_filename`/`save_best_params`),
    per-run CSVs in `.../reeval_v2/`, full table in `reeval_v2_results.json` (per-seed scores and
    eval histories). `--top-k` is accepted but ignored.
  - Printed line format `reeval rank=N (search trial #T, search score S): mean=M std=D over K seeds`
    is kept for stage-2 survivors (RUN_002's log-recovery procedure greps it); stage-1 lines are
    prefixed `stage1`.
- `scripts/run_best_ablation.py` (final eval, DQN branch only): `--architecture`, `--variant`,
  `--seed-offset` (use >= 2000), `--n-evaluations`, `--n-eval-episodes`, `--final-eval-episodes`
  (default 100), `--last-k`. Writes `final_eval_seeds.jsonl` per cell (primary score = last-K greedy
  mean; also last-1, final-policy eval, online return, timings, size). Deep results go to
  `dqn_deep/...`. A2C/Nengo imports are now lazy — the script previously could not even import
  in the Stage-0 env (it imported `nengo` at the top).
- `scripts/pilot_greedy_rank_check.py`: the pilot (see below), resumable via
  `outputs/pilot/pilot_runs.jsonl`; `--analyze` prints the tables.

Findings made while implementing (not changed — flagged for a decision):

1. **Every DQN run starts from identical network weights.** `rl/networks/dqnBasic.py` calls
   `torch.manual_seed(0)` inside `DQN.__init__`, so the `seed` argument never changes weight
   init (for linear AND deep). Together with the unseeded `env.reset()`/`action_space.sample()`
   (Gymnasium seeds those from OS entropy), `seed` only controls numpy-driven randomness
   (epsilon draws, replay indices). Consequences: (a) seed-to-seed std in reeval/final eval
   excludes weight-init variance, so it probably UNDERSTATES true across-seed variability,
   especially for deep nets; (b) a run cannot be reproduced exactly from its seed. Changing this
   would make new runs incomparable with the 500-trial searches, so it is left alone; decide
   whether the final eval should vary init (that would be a documented deviation from the search).
2. Local-run caveat for the pilot: it runs on the dev machine's torch (2.14), not the pod image.
   Fine for comparing ranks; do not read its absolute numbers against pod results to the point.
3. Operational: the pilot with 12 spawn workers used ~11 of 15 GB RAM (~0.9 GB RSS per worker); it
   was restarted with 8. Killing the parent leaves orphaned workers (parent pid 1) — kill by the
   venv-path `multiprocessing` command line.

## Pilot: how to run it on RunPod (local was too slow)

The pilot (`scripts/pilot_greedy_rank_check.py`, 200 runs over 30 trials) was started locally with 8
workers and stopped: nothing finished in 8+ minutes. Reason: the pilot trials are the STRONGEST
configs, which do the most work per run (500-step episodes all the way), and the dev machine is
several times slower per run than a pod. Pod data for reference (R2 per-trial metadata, completed
search trials with online return >= 450 in ssp/none/deep): median 2.4 min/run, p90 2.7 min. Rough
total: ~4–5 core-hours of work, i.e. ~25 min on 16 vCPUs / ~45 min on 8 / ~80 min on 4, at roughly
$0.03 per vCPU-hour (the search pods' 4 vCPU = $0.12/hr) — well under $1.

Everything needed is built, except a RunPod API key (none was configured: `~/.runpod/config.toml`
has an empty `apikey`; `.env` only had the R2_* keys):

1. Put `RUNPOD_API_KEY=...` in the repo `.env` (gitignored, excluded from docker builds), or run
   `runpodctl doctor`.
2. Image `kevinkyawthiha/dqn-ssp-centering:pilot-greedy` (NEW tag; `:stage0` was not touched) has the
   updated code. `scripts/entrypoint_dqn_search.sh` gained a `pilot` mode that skips the per-study
   restore and runs the pilot script. Verified locally: `docker run ... pilot --r2 --dry-run` fetches
   the two study DBs from R2 (read-only) and plans the identical 200 runs. NOTE: confirm the push
   finished (`docker push` is slow because of the 6 GB dependency layer) before launching.
3. `python scripts/launch_pilot_pod.py --dry-run` shows the request (secrets redacted);
   `python scripts/launch_pilot_pod.py` creates the pod via REST v2 (`POST https://api.runpod.io/v2/pods`,
   `cpu: {id: cpu3c, vcpuCount: N}`; tries 16, 8, 4 vCPUs in turn). `runpodctl pod create` cannot set
   the vCPU count, hence the script.
4. The pod writes results ONLY to `stage0_cartpole/pilot/pilot_runs.jsonl` on R2 (new prefix),
   resumes from it after a restart, and idles when finished (so RunPod does not restart-loop it).
   Watch with `runpodctl pod logs <id>`; get results with
   `python scripts/pilot_greedy_rank_check.py --analyze --pull-results`; then DELETE the pod.

## Pilot results (run 2026-10-05 on RunPod; 200 runs, 0 errors)

Setup: pod `qdghew73oy1pk1`, 16 vCPU cpu3c, 16 parallel runs, code delivered via R2 tarball on the
existing `:stage0` image (a new Docker image push was abandoned — the 6 GB dependency layer kept
retrying over a ~14 Mbit/s uplink). Results: `stage0_cartpole/pilot/pilot_runs.jsonl` on R2 (also
`outputs/pilot/pilot_runs.jsonl` locally); reproduce the tables with
`python scripts/pilot_greedy_rank_check.py --analyze --pull-results`. Pod deleted afterwards.

**ssp/simple/deep (27 trials, 5–10 fresh greedy seeds each)**
- Search rank does NOT predict fresh-seed performance: Spearman(search last-5, 5-seed mean) =
  0.05 (p=0.80, n=27). Band means (reeval greedy last-5, 5 seeds): top 1% 289, 5–10% 316,
  10–25% 257, around the median 308. The five top-ranked trials (search last-5 = 500 AND last-1
  = 500) came back at 270–340. Nothing below the top ranks did worse.
- Variance decomposition (one-way ICC, 27 trials x 5 seeds): within-config (seed) sd ~ 90,
  between-config sd ~ 30, ICC = 0.10. The observed sd of 5-seed means (49.9) is barely above what
  seed noise alone predicts (40.2). So ~90% of the score variance is seed noise, and configs in this
  top set are largely indistinguishable at 5–10 seeds.
- Only 4.7% of all 170 fresh-seed runs reach greedy last-5 >= 450 (12.9% >= 400, max 500). The
  "500 in search -> ~300 on fresh seeds" gap is selection on noise, not a metric artifact: with the
  greedy metric on both sides the gap persists. The best 5-seed means (~390) are themselves
  inflated by picking the max of 27 noisy means (SE ~ 40).

**ssp/none/deep (3 trials, 10 seeds each):** all genuinely good — trial 373 (rank 0, search 500)
468 ± 25, trial 344 (rank 25) 484 ± 21, trial 154 (rank 75, search last-5 484.6) 497 ± 4. The order
by search score is reversed on fresh seeds, but all are 470–497, so selection matters little here.
87% of its fresh-seed runs score >= 450.

**Cost / timing facts (16 parallel on 16 vCPU):** median 8.0 min/run for ssp/none/deep (strong
configs run full-length episodes), 2.8 min/run for ssp/simple/deep. Greedy eval is only 6–10% of
run time. Whole pilot ~48 min wall. These are far above the 2.4 min I had estimated from search
pods (strong configs and 16-way contention); budget production re-filters accordingly.

**Implications for the plan above (decisions pending with the user):**
1. The last-5 ranking is harmless but is NOT a strong filter — the single-seed search score,
   smoothed or not, carries almost no fine-grained information within the top set. The earlier
   decision rule ("rank response flat => deeper pool is worth it") was too glib: flat here means
   the candidates are all about equally good (config sd ~30 vs seed sd ~90), so a bigger pool buys
   little; seeds per candidate and the fresh-seed final eval are the levers.
2. A 3-seed first cut (SE ~ 52) is essentially a coin flip among such candidates. Suggest
   `--pool-max 20 --stage1-keep 1.0` (= flat 20 x 10 seeds; no code change, CLI flags already exist),
   or keep staging only as a disaster screen. Cost per cell at 16 vCPU: roughly 0.4–1.6 h,
   $0.2–0.8.
3. Final eval (20 fresh seeds, mean ± std unchanged) is where cell values should be read from; the
   re-filter winner's own mean is optimistic (max of noisy means).
4. Because configs differ little in cells like ssp/simple/deep, cross-cell comparisons will be
   driven by seed noise and by the representation x centering effect, not by tuning luck.

**Process notes:** (a) my background watcher swallowed R2 read errors and falsely reported "stalled
at 100/200" while the pod had actually finished at 01:19 UTC; the pod then sat idle ~1h45m (~$0.85)
before I noticed and deleted it. Total pod spend ~$1.2. Future watchers must surface errors, and
pods should be launched with a cost guard. (b) `runpodctl` did not read `RUNPOD_API_KEY` from the
repo `.env` (export it per command: `RUNPOD_API_KEY=$(grep ...)`); Cloudflare rejects Python's
default urllib User-Agent on api.runpod.io (HTTP 403 / 1010) — scripts must set one. (c) RunPod's
log stream returned nothing for this CPU pod; R2 results pushes were the only progress signal.

## Follow-up (2026-10-05): init_seed change + init/centering experiments

Implemented (user approved): `DQN.__init__(..., init_seed=0)` (`rl/networks/dqnBasic.py`) — 0 reproduces
the legacy `torch.manual_seed(0)` weights exactly (verified: identical to a manual-seed-0 net; init_seed=1
is repeatable and differs, mean |diff| 0.075). `DQNTrial` param `init_seed` (0 = legacy default, <0 = use
the run seed). The search objective is unchanged (default 0). `_run_reeval_seed` now defaults to
`init_seed=-1` (weights follow the seed; callers can override), and `run_best_ablation.py` does the same
unless `--fixed-init`. NOTE: pilot numbers above were all fixed-init; reeval/final-eval numbers from now on
are not directly comparable with them.

Experiments on a pod (`scripts/pilot_init_centering.py`, 40 runs, same seeds 1000..1009 as the pilot):
(init) trials 154 (ssp/none/deep) and 285 (ssp/simple/deep) with init following the seed vs the pilot's
fixed init; (cent) trials 285 and 42 (ssp/simple/deep) with `reward_center_mode="none"` vs the pilot's
simple centering, fixed init. Motivation for (cent): in ssp/simple/deep the mean greedy-eval curve over
all 440 trials rises to ~350 by checkpoint 3 and then declines to ~267 (72% of trials that ever reach 450
end below 350; ssp/none/deep: flat ~410, 27%), and the centering step size beta has no effect there
(Spearman -0.07), so the cause is unproven. Results: `python scripts/pilot_init_centering.py --analyze
--pull-results`. New helper `scripts/watch_pod_results.py` waits for the results and deletes the pod
(cost guard; it also deletes at its time cap and prints R2 read errors). `launch_pilot_pod.py` gained
`--script`/`--name`.

### Results of the init + centering experiments (pod `0mwqmauarhmqth`, 40 runs, 0 errors, ~12 min, ~$0.10; pod auto-deleted)

**init (fixed init vs weights follow the seed; paired by seed):**
- ssp/none/deep #154: 496.9 ± 4.3 -> 498.7 ± 2.9 (paired diff +1.7, sd 3.1). No effect.
- ssp/simple/deep #285: 353.4 ± 104.0 -> 316.6 ± 54.4 (paired diff -36.8, sd 143; SE ~45, not
  significant; spread ratio F~3.7, borderline). Within noise.
- Reading (n=10 per cell, two configs, both SSP deep): varying init did not noticeably change mean or
  spread, so the fixed-init worry looks small for these configs. Varying it in reeval/final eval costs
  nothing and is the honest default; the search stays at init 0.

**cent (simple centering vs centering none, SAME other hyperparameters, fixed init):**
- #285: simple 353.4 ± 104.0 vs none 354.4 ± 63.3 (paired diff +1.0, sd 100).
- #42: simple 379.5 ± 65.4 vs none 352.6 ± 102.6 (paired diff -27.0, sd 80; ~1 SE).
- Turning centering off did NOT fix ssp/simple/deep's configs. So centering is not what limits THESE
  configs; the earlier "reward centering is invalid on episodic CartPole" explanation is not
  supported by this test.

**What differs between the cells: the search converged to different hyperparameter regions.**
Median of the top-20 trials (by last-5 greedy mean), ssp/none/deep vs ssp/simple/deep:
eps 0.023 vs 0.42; length_scale 0.11 vs 0.82; lr 4.6e-4 vs 9.4e-3; n_rotates 19 vs 14.
In ssp/none/deep, lr is strongly tied to score (Spearman -0.70) and eps -0.31; the simple cell's top
trials sit in a high-exploration / wide-kernel / high-lr region that tops out near 300–380.
Hypotheses: (a) the ssp/simple/deep search got stuck in a poor basin (TPE, 500 trials, noisy 1-seed
objective), or (b) simple centering genuinely makes the good basin unreachable/worse. Decisive next test
(not run yet): take a good ssp/none/deep config (e.g. #154) and switch simple centering ON (a few beta
values), 10 seeds. ~10 runs at ~8 min; ~$0.15. Still ~stable at ~495 => centering is benign and the
simple_deep search failed to find the good basin (a search problem, e.g. widen/seed the search).
Drops to ~300 => centering is the cause.

**Stray pod seen:** `cerberus-w15a` ($0.74/hr, id yw8564ao98xtr8, not created by these scripts) was running
when the experiment finished; left untouched, flagged to the user.

### Reverse centering test (pod `z5piq5p7t6z0vb`, 30 runs, 0 errors, ~12 min, ~$0.10; pod auto-deleted)

A GOOD ssp/none/deep config (#154; centering none = 496.9 ± 4.3 on seeds 1000..1009, fixed init) with
simple centering switched ON (same seeds, fixed init, `scripts/pilot_init_centering.py --exp-set reverse`):
- beta 1e-5: 468.6 ± 36.8 (paired diff vs none -28.3)
- beta 1e-4: 338.2 ± 56.7 (-158.7); eval curve climbs to ~495 at checkpoint 6 then decays to ~330
- beta 1e-2: 270.9 ± 82.5 (-226.1); peaks ~400, noisy

**Conclusion: simple centering (as implemented) hurts, dose-dependently, even on a config that is
excellent without it, and reproduces ssp/simple/deep's learn-then-decay curve.** This supports the original
"centering is invalid on episodic CartPole" explanation: every CartPole reward is exactly +1 (including the
terminating step), so the running average is exactly 1-(1-beta)^n_updates and the centered reward -> 0 for
every transition, removing the failure signal. (Not directly instrumented; formula exact for this task.)
Together with the earlier ablation (turning centering OFF on the simple-tuned configs did not help), the
reading is: under simple centering the search settles in a degraded compromise region (eps ~0.4, lr ~1e-2,
wide kernel) that is not good without centering either, so ssp/simple/deep's ~300 is an honest property of
that cell, not a filtering artifact or a search failure.
Implications / decisions pending: (1) the search range for beta (1e-5..1e-1) has no "centering ~off"
regime; best trials in tile_coding/simple and discrete/simple already sit at the lower bound (1.4e-5,
1.6e-5) — widening to ~1e-7 would let the cell approach centering=none (consider whether the study needs a
new generation); (2) more fundamentally, centering comparisons on episodic CartPole conflate "centering"
with "centering invalid for episodic tasks" (docs/01 and 05 say episodic tasks need conversion to a continuing
formulation, e.g. penalty-and-reset); decide whether Stage-0 CartPole should be converted before the
centering results are interpreted as evidence for/against centering. `value` centering behaved much
better (ssp/value: 57% strong trials) — its update is tied to the TD error, not the raw reward mean.

## Build 2026-10-05: chained per-cell pipeline (parallel + resumable filter, final eval, zombie handling)

Decisions with the user (2026-10-05): flat filter (pool 20 x 10 seeds), run all 18 cells including `simple`
centering (code verified against Naik et al. 2024 — no bug; the paper says centering is not applicable to
episodic tasks, see decision log 17), stale RUNNING trials -> FAIL + extra trials, final eval chained into the
same pod as the filter. All of this is implemented; nothing is committed.

What a cell pod now does (one pod per cell, 4 vCPU, 4 workers): restore its study DB from R2 -> mark stale
RUNNING trials FAIL and raise `target_n_trials` by that count (so 500 FINISHED trials are reached) -> top-up
search -> greedy filter: top 20 by last-5 greedy mean x 10 seeds (1000..1009), 4 processes, each finished run
appended to `reeval_v2_runs.jsonl` and pushed to R2 -> winner (best mean, tie -> lower np.std) -> final eval:
20 seeds (2000..2019), greedy last-5 mean primary, plus last-1, 100-episode greedy eval of the final policy,
online return, timings, size -> push -> pod deletes itself (`--self-delete`; idles instead if any upload failed
or the pod env lacks RUNPOD_POD_ID / RUNPOD_API_KEY). On any exception it idles instead of exiting (RunPod
would restart-loop the entrypoint).

Code: `scripts/parallel_jobs.py` (runner: spawn pool, JSONL repair/resume, R2 push/pull, retry rounds,
rebuilds the pool if a worker dies), `scripts/run_hparam_search_dqn.py` (new flags `--workers --final-eval
--final-seeds --final-seed-offset --final-eval-episodes --idle-after --self-delete`; defaults changed:
`POOL_MAX 20`, `STAGE1_KEEP_FRAC 1.0`; `reevaluate_top_trials` now returns a 4-tuple incl. the winner's
search trial number; `final_evaluate`), `scripts/launch_cell_pod.py` (launcher; content-hashed code tarball
on R2; `--backup-dbs`), `scripts/watch_cells.py` (progress + cost guard; prints R2 errors; deletes pods when
done or at the time cap). `scripts/smoke_test_refilter.py` is the local end-to-end test (tiny budget; also
covers stale-RUNNING handling and resume from a truncated JSONL). `run_best_ablation.py` is unchanged: it
remains the local serial final-eval tool; the pod path is the chained `--final-eval`.

R2 outputs (all new): `stage0_cartpole/refilter_v2/<study>/{reeval_v2_runs.jsonl, reeval_v2_results.json,
final_eval_runs.jsonl, final_eval_summary.json}`, `stage0_cartpole/best_params/dqn_<rep>_<cen>[_deep]_v2.yaml`,
Optuna attrs `reeval_v2_*` (+ `reeval_v2_best_trial`, `stale_running_failed`). Each cell's Optuna DB key is
still rewritten by the pod's sync (it only gains rows/attributes); all 18 v4 DBs were copied first to
`stage0_cartpole/backup_2026-10-05/optuna_dbs/`. The code tarball is in `stage0_cartpole/refilter_v2/_code/`.

Launch: `python scripts/launch_cell_pod.py --cells discrete/none/linear,ssp/none/deep` (or `--cells all`);
monitor: `python scripts/watch_cells.py --cells ... --max-hours N` (run in background). `RUNPOD_API_KEY` is read
from `.env`. Cell syntax: `<rep>/<centering>/<linear|deep>`.

Housekeeping found while building: `scripts/.s3cfg/s3cfg` contains R2 credentials, was NOT gitignored, and was
being packed into the pilot's code tarball on R2 (`stage0_cartpole/pilot/pilot_code.tar.gz` — consider deleting
that object). Now gitignored and excluded from tarballs (also PNGs/outputs; tarball 13 MB -> ~0.1 MB).

Known limits: per-run wall times are contended (4 runs share a pod) — comparable across cells only
loosely; the long top-up cells (`tile_coding/value` ~199 trials, `tile_coding/value/deep` ~289) run the search
serially (`--n-jobs 1`, as before) and will take many hours; the image has no sshd so `startSsh` debugging is
not available; Bug #2 (mid-reeval hang) is still unexplained but now costs at most 4 runs.

Validation (2026-10-05): pods for `discrete/none/linear` (ntck4ltzn2d1lm) and `ssp/none/deep`
(snwjr4wmqok5jl) launched as the end-to-end check before the other 16 cells.

### Live rollout update (2026-10-05 07:08 UTC)

All 18 cells have now been launched in RunPod (4 vCPU per cell); 17 pods remain RUNNING, while `discrete/none/linear`
finished and was deleted by the watcher. No pods are running locally. Launch IDs: `discrete/none/linear`
ntck4ltzn2d1lm (complete); `ssp/none/deep` snwjr4wmqok5jl; `tile_coding/value/deep` 8n7hcfhwwuvkaz;
`tile_coding/value/linear` xb69flqz7dlnnl; `tile_coding/none/linear` 3vpx2mqqmnok13;
`tile_coding/none/deep` 1iqxvjw2x3zg65; `tile_coding/simple/linear` 7lvzq1knpuhvcg;
`tile_coding/simple/deep` jeqomrdimhysp3; `ssp/value/deep` auuxnqsua0srmi;
`discrete/none/deep` 7nccrqyem0ejg2; `discrete/simple/linear` 8ttk8d9daklu9t;
`discrete/simple/deep` cks1spelnark3o; `discrete/value/linear` u8r1hrcxsm8fey;
`discrete/value/deep` xow3nm2o0x8lea; `ssp/none/linear` x6pknm6mcldq4w;
`ssp/simple/linear` 9nnxqnr3msas5h; `ssp/simple/deep` oqtmc3ypbh7va7;
`ssp/value/linear` 0z17n2xqt515zy.

Validation cell `discrete/none/linear` results are all present on R2: filter 200/200 runs, final eval 20/20 seeds,
summary and v2 best yaml. Filter winner was search trial 441 with 299.5 +/- 83.9 (10 seeds); final eval was
260.4 +/- 68.0 over 20 seeds, 0% solved, vs the final-policy 100-episode metric 261.6 +/- 86.0. The winner's
filter mean was optimistically high as expected. R2 Optuna DB: 239 COMPLETE, 261 PRUNED, 1 FAIL (the stale
RUNNING zombie), target 501. The watcher deleted this pod immediately when summary+yaml appeared, interrupting
the *secondary* final-eval CSV tree upload: only 52 final_eval_v2 CSV objects are in R2, though all 20 JSONL
records and the summary are intact. Watcher now waits 12 min after completion for the pod's final sync/self-delete
before fallback deletion; future cells should upload full CSV trees. The final summary and primary data are safe.

At the same timestamp, `ssp/none/deep` had 156/200 filter runs; remaining cells were starting or filtering.
All existing pods had restored their v4 DB and found the expected 20-candidate pool when checked; no startup
errors. Long top-ups for tile_coding/value and tile_coding/value/deep continue before those cells filter.

Test suite after adding safe Torch inter-op thread initialization: `pytest -q` -> 11 passed. Local chained
smoke test passed search -> stale-RUNNING compensation -> filter -> final eval -> JSONL-truncation recovery;
R2 JSONL push/restore roundtrip and worker-death retry passed. There is a tracked watcher for all remaining
17 cells (two groups; ordinary max 16 h, slow value cells max 40 h; 12-min completion grace). These watchers
print R2 errors and delete pods at finish or time cap. Do not launch more cells; all 18 are already running or
complete. Continue monitoring until each final summary appears and each pod is deleted.

### Results as of 2026-10-05 09:56 UTC (9 cells complete)

All means/std below are the final eval across 20 fresh seeds; `solved` = share of seeds whose last-5
checkpoint mean is >=475. Each listed pod was deleted after its result bundle was safe on R2 (for the first
`discrete/none` cell the summary/YAML/JSONL are intact, but 8 seed CSVs were lost to the premature watcher
cleanup noted above).

- discrete/none (linear): 260.4 +/- 68.0; solved 0%; filter winner 299.5 +/- 83.9.
- discrete/simple (linear): 255.1 +/- 50.5; solved 0%; filter 263.6.
- ssp/none/deep: 486.4 +/- 21.4; solved 80%; filter 496.2.
- ssp/simple (linear): 311.5 +/- 101.6; solved 5%; filter 317.0.
- ssp/simple/deep: 363.6 +/- 67.3; solved 0%; filter 388.4.
- ssp/value (linear): 468.7 +/- 34.6; solved 60%; filter 492.5.
- tile_coding/none (linear): 498.6 +/- 3.6; solved 100%; filter 499.6.
- tile_coding/simple (linear): 496.3 +/- 7.1; solved 100%; filter 498.6.
- tile_coding/simple/deep: 479.5 +/- 18.4; solved 60%; filter 461.7 (final evaluation is fresh-seed confirmation, not a filter winner re-selection).

The remaining 9 cells are filtering (7) or still topping up search (tile_coding/value linear/deep). At 09:56
UTC, tile_coding/value/linear had synced 267 COMPLETE + 125 PRUNED + 2 FAIL + 1 RUNNING; its pod log showed
search at 94/199 additional trials. tile_coding/value/deep had 195 COMPLETE + 47 PRUNED + 2 FAIL + 1 RUNNING;
its current trial was progressing (episode bar 693/1000), but it had not completed enough new trials for a
recent periodic DB sync yet. tile_coding/none/deep was already filtering; its R2 DB is intentionally stale
while the search process does not sync the study DB again until the pipeline ends.

### Where to continue (added 2026-10-05, late)

Preliminary results (16 of 18 cells), the conclusions drawn so far, the post-pipeline EVALUATION plan (policy saving,
robustness suite, A2C port, DQN-settings test), the strengths of SSP, and the prioritized next experiments are in
`docs/handoffs/03-prelim-results-and-eval-plan.md`. Read that file next.
