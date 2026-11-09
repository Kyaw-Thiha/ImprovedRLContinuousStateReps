# Handoff: the G1 wrapper finding, what replicated and what did not, and the pilot in flight

Written 2026-11-01 21:30 for an agent picking this up cold. Authoritative records: `../02-decisions.md` entries
73-80, `../runs/RUN_M04_g1_diagnosis.md`, `../runs/RUN_M05_pilot_round1.md`. Live campaign: `pilot1-9f3a1c`.

## 0. UPDATE 21:45 - decision 81: the new default had a bug, and this supersedes sections 3 and 5

**Removing the fall termination also removed the fall reset.** mjlab auto-resets an environment when a termination
fires, so popping `fell_over` did not just stop *calling* a fall terminal - it left the fallen robot lying on the floor
until the next time-out. Verified on G1 with the shipped default: at the first fall the episode buffer kept climbing
(11 -> 12 -> 13 -> 14) and the root height fell (0.41 -> 0.37 -> 0.33 -> 0.29), with dones 0 and no truncation.

That is the explanation for the 700-1000 falls per 1k steps in the new-protocol arms: after the first fall the robot is
flagged fallen for the rest of the 20 s episode. A humanoid cannot learn a gait from the floor, so every new-protocol
G1 cell in campaign `pilot1-9f3a1c` measures that pathology, not the centering question. It also means the Go1 cells
must not be read from that protocol either.

**Corrected default (decision 81).** Keep both terminations in the env and hide only the fall from PPO:

- `build_env_cfg(falls_reset=True, time_outs=True)` - `fell_over` stays (the env resets the robot in place), `time_out`
  stays (the segment is truncated and bootstrapped).
- `ContinuingVecEnv(time_outs=True)` - `dones` comes from `truncated` only; a fall is a reset without a done. That is
  the Wan-style "continuing with predefined resets" the study always meant.
- Verified after the fix: at the fall step the buffer is 0, the root height is back at 0.80, dones 0, `time_outs`
  False. Two regression tests cover it (one config-level, one driving G1 with saturating actions).
- `falls_terminal` is renamed `falls_reset`; "robot stays on the floor" survives only as `--no-falls-reset`.

**What that did to the money already spent.** The 10 completed `pilot1-9f3a1c` jobs (4 replication arms, 6 gamma .999
G1 cells) ran the buggy protocol and are kept as a labelled negative control (v1). Campaign 1 was cut back to its two
episodic cells, which are unaffected because the episodic path never popped `fell_over`. The rest were paused, and the
corrected pilot is campaign **`fix1-b7c40a`**, 11 jobs, caps $6 soft / $8 hard:

| block | cells | purpose |
|---|---|---|
| G1 (priority 0) | `fix_g1_raw_none_o0_s{1000,1001,1002}`, gamma .99, offset 0, penalty 0 | does G1 walk at all with the fall reset restored |
| Go1 (priority 1) | `fix_go1_raw_{none,value}_o{0,c}_s{1000,1001}`, gamma .999, penalty 10 | does the centering main effect survive the protocol change |

**Guards added (decisions 81-82).** `test_a_fall_resets_the_robot_without_becoming_a_done` asserts on a driven G1 that
the fall step resets the buffer to 0 and the root back above 0.5 m while dones stays 0 and `time_outs` is false;
`test_penalty_is_charged_once_on_the_fall_step_only` asserts the same, plus that the reward is `task - P + offset`
exactly once on that step, that the fall flag is false on the next step, and that the cumulative charge equals P x
falls. Suite: 80 passed.

**Waves 1-9 are unaffected (decision 83).** The legacy protocol never popped `fell_over` - it removed only `time_out`
- so a fall still terminated and reset the robot while `dones` stayed zero. Code check plus data check: legacy Go1 fall
rate collapses to 0.02 per 1k, and the legacy G1 probe's `steps_since_reset_max` grows to ~39,400 with isolated
single-fall resets, which is impossible without resets. Only the 10 v1 pilot jobs were affected.

**Penalty (decision 82).** The pilot's penalty 0 matches RUN_M04's walking cell for the diagnosis only; it is not the
protocol. If `fix_g1_raw_none_o0_s1000` walks, `fix_g1_pen10_none_o0_s1000` runs (~$0.3) before any G1 centering cell. If
penalty 10 breaks G1, the per-robot rule is `P = 130 x standing per-step task reward` (Go1 10 as frozen, G1 5.7), and
that deviation is logged with the result.

**G1 SECOND ROBOT, final (decision 98, 14:20).** G1, gamma .99, offset 1, penalty 5.7, **6 seeds per arm**: uncentered
**0/6** walk (mean 0.312, J 0.0584) against centered **6/6** (0.793, 0.0883), paired **dJ +0.0299 (se 0.0012)**,
dtrack +0.481. At gamma .999
offset 0 the humanoid is unstable and centering does not help (0.407/0.502/0.671 vs 0.408/0.376/0.408, dJ -0.0036), so
no claim there. Penalty correction: the full penalty-10 record run reaches 0.550, so "penalty 10 breaks G1" becomes
"penalty 10 costs most of the tracking reward at this budget" (5.7 gives 0.787 at the same cell).

**OFFSET-0 REFUTATION (decision 94, 05:00).** Ten fresh seeds at gamma .999, offset 0: uncentered **walks 10/10**
(track 0.882, J 0.0993), centered 10/10 (0.884, 0.1000), paired dJ +0.0007 (se 0.0003) - a tie. At gamma .99 also a
tie (0.888 vs 0.890). The 2-seed pilot that suggested centering was needed without an offset was seed noise; the
supported claim is that the constant offset is what makes centering necessary.

**SHOCK RESULT (decision 95).** Against each run's own pre-shock level: +0.1 offset (3 seeds) no measurable dip; action-rate
weight x5 (5 seeds) costs 0.05-0.07 track and recovers 0.38-0.54 of it within 800 iterations.

**MAIN RESULT, 20 seeds (decision 93, 02:45).** Go1, gamma .999, offset 1.0, 2000 iterations, seeds 3000-3019:
uncentered 0/20 walks (track 0.307, J 0.0459, max 0.328); centered 18/18 completed seeds walking (track 0.884,
J 0.0999, band 0.875-0.891). Paired dJ **+0.0534 (se 0.0019, 95% CI [+0.0497, +0.0571])**, dtrack +0.578, every
per-seed difference positive. This is the paper's claim-grade number under the corrected protocol.

**MAIN RESULT, corrected protocol (decision 89, 00:20).** Go1 at gamma .999, 2000 iterations, 2 seeds per cell:
offset 0 no centering 0.309 track / 0.0710 J, offset 0 centered **0.885 / 0.0998**; offset 1.0 no centering
0.303 / 0.0384, offset 1.0 centered **0.880 / 0.0994**. Paired dJ +0.0288 (offset 0) and +0.0611 (offset 1.0), dtrack
+0.576 / +0.577 in both. So the main effect survives the protocol change, and at gamma .999 centering is needed even
with no reward offset. G1 at penalty 0 walks 3/3 (0.732); penalty 10 is at the floor, so the 5.7 rule is expected to
apply to the G1 centering cells.

**G1 gate (decision 84), armed and running.** `analysis/g1_gate.py --campaign fix1-b7c40a --wait-min 200 --launch`
waits for the three corrected G1 seeds and the penalty-10 arm, then applies: gate passes iff >= 2 of 3 seeds walk
(last-5 `track_lin` >= 0.5) AND the penalty-10 arm walks. On a pass it builds `manifests/g1c.json` (4 cells x 3 seeds:
{none, value} at gamma .99 offset 1, and {none, value} at gamma .999 offset 0) with the resolved G1 penalty (10 if the
penalty-10 arm walks, else 5.7) and starts that campaign detached with a $6 hard cap. On a fail it records the walking
rate and stops - no retry, no re-scoping; G1 is then reported as "walked on N of 3 seeds".

**Episodic fresh-seed result (decision 85, 22:20).** `g1rep_episodic_none_o0_s1003` 0.309 and `s1004` 0.298 - both at
the floor. With seed 1000 at 0.813 the episodic walk rate in this stack is **1 of 3**, so G1 walking is seed-dependent
even on mjlab's own env: the wrapper-defect branch is closed, and G1 claims must be walking rates, never means. The
decision-81 bug is still a real bug (without the fall reset nothing walks, including seed 1000), and the corrected
protocol walks on seed 1000 (0.745 at iteration 1999, 12.9 falls per 1k, std 0.39); whether it is less seed-fragile is
what the gate measures.

**Go1 reading rule.** The 8 queued Go1 pilot cells are the only test of whether the centering main effect survives the
protocol change (time-outs now kept). Until they land, the waves 1-9 Go1 effect is quoted as "shown under the
no-time-out protocol" and nothing stronger.

**Standing rules from the user, to be applied when reading results:**
1. Do not interpret any Go1 new-protocol cell until G1 walks in this harness. 2. Do not run G1 centering cells at
gamma .999 while G1 has never walked at gamma .99. 3. The replication cells decide the branch: episodic walks on
1003/1004 => our wrapper path; episodic fails on fresh seeds => our harness or the pins, in which case run one G1 seed
with mjlab's own `mjlab.scripts.train` at our pinned commit; episodic is seed-fragile => G1 claims need a walking rate
over many seeds, never a mean. 4. G1 gets at most ~$2 more; if it cannot be made to walk reliably, drop it from this
paper as "G1 in this stack walked on 1 of N seeds" and spend the money on the Go1 20-seed confirmation.

## 1. TL;DR of the reading update

Three hours ago the claim was: **"G1 does not walk because the wrapper removed the time-outs; keeping them while a fall
stays non-terminal walks as well as the fully episodic env."** That rested on four single-seed runs (RUN_M04).

Two fresh seeds added today **do not reproduce it**: the time-outs-kept arm sits at the standing floor and is
statistically indistinguishable from the no-time-outs arm. So the earlier claim is now an unresolved three-way
question, not a result:

1. **Seed variance** - only seed 1000 has ever walked, and both RUN_M04 arms that walked were that same seed, so they
   were one observation, not two.
2. **The new wrapper path learns worse than the episodic path** - RUN_M04's walking arm ran on mjlab's
   `RslRlVecEnvWrapper`; the current default routes the same env through our `ContinuingVecEnv` with time-outs passed
   through. A defect there would explain it.
3. **Gamma** - every G1 pilot cell at gamma .999 is at the floor so far, and the only G1 walking ever observed was at
   gamma .99.

Tests for all three are already queued inside the running campaign (section 5).

## 2. Verified numbers

G1 velocity tracking in mjlab, 2048 envs, 2000 iterations, raw input, reported as the mean of the last 5 evals.
Reference levels: a standing G1 scores `track_lin_frac` 0.267 and `J_task` 0.0436. Values in the 0.28-0.35 band are a
pseudo-null (see section 7), so "walks" was pre-defined as `track_lin_frac >= 0.5`.

RUN_M04 (2026-11-01, seed 1000, gamma .99, offset 0, no centering):

| arm | vec path | time-outs | falls terminal | track_lin | J_task |
|---|---|---|---|---|---|
| wrapped | ContinuingVecEnv, dones forced 0 | no | no | 0.307 | 0.0472 |
| wrapped + centering (offset 1) | ContinuingVecEnv | no | no | 0.309 | 0.0458 |
| episodic | RslRlVecEnvWrapper | yes | yes | **0.813** | 0.0926 |
| time-outs kept, falls non-terminal | RslRlVecEnvWrapper | yes | no | **0.811** | 0.0924 |

RUN_M05 replication (2026-11-01, gamma .99, offset 0, no centering), same env as the walking arm, different seeds:

| arm | seed | track_lin | J_task |
|---|---|---|---|
| no time-outs (legacy) | 1001 | 0.308 | 0.0634 |
| no time-outs (legacy) | 1002 | 0.324 | 0.0547 |
| time-outs kept | 1001 | 0.309 | 0.0605 |
| time-outs kept | 1002 | 0.300 | 0.0537 |

So on fresh seeds **both** arms fail and they are indistinguishable. Only seed 1000 walks.

Gamma .999 pilot cells completed so far (new wrapper default, 2000 iterations):

| cell | n done | track_lin | J_task | action std |
|---|---|---|---|---|
| g1, none centering, offset 0 | 3/3 | 0.299 | 0.0542 | 0.73 |
| g1, none centering, offset 1 | 2/3 | 0.290 | -0.0753 | 1.32 |

The offset-1 cell's action std growing to 1.32 is the same signature the legacy Go1 uncentered arms showed while
failing (REPORT_002 section 2.2), which is at least consistent with the reward offset hurting again at gamma .999.

## 3. Protocol semantics (what "time-outs kept" means in code)

Decision 77 made this the default:

- `build_env_cfg(..., continuing=True, time_outs=True, falls_terminal=False)` - mjlab's `time_out` termination stays,
  the `fell_over` termination is removed, episodes are 20 s again.
- `ContinuingVecEnv(..., time_outs=True, fall_check_deg=70)` - `dones` comes from `truncated` instead of being forced
  to zero, `extras["time_outs"]` is preserved so rsl-rl bootstraps on the truncation, and falls are still *measured* by
  evaluating `bad_orientation` at 70 deg.
- The legacy protocol (waves 1-9) is now opt-in: `time_outs=False, falls_terminal=True`, penalty 10.
- Pilot cells use `penalty=0` and `offset` 0 or c.

Continuation semantics that the study needs survive: no terminal state and no episode-return objective; only
truncated segments that are bootstrapped. That was the argument for adopting it.

## 4. The campaign in flight

- Campaign id `pilot1-9f3a1c`, manifest `mjlab_centering/manifests/pilot1.json` (also mirrored on R2 as
  `pilot1-9f3a1c/manifest.json`; the pods read the R2 copy, so editing the file without `put_json` changes nothing).
- 33 jobs, 9 done at 21:30, two pods (RTX 5090 COMMUNITY, $0.69/h each), 0.51 s/iteration, ~17 min per 2000-iteration
  job. Spend $2.71 of a $12 soft / $14 hard cap.
- Orchestrator runs detached (`setsid`, own session) with its log at `mjlab_centering/results/orch_pilot1.log`; it
  deletes pods when the queue drains. A workstation reboot cleared `/tmp` and killed an earlier copy at 21:14 - no job
  was lost, the pods kept claiming from R2.
- The campaign grew from the 24 jobs the user specified to 33 with decision 79's additions. Money spent so far is
  roughly $2.7, and the whole thing projects to about $5-6.

## 5. Jobs queued to settle the three hypotheses (decision 79, readings fixed in advance)

| job(s) | what it separates |
|---|---|
| `g1rep_episodic_none_o0_s1003`, `s1004` (gamma .99) | episodic path on fresh seeds. Walks here but not in the wrapper path => the wrapper has a defect |
| `pilot99_g1_raw_none_o0_s1000` | RUN_M04's walking seed routed through the *new* wrapper path. Walks => seed variance is enough to explain the replication failure |
| `pilot99_g1_raw_none_o0_s1003`, `s1004` and `pilot99_g1_raw_value_o0_s1000`, `s1003` | the rate at gamma .99, plus whether centering matters there |
| `g1long_{timeouts,legacy}_none_o0_s1001_i4000` (priority 10, last) | "G1 is slow" vs "G1 is stuck": 4000 iterations on one seed per arm |
| 8 `pilot_go1_*` cells (gamma .999) | does the Go1 centering main effect survive the protocol change, and where does the curve flatten (run length) |
| 6 `pilot_g1_raw_value_*` cells (gamma .999) | does centering lift G1 at the study's main gamma |

Fixed interpretations:

- seed 1000 walks under the new wrapper and 1003/1004 do not => G1 walking is seed-fragile, decision 75 is downgraded
  to a single-seed observation, and the paper cannot lean on the wrapper story.
- the episodic path walks on 1003/1004 where the wrapper path does not => the new wrapper has a defect and must not be
  the study default until fixed.
- no G1 cell walks => G1 is not learnable within 2000 iterations under any wrapper at these two gammas, and the humanoid
  drops out of the centering claim.

## 6. Data-integrity defect found and handled (decision 80)

`pilot_g1_raw_none_o1_s1000`'s mirrored `iter.jsonl` holds 2336 rows with 500 duplicated iteration numbers: the job
restarted at iteration 0 and appended. Cause: the `podrun` queue cleared stale artefacts before a retry, the worker
path (used by every pod) did not. Fixed in `worker.clear_stale()` with a unit test, and the readout now trims to the
newest attempt and prints `[integrity]`. A scan of all 33 jobs found exactly this one affected. Pods already booted keep
the old code, so the fix protects future campaigns only.

## 7. Metric caveats a reader or agent must not get wrong

- **Fall rate is not comparable across protocols.** In the new protocol it is a real measurement
  (`bad_orientation`); in the legacy protocol it reads ~0 because falls were terminations the wrapper hid. New-protocol
  G1 arms report 700-1000 falls per 1k steps, i.e. at the ground most of the time; a direct check on an upright G1
  returns `False`, so the counter itself is not broken, but do not compare that column to legacy runs.
- **`track_lin_frac` 0.28-0.35 is a pseudo-null.** A standing G1 scores 0.267. Any policy that sits still, stands or
  lies down lands in that band; the informative jump is to ~0.8. Treat 0.30 as failure, not as partial success.
- **`J_task` is not comparable across robots.** G1 standing 0.0436, Go1 standing 0.077, Go1 trained ~0.10.
- Single-seed numbers are diagnoses, never claims; the seed rule for claims is 20 fresh seeds.

## 8. Where everything lives

- Code: `mjlab_centering/src/mjlab_centering/{env_cfg,wrapper,train,waves}.py`; tests `mjlab_centering/tests`
  (76 passing at the last full run).
- Analysis: `analysis/g1diag.py` (per-run readout), `analysis/pilot_readout.py` (grouped cells, paired centering gap,
  attempt dedupe), `analysis/status.py` (campaign status from R2), `analysis/restore_runs.py` (mirror -> local),
  `analysis/wait_for_rows.py` (smoke watchers).
- Data: R2 `mjlab_ssp_centering/<campaign>/runs/<job_id>/{iter.jsonl,eval.jsonl,final.json,DONE}`; restore with
  `python analysis/restore_runs.py --out results/merged_r2`; read with
  `python analysis/pilot_readout.py --campaign pilot1-9f3a1c`.
- Launch: `python orchestrator.py --campaign <id> --manifest manifests/pilot1.json --launch 2 --poll 120
  --hard-cap 14 --soft-cap 12` (detached: `setsid nohup ... &`).

## 9. Suggested next steps once the queue drains

1. Report the four decisive readings in section 5 and re-open or close decision 75 accordingly. Nothing else is worth
   reading until that is settled.
2. If the wrapper path is implicated, bisect it: the only differences from the episodic path are (a) `dones` from
   `truncated` + `extras["time_outs"]`, (b) the reward is `task + offset` rather than the env's own reward, (c) the
   `bad_orientation` call each step, (d) `extras["cc"]` present. A single 2000-iteration run with (c) disabled would
   cost about $0.3 and is the cheapest first bisect step.
3. If seed variance is implicated, G1 needs a seed-rate table (6-10 seeds at gamma .99, new wrapper, offset 0) before
   any humanoid claim, about $2-3, and the Go1 main-effect confirmation should not wait for it.
4. The shock test and the 20-seed confirmations in the session plan remain untouched by all of this.

## 10. What not to conclude

- Not "the wrapper change fixed G1" - that was one seed and it did not replicate.
- Not "G1 cannot learn" - the episodic path walked once at 0.81, and the 4000-iteration pair is still queued.
- Not "centering does not work on G1" - the G1 centered cells are mostly unrun, and the only completed ones are at
  gamma .999 where nothing walks.
