# RUN_M05 — round 1 pilots under the time-outs-kept wrapper

Started 2026-11-01. Design: decisions 77-78 in `../02-decisions.md`. Data: campaign `pilot1-9f3a1c` on R2
(`mjlab_ssp_centering/pilot1-9f3a1c/runs/<job_id>/`), restored with `analysis/restore_runs.py`; readout with
`analysis/pilot_readout.py`.

## 0. Why

RUN_M04 (decision 75) found that the wrapper's *removal of the time-outs* is what stopped G1 learning: with time-outs
kept and a fall non-terminal, G1 walks (`track_lin_frac` 0.811) exactly as in mjlab's episodic env (0.813), while the
no-time-out wrapper sits at the standing floor (0.307) and centering does not rescue it (0.309). Decision 77 therefore
made time-outs-kept the default protocol. Every Go1 cell measured so far used the legacy protocol, so the pilot asks
whether the published main effect survives the change, and what run length it needs.

## 1. Runs (`manifests/pilot1.json`, 2000 iterations, 2048 envs, raw input, gamma .999)

| block | cells | seeds | runs |
|---|---|---|---|
| pilot, Go1 | {none, value} x {offset 0, offset c} | 1000, 1001 | 8 |
| pilot, G1 | {none, value} x {offset 0, offset c} | 1000, 1001, 1002 | 12 |
| g1rep, G1 wrapper replication | no-time-outs (legacy, penalty 10) vs time-outs-kept (penalty 0), offset 0, no centering | 1001, 1002 | 4 |

All pilot cells use the new protocol: `time_outs=True`, `falls_terminal=False`, `penalty=0`, value centering at
eta 0.03 with `r_bar` initialised from the first batch.

## 2. Readings fixed before launch (decision 78)

- "Walks" = mean `track_lin_frac` of the last 5 evals >= 0.5.
- The centering gap is the paired difference (value - none) on shared seeds, reported with its seed count; 2 Go1 seeds
  and 3 G1 seeds are pilots, not claims.
- Run length is read off where the Go1 centered curve flattens, to set the length of the 20-seed confirmation.
- If either legacy replication seed walks, the RUN_M04 conclusion is weakened to seed-specific and re-examined first.

## 3. Launch

Two pods, RTX 5090 COMMUNITY at $0.69/h (the 4090 COMMUNITY, A6000 and A40 pools were empty at launch), campaign
`pilot1-9f3a1c`, orchestrator hard cap $14 / soft cap $12.

## 3b. Protocol verification (before reading any result)

- Shipped-code check: the R2 tarball `pilot1-9f3a1c/_code/mjlab_centering.tar.gz` contains `env_cfg(..., time_outs=True)`
  as the default, `RunSpec.time_outs=True` / `falls_terminal=False`, and the wrapper's `time_outs` / `fall_check_deg`
  paths, so the pods are not running a stale build.
- Local end-to-end smoke of the exact training path (Go1, 64 envs, value centering, offset 1, 4 iterations):
  `offset_now` 1.0, `r_bar` initialised from the first batch (0.988), falls counted through `bad_orientation`
  (5 -> 14 -> 13 events, 3.3-9.1 per 1k) although a fall is no longer terminal, and eval rows written normally.
- Throughput on the 5090 pods: 0.51 s/iteration, i.e. ~17 min per 2000-iteration job.

## 4. Results so far (campaign re-scoped mid-flight, decision 79)

### 4.1 The G1 wrapper replication does not reproduce RUN_M04

| arm (offset 0, no centering, gamma .99) | seed | track_lin (last 5) | J_task |
|---|---|---|---|
| no time-outs (legacy) | 1001 | 0.308 | 0.0634 |
| no time-outs (legacy) | 1002 | 0.324 | 0.0547 |
| time-outs kept | 1001 | 0.309 | 0.0605 |
| time-outs kept | 1002 | 0.300 | 0.0537 |
| time-outs kept (RUN_M04) | 1000 | 0.811 | 0.0924 |

Neither arm walks on the two fresh seeds, and the two arms are indistinguishable, so RUN_M04's "time-outs are the
fix" rests on a single seed that also produced the walking episodic arm. Note also that the fall metric is now real:
the new-protocol G1 runs sit at 700-1000 falls per 1k steps, i.e. on the ground, while the legacy arm reads near 0
because its falls are terminations that the wrapper hides.

### 4.2 Cells added while the campaign runs (decision 79)

- `pilot99_g1_raw_none_o0_s{1000,1003,1004}` and `pilot99_g1_raw_value_o0_s{1000,1003}`, gamma .99, offset 0: seed
  1000 is RUN_M04's walking seed routed through the new wrapper path; the others turn a point into a rate.
- `g1rep_episodic_none_o0_s{1003,1004}`: the episodic path on fresh seeds, separating "lucky seed" from "the wrapper
  path learns worse".
- `g1long_{timeouts,legacy}_none_o0_s1001_i4000`: 4000 iterations on one seed per arm, priority 10, to separate
  "G1 is slow" from "G1 is stuck".

### 4.3 Pilot cells (gamma .999) so far

All completed G1 gamma .999 cells (offset 0 and offset c, no centering) sit at 0.28-0.34 `track_lin`, i.e. the standing
floor; Go1 gamma .999 cells and the G1 centered cells are still pending.

### 4.3b Operations

The workstation cleared `/tmp` and killed the local orchestrator at about 21:14 (the pods are remote and unaffected;
this is the situation the "safe to close the terminal" question was about). No job was lost: the pods kept claiming
from R2. The orchestrator was restarted detached (`setsid`, own session) with its log moved from `/tmp` to
`results/orch_pilot1.log` so it survives a `/tmp` wipe; it picked up the same state file (`spend` was $2.52 with two
pods live at 25 jobs pending).

The campaign grew from 24 to 33 jobs with decision 79's additions, at the same $12 soft / $14 hard caps.

### 4.5 The episodic control on fresh seeds fails (decision 85)

`g1rep_episodic_none_o0_s1003` 0.309 and `s1004` 0.298 (last-5 `track_lin`), J_task 0.0597 / 0.0600. With seed 1000 at
0.813 the episodic path walks on **1 of 3 seeds**, so G1's difficulty is seed-dependent even on mjlab's own env and the
"our wrapper broke it" branch is closed. The separate, real finding stands: with the fall reset removed (decision 81)
nothing walks, including seed 1000; with it restored the corrected protocol walks on seed 1000 (0.745 at iteration 1999,
12.9 falls per 1k, action std 0.39).

### 4.6 Campaign v1: what it cost and what it is

10 jobs before the cut, plus 3 that were already claimed, about $2.7 of the campaign. All of them ran the decision-81
protocol and are kept only as the negative control: G1 at `track_lin` 0.28-0.34 with 700-1000 falls per 1k.

### 4.7 Corrected protocol (campaign `fix1-b7c40a`, decisions 81-86)

G1, gamma .99, offset 0, penalty 0, 2000 iterations - the decisive three seeds:

| seed | track_lin (last 5) | J_task |
|---|---|---|
| 1000 | 0.73 | 0.092 |
| 1001 | 0.74 | 0.091 |
| 1002 | 0.73 | 0.091 |

3 of 3 walk, take-off between iterations 49 and 349, falls 11.7-12.9 per 1k (healthy, unlike the 958 of the buggy
protocol), action std 0.38. Against 1 of 3 on the episodic path - not paired, because the episodic seeds were
1000/1003/1004.

Go1, gamma .999, offset 0, penalty 10, uncentered: flat at `track_lin` 0.29-0.32 from iteration 49 to 1849, J_task
0.069-0.071 against a standing baseline of 0.077, action std 0.72-0.80, falls 0.0 per 1k. Not an artefact of a stuck
protocol: `steps_since_reset_max` is pinned at 999, so the 20 s time-outs fire as designed. Interpretation waits on the
centered counterpart at the same cell and, if that also fails, one gamma .99 corrected-protocol control.

### 4.8 Corrected-protocol factorial (fix1-b7c40a, decisions 89)

Go1, gamma .999, 2000 iterations, 2 seeds per cell, corrected wrapper:

| cell | track_lin | J_task | walks |
|---|---|---|---|
| offset 0, no centering | 0.309 | 0.0710 | 0/2 |
| offset 0, value centering | 0.885 | 0.0998 | 2/2 |
| offset 1.0, no centering | 0.303 | 0.0384 | 0/2 |
| offset 1.0, value centering | 0.880 | 0.0994 | 2/2 |

Paired gaps (value - none), same seeds: offset 0 dJ +0.0288 (se 0.0011, per-seed +0.0298/+0.0277), dtrack +0.576;
offset 1.0 dJ +0.0611 (se 0.0051, per-seed +0.0560/+0.0662), dtrack +0.577. So the centering effect reproduces under
the protocol the study will use, and at gamma .999 it needs no reward offset: uncentered sits at the floor with offset
0 too. Two seeds here; the 20-seed confirmation is `w2main`.

G1 under the same protocol (penalty 0): 3/3 seeds walk, 0.732 track, J 0.0913. The penalty-10 arm is at the floor
(0.304 track at the time of writing), so the pre-registered 5.7 fallback is expected to fire for the G1 centering cells.

### 4.9 Claim-grade confirmation, 20 fresh seeds (decision 93)

Go1, gamma .999, offset 1.0, 2000 iterations, seeds 3000-3019, corrected wrapper:

| arm | n | track_lin | J_task | walks |
|---|---|---|---|---|
| no centering | 20 | 0.307 | 0.0459 | 0/20 (max 0.328) |
| value centering | 18 completed | 0.884 | 0.0999 | 18/18 (0.875-0.891) |

Paired: dJ +0.0534, se 0.0019, 95% CI [+0.0497, +0.0571], dtrack +0.578; all 18 per-seed differences positive.

### 4.10 Offset-0 block and shock block (decisions 94-95)

Offset 0, gamma .999, 10 fresh seeds: uncentered 0.882 / J 0.0993 (10/10 walk), centered 0.884 / 0.1000 (10/10), paired
dJ +0.0007 (se 0.0003). At gamma .99: uncentered 0.888, centered 0.890 (3 seeds each). So the 2-seed pilot reading
("centering needed without an offset") was seed noise; the effect requires the constant offset.

Shock at iteration 1200, Go1 centered offset c gamma .999, measured against each run's own iteration-1149 level:
+0.1 offset (3 of 5 seeds) leaves no measurable dip (end 0.86-0.88); action-rate weight x5 (5 of 5) costs 0.05-0.07
track and recovers 0.38-0.54 of the dip by iteration 1999 (end 0.84-0.85).

### 4.11 G1 second robot, 6 seeds per arm (decision 98)

G1, gamma .99, offset 1, penalty 5.7, corrected wrapper, 2000 iterations, seeds 1000-1005:

| arm | n | track_lin | J_task | walks |
|---|---|---|---|---|
| no centering | 6 | 0.312 | 0.0584 | **0/6** |
| value centering | 6 | 0.793 | 0.0883 | **6/6** |

Paired dJ +0.0299 (se 0.0012), dtrack +0.481. At gamma .999 offset 0 the humanoid is unstable (none 0.527, 2/3 walk;
value 0.398, 0/3; dJ -0.0036), so no claim there.

### 4.4 Open

The centering main effect on Go1 under the new wrapper, the G1 centered cells, and the verdict in decision 79.
