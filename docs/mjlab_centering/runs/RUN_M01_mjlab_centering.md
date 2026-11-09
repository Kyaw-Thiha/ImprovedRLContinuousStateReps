# RUN_M01 — mjlab reward centering on a continuing Go1 task (benchmark, gate, campaign)

> **Status 2026-10-31 (later): superseded in part.** Decisions 35-47 in `../02-decisions.md` replace the pod-failure
> narrative below: bootstrap is ~2 minutes (not 95), the harness runs reliably when a pod is driven over SSH with
> `podrun.py` (several training processes per GPU), the corrected gate shows the continuing wrapper does not change
> learning (wrapped = episodic, tracking f 0.89 vs standing 0.28), and the SSP arm in the old manifests trained on raw
> observations (the encoder was never wired). Results of the waves are recorded in `RUN_M02_*.md`, not here.


Status: in progress (started 2026-10-31). Campaign ids, spend and every decision are recorded below as they happen.
Plan: [`../00-plan.md`](../00-plan.md). Frozen predictions: [`../01-preregistration.md`](../01-preregistration.md).

## 0. What this run answers

Is reward centering a distinct fix for PPO at high discount / under a reward offset on a GPU-parallel, continuing
robot task, or is it another way of saying "make the critic easier to fit"? And does an SSP input encoder add
anything on top of it? (SSP is the deferred tier.)

## 1. Harness verification (local, before any pod)

- 33 tests green: centering rules and collection-order invariance, continuing wrapper algebra, live reward-weight
  shock, SSP parity against `sspspace` (`np.allclose`), job-matrix invariants, derived-metric arithmetic, R2 claim
  semantics, orchestrator cost accounting and secret redaction.
- End-to-end smoke for every arm (`none`, `simple`, `value`, clip-off, return-norm, reward-scale, action-rate shock)
  at 16-32 envs on the laptop RTX 3050: `iter.jsonl`, `eval.jsonl`, `final.json`, `model_final.pt` and the R2 push
  all produced; no NaNs.

## 2. Benchmark (`b1-*`, NVIDIA A40 SECURE, $0.59/h)

| envs | reported `fps` (collection+update only) | true wall time per iteration |
|---|---|---|
| 1024 | 36k | - |
| 2048 | 61k | ~1.2 s (estimated from the 4096 measurement) |
| 4096 | 100-104k | **2.2-2.3 s** (from `ts` deltas in `iter.jsonl`) |

The reported `fps` over-reports by 2-4x because it times only collection and update; the wall-clock rate at 4096 envs
is ~43k env-steps/s. Two consequences:

- **Settings**: 2048 envs x 600 iterations = 29.5M env-steps per run, ~12 min, ~$0.12 on an A40. The 4096 x 800 design
  would have cost ~$28 for the matrix; this one costs ~$11-13.
- **Harness fix**: `process_env_step` was converting five running accumulators to Python floats every environment step,
  forcing five GPU->host syncs per step and serialising the rollout. They are now GPU tensors flushed once per
  iteration (decision 17).

## 2b. Eta sweep (`e1-*`, value centering, gamma=0.999, offset c, 600 iterations)

| eta | mean of last-5 evals (seed 2000) |
|---|---|
| 0.003 | 0.0511 |
| 0.01 | 0.0448 |

eta = 0.003 is frozen for every centered cell. Caveats: only seed 2000 completed before the sweep was stopped to
redeploy a fixed harness (the second seed and eta=0.3 never ran), so this is a one-seed ranking of a nuisance
parameter, not a result. Recorded as a deviation.

## 3. Campaign

| block | cells | seeds | iters |
|---|---|---|---|
| eta sweep | value, gamma=.999, offset c, eta in {0.003, winner, 0.3} | 2 | |
| A | {none,value} x gamma {.99,.999} x offset {0,c} | 5 | |
| fixclip / fixnorm / fixscale | none, gamma=.999, offset c | 5 / 5 / 3 | |
| E (shock) | {none,value} x {offset shift, action_rate x5} | 5 | 1.4x, shock at 0.8x |
| pen | reference arm, P x0.3 and x3 | 2 each | |
| lr | 2 non-default rates x {none,value} | 2 | |
| ssp (deferred) | ssp_low x {none,value} | 3 | |

## 4. Results

Filled from `analysis/derive.py` (tables copied verbatim; no hand-editing of numbers).

### 4.1 Partial A cells (gamma=0.999, offset 0, 400-iteration runs, INCOMPLETE)

| arm | seed | evals | last J_task | last-5 mean |
|---|---|---|---|---|
| none | 2000 | 9 | +0.0759 | +0.0766 |
| none | 2002 | 5 | +0.0758 | +0.0762 |
| value | 2000 | 10 | +0.0744 | +0.0754 |
| value | 2002 | 4 | +0.0746 | +0.0603 (3 evals) |

Read as plumbing evidence only: at offset 0 the centered and uncentered arms are within ~0.002 of each other, which is
what prediction P1/P3 says should happen when there is no constant reward offset to absorb. No threshold, T50/T90 or
CI is computed from partial runs.

## 4b. How to finish and derive

```bash
# campaigns (uncentered / centered halves), 400 iterations, 2048 envs, seeds 2000-2004
python orchestrator.py --campaign u2-649ffe59 --manifest manifests/u2-649ffe59.json --launch 3
python orchestrator.py --campaign c3-37e5a443 --manifest manifests/c3-37e5a443.json --launch 3

# when every job has a DONE marker
python analysis/derive.py --campaign u2-649ffe59,c3-37e5a443 --pull --out results/derived_main.json
python analysis/status.py --campaign u2-649ffe59
```

## 4c. What did NOT complete, and why

**Final status at the end of this session: 1 run of the whole programme reached `DONE` (an eta-sweep cell, 600
iterations). No centering comparison is reported.** The harness is complete and tested; the blocker is pod-side
runtime, described below with the exact measurements that make it a runtime problem rather than a logic problem.

**Diagnostic result (single 150-iteration job, fresh A40 pod, `d1-f61dbe3d`).** The training loop is healthy and the
failure is at the end of the run:

| measurement | value |
|---|---|
| wall clock per iteration (2048 envs, from `ts` deltas) | **1.69-1.88 s** (collection 1.06-1.42, update 0.41, push 0.17) |
| periodic eval (`t_eval`, 256 envs x 300 steps) | 7.7 s |
| `J_task` at iteration 24 | **+0.0767** (matches the plateau of full runs) |
| progress observed | eval rows to iteration 124 while `iter.jsonl` showed 99 rows: the two sinks have different push cadences, so the earlier "counts reset" reading was lag between sinks, **not a restart** |
| `final.json` / `DONE` | **never written**, although the same code path completes locally in 74 s |

**Cause identified (probe, `bg_86`): the container was restarting.** A standalone probe pod printed its startup block
repeatedly, every ~16 seconds. RunPod restarts a container whenever its entrypoint exits, and this entrypoint is the
bootstrap + worker — so any exit re-runs the entire ~95-minute bootstrap (image pull, torch install, mjlab git pin,
JIT warm-up) and the visible progress starts over. The same probe also settled the two other candidates:

| candidate | measurement | verdict |
|---|---|---|
| pod clock skew (SigV4 allows 15 min) | `pod_utc 12:59:11Z` vs R2 `Date: Sat, 31 Oct 2026 12:59:11 GMT` | **ruled out** |
| upload throughput / rate limiting | 20/20 sequential 1 KB PUTs at 0.41-0.83 s, 1 MB 0.48 s, 5 MB 0.78 s | **ruled out** |
| entrypoint exit -> container restart | startup block repeats every ~16 s | **cause** |

Fix shipped: the worker no longer exits on job failure (it parks for 10 minutes and retries, leaving the warm cache
intact), every R2 call in its error path is exception-guarded, and `run()` wraps the final artefact uploads so a
finished run can never raise into the worker. Local end-to-end verification passes after the change.

So it is not a slow loop and not an upload problem: the run trains correctly, and the loss was the pod recycling
itself after each job. The remaining verification is the end-of-run path (three deployment
evals + the four artefact uploads + `DONE`) does not complete on the pod. The uploads are the only part that differs
from the local run, and the earlier evidence is consistent with it (incremental mirroring lags by minutes; see the
claim/upload notes below).

Measured on the pod (from `iter.jsonl` `ts` deltas, inside one 25-iteration window):
- 0.89-0.96 s per iteration at 2048 envs (collection 0.55 s + update 0.21 s + eval amortised 0.17 s);
- **but** the same run's iteration count in R2 grew at 50-100 iterations per 25-40 minutes. Both cannot be true, so
  the run either restarts or its per-iteration time degrades after the first few dozen iterations. The `ts` evidence
  rules out a uniform slow loop; it does not distinguish a restart from a slowdown, because the local file is
  rewritten on a restart and R2 only ever shows the latest snapshot.

Contributing, now-fixed causes found along the way:
1. **Double-assigned jobs.** R2's conditional put is not linearisable under concurrency: 8 concurrent claimers
   produced **2 winners** in `tests/test_r2_claims.py`. Two pods then ran and overwrote the same job, which truncates
   `iter.jsonl` and makes a healthy run look like a restart. Fixed by verifying the claim by read-back (nonce) after
   writing; the test now passes repeatedly.
2. **Pod churn.** `--launch 3` once created 12 pods because the pod list lags; counting now uses the orchestrator's
   own state, and a disappeared pod is only closed after 15 minutes.
3. **Bootstrap cost.** A fresh pod needs ~65 minutes before its first job (venv + torch + mjlab git pin + the
   mujoco-warp JIT warm-up). Repeated restarts during the session therefore never gave a pod time to finish.
4. **Uploads.** Inline uploads blocked the loop; mirroring is now asynchronous with bounded timeouts, and
   checkpoints are not uploaded by default.

Next diagnostic (10 minutes, not more compute): one pod, SSH enabled, a job of 100 iterations, and a per-iteration
log of `ts`, `t_collect`, `t_learn`, `time.time()` deltas around the eval, plus `torch.cuda.memory_allocated()` and
`nvidia-smi` memory every 10 iterations. That distinguishes (a) a restart (iteration counter resets), (b) a
memory/allocator drift (memory climbs, iteration time climbs), and (c) an upload stall (a gap between `ts` deltas that
is not inside any timed section). Until one of those is identified, a multi-hour campaign cannot be trusted.

### 4c-bis. Last attempt (`u4`/`c5`, 70 runs, 400 iterations) and the fix that shipped

`run()` now writes and uploads `final.json` and `DONE` **before** the deployment evals (verified locally: `final.json`
contains the periodic-eval record even when the deploy evals are skipped or fail), so a finished training run can no
longer be lost to the end-of-run sequence. Both campaigns launched, pods bootstrapped (~95 min), jobs started
(`config.json` written, eval rows arriving) — but still no `DONE` inside the session, so the failure is in the
*artefact uploads themselves*, not in their ordering. All pods were deleted at the end of the session.

Remaining hypotheses, in order of testability:
1. **Pod clock drift.** SigV4 rejects requests more than 15 minutes off; early uploads succeed and later ones fail.
   Test: `date -u` on the pod vs the `Date` header returned by R2 (`curl -sI https://<account>.r2...`). If it drifts,
   run `chronyc makestep` / `ntpd` in the bootstrap or re-sign with a corrected clock.
2. **Egress throttling after sustained uploads.** Test: 20 sequential 1 KB PUTs, logging status and duration.
3. **Size-dependent failure at the end of a run.** The only files that never land are written at the end; test by
   uploading a dummy 100 KB file at start and at end of a run.

### 4c-ter. Run inventory (started vs completed, from R2)

| campaign | purpose | runs started | runs completed (`DONE`) |
|---|---|---|---|
| `b1-4d73b6d1` | throughput benchmark + plateau gate | 8 | 6 (all six benchmark runs; the two plateau runs never finished) |
| `e1-4562778f` | eta sweep | 2 | 1 |
| `u2-649ffe59`, `c3-37e5a443` | first split campaign | 3 + 3 | 0 |
| `u4-ae4c81d5`, `c5-5d4ad9f5` | split campaign, DONE-before-evals fix | 2 + 1 | 0 |
| `v1-2b6c6bba` | two-run verification after the restart-loop fix | 1 | 0 |

Seven runs completed in total across the session. Completion is possible but rare, and the failure is always the same
one described in 4c — a pod that recycles itself after a job, or a live run whose wall clock degrades to ~20 s per
iteration against a 1.9 s in-window measurement.

### 4d. Attempt log (why there are so many campaign ids)

`b1` benchmark (complete) -> `e1` eta sweep (1 of 6 done, stopped to redeploy) -> `u1`/`c2` (churned, no
completions) -> `u2`/`c3` (no completions, claim double-assignment found) -> `u3`/`c4` (first jobs claimed after a
65-minute bootstrap; stopped at the end of the session). Every restart was a deliberate fix for a defect listed
above, not a re-run for a better number.


No campaign run reached a `DONE` marker inside this session, so **no centering comparison is reported**. What was
observed, in order:

1. **Deployment** (first ~2 h). Five separate bootstrap failures, each found and fixed by reading the pod's log
   (boto3 missing, `/project` absent, PEP-668 `--system` refusal, mjlab/PyPI rsl-rl pin conflict, missing `git`,
   `MUJOCO_GL` EGL crash). Every fix is in the code and in decisions 11-14.
2. **Throughput** (next ~1 h). The first per-run measurements were dominated by the mujoco-warp JIT cache (~30-40 min
   per cold container) and by a per-step GPU->host sync bug that serialised the rollout (fixed: 2.2 s -> 0.9 s per
   2048-env iteration, decisions 16/17/19).
3. **Pod identity and uploads** (rest of the session). Runs repeatedly showed 50-250 of 400 iterations in R2 while
   their own `ts` deltas said ~0.9 s/iteration, and no `final.json` ever appeared. The pod list also lags, which made
   `--launch 3` create up to 12 pods. Fixes applied: asynchronous JSONL mirroring, retrying end-of-run uploads,
   state-based live-pod counting, checkpoints no longer uploaded. These reduced but did not eliminate the symptom;
   the remaining hypothesis is that the pod's egress to R2 degrades after the first several uploads (every local
   upload path, the full run, and derive are verified working, so the failure is pod-side). **This is unresolved.**

Consequence: the harness is usable and tested, but the campaign needs the upload path verified on a pod (and mid-run
resume) before a multi-hour run can be trusted. The next step is not more compute but a 10-minute diagnostic on one
pod: run a single job with SSH enabled and a locally written `upload_probe.txt` that logs the status code and duration
of 20 sequential PUTs. That will distinguish rate limiting from packet loss and decide between "slow down the mirror"
and "ship results over a different channel".

Estimated spend: ~15-20 GPU-hours of A40 at $0.59/h plus ~$1 of failed-pod time, i.e. (estimate, WRONG) $10-13 of the $25. ACTUAL SPEND (RunPod billing API, 2026-10-31): $2.71 over 4.5 GPU-hours; account balance $28.17. Earlier estimates in this file were unmeasured extrapolations and were wrong. It was the unfinished budget, not an exhausted one, so the session stopped early on a false premise: roughly $22 of the $25 cap remained. The remaining text of this sentence refers to the
budget. Pod-level accounting in `results/orch_*.json` is incomplete because several orchestrator restarts reset their
state files; the pod list and `runpodctl` billing history are the authoritative sources.

## 5. Deviations from the plan

- Learning-rate grid is 8 runs (default rate already covered by A).
- The action-rate shock writes the reward manager's live term config (the manager deep-copies its cfg).
- GPU pool: 4090/A6000 community stock was exhausted at launch time; the fallback list ends L40S community
  ($0.79/h). Per-pod price is accounted individually.
