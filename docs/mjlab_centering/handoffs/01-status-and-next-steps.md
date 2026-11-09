# Handoff: mjlab centering track, status 2026-10-31 (end of session)

Read first: [`../00-plan.md`](../00-plan.md), [`../01-preregistration.md`](../01-preregistration.md),
[`../02-decisions.md`](../02-decisions.md) (26 entries, including every defect found), and
[`../runs/RUN_M01_mjlab_centering.md`](../runs/RUN_M01_mjlab_centering.md) §4c-4d.

## State

- **Code**: `mjlab_centering/` is complete and tested (33 tests green, `pytest tests -q`). Every arm of the matrix is
  smoke-tested locally end-to-end (training, periodic eval, deployment eval, R2 mirroring, claims, derivation).
- **Compute**: nothing is running; every pod was deleted and billing stopped. `results/orch_*.json` holds whatever
  spend accounting survived the restarts; treat `runpodctl` billing history as authoritative.
- **Results**: one eta-sweep cell completed (600 iterations). **No centering comparison exists yet.** Do not write up
  a centering result from this state.

## The one blocker (narrowed by the 150-iteration diagnostic)

A single 150-iteration job on a fresh A40 trained normally — 1.69 s/iteration at 2048 envs, `J_task` +0.0767 by
iteration 24, evaluator on schedule — but never wrote `final.json` or `DONE`, while the same end-of-run path completes
locally in 74 s. The training loop is fine; the **end-of-run sequence** (three deployment evals + four artefact
uploads + `DONE`) is what does not survive the pod. Fix `run()` in `src/mjlab_centering/train.py` in this order:

1. write `final.json` and `DONE` *before* the deployment evals, then upload the eval results separately;
2. make the deployment evals optional per campaign (`--final-steps 0`);
3. if uploads are still the problem, inline iter/eval/final into one small JSON and push that once.

Historical note (still useful):

A run's `iter.jsonl` shows 0.89-0.96 s per iteration at 2048 envs (from `ts` deltas inside one 25-iteration window),
yet the same run's visible iteration count grows at 50-100 per 25-40 minutes and `final.json`/`DONE` never appear.
Either the run restarts or its iteration time degrades after the first few dozen iterations; R2's snapshot cannot tell
them apart because a restart rewrites the local file.

## Next step (one pod, `analysis/profile_pod.py`)

`python analysis/profile_pod.py --iters 60 --num-envs 2048 > /tmp/profile.jsonl` on a bootstrapped pod. It prints, per
iteration, `wall_s`, `cpu_s`, each phase, the *unaccounted remainder*, GPU memory and RSS. Read it as:

- `other_s` consistently large -> the R2 mirror threads (disable them with `--r2 ''` and compare);
- `collect_s` growing with iteration number, GPU memory growing with it -> allocator/env drift;
- `wall_s` >> `cpu_s` with no phase accounting for it -> the container is descheduled (host contention), i.e. the pod
  type is the problem, not the code.

## Older plan (kept for reference)

1. Launch one pod with `--launch 1` and a campaign whose single job is `iters=100, num_envs=2048`.
2. On that pod (SSH now starts in the bootstrap), log per iteration: `ts`, `t_collect`, `t_learn`, wall time around
   the eval, `torch.cuda.memory_allocated()`, and `nvidia-smi` memory. Write these to a local file **and** stdout.
3. One of three outcomes decides the fix:
   - **iteration counter resets** -> the job is being re-claimed: check `claims/` for two pods with the same job and
     confirm the nonce-verified claim path is running (decision 26);
   - **memory climbs with iteration time** -> allocation drift in the env/optimiser; reduce `num_envs` or find the
     leak before any campaign;
   - **a gap in `ts` deltas outside every timed section** -> the upload/mirror path; move results off R2 (network
     volume or stdout-only with a final pull).

## Then

```bash
cd mjlab_centering
# reduced matrix already generated (400 iterations, 2048 envs): 70 runs
python -m mjlab_centering.jobs --blocks A,fixclip,fixnorm,E --centers none --iters 400 --tag u3
python -m mjlab_centering.jobs --blocks A,E --centers value --iters 400 --eta 0.003 --tag c4
# one orchestrator per campaign, 2-3 pods each, then leave them alone (restarts lose progress)
python orchestrator.py --campaign <id> --manifest manifests/<id>.json --launch 2 --hard-cap 4.5
# when every job has DONE:
python analysis/derive.py --campaign <u-id>,<c-id> --pull --out results/derived_main.json
```

Budget: ACTUAL SPEND (RunPod billing API, 2026-10-31): $2.71 over 4.5 GPU-hours; account balance $28.17. Earlier estimates in this file were unmeasured extrapolations and were wrong. (the old line read ~$12-13 of $25). The reduced 70-run matrix costs ~$8-10 at
2048 envs x 400 iterations (~$0.10 per run on an A40, once the pod is warm), so it fits the remainder.

## Things not to redo

- Do not re-litigate: centering must be applied at collection time, before rsl-rl's time-out bootstrap (decision 3);
  the continuing wrapper must remove `time_out` (decision 4); mjlab must be pinned to the git commit, not PyPI
  (decision 12); `MUJOCO_GL=disable` on pods (decision 13); checkpoints off the mirror by default (decision 21).
- Do not add pods to compensate for slowness until the diagnostic above is done: the cost is linear in pod-hours and
  the failure mode is per-run, not per-pod.
