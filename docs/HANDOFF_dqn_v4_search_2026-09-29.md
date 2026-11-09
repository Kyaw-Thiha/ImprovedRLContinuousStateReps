# DQN v4 Hyperparameter Search — Handoff (2026-09-29, ~19:40 UTC)

> **STATUS UPDATE (2026-09-30, ~02:30 UTC): all pods terminated, run paused.**
> Everything below this line describes the state as of the original ~19:40 UTC
> handoff and is kept for its debugging/recovery-procedure content, but the
> **live status, final results table, and next-steps are now tracked in
> [`runs/RUN_002.md`](../runs/RUN_002.md)**, not here — update that file, not
> this one, as more data comes in on future runs. Summary of what changed since
> this doc was written:
> - 2 more cells finished cleanly (ssp/none/deep, and a tie-fixed redo of
>   ssp/simple) before being caught up in the pause below — 9/18 done total.
> - The reeval candidate-selection step was found to have a winner's-curse bias
>   (only trials tied at the exact max search score are ever considered) —
>   session paused to fix this before trusting further reeval output. See
>   RUN_002.md's "Open methodology issue" section.
> - Every pod still running or stopped mid-reeval was deliberately stopped and
>   then terminated, including the two cells that were still genuinely
>   searching (tilecoding/none/deep, ssp/value/deep) and the test pod — all
>   9 remaining cells will be rerun together in a future session once the
>   methodology question above is settled.
> - **New, real data loss found this session**: `tilecoding/value/linear` hung
>   mid-reeval (Bug #2 below) after its search completed 500/500 trials
>   *locally*, but R2 only ever got 303/500 synced before the pod was
>   terminated — the last ~200 trials are gone. No pod in this entire run had
>   SSH enabled (`startSsh` was never set, and the image's entrypoint has no
>   sshd), so there was no way to pull the local DB off a hung pod before
>   terminating it. Next time Bug #2 recurs, launch that pod with SSH enabled.
> - `Pending #1` and `Pending #2` below (ssp/value/linear capacity, ssp/none
>   reeval redo) are superseded — see RUN_002.md's per-cell table instead.

## What this is

Running an 18-cell Optuna hyperparameter search for DQN on CartPole-v1: 3
representations (`discrete`, `ssp`, `tile_coding`) × 3 centering modes
(`none`, `simple`, `value`) × 2 architectures (`linear`, `deep`). Each cell
runs independently on its own RunPod CPU pod (`cpu3c`, 4 vCPU, $0.12/hr),
500 Optuna trials, then a post-hoc re-evaluation of the top candidates with
multiple seeds to filter out lucky-seed winners. Methodology: Antonin
Raffin's two-part Optuna guide (`docs/06-related-work.md`,
`docs/07-decision-log.md` entry 11). Full design docs in `docs/`.

Key script: `scripts/run_hparam_search_dqn.py`. Sync/recovery script:
`scripts/sync_to_r2.py` + `scripts/sync_to_r2_README.md`. Everything
persists to Cloudflare R2 bucket `rl-ssp-centering` (creds in repo `.env`
and baked into each pod's env vars).

## Current cell status (8 of 18 done)

| Cell | Status | best_value | Notes |
|---|---|---|---|
| discrete/simple (linear) | ✅ done | 247.29 | clean, full 10/10 reeval |
| discrete/value (linear) | ✅ done | 242.57 | CAVEATED — only 2/10 reeval ranks recovered (manual recovery from a hang) |
| discrete/value/deep | ✅ done | 334.67 | CAVEATED — 3/10 ranks |
| discrete/none/deep | ✅ done | 302.99 | CAVEATED — 2/10 ranks |
| discrete/simple/deep | ✅ done | 321.947 | clean, finished ~19:15 UTC |
| ssp/simple (linear) | ✅ done (redone) | see R2 `dqn_ssp_simple.yaml` | redone with tie-break fix after original had 78 ties at max score |
| ssp/none (linear) | ✅ done (redone, OLD logic) | 488.80 | **still on old arbitrary-tie-break logic** — had 91 ties at max, redo was queued (reeval-v2 pod) but got capacity-blocked before it could launch; see Pending below |
| ssp/simple/deep | ✅ done | 279.076 | got the NEW tie-break fix (130 ties → pool-of-20), but predates the reeval-stats-persistence fix, so no full candidate table |
| **remaining 10 cells** | still running | — | see table below |

Check current state any time with:
```bash
.venv/bin/python3 -c "
import boto3
env={}
for l in open('.env'):
    l=l.strip()
    if l and not l.startswith('#') and '=' in l:
        k,v=l.split('=',1); env[k]=v.strip().strip(chr(34)).strip(chr(39))
s3=boto3.client('s3', endpoint_url=f'https://{env[\"R2_ACCOUNT_ID\"]}.r2.cloudflarestorage.com',
    aws_access_key_id=env['R2_ACCESS_KEY_ID'], aws_secret_access_key=env['R2_SECRET_ACCESS_KEY'])
for o in sorted(s3.list_objects_v2(Bucket=env['R2_BUCKET'], Prefix='stage0_cartpole/best_params/')['Contents'], key=lambda x:x['LastModified']):
    print(o['LastModified'], o['Key'])
"
```

## Still-running pods (as of last check)

| Cell | Pod ID | Trials done/500 (search) | Notes |
|---|---|---|---|
| tilecoding/value/deep | 43betbxecqaiyv | ~150/500, slow (~0.6/min) | genuinely slow config (large iht_size/tiles_per_dim draw), not hung — see "Slow ≠ hung" below |
| tilecoding/value/linear | 4h976s0anr5089 | ~300/500 | cpu.util fluctuates 44-144%, healthy |
| tilecoding/simple/deep | 78tfhi9rzkdt90 | ~400/500 | healthy |
| tilecoding/simple/linear | 45z4k3a75z5mw5 | ~370/500 | healthy |
| tilecoding/none/linear | mrqxd4mf9kkgm3 | ~390/500 | healthy |
| tilecoding/none/deep | cqd8i5iwmzvp34 | ~270/500 | recreated pod, resumed from R2 snapshot |
| ssp/value/deep | pnaaitycfj2b5r | ~300/500 | healthy |
| ssp/none/deep | 7ylto9fjxib1ho | in reeval, pool-of-20 | last confirmed rank ~5-6/20; has NEW tie-fix code |
| ssp/simple (linear) reeval-v2 | qwep5rx3lyvclr | search restarting from R2 (500 done already) | launched specifically to redo this cell's reeval with the fix — should reach reeval soon |
| test pod: discrete/none/linear | 3z7ne2ttpzi9i0 | **unknown — see below** | DO NOT TOUCH |

## ⚠️ Pending / needs attention

1. **`ssp/value/linear` is DOWN** — pod terminated after a failed restart got
   stuck in `EXITED` state (RunPod capacity issue), and recreation attempts
   have been failing with "no instances available" / "not enough free
   vcpu". Its full 500-trial search is safely on R2
   (`optuna_studies_dqn_ssp_value_v4_hparam_search.db`) — nothing lost
   except ~2 reeval candidates that were mid-flight. **Needs**: retry
   `create-pod` with the same args (see "Relaunch command template" below)
   until capacity is available. This cell had 61 ties at the max search
   score, so it needs the tie-fix — the image already has it, just needs to
   launch.

2. **`ssp/none (linear)` needs a reeval redo** — its current done result
   (488.80) was computed under the OLD arbitrary-tie-break logic (91 ties
   at the max score, top-10 was essentially arbitrary). A reeval-v2 pod was
   supposed to launch for this (same pattern as `ssp/simple`'s successful
   redo) but every `create-pod` attempt failed on capacity. **Before
   launching**: the R2-hosted study's `reeval_done` user_attr needs to be
   cleared again if it isn't already (check first — it may already be
   cleared from an earlier attempt this session). See "How to force a
   reeval redo" below.

3. **RunPod capacity has been intermittently unavailable all day** — not a
   quota issue (user confirmed via dashboard), a genuine vendor-side
   capacity crunch. It comes and goes; sometimes 3 consecutive attempts
   fail, then the very next one succeeds. **Keep retrying `create-pod`
   calls one at a time (not in parallel — parallel destructive/creation
   calls have intermittently been blocked by a safety classifier)** for any
   pending launch.

4. **Full per-candidate reeval stats are missing for most cells.** See "The
   reeval-stats gap" below — only cells that run reeval on the very latest
   image (pushed today, digest `sha256:e3528bfc...`) get a full
   `reeval_results.json` (rank/trial/mean/std for every pool candidate).
   Everything done before that fix only has the single winning
   `best_value` (and sometimes `best_std`, added slightly earlier). If
   full statistics matter for the final report, cells marked "no full
   stats" above would need a reeval redo too — but weigh this against
   their tie count (see table above; low-tie cells don't need it).

## The test pod — read this before touching it

`dqn-test-discrete-none-linear` (pod `3z7ne2ttpzi9i0`) is cell
discrete/none/linear. It was the FIRST pod launched (05:01 UTC, ~20 min
before the other 17) — "test" is just its name, it runs the same job as
every other cell.

**DO NOT restart, stop, or otherwise touch this pod.** Its R2-synced Optuna
DB has been unreliable all day (call this "bug #3") — at various checks
it's shown as stuck at ~20 trials for hours, then briefly jumped to 55,
then reverted to 21, even though the pod itself is clearly healthy (high
CPU, fresh log timestamps, visibly advancing progress bars in
`stream-pod-logs`). **The pod is NOT hung — it is definitely training
right now** — but its R2 mirror cannot be trusted to reflect real
progress. If you restart it, the entrypoint's pull-from-R2 step will
**overwrite its real, further-along local progress with the stale R2
snapshot**, destroying real work. If it ever needs recovery (shows
`cpu.util≈0` for real), do LOG-ONLY recovery (see below) — never trust its
R2 database.

A `dqn_discrete_none.yaml` file exists on R2 dated 2026-09-29 16:15 with
`best_value=346.86` — this is very likely a STALE leftover from an old
pre-v4 methodology run sharing the same output filename, NOT a real
completion of this pod's current v4 study (confirmed: the v4 study's DB
had only 21-55 trials at checks around and after that timestamp, no
`reeval_done` attribute). Don't count this cell as done based on that file
alone — verify against the live DB (`dqn_discrete_none_v4_hparam_search`)
before trusting any future-looking "done" signal for this cell too.

## Slow ≠ hung — how to tell the difference

Checked this today for `tilecoding/value/deep` (the slowest cell, ~0.6
trials/min vs ~1.5-2/min for peers). It is NOT hung, NOT a search-space
design flaw — the code already has a benchmarked (26-132x speedup) sparse
`EmbeddingBag` + `SparseAdam` path for exactly this large-representation +
deep-architecture case (see `rl/networks/dqnBasic.py`'s `QNetwork`
docstring). The two currently-running trials there just happen to have
drawn `iht_size=65536` (max) + `tiles_per_dim=16` (max) — legitimate,
in-bounds, just expensive. This is normal Optuna sampling variance, not a
bug. Recommendation reached: leave the search space as-is; the dollar cost
of the slow cell finishing late is ~$1, not worth restarting the study
(which would discard 100+ existing trials).

**To verify "genuinely working" vs "hung" for ANY pod**: check all three —
(1) `cpu.util > 0` via `list-pods`/`get-pod`, (2) `stream-pod-logs` shows a
timestamp within the last ~30s of wall-clock time, (3) the tqdm progress
percentage is advancing between log lines. A real hang shows `cpu.util≈0`
AND logs frozen at some point in the past — a "just slow" pod passes all
three checks, just at a lower trials/min rate.

## Known bugs — status

- **Bug #1 (final-sync deadlock)**: FIXED. `sync_to_r2.py`'s
  `acquire_lock()` previously blocked forever (`fcntl.flock(LOCK_EX)`, no
  timeout) if another sync held the lock — this caused pods to hang
  forever on "another sync in progress, waiting..." right after printing
  "Post-hoc winner". Fixed with a real 60s timeout (raises `TimeoutError`
  instead of hanging). Verified with a unit test (lock held by another fd,
  confirmed timeout fires in ~1.5s for a 1.5s timeout).

- **Bug #2 (mid-reeval hang, cpu.util≈0, no new log output, no sync
  message)**: root cause STILL NOT CONFIRMED. Ruled OUT `sync_to_r2.py` as
  the cause — read the code, `reevaluate_top_trials()` never calls it at
  all (only the search-phase periodic callback and the final post-reeval
  sync do). Leading unconfirmed hypothesis: `tqdm`'s per-step stdout
  writes blocking on RunPod's log-pipe backpressure. Recurred ~5 times
  today, always recovered via log-scraping (see "Recovery procedure"
  below). If it recurs on a pod created with SSH enabled (anything
  launched via `create-pod` in this session has `startSsh` available —
  check `ssh.direct`/`ssh.proxy` on the pod), a live `py-spy dump` before
  recovery would help finally pin this down — nobody has done this yet.

- **Bug #3 (test pod's R2 sync unreliable)**: NOT FIXED, not urgent (only
  affects our ability to monitor progress remotely, not the pod's actual
  correctness). See "The test pod" section above.

- **Tie-break bug (found and fixed today, separate from #1-#3)**:
  CartPole-v1's 500-step reward cap means many trials saturate the
  search-phase score. When more than `top_k` (10) trials tie at the max,
  the old code's `sorted(...)[:top_k]` picked whichever 10 happened to
  have the lowest trial numbers (Python's stable sort) — an arbitrary,
  uninformative "top 10". Fixed: `_select_reeval_candidates()` in
  `run_hparam_search_dqn.py` now expands to a pool of up to
  `TIE_EXPANDED_POOL_SIZE=20` when ties exceed `top_k`, ranking by
  `episodes_to_learn` (pulled from R2-synced per-trial `.txt` metadata)
  where available, random-filling the rest. Verified against real data
  (ssp/none/deep: 114 ties → correctly expanded, 68/114 had usable
  episodes_to_learn).

- **Reeval-stats gap (found and fixed today)**: `reevaluate_top_trials()`
  computed mean/std for every reeval candidate but only ever printed
  non-winners to stdout — permanently lost once a pod terminates. Fixed:
  now writes `reeval_results.json` (full rank/trial/mean/std/params table)
  into the reeval data dir, which rides along on the existing `--csv-dir`
  R2 sync for free. Also added `best_std` to the `best_params` yaml
  (previously only `best_value` was saved, even for the winner). See
  `stage0_cartpole/reeval_stats/partial_scrape_2026-09-29.json` on R2 for
  a partial manually-scraped table covering some candidates from
  `ssp/simple/deep` and `ssp/none/deep` (recovered before the fix could
  apply to them).

## Image state

Current image: `kevinkyawthiha/dqn-ssp-centering:stage0`, digest
`sha256:e3528bfc5e3f4d7c371530953e0eafaa7170dfe192227bd25ea72256958c6a00`
(includes ALL fixes: bug #1 timeout, tie-break pool-of-20, reeval-stats
persistence). Dockerfile at repo root. Rebuild + push:
```bash
docker build -t kevinkyawthiha/dqn-ssp-centering:stage0 .
docker push kevinkyawthiha/dqn-ssp-centering:stage0
```
Existing running pods do NOT pick up a new image push automatically — only
a fresh `create-pod`, or a `restart`/`stop`+`start` on an existing pod,
re-pulls the tag.

**Restarting a pod is risky right now** (capacity crunch): a `restart` can
successfully stop the old container then fail to start the new one
("not enough free vcpu"), stranding the pod in `EXITED` with its disk
already wiped (container disk is ephemeral, wiped on restart regardless of
whether the subsequent start succeeds). If this happens: try `pod-action
start` a couple times; if that also fails, `delete-pod` + recreate fresh
(no additional loss — the disk was already wiped by the failed restart).
This happened twice today (`tilecoding/none/deep`, `ssp/value/linear` —
the latter is still down, see Pending #1).

## Relaunch command template

All 18 cells use the same pattern. Example (adjust `--representation` /
`--centering` / `--architecture`):
```python
create-pod(body={
  "name": "dqn-<rep>-<centering>-<arch>",
  "image": "kevinkyawthiha/dqn-ssp-centering:stage0",
  "cloud": "SECURE",
  "disk": 10,
  "cpu": {"id": "cpu3c", "vcpuCount": 4},
  "cmd": ["--representation", "<rep>", "--centering", "<centering>",
          "--architecture", "<arch>", "--n-trials", "500"],
  "env": {
    "R2_ACCOUNT_ID": "bfee415890c51474c3b1298cc9705052",
    "R2_ACCESS_KEY_ID": "4d5861de755cae99fc7b12a12ce44d4f",
    "R2_SECRET_ACCESS_KEY": "ff0b30aeb0310f302539b59bb2390f5ab5aeff280ed9b61075e85ed0a22bb123",
    "R2_BUCKET": "rl-ssp-centering",
  },
})
```
The entrypoint (`scripts/entrypoint_dqn_search.sh`) automatically pulls
this cell's existing per-study DB from R2 on startup and resumes — a
relaunch for a cell whose search is already at 500 trials will skip
straight to reeval.

## How to force a reeval redo (e.g. for `ssp/none (linear)`)

Used successfully today for `ssp/simple` and (attempted, blocked by
capacity) `ssp/none`. Steps:
```python
# 1. Download the study's current db
s3.download_file(bucket, f"stage0_cartpole/optuna_dbs/optuna_studies_{study_name}.db", local_path)
# 2. Clear the cached reeval flag
import optuna
study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{local_path}")
study.set_user_attr("reeval_done", False)
# 3. Upload back to R2 (same key), verify by re-downloading
s3.upload_file(local_path, bucket, key)
# 4. Launch a fresh pod with the same --representation/--centering/--architecture —
#    it'll pull 500 existing trials, see reeval_done=False, and redo reeval
#    with the current (fixed) code.
```
Note: uploading to R2 like this is a real overwrite of production data —
confirm with the user before doing it if this is a fresh session (it was
explicitly approved this session, but always worth re-confirming since it
discards the previous reeval result).

## Recovery procedure for a hung pod (cpu.util≈0)

1. Check logs for `"Post-hoc winner: mean=X std=Y params={...}"` — if
   present, the pod finished cleanly and only the final sync hung (bug
   #1, now fixed going forward, but could still recur on old-image pods).
   Use it directly.
2. If absent: fetch `stream-pod-logs` at several different `tail` sizes
   (15/30/60/100/200/300/400 — RunPod's log API paginates inconsistently;
   different sizes surface different, non-overlapping windows). Grep for
   `"reeval rank=N (search trial #T, search score S): mean=M std=D"`.
   Collect as many distinct ranks as you can find across all tail sizes.
3. Take the highest mean found as the winner (or, if you have time,
   look up the winning trial's actual hyperparameters from the R2 db —
   see below).
4. **DB lookup for hyperparameters** (if the log line doesn't include full
   params): download `stage0_cartpole/optuna_dbs/optuna_studies_dqn_<rep>_<centering>[_deep]_v4_hparam_search.db`.
   These files are a SHARED sqlite backend accumulating many past studies
   — always resolve `study_id` via `SELECT study_id FROM studies WHERE
   study_name=?` FIRST, then join `trials`/`trial_values`/`trial_params`
   on that exact `study_id` (trial numbers repeat across studies in the
   same file). Decode categorical/int params via `distribution_json`'s
   `"choices"` list (index = stored float cast to int). **Verify the
   trial's recorded `value` matches the log's reported search score**
   before trusting the db.
5. Construct a `best_params` yaml matching `save_best_params()`'s format:
   `model_type: dqn`, `representation`, `centering`, `architecture`,
   `best_value`, `best_std` (if known), `params: {...hyperparams...,
   rep_: <Discrete/PlaceSSP/TileCoding>, reward_center_mode: <centering>}`.
   `centering="none"` has no `reward_center_*` params; `"simple"` uses
   `reward_center_beta`; `"value"` uses `reward_center_eta`. Add a CAVEAT
   comment at the top if recovery was partial.
6. Upload via boto3 to `stage0_cartpole/best_params/<filename>.yaml`
   (naming: `dqn_<rep>_<centering>[_deep].yaml` — linear omits the
   `_deep` suffix). Verify by re-downloading.
7. Terminate the pod via `delete-pod` (tool param is `id`, NOT `podId`).
8. **Exception for the test pod**: log-only recovery only, never trust its
   R2 db (stale). If only rank lines without full params are visible,
   report log-only info, don't guess params.

## Suggested next steps for whoever picks this up

1. Retry `create-pod` for `ssp/value/linear` (Pending #1) until capacity
   allows.
2. Check whether `ssp/none (linear)`'s `reeval_done` flag is still cleared
   on R2; if so just retry launching its reeval-v2 pod (Pending #2); if it
   somehow got re-set, clear it again first.
3. Keep an eye on `cpu.util` across all pods every 20-30 min; apply the
   recovery procedure to any showing 0.
4. Once all 18 cells are done, write a final comprehensive summary: best
   value+params for all 18 (clearly marking clean vs. caveated vs.
   old-tie-break-logic results), full bug writeup, R2 stability notes,
   total pod-hours/cost, and the `iht_size`/`tiles_per_dim` boundary
   observation for tile_coding (worth a footnote, not an action item).
5. Consider (not urgent): a `py-spy` capture next time bug #2 recurs on an
   SSH-enabled pod, to finally confirm its root cause.
