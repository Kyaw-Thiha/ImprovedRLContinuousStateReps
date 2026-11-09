# R2 Sync — Usage

`scripts/sync_to_r2.py` pushes experiment artifacts to the Cloudflare R2 bucket
specified in `.env` (`R2_BUCKET=rl-ssp-centering`). Backed by **boto3**, not
rclone — see "Why not rclone?" below.

## Quick reference

```bash
.venv/bin/python scripts/sync_to_r2.py trial             # intra-search sync
.venv/bin/python scripts/sync_to_r2.py phase <stage>     # end-of-phase full sync
.venv/bin/python scripts/sync_to_r2.py dry-run           # list bucket contents
.venv/bin/python scripts/sync_to_r2.py list <stage>       # list under <stage>/
.venv/bin/python scripts/sync_to_r2.py init-bucket       # create stage2/3 placeholders
```

## Subcommands

### `trial`
Push this cell's **Optuna DB** (per-study R2 key), and optionally:
- `--best-params-file <name>.yaml` — upload exactly ONE named best_params file
  (never the whole `best/` directory — see below for why).
- `--csv-dir <path>` — upload one cell's per-trial CSV directory (e.g.
  `dqn/ssp/none`), relative to `outputs/hparam_search/`. Safe to upload the
  whole directory: already namespaced per (representation, centering,
  architecture), so no cross-cell collision is possible.

Designed to be called frequently during a search (every 10 completed trials,
via `--sync-every-n-trials`) so that process death / RunPod pod restart loses
at most ~10 trials of search progress. `run_hparam_search_dqn.py`'s periodic
callback passes `--csv-dir` but omits `--best-params-file` (nothing to push
there until the search actually completes); its one final post-completion
sync passes both.

**Never pass `--best-params-file` and rely on some other invocation to upload
the whole `best/` directory instead.** An earlier version of this script did
exactly that (`upload_dir()` over the entire local `best/` folder on every
sync) — on a multi-pod launch, every pod re-uploaded every OTHER cell's yaml
too, most of which were stale image-baked copies from before that pod
started, and whichever pod's sync landed last clobbered every other cell's
already-correct R2 result. Confirmed recurring 10+ times across two overnight
multi-pod runs before the fix — see `docs/07-decision-log.md` entry 12.

- Idempotent: overwrites the same keys.
- Locked with `flock` so concurrent invocations don't overlap.
- Returns non-zero exit code on any failure.

### `phase <stage>`
End-of-phase sync. Pushes:
- `code_snapshot.tar.gz` (git archive of HEAD at sync time)
- All Optuna DBs (`optuna_studies_dqn.db`, `optuna_studies_a2c.db`, `optuna_studies_old.db`)
- `best_params/*.yaml` (tuned hyperparameters for all 9 cells of the algo×rep×centering grid)
- `tuned/<algo>/<rep>/<centering>/<trial_id>/` per-trial CSVs + metadata
- `logs/*.log` from the most recent search run

Examples:
```bash
.venv/bin/python scripts/sync_to_r2.py phase stage0_cartpole
.venv/bin/python scripts/sync_to_r2.py phase stage2_manipulation_lift_cube_yam
```

### `dry-run`
List everything in the bucket (root-level). Useful for sanity-checking that
syncs completed successfully.

### `list <stage>`
List everything under `<stage>/` in the bucket. Example:
```bash
.venv/bin/python scripts/sync_to_r2.py list stage0_cartpole
```

### `init-bucket`
One-time setup: creates `stage2_manipulation_lift_cube_yam/`, `stage3_locomotion_go1/`,
`stage3_locomotion_g1/` placeholder README files so the bucket layout is
pre-defined for future phases.

## Integration with `run_hparam_search_dqn.py`

The Optuna search script automatically invokes `trial` every N completed
trials via a callback (`--sync-every-n-trials`, default 10), passing
`--csv-dir` for this cell's per-trial CSVs. The one final sync, right after
`save_best_params()` writes this cell's result, additionally passes
`--best-params-file`.

To disable per-search syncing:
```bash
.venv/bin/python scripts/run_hparam_search_dqn.py ... --no-sync
```

## Why not rclone?

We initially planned to use rclone, but discovered an incompatibility with
Cloudflare R2 + rclone 1.74+: rclone's request signing adds `Amz-Sdk-Invocation-Id`
and `Amz-Sdk-Request` headers (AWS SDK Go v2 behavior). Cloudflare R2's SigV4
validator rejects signatures that include these non-standard headers. The same
credentials work fine with `boto3` and `awscli`, neither of which add those
headers. boto3 also gives us per-file error handling and a Python-native
script that's easier to maintain than a bash wrapper around rclone.

## Bucket layout

```
r2://rl-ssp-centering/
├── README.md                       (not auto-created; add manually)
├── stage0_cartpole/
│   ├── README.md                   (recommended: commit hash + env version)
│   ├── code_snapshot.tar.gz
│   ├── optuna_dbs/
│   │   ├── optuna_studies_dqn.db
│   │   ├── optuna_studies_a2c.db
│   │   └── optuna_studies_old.db
│   ├── best_params/
│   │   └── dqn_<rep>_<centering>[_deep].yaml × 18 files (9 cells × 2 architectures)
│   ├── hparam_search_csvs/          (pushed by `trial`, incrementally, per cell)
│   │   └── dqn/<rep>/<centering>[_deep]/<trial_dir>/
│   │       └── rewards.csv, states.csv, values.csv
│   ├── tuned/                       (pushed by `phase`, once, cross-experiment)
│   │   └── <algo>/<rep>/<centering>/<trial_id>/
│   │       └── rewards.csv, states.csv, values.csv + .txt
│   └── logs/
│       └── dqn_<rep>_<centering>.log × 3 files
├── stage2_manipulation_lift_cube_yam/
│   └── README.md
├── stage3_locomotion_go1/
│   └── README.md
└── stage3_locomotion_g1/
    └── README.md
```

## Cost

Cloudflare R2 storage: $0.015/GB/month for standard tier, no egress fees.
Total valuable artifacts per stage: ~120 MB. Monthly cost: effectively $0.

## Failure modes

- `boto3` missing → run `uv pip install --python .venv/bin/python boto3`
- `.env` missing or invalid creds → error message points at `.env`
- Network error during sync → logged per-file; sync continues
- Lock contention → "r2_sync: another sync in progress, waiting..." (max 60s, then exits)

## eval_v3 artifacts

The post-search evaluation phase uses a separate prefix and never overwrites search artifacts:

```text
stage0_cartpole/eval_v3/
  _code/<content-hashed code tarball>
  _shared/probe_states.npz
  <study>/policies/seed_<seed>.pt
  <study>/train_runs.jsonl
  <study>/train_summary.json
  <study>/manifest.json
  <study>/robustness_runs.jsonl
  <study>/deployment.json
  dqn_settings/<study>/<variant>/{train_runs.jsonl,train_summary.json,manifest.json}
```

Keys are generated by `scripts/eval_v3_layout.py`. The local mirror is `outputs/eval_v3/`. Normal policy seeds are
3000-3019; DQN-settings variants use 3000-3009 and do not save policies. `run_policy_pass.py` resumes from
`train_runs.jsonl` and a policy is complete only after exact reload verification. `run_robustness.py` and the
deployment runner also resume their per-cell artifacts. The eval_v3 code does not write `refilter_v2/`,
`best_params/`, `optuna_dbs/` or backup prefixes.
***
