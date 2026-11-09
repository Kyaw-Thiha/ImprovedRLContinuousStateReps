#!/usr/bin/env python3
"""Sync artifacts to Cloudflare R2 for persistence.

Uses boto3 directly (not rclone) because rclone 1.74+ signs requests
with extra `amz-sdk-*` headers that Cloudflare R2's SigV4 validator
rejects. boto3/awscli both work fine with the same credentials.

Usage:
    sync_to_r2.py trial                    # intra-search: DB only (push)
    sync_to_r2.py trial --best-params-file dqn_ssp_none.yaml   # + one best_params file
    sync_to_r2.py phase <stage-name>       # end-of-phase: full sync (push)
    sync_to_r2.py pull                     # restore DB + best_params from R2 (pod startup)
    sync_to_r2.py dry-run                  # show what's there
    sync_to_r2.py list <stage>             # list files under <stage>/

Reads R2_* from .env in repo root if present, otherwise from os.environ.
"""
import os
import sys
import subprocess
import argparse
import datetime
import fcntl
import time
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

REPO = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO / ".r2_sync.lock"


def load_env():
    """Load R2 credentials from REPO/.env if present, else rely on os.environ.

    Permissive: if the file is missing, just warn and return. The caller
    (s3_client / cmd_pull) will then read R2_* from os.environ, which is
    how the Docker entrypoints populate them — RunPod pod env vars are
    already in os.environ by the time this script runs.
    """
    env_path = REPO / ".env"
    if not env_path.exists():
        print(f"[sync_to_r2] NOTE: {env_path} not found — using os.environ for R2_* vars",
              file=sys.stderr)
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


def s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def acquire_lock(timeout: float = 60.0, poll_interval: float = 0.5):
    """Serialize concurrent syncs to avoid overlapping uploads.

    Bounded to `timeout` seconds (matches the "max 60s, then exits" behavior
    documented in sync_to_r2_README.md, which the previous implementation
    never actually implemented — the fallback path called a bare blocking
    fcntl.flock(LOCK_EX) with no timeout at all, so a lock-holder that died
    mid-upload or itself hung would wedge every subsequent sync forever.
    Confirmed as the cause of the "another sync in progress, waiting..."
    deadlock on the final post-reeval sync (see docs/07-decision-log.md) —
    that hang only ever cleared via manual pod recovery, never on its own.

    Raises TimeoutError on expiry rather than proceeding without the lock;
    callers should let this propagate (non-zero exit), matching the
    "returns non-zero exit code on any failure" contract in the README.
    """
    LOCK_PATH.touch()
    fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        pass

    print("r2_sync: another sync in progress, waiting...", file=sys.stderr)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            time.sleep(poll_interval)

    fd.close()
    raise TimeoutError(f"r2_sync: timed out after {timeout}s waiting for lock at {LOCK_PATH}")


def put_file(s3, local: Path, key: str, bucket: str):
    """Upload a single file. Silent on success; prints on failure."""
    try:
        s3.upload_file(str(local), bucket, key)
    except ClientError as e:
        print(f"  FAILED {key}: {e.response['Error']['Code']}: {e.response['Error']['Message']}",
              file=sys.stderr)
        return False
    return True


def upload_dir(s3, local_dir: Path, bucket: str, prefix: str, include_globs=None):
    """Recursively upload local_dir to bucket at key prefix."""
    if not local_dir.exists():
        print(f"  skip {prefix} (local path missing: {local_dir})")
        return 0, 0
    uploaded = 0
    failed = 0
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        if include_globs and not any(local_file.match(g) for g in include_globs):
            continue
        rel = local_file.relative_to(local_dir)
        key = f"{prefix}/{rel.as_posix()}"
        if put_file(s3, local_file, key, bucket):
            uploaded += 1
    return uploaded, failed


def download_file(s3, bucket: str, key: str, dest: Path) -> bool:
    """Download a single object. Returns True on success."""
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except ClientError as e:
        print(f"  FAILED download {key}: {e}", file=sys.stderr)
        return False


def download_dir(s3, bucket: str, prefix: str, dest_dir: Path, suffix: str = None) -> int:
    """Recursively download all objects under prefix into dest_dir.

    Mirrors upload_dir(): prefix becomes the local dest_dir root; each object's
    relative path is preserved. `suffix` filters by file extension. Returns
    count of files downloaded.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if suffix and not key.endswith(suffix):
                    continue
                rel = key[len(prefix):]
                if not rel:
                    continue
                local = dest_dir / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                if download_file(s3, bucket, key, local):
                    n += 1
    except ClientError as e:
        print(f"  FAILED list {prefix}: {e}", file=sys.stderr)
    return n


def cmd_trial(s3, bucket, db_path: str = None, best_params_file: str = None, csv_dir: str = None):
    """Fast intra-search sync: the Optuna DB, plus (only when given) ONE
    specific best_params file.

    The local DB is uploaded to a per-study key
    (stage0_cartpole/optuna_dbs/optuna_studies_<study_name>.db) so concurrent
    syncs from multiple pods (different reps) do not overwrite each other.

    For backward compatibility, the local DB is also uploaded to the legacy
    shared key (optuna_studies.db) as a fallback — but with the per-study key
    taking precedence for restoration in `cmd_pull`.

    `db_path` should match whatever --db-path was passed to
    run_hparam_search_dqn.py for THIS run (defaults to the legacy
    "optuna_studies.db" name if omitted, preserving old behavior for runs
    that don't use a custom path). Without this, a custom --db-path run's
    real Optuna database is never backed up at all — only the (irrelevant)
    default-path file is — a real data-loss bug found and fixed 2026-09-26
    after discrete/ssp's final Phase A reruns lost their full trial history
    this way (best_params/*.yaml survived independently, but per-trial
    convergence/variance data did not).

    `best_params_file` scopes the best_params upload to exactly one file
    (e.g. "dqn_ssp_none.yaml") instead of the whole best/ directory. This
    used to call upload_dir() over the ENTIRE local best/ directory on every
    sync — which on a multi-pod launch meant every pod re-uploaded every
    OTHER cell's yaml too, most of which were stale image-baked copies from
    before any pod started. Whichever pod's sync landed last clobbered every
    other cell's already-correct R2 result. Confirmed recurring 10+ times
    across two separate overnight multi-pod runs (2026-09-27/28) before this
    fix — see docs/07-decision-log.md. Since a cell's best_params file is
    only ever written once, at search completion (save_best_params(), not
    per-trial), there is nothing meaningful to upload here until then — omit
    this argument entirely for the periodic in-search sync and pass it only
    on the one final post-completion sync.

    `csv_dir` uploads one cell's raw per-trial artifacts (rewards.csv/
    states.csv/values.csv, written by trial_cartpole_dqn.py's evaluate() —
    one subfolder per trial) — relative to outputs/hparam_search/, e.g.
    "dqn/ssp/none". Unlike best_params, these were NEVER synced at all before
    this was added (a real, if lower-stakes, gap: a lost pod meant losing
    every episode-level reward/state/value curve for that cell's trials,
    though each trial's final score/params still survived in the synced DB).
    Safe to upload_dir() over the whole tree on every sync — unlike the old
    best_params bug, this path is already namespaced per (representation,
    centering, architecture) by the caller (_cell_data_dir in
    run_hparam_search_dqn.py), so no cross-cell collision is possible. Each
    sync re-uploads already-synced older trials' files too (they're
    immutable once written, so this is wasted bandwidth, not incorrect) —
    accepted for simplicity over tracking an incremental diff.
    """
    uploaded = 0
    failed = 0

    db_local = REPO / (db_path or "optuna_studies.db")
    if db_local.exists():
        # Discover which studies this DB contains. Each pod typically runs one
        # Optuna study (e.g. dqn_ssp_none_hparam_search), but multi-study DBs
        # are supported for symmetry.
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_local))
            studies = [r[0] for r in conn.execute(
                "SELECT study_name FROM studies"
            ).fetchall()]
            conn.close()
        except Exception as e:
            print(f"  failed to enumerate studies in {db_local}: {e}")
            studies = []

        if not studies:
            print(f"  skip optuna_studies.db (no studies found)")
        else:
            for study_name in studies:
                # Per-study key — safe under concurrent multi-pod sync.
                safe_name = study_name.replace("/", "_")
                per_study_key = f"stage0_cartpole/optuna_dbs/optuna_studies_{safe_name}.db"
                if put_file(s3, db_local, per_study_key, bucket):
                    uploaded += 1
                else:
                    failed += 1
            # Legacy shared key — kept for backward compat with older pull
            # scripts / external consumers. If the per-study upload fails,
            # this also fails (same file), but if R2 storage is misconfigured
            # in a per-study-prefix way, this still gives a single fallback.
            if put_file(s3, db_local, "stage0_cartpole/optuna_dbs/optuna_studies.db", bucket):
                uploaded += 1
            else:
                failed += 1
    else:
        print(f"  skip optuna_studies.db (not found)")

    if best_params_file:
        best_local = REPO / "outputs" / "hparam_search" / "best" / best_params_file
        if best_local.exists():
            if put_file(s3, best_local, f"stage0_cartpole/best_params/{best_params_file}", bucket):
                uploaded += 1
            else:
                failed += 1
        else:
            print(f"  skip {best_params_file} (not found — not written by this run yet)")
    else:
        print("  skip best_params (no --best-params-file given — nothing new to push yet)")

    if csv_dir:
        csv_local = REPO / "outputs" / "hparam_search" / csv_dir
        u, f = upload_dir(s3, csv_local, bucket, f"stage0_cartpole/hparam_search_csvs/{csv_dir}")
        uploaded += u
        failed += f

    print(f"trial-sync complete: {uploaded} uploaded, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_pull(s3, bucket, study_name: str = None, db_path: str = None):
    """Restore Optuna DB + best_params from R2 (used at pod startup).

    Inverse of cmd_trial:
        stage0_cartpole/optuna_dbs/optuna_studies_<study_name>.db  -> REPO/(db_path or optuna_studies.db)
        stage0_cartpole/optuna_dbs/optuna_studies.db (legacy shared) -> fallback
        stage0_cartpole/best_params/*.yaml                              -> REPO/outputs/hparam_search/best/

    If `study_name` is given, prefer the per-study key
    (optuna_studies_<study_name>.db). Otherwise fall back to the legacy
    shared key. Best-effort: any failure means the search starts fresh from
    warm-start trial 0 (image-baked A2C YAMLs) — never fatal.

    `db_path` MUST match whatever --db-path the corresponding
    run_hparam_search_dqn.py invocation will use — this mirrors the same
    fix applied to cmd_trial (see its docstring). Without this, restoring
    always overwrote the default optuna_studies.db regardless of which file
    the search actually reads from, so a run launched with a custom
    --db-path would silently fail to resume AND, worse, a subsequent `trial`
    push from that "resumed" (actually fresh) run would overwrite the
    correct per-study R2 snapshot with an empty/partial one. Found while
    designing the multi-rep-per-pod Phase A entrypoint (each rep gets its
    own --db-path so restoring rep N never clobbers rep N-1's already-loaded
    local state) — see runs/RUN_001.md bug #5 for the push-side version of
    this same class of bug.
    """
    local_optuna = REPO / (db_path or "optuna_studies.db")
    local_optuna.parent.mkdir(parents=True, exist_ok=True)

    restored = False
    if study_name:
        safe_name = study_name.replace("/", "_")
        per_study_key = f"stage0_cartpole/optuna_dbs/optuna_studies_{safe_name}.db"
        if download_file(s3, bucket, per_study_key, local_optuna):
            print(f"[pull] optuna_studies.db restored from {per_study_key}")
            restored = True
        else:
            print(f"[pull] WARNING: failed to restore {per_study_key}")

    if not restored:
        legacy_key = "stage0_cartpole/optuna_dbs/optuna_studies.db"
        if download_file(s3, bucket, legacy_key, local_optuna):
            print(f"[pull] optuna_studies.db restored from legacy {legacy_key}")
            restored = True
        else:
            print(f"[pull] WARNING: failed to restore {legacy_key} — search starts fresh")

    local_best = REPO / "outputs" / "hparam_search" / "best"
    local_best.mkdir(parents=True, exist_ok=True)
    n_yaml = download_dir(s3, bucket, "stage0_cartpole/best_params/", local_best, suffix=".yaml")
    print(f"[pull] best_params: {n_yaml} file(s) restored")

    return 0


def cmd_phase(s3, bucket, stage):
    """End-of-phase sync: full sync + code snapshot."""
    uploaded = 0
    failed = 0

    # Code snapshot (git archive of HEAD)
    snapshot_path = REPO / "code_snapshot.tar.gz"
    try:
        subprocess.run(
            ["git", "-C", str(REPO), "archive", "--format=tar.gz",
             "-o", str(snapshot_path), "HEAD"],
            check=True, capture_output=True,
        )
        if put_file(s3, snapshot_path, f"{stage}/code_snapshot.tar.gz", bucket):
            uploaded += 1
        else:
            failed += 1
        snapshot_path.unlink(missing_ok=True)
    except subprocess.CalledProcessError as e:
        print(f"  FAILED code_snapshot.tar.gz: git archive error: {e.stderr.decode()}")
        failed += 1

    # Optuna DBs (copy whichever exist)
    for db_name in ["optuna_studies.db", "optuna_studies_a2c.db", "optuna_studies_old.db"]:
        db_local = REPO / db_name
        if db_local.exists():
            if put_file(s3, db_local, f"{stage}/optuna_dbs/{db_name}", bucket):
                uploaded += 1
            else:
                failed += 1

    # Best params (cross-experiment)
    best_dir = REPO / "outputs" / "hparam_search" / "best"
    if best_dir.exists():
        u, f = upload_dir(s3, best_dir, bucket, f"{stage}/best_params",
                          include_globs=["*.yaml"])
        uploaded += u
        failed += f

    # Tuned ablation runs (per-trial CSVs + metadata)
    tuned_dir = REPO / "outputs" / "tuned"
    if tuned_dir.exists():
        u, f = upload_dir(s3, tuned_dir, bucket, f"{stage}/tuned",
                          include_globs=["*.csv", "*.txt"])
        uploaded += u
        failed += f

    # Logs from the Optuna search (most recent only)
    logs_dir = REPO / "outputs" / "hparam_search" / "logs"
    if logs_dir.exists():
        u, f = upload_dir(s3, logs_dir, bucket, f"{stage}/logs",
                          include_globs=["*.log"])
        uploaded += u
        failed += f

    print(f"phase-sync complete → r2://{os.environ['R2_BUCKET']}/{stage}/")
    print(f"  {uploaded} uploaded, {failed} failed")
    return 0 if failed == 0 else 1


def cmd_dry_run(s3, bucket, prefix=""):
    """List what's in the bucket under the given prefix (or root)."""
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    print(f"=== r2://{bucket}/{prefix} ===")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for common_prefix in page.get("CommonPrefixes", []):
                print(f"  [DIR]  {common_prefix}")
            for obj in page.get("Contents", []):
                size_kb = obj["Size"] / 1024
                print(f"  {size_kb:8.1f} KB  {obj['Key']}  ({obj['LastModified'].isoformat()})")
                count += 1
        print(f"=== {count} objects ===")
    except ClientError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


def cmd_list(s3, bucket, stage):
    return cmd_dry_run(s3, bucket, prefix=stage + "/")


def cmd_mkdir_placeholder(s3, bucket, key, content):
    """Create a zero-byte placeholder file with content as metadata.

    R2 has no real directories, so we create empty objects whose names
    end in '/' to simulate them."""
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode())
        print(f"  created {key}")
        return True
    except ClientError as e:
        print(f"  FAILED {key}: {e}")
        return False


def cmd_init_bucket(s3, bucket):
    """Create the placeholder structure for stage2/stage3."""
    readme_template = ("# Stage placeholder\n\n"
                       "Empty until the corresponding Stage is run.\n"
                       "See docs/02-experimental-design.md for stage descriptions.\n")
    ok = 0
    fail = 0
    for stage in ["stage2_manipulation_lift_cube_yam",
                  "stage3_locomotion_go1",
                  "stage3_locomotion_g1"]:
        for f in ["README.md"]:
            if cmd_mkdir_placeholder(s3, bucket, f"{stage}/{f}",
                                     readme_template):
                ok += 1
            else:
                fail += 1
    print(f"init-bucket complete: {ok} created, {fail} failed")
    return 0 if fail == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Sync artifacts to Cloudflare R2 (uses boto3)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_trial = sub.add_parser("trial", help="intra-search sync (DB + best_params, push)")
    p_trial.add_argument(
        "--db-path", default=None,
        help="Path to the Optuna SQLite DB actually in use for this run "
             "(pass through whatever --db-path was given to "
             "run_hparam_search_dqn.py). Defaults to optuna_studies.db.",
    )
    p_trial.add_argument(
        "--best-params-file", default=None,
        help="Filename (not path) of the ONE best_params yaml belonging to this "
             "run's own cell, e.g. dqn_ssp_none.yaml. Omit for periodic in-search "
             "syncs (nothing to push before search completion); pass it only on "
             "the final post-save_best_params() sync. Never uploads the whole "
             "best/ directory — see cmd_trial's docstring for why.",
    )
    p_trial.add_argument(
        "--csv-dir", default=None,
        help="One cell's per-trial CSV directory, relative to outputs/hparam_search/ "
             "(e.g. dqn/ssp/none). Safe to sync as a whole directory — already "
             "namespaced per cell, unlike best_params. See cmd_trial's docstring.",
    )
    p_phase = sub.add_parser("phase", help="end-of-phase full sync (push)")
    p_phase.add_argument("stage", help="Stage folder name in the bucket")
    sub.add_parser("dry-run", help="list bucket contents")
    p_pull = sub.add_parser("pull", help="restore DB + best_params from R2 (for pod startup)")
    p_pull.add_argument(
        "study_name",
        nargs="?",
        default=None,
        help="Optuna study name (e.g. dqn_ssp_none_hparam_search). "
             "If given, pulls the per-study DB optuna_studies_<study_name>.db; "
             "otherwise falls back to the legacy shared key.",
    )
    p_pull.add_argument(
        "--db-path", default=None,
        help="Local path to restore the DB to — MUST match whatever --db-path "
             "the following run_hparam_search_dqn.py invocation will use. "
             "Defaults to optuna_studies.db.",
    )
    p_list = sub.add_parser("list", help="list under <stage>/")
    p_list.add_argument("stage", help="Stage prefix to list")
    sub.add_parser("init-bucket", help="create stage2/stage3 placeholder structure")
    args = parser.parse_args()

    load_env()
    bucket = os.environ["R2_BUCKET"]
    s3 = s3_client()

    if args.cmd in ("trial", "phase"):
        with acquire_lock():
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            print(f"[{ts}] r2_sync {args.cmd} starting")
            if args.cmd == "trial":
                rc = cmd_trial(s3, bucket, db_path=args.db_path,
                                best_params_file=args.best_params_file,
                                csv_dir=args.csv_dir)
            else:
                rc = cmd_phase(s3, bucket, args.stage)
    elif args.cmd == "dry-run":
        rc = cmd_dry_run(s3, bucket)
    elif args.cmd == "list":
        rc = cmd_list(s3, bucket, args.stage)
    elif args.cmd == "init-bucket":
        rc = cmd_init_bucket(s3, bucket)
    elif args.cmd == "pull":
        rc = cmd_pull(s3, bucket, study_name=args.study_name, db_path=args.db_path)

    sys.exit(rc)


if __name__ == "__main__":
    main()
