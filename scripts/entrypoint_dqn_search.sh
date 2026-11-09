#!/usr/bin/env bash
# Single-cell entrypoint for the DQN hparam search Docker image. Every one of
# the 9 (representation, centering) cells, per architecture, is tuned fully
# independently — see docs/02-experimental-design.md — so this script's job
# is just: restore this ONE cell's state from R2, then exec the search.
#
# Behavior:
#   1. Materialize R2 credentials into /project/.env. Two sources, in priority order:
#      a. Pod env vars R2_* (set via RunPod create-pod env block) — preferred path,
#         no network volume needed.
#      b. File at $ENV_FILE (default /vol/.env) — fallback for setups where R2
#         creds are mounted via a volume, or for local docker testing with --env-file.
#      If neither is present, the search runs in --no-sync mode (results stay local).
#   2. Verify R2 connectivity (matches setup_runpod.sh fail-fast pattern).
#   3. Restore Optuna DB + best_params from R2 (best-effort; failure → search
#      starts fresh from warm-start trial 0, same as a clean image — never fatal).
#   4. exec scripts/run_hparam_search_dqn.py with all forwarded CLI args so
#      SIGTERM/SIGINT from `docker stop` propagate to Python and let the
#      in-flight Optuna trial finish + write its row before exit.
#
# Usage (one pod per cell, no /vol/ needed):
#   docker run --rm \
#       -e R2_ACCOUNT_ID=... -e R2_ACCESS_KEY_ID=... -e R2_SECRET_ACCESS_KEY=... \
#       -e R2_BUCKET=rl-ssp-centering \
#       dqn-ssp-centering:stage0 \
#       --representation discrete --centering none --n-trials 500

set -euo pipefail

ENV_FILE="${ENV_FILE:-/vol/.env}"
HAS_R2_ENV=false

if [ -n "${R2_ACCOUNT_ID:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ] && \
   [ -n "${R2_SECRET_ACCESS_KEY:-}" ] && [ -n "${R2_BUCKET:-}" ]; then
    echo "[entrypoint] using R2 creds from pod env (R2_*)"
    # printf %s\n preserves special chars (base64 secrets can contain +/=) without
    # shell expansion. sync_to_r2.py's load_env() reads REPO/.env from disk, so we
    # materialize the env vars into the file before invoking it.
    printf 'R2_ACCOUNT_ID=%s\nR2_ACCESS_KEY_ID=%s\nR2_SECRET_ACCESS_KEY=%s\nR2_BUCKET=%s\n' \
        "$R2_ACCOUNT_ID" "$R2_ACCESS_KEY_ID" "$R2_SECRET_ACCESS_KEY" "$R2_BUCKET" \
        > /project/.env
    HAS_R2_ENV=true
elif [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
    echo "[entrypoint] sourced $ENV_FILE"
    ln -sf "$ENV_FILE" /project/.env
    HAS_R2_ENV=true
else
    echo "[entrypoint] WARNING: no R2 creds found — running without R2 sync."
    echo "[entrypoint]   Set R2_* env vars on the pod, or mount $ENV_FILE."
    set -- --no-sync "$@"
fi

# Pilot mode (added 2026-10-04): `dqn-entrypoint pilot [args...]` runs
# scripts/pilot_greedy_rank_check.py instead of a search. It fetches its own inputs from
# R2 (read-only) and writes results under the NEW prefix stage0_cartpole/pilot/, so the
# per-study restore below (and anything it could overwrite) is deliberately skipped.
if [ "${1:-}" = "pilot" ]; then
    shift
    cd /project
    exec python scripts/pilot_greedy_rank_check.py "$@"
fi

if [ "$HAS_R2_ENV" = "true" ]; then
    if python scripts/sync_to_r2.py dry-run >/dev/null 2>&1; then
        echo "[entrypoint] R2 connectivity: OK"
        echo "[entrypoint] restoring state from R2..."

        # Extract --representation/--centering/--architecture/--db-path from args
        # to construct the Optuna study name and target the SAME local DB path
        # the search itself will use, so pull picks the right per-study DB
        # instead of the legacy shared key (avoids cross-pod clobbering under
        # concurrent sync) AND restores to the path run_hparam_search_dqn.py
        # will actually read (a run launched with a custom --db-path that pull
        # doesn't also honor would silently fail to resume, and a later
        # "trial" push from that falsely-fresh run would overwrite the correct
        # R2 snapshot with an empty one — see sync_to_r2.py's cmd_pull docstring).
        ARGS_STR="$*"
        REP=$(echo "$ARGS_STR" | grep -oP -- '--representation \K\S+' || true)
        CEN=$(echo "$ARGS_STR" | grep -oP -- '--centering \K\S+' || true)
        ARCH=$(echo "$ARGS_STR" | grep -oP -- '--architecture \K\S+' || true)
        DBP=$(echo "$ARGS_STR" | grep -oP -- '--db-path \K\S+' || true)
        ARCH_SUFFIX=""
        if [ -n "$ARCH" ] && [ "$ARCH" != "linear" ]; then
            ARCH_SUFFIX="_${ARCH}"
        fi
        STUDY_NAME=""
        if [ -n "$REP" ] && [ -n "$CEN" ]; then
            # _v4 must match STUDY_GEN in run_hparam_search_dqn.py's main().
            STUDY_NAME="dqn_${REP}_${CEN}${ARCH_SUFFIX}_v4_hparam_search"
        fi
        DBP_ARGS=()
        if [ -n "$DBP" ]; then
            DBP_ARGS=(--db-path "$DBP")
        fi

        if [ -n "$STUDY_NAME" ]; then
            if python scripts/sync_to_r2.py pull "$STUDY_NAME" "${DBP_ARGS[@]}"; then
                echo "[entrypoint] R2 restore: OK (study=$STUDY_NAME)"
            else
                echo "[entrypoint] R2 restore: FAILED — search starts fresh"
            fi
        else
            if python scripts/sync_to_r2.py pull "${DBP_ARGS[@]}"; then
                echo "[entrypoint] R2 restore: OK (legacy fallback)"
            else
                echo "[entrypoint] R2 restore: FAILED — search starts fresh"
            fi
        fi
    else
        echo "[entrypoint] R2 connectivity: FAILED — sync will fail at trial boundaries (search itself will still run)"
    fi
fi

cd /project

# exec so the python process becomes PID 1 — signals from `docker stop` reach
# Optuna directly, which lets the in-flight trial finish + flush its DB row.
exec python scripts/run_hparam_search_dqn.py "$@"
