# Stage-0 DQN image for RunPod CPU pods.
#
# Builds the Stage-0 Python env from requirements-stage0.txt plus boto3, bakes
# the source tree + conf/ + A2C warm-start YAMLs into the image, and exposes
# one entrypoint:
#
#   dqn-entrypoint (default) -> single-cell Optuna search for one
#                                (representation, centering, architecture)
#                                combination. Every one of the 9 cells (3
#                                representations x 3 centering modes, per
#                                architecture) is tuned fully independently —
#                                launch one dedicated pod per cell, overriding
#                                CMD at pod creation time. There is no
#                                staged/locked-hyperparameter shortcut; see
#                                docs/02-experimental-design.md.
#
# Build:    docker build -t kevinkyawthiha/dqn-ssp-centering:stage0 .
# Test:     docker run --rm --env-file .env \
#                 -e R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e R2_BUCKET \
#                 kevinkyawthiha/dqn-ssp-centering:stage0 \
#                 --representation discrete --centering none --n-trials 5 --no-sync
# Push:     docker push kevinkyawthiha/dqn-ssp-centering:stage0
# RunPod:   create-pod --image kevinkyawthiha/dqn-ssp-centering:stage0 \
#                    --env R2_*=...  (then override CMD per pod, --no-volume needed)
#
# Source: docs/02-experimental-design.md "Hyperparameter tuning methodology".

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /project

# git is required to fetch the pinned sspspace wheel from GitHub.
# ca-certificates is already in python:slim but we re-pin it for explicitness.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---- Stage-0 deps (one layer, caches well across source-only rebuilds) ----
COPY requirements-stage0.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements-stage0.txt \
 && pip install boto3

# ---- Source tree ----
# experiments/  : DQNTrial() — the actual training runner
# rl/           : networks, replay buffer, centering math
# scripts/      : run_hparam_search_dqn.py, sync_to_r2.py, entrypoints
# conf/         : Optuna search-space YAMLs (kept for completeness; not used by
#                 run_hparam_search_dqn.py directly but referenced by sibling tools)
# outputs/hparam_search/best/  : A2C best-params YAMLs that warm-start DQN trial #0
#                                 (see _load_a2c_best in run_hparam_search_dqn.py)
COPY experiments/ ./experiments/
COPY rl/          ./rl/
COPY scripts/     ./scripts/
COPY conf/        ./conf/
COPY outputs/hparam_search/best/ ./outputs/hparam_search/best/

# Pre-create writable output dirs. At RunPod runtime these get overlaid by the
# /vol/ network volume mount, so per-cell DBs and logs survive pod restarts.
RUN mkdir -p outputs/hparam_search/dqn outputs/hparam_search/logs

# Install the entrypoint script into PATH.
COPY scripts/entrypoint_dqn_search.sh /usr/local/bin/dqn-entrypoint
RUN chmod +x /usr/local/bin/dqn-entrypoint

# Default: the discrete/none/linear cell as a sane fallback. Override CMD at
# pod creation to point at a different (representation, centering,
# architecture) cell — see scripts/setup_runpod.sh for launch examples.
ENTRYPOINT ["dqn-entrypoint"]
CMD ["--representation", "discrete", \
     "--centering", "none", \
     "--n-trials", "500"]
