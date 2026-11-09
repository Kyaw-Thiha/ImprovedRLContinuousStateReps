#!/usr/bin/env bash
# Idempotent RunPod CPU pod setup for Stage-0 DQN hyperparameter search.
# Reproducible from the repo alone (apart from .env, which must be supplied).
#
# Usage on a fresh RunPod CPU pod:
#   1. Copy the repo (rsync / git clone)
#   2. Copy your .env into the repo root (R2 creds)
#   3. Run this script: bash scripts/setup_runpod.sh
#   4. Launch the search for one (representation, centering, architecture)
#      cell — every cell is tuned fully independently, one dedicated pod each
#      (see docs/02-experimental-design.md):
#        .venv/bin/python scripts/run_hparam_search_dqn.py \\
#            --representation discrete --centering none --n-trials 500
#
# Cost reference (RunPod CPU pods, Sep 2026):
#   - cpu3c 4-vCPU @ \$0.12/h — ~9-10h wall per cell at the current 500-trial
#     + top-10/10-seed post-hoc validation budget (single-threaded training,
#     so 4 vCPU is functionally equivalent to more for a lower price).
#   See https://runpod.io/pricing for current rates.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# 1. rclone is no longer used (boto3 is the sync backend; see scripts/sync_to_r2.py).
#    We removed rclone after diagnosing AWS-SDK-Go-v2 + Cloudflare R2 SigV4
#    incompatibility. boto3 works fine with the same creds.

# 2. Ensure uv is installed (RunPod Ubuntu base images ship without it).
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Create .venv if missing (idempotent — uv venv is a no-op if it exists).
if [ ! -d .venv ]; then
    echo "[setup] creating .venv with python 3.14..."
    uv venv --python /usr/bin/python3.14 .venv
fi

# 4. Install Stage-0 deps. Always reinstall so this script is also the upgrade
#    path when requirements-stage0.txt changes.
echo "[setup] installing requirements-stage0.txt..."
uv pip install --python .venv/bin/python -r requirements-stage0.txt

# 5. Add boto3 (used by sync_to_r2.py). Was added at runtime during the
#    Sep 22 R2 debugging session; now declared as a dep.
echo "[setup] ensuring boto3 is installed..."
uv pip install --python .venv/bin/python boto3

# 6. Verify .env exists and has R2 creds.
if [ ! -f .env ]; then
    echo "[setup] WARNING: .env not found at $REPO/.env"
    echo "[setup]   copy your local .env to this pod before running search"
fi

# 7. Verify r2 sync connectivity. The first run will validate creds.
if [ -f .env ]; then
    echo "[setup] verifying R2 connectivity..."
    if .venv/bin/python scripts/sync_to_r2.py dry-run >/dev/null 2>&1; then
        echo "[setup] R2 connectivity: OK"
    else
        echo "[setup] R2 connectivity: FAILED — check .env"
    fi
fi

echo "[setup] done. Repo ready at $REPO"
echo ""
echo "To launch a subset of the 9 independent cells (one dedicated pod each is"
echo "the normal way to run these; nohup here is only for local/single-pod testing):"
cat <<'EOF'
mkdir -p outputs/hparam_search/logs
nohup .venv/bin/python scripts/run_hparam_search_dqn.py \
    --representation discrete    --centering none --n-trials 500 \
    > outputs/hparam_search/logs/dqn_discrete_none.log 2>&1 &
nohup .venv/bin/python scripts/run_hparam_search_dqn.py \
    --representation ssp         --centering none --n-trials 500 \
    > outputs/hparam_search/logs/dqn_ssp_none.log 2>&1 &
nohup .venv/bin/python scripts/run_hparam_search_dqn.py \
    --representation tile_coding --centering none --n-trials 500 \
    > outputs/hparam_search/logs/dqn_tile_coding_none.log 2>&1 &
EOF
