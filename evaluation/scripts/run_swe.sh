#!/bin/bash
# Fixed Kimi SWE-bench Verified reproduction: 50 published IDs, single + evolve.
#
# Default execution runs the orchestrator in a Modal Sandbox. The named Modal
# Secret supplies LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_MAP, LLM_PRICE_MAP,
# LLM_BACKEND=compatible, and LLM_DISABLE_OUTPUT_LIMITS=1. It should also supply
# MODAL_TOKEN_ID / MODAL_TOKEN_SECRET if nested Modal auth is not inherited.
# No key value is accepted on the command line.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXPECTED_MODEL="kimi-k2.6"
if [ "$#" -gt 0 ]; then
    if [ "$#" -ne 1 ] || [ "$1" != "$EXPECTED_MODEL" ]; then
        echo "ERROR: this hardened path accepts only one model: $EXPECTED_MODEL" >&2
        exit 2
    fi
fi

EXECUTION_BACKEND="${EXECUTION_BACKEND:-modal}"
DATA_PATH="${DATA_PATH:-final_dataset/swe_bench_verified_final.json}"
OFFICIAL_DATASET_PATH="${OFFICIAL_DATASET_PATH:-tmp/source-data/swe_bench_verified_princeton.parquet}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_NAME="${RUN_NAME:-kimi-k2.6}"
RUN_DIR="${RUN_DIR:-evaluation/swe_runs/$RUN_NAME}"
DRY_RUN="${DRY_RUN:-0}"

case "$EXECUTION_BACKEND" in
    modal)
        command -v modal >/dev/null 2>&1 || {
            echo "ERROR: Modal CLI is not installed." >&2
            exit 1
        }
        MODAL_SECRET_NAME="${MODAL_SECRET_NAME:-}"
        if [ -z "$MODAL_SECRET_NAME" ]; then
            echo "ERROR: set MODAL_SECRET_NAME to a Modal Secret name, never a key value." >&2
            exit 1
        fi
        MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-evolving-intent-swe-results}"
        modal_args=(
            modal run evaluation/swe_bench/modal_app.py
            --secret-name "$MODAL_SECRET_NAME"
            --data-path "$DATA_PATH"
            --official-dataset-path "$OFFICIAL_DATASET_PATH"
            --run-name "$RUN_NAME"
            --workers "$NUM_WORKERS"
            --volume-name "$MODAL_VOLUME_NAME"
        )
        [ "$DRY_RUN" = "1" ] && modal_args+=(--dry-run)
        "${modal_args[@]}"
        ;;
    local-swerex-modal)
        local_args=(
            python -m evaluation.swe_bench.run
            --data-path "$DATA_PATH"
            --official-dataset-path "$OFFICIAL_DATASET_PATH"
            --run-dir "$RUN_DIR"
            --workers "$NUM_WORKERS"
        )
        [ "$DRY_RUN" = "1" ] && local_args+=(--dry-run)
        "${local_args[@]}"
        ;;
    *)
        echo "ERROR: EXECUTION_BACKEND must be modal or local-swerex-modal." >&2
        exit 2
        ;;
esac
