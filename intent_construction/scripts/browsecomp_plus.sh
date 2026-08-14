#!/bin/bash
# =============================================================================
# BrowseComp-Plus data-construction pipeline (search/retrieval domain)
# =============================================================================
# Stage 1: Extraction (function + arguments) + auto-retry
# Stage 2: Argument counterfactual (search)
# Stage 3: Function predecessor via predecessor inference (browsecomp)
#
# NOTE: BrowseComp-Plus queries are an external dependency. Before running,
# follow intent_construction/intent_extraction/dataset_impl/browsecomp_plus/README.md
# to place the decrypted queries at
#   intent_construction/intent_extraction/output/browsecomp_plus/raw_data.jsonl
#
# Usage:
#   ./scripts/browsecomp_plus.sh [WORKERS=20] [MODEL=gpt-5.1] [ARG_CF=4] [FUNC_CF=3]
# =============================================================================

set -e

WORKERS="${1:-20}"
MODEL="${2:-gpt-5.1}"
NUM_COUNTERFACTUALS="${3:-4}"
NUM_PREDECESSORS="${4:-3}"
RETRY_MODEL="gpt-5.1"

DATASET="browsecomp_plus"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONSTRUCT_DIR}/.." && pwd)"

EXTRACTION_DIR="${CONSTRUCT_DIR}/intent_extraction"
COUNTERFACTUAL_DIR="${CONSTRUCT_DIR}/retrospective_expansion/counterfactual"
PREDECESSOR_DIR="${CONSTRUCT_DIR}/retrospective_expansion/predecessor"
FINAL_DATASET_DIR="${REPO_ROOT}/final_dataset"

STAGE1_OUTPUT="${EXTRACTION_DIR}/output/${DATASET}/extracted_test.json"
STAGE2_OUTPUT="${COUNTERFACTUAL_DIR}/output/${DATASET}/argument_counterfactual.json"
STAGE3_OUTPUT="${PREDECESSOR_DIR}/output/${DATASET}/predecessor.json"
FINAL_OUTPUT="${FINAL_DATASET_DIR}/${DATASET}_final.json"

echo "============================================================"
echo "BrowseComp-Plus Pipeline | model=${MODEL} workers=${WORKERS}"
echo "============================================================"

# -----------------------------------------------------------------------------
# Stage 1: Extraction (+ retry)
# -----------------------------------------------------------------------------
echo "[Stage 1/3] Extraction (function + arguments)"
cd "${EXTRACTION_DIR}"
python generate.py --dataset "${DATASET}" --split test --model "${MODEL}" \
    --batch --batch_size "${WORKERS}"

if [ ! -f "${STAGE1_OUTPUT}" ]; then
    echo "Error: Stage 1 output not found at ${STAGE1_OUTPUT}" >&2
    exit 1
fi

echo "[Stage 1.5] Retrying failed extractions with ${RETRY_MODEL}"
python retry_failed.py --dataset "${DATASET}" --split test \
    --model "${RETRY_MODEL}" --batch_size "${WORKERS}"

# -----------------------------------------------------------------------------
# Stage 2: Argument counterfactual (search)
# -----------------------------------------------------------------------------
echo "[Stage 2/3] Argument counterfactual (search)"
cd "${COUNTERFACTUAL_DIR}"
python generate_counterfactuals.py \
    --input "${STAGE1_OUTPUT}" --output "${STAGE2_OUTPUT}" \
    --num_counterfactuals "${NUM_COUNTERFACTUALS}" --model "${MODEL}" \
    --dataset_type search --batch --batch_size "${WORKERS}"

if [ ! -f "${STAGE2_OUTPUT}" ]; then
    echo "Error: Stage 2 output not found at ${STAGE2_OUTPUT}" >&2
    exit 1
fi

echo "[Stage 2.5] Retrying failed argument counterfactuals with ${RETRY_MODEL}"
python retry_failed.py \
    --input "${STAGE1_OUTPUT}" --output "${STAGE2_OUTPUT}" \
    --model "${RETRY_MODEL}" --num_counterfactuals "${NUM_COUNTERFACTUALS}" \
    --dataset_type search --batch_size "${WORKERS}"

# -----------------------------------------------------------------------------
# Stage 3: Function predecessor (predecessor inference, browsecomp)
# -----------------------------------------------------------------------------
echo "[Stage 3/3] Function predecessor (predecessor inference)"
cd "${PREDECESSOR_DIR}"
python generate_predecessors.py \
    --input "${STAGE2_OUTPUT}" --output "${STAGE3_OUTPUT}" \
    --num_predecessors "${NUM_PREDECESSORS}" --model "${MODEL}" \
    --fallback_model "${RETRY_MODEL}" --dataset_type browsecomp --parallel "${WORKERS}"

echo "[Stage 3.5] Retrying failed function predecessors with ${RETRY_MODEL}"
python retry_predecessor_failed.py \
    --input "${STAGE2_OUTPUT}" --output "${STAGE3_OUTPUT}" \
    --model "${RETRY_MODEL}" --dataset_type browsecomp \
    --num_predecessors "${NUM_PREDECESSORS}" --parallel "${WORKERS}"

# -----------------------------------------------------------------------------
# Copy final output
# -----------------------------------------------------------------------------
mkdir -p "${FINAL_DATASET_DIR}"
cp "${STAGE3_OUTPUT}" "${FINAL_OUTPUT}"
echo "============================================================"
echo "Pipeline complete. Final dataset: ${FINAL_OUTPUT}"
echo "============================================================"
