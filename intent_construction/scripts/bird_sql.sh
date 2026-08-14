#!/bin/bash
# =============================================================================
# BIRD-SQL complex-query data-construction pipeline
# =============================================================================
# Stage 0: Select complex queries from BIRD train+dev
# Stage 1: Extract function+arguments for the selected subset
# Stage 2: Argument counterfactual (WHERE + HAVING + LIMIT)
# Stage 3: Function predecessor via LLM multi-clause rewrite (+ naturalizer)
#
# Usage:
#   ./scripts/bird_sql.sh [N=200] [MODEL=gpt-5.1] [WORKERS=8] [COUNTERFACTUALS=3] [TAG=v2_n200]
# =============================================================================

set -e
set -o pipefail

N="${1:-200}"
MODEL="${2:-gpt-5.1}"
WORKERS="${3:-8}"
COUNTERFACTUALS="${4:-3}"
TAG="${5:-v2_n200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONSTRUCT_DIR}/.." && pwd)"

DATASET="bird_sql"

BIRD_DIR="${CONSTRUCT_DIR}/intent_extraction/dataset_impl/bird_sql"
EXTRACTION_DIR="${CONSTRUCT_DIR}/intent_extraction"
COND_DIR="${CONSTRUCT_DIR}/retrospective_expansion/counterfactual"
PREDECESSOR_DIR="${CONSTRUCT_DIR}/retrospective_expansion/predecessor"
FINAL_DATASET_DIR="${REPO_ROOT}/final_dataset"

SELECTED="${BIRD_DIR}/data/selected_complex_${TAG}.json"
EXTRACTED="${EXTRACTION_DIR}/output/${DATASET}/extracted.json"
COND_OUT="${COND_DIR}/output/${DATASET}/argument_counterfactual.json"
STAGE3_OUTPUT="${PREDECESSOR_DIR}/output/${DATASET}/predecessor.json"
FINAL_OUTPUT="${FINAL_DATASET_DIR}/${DATASET}_final.json"

mkdir -p "$(dirname "${EXTRACTED}")" "$(dirname "${COND_OUT}")" "$(dirname "${STAGE3_OUTPUT}")"

echo "=================================================================="
echo "BIRD-SQL Complex Pipeline"
echo "------------------------------------------------------------------"
echo "  N samples:         ${N}"
echo "  Model:             ${MODEL}"
echo "  Workers:           ${WORKERS}"
echo "  Counterfactuals/sample:   ${COUNTERFACTUALS}"
echo "  Final output:      ${FINAL_OUTPUT}"
echo "=================================================================="

# -----------------------------------------------------------------------------
# Stage 0: Selection
# -----------------------------------------------------------------------------
if [ -f "${SELECTED}" ]; then
    EXISTING=$(python -c "import json; print(len(json.load(open('${SELECTED}'))))")
    echo "[Stage 0] Selection cache hit (${EXISTING} samples) — skipping."
else
    echo "[Stage 0] Selecting ${N} complex BIRD samples..."
    python "${BIRD_DIR}/select_complex.py" \
        --n "${N}" \
        --num_workers "${WORKERS}" \
        --out "${SELECTED}"
fi

# -----------------------------------------------------------------------------
# Stage 1: Extraction (function + arguments for the selected subset)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 1] Extracting function+arguments for the selected subset..."
python "${BIRD_DIR}/extract_selected.py" \
    --input  "${SELECTED}" \
    --output "${EXTRACTED}" \
    --model  "${MODEL}" \
    --num_workers "${WORKERS}"

# -----------------------------------------------------------------------------
# Stage 2: Argument counterfactual (WHERE + HAVING + LIMIT)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 2] Argument counterfactual (WHERE + HAVING + LIMIT)..."
python "${COND_DIR}/generate_counterfactuals_sql.py" \
    --input  "${EXTRACTED}" \
    --output "${COND_OUT}" \
    --num_counterfactuals "${COUNTERFACTUALS}" \
    --num_workers "${WORKERS}"

# -----------------------------------------------------------------------------
# Stage 3: Function predecessor (LLM multi-clause + naturalizer)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 3] Function predecessor (LLM multi-clause rewrite + naturalizer)..."
python "${PREDECESSOR_DIR}/generate_predecessors_sql_llm.py" \
    --input  "${COND_OUT}" \
    --output "${STAGE3_OUTPUT}" \
    --num_predecessors "${COUNTERFACTUALS}" \
    --model  "${MODEL}" \
    --num_workers "${WORKERS}"

# -----------------------------------------------------------------------------
# Copy final output
# -----------------------------------------------------------------------------
mkdir -p "${FINAL_DATASET_DIR}"
cp "${STAGE3_OUTPUT}" "${FINAL_OUTPUT}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "✓ Pipeline complete"
echo "------------------------------------------------------------------"
echo "  Selected:           ${SELECTED}"
echo "  Extracted:          ${EXTRACTED}"
echo "  Argument counterfactual:  ${COND_OUT}"
echo "  Function predecessor:       ${STAGE3_OUTPUT}"
echo "  Final dataset:      ${FINAL_OUTPUT}"
echo "=================================================================="
echo ""
echo "Next steps:"
echo "  1. Build the user-simulation dataset from ${FINAL_OUTPUT}."
echo "  2. Run evaluation with execution accuracy:"
echo "       python evaluation/runners/run_experiment.py \\"
echo "         --dataset_name bird_sql --data_path ${FINAL_OUTPUT} --models ${MODEL}"
