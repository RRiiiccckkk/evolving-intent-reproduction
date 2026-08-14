#!/bin/bash
# =============================================================================
# SWE-bench Verified function-change data-construction pipeline
# =============================================================================
# SWE function-switch scenarios cannot swap in an arbitrary distractor bug (the
# docker container is pinned to one instance's base_commit / tests), so the
# function chain is built as:
#
#   Turn 1: G1 orientation question (generate_g1_swe.py)
#   Turn 2: G2 implementation-planning precursor (generate_impl_precursors_swe.py)
#   Turn 3: G3 target bug fix (the original SWE-bench instance)
#
# Stages:
#   Stage 1: Extraction (function + arguments) for the SWE-bench Verified pool
#   Stage 2: Argument counterfactual (SWE-specific, category-aware)
#   Stage 3: Real-bug pairing      — attach a same-repo predecessor (G2)
#   Stage 4: G1 orientation function    — prepend a repo-orientation question
#   Stage 5: Impl-precursor function     — replace G2 with an implementation plan
#
# The design rationale lives in evaluation/SWE_README.md
# ("Function-change design (impl-precursor)").
#
# Usage:
#   ./scripts/swe_bench_verified.sh [MODEL=gpt-5.1] [WORKERS=8] [COUNTERFACTUALS=2] [SEED=42] [TARGET=<extracted.json>]
#
# TARGET is the curated G3 target subset fed into argument counterfactual.
# It defaults to the full Stage-1 extraction output; point it at a curated
# subset (e.g. extracted_filtered_n251.json) to reproduce the paper set.
# =============================================================================

set -e
set -o pipefail

MODEL="${1:-gpt-5.1}"
WORKERS="${2:-8}"
COUNTERFACTUALS="${3:-2}"
SEED="${4:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONSTRUCT_DIR}/.." && pwd)"

EXTRACTION_DIR="${CONSTRUCT_DIR}/intent_extraction"
COND_DIR="${CONSTRUCT_DIR}/retrospective_expansion/counterfactual"
PREDECESSOR_DIR="${CONSTRUCT_DIR}/retrospective_expansion/predecessor"
FINAL_DATASET_DIR="${REPO_ROOT}/final_dataset"

EXTRACTED="${EXTRACTION_DIR}/output/swe_bench_verified/extracted_test.json"
COND_OUT="${COND_DIR}/output/swe_bench_verified/argument_counterfactual.json"
PAIRED="${PREDECESSOR_DIR}/output/swe_bench_verified/paired.json"
PAIRED_G1="${PREDECESSOR_DIR}/output/swe_bench_verified/paired_g1.json"
FINAL_OUT="${PREDECESSOR_DIR}/output/swe_bench_verified/paired_g1_implprec.json"
FINAL_OUTPUT="${FINAL_DATASET_DIR}/swe_bench_verified_final.json"

# Curated G3 target subset for argument counterfactual (defaults to full pool).
TARGET="${5:-${EXTRACTED}}"

mkdir -p "$(dirname "${COND_OUT}")" "$(dirname "${PAIRED}")"

echo "=================================================================="
echo "SWE-bench Verified Function-Change Pipeline"
echo "------------------------------------------------------------------"
echo "  Model:             ${MODEL}"
echo "  Workers:           ${WORKERS}"
echo "  Argument counterfactuals:${COUNTERFACTUALS}"
echo "  Seed:              ${SEED}"
echo "  Target subset:     ${TARGET}"
echo "=================================================================="

# -----------------------------------------------------------------------------
# Stage 1: Extraction (function + arguments)
# -----------------------------------------------------------------------------
if [ -f "${EXTRACTED}" ]; then
    echo ""
    echo "[Stage 1] Extraction cache hit — skipping (${EXTRACTED})."
else
    echo ""
    echo "[Stage 1] Extracting function+arguments for SWE-bench Verified..."
    python "${EXTRACTION_DIR}/generate.py" \
        --dataset swe_bench_verified \
        --split test \
        --model "${MODEL}" \
        --batch --batch_size "${WORKERS}" \
        --output "${EXTRACTED}"
fi

# -----------------------------------------------------------------------------
# Stage 2: Argument counterfactual (SWE-specific, category-aware)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 2] Argument counterfactual (constraint/scope/approach/location)..."
python "${COND_DIR}/generate_counterfactuals_swe.py" \
    --input  "${TARGET}" \
    --output "${COND_OUT}" \
    --model  "${MODEL}" \
    --num_counterfactuals "${COUNTERFACTUALS}" \
    --num_workers "${WORKERS}"

# -----------------------------------------------------------------------------
# Stage 3: Real-bug pairing (attach a same-repo predecessor function G2)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 3] Pairing each target (G3) with a same-repo predecessor (G2)..."
python "${PREDECESSOR_DIR}/pair_swe_bugs.py" \
    --target_input "${COND_OUT}" \
    --pool_input   "${EXTRACTED}" \
    --output       "${PAIRED}" \
    --seed         "${SEED}"

# -----------------------------------------------------------------------------
# Stage 4: G1 orientation function (prepend a repo-orientation question)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 4] Generating G1 orientation functions..."
python "${PREDECESSOR_DIR}/generate_g1_swe.py" \
    --input  "${PAIRED}" \
    --output "${PAIRED_G1}" \
    --model  "${MODEL}" \
    --num_workers "${WORKERS}" \
    --seed   "${SEED}"

# -----------------------------------------------------------------------------
# Stage 5: Implementation-precursor function (replace G2 with an impl plan)
# -----------------------------------------------------------------------------
echo ""
echo "[Stage 5] Replacing G2 with an implementation-planning precursor..."
python "${PREDECESSOR_DIR}/generate_impl_precursors_swe.py" \
    --input  "${PAIRED_G1}" \
    --output "${FINAL_OUT}" \
    --model  "${MODEL}" \
    --num_workers "${WORKERS}" \
    --seed   "${SEED}" \
    --reasoning_effort medium

# -----------------------------------------------------------------------------
# Copy final output
# -----------------------------------------------------------------------------
mkdir -p "${FINAL_DATASET_DIR}"
cp "${FINAL_OUT}" "${FINAL_OUTPUT}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "✓ Pipeline complete"
echo "------------------------------------------------------------------"
echo "  Extracted:          ${EXTRACTED}"
echo "  Argument counterfactual:  ${COND_OUT}"
echo "  Paired (G2):        ${PAIRED}"
echo "  G1 orientation:     ${PAIRED_G1}"
echo "  Final (impl-prec):  ${FINAL_OUT}"
echo "  Final dataset:      ${FINAL_OUTPUT}"
echo "=================================================================="
echo ""
echo "Next steps:"
echo "  1. (Optional) Select the canonical 50-task eval subset listed in"
echo "       intent_construction/eval_indices/swe_bench_verified_eval_ids.json."
echo "  2. Pre-pull docker images:  bash evaluation/scripts/prepull_swe_images.sh"
echo "  3. Run evaluation:"
echo "       python evaluation/runners/run_swe_mini_agent.py \\"
echo "         --data_path ${FINAL_OUTPUT} --models ${MODEL} \\"
echo "         --use_tool_calling --num_turns 3 --num_switches 2"
