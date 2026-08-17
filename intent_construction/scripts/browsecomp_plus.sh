#!/bin/bash
# =============================================================================
# BrowseComp-Plus data-construction pipeline (search/retrieval domain)
# =============================================================================
# Stage 1: Extraction (function + arguments) + auto-retry
# Stage 2: Argument counterfactual (search)
# Stage 3: Function predecessor via predecessor inference (browsecomp)
#
# NOTE: All construction runs on a high-memory Modal CPU. The named Secret
# evolving-intent-kimi-k2-6 supplies compatible Kimi credentials and pricing;
# decrypted gold documents remain on the private Modal Volume.
#
# Usage:
#   ./scripts/browsecomp_plus.sh [WORKERS=20] [MODEL=gpt-5.1] [ARG_CF=4] [FUNC_CF=3]
# =============================================================================

set -e

WORKERS="${1:-20}"
MODEL="${2:-kimi-k2.6}"
NUM_COUNTERFACTUALS="${3:-4}"
NUM_PREDECESSORS="${4:-3}"
PLAN_A_MODEL="kimi-k2.6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONSTRUCT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [ "$MODEL" != "$PLAN_A_MODEL" ]; then
    echo "ERROR: Plan A requires exactly one model: $PLAN_A_MODEL" >&2
    exit 2
fi
if [ "$NUM_COUNTERFACTUALS" != "4" ] || [ "$NUM_PREDECESSORS" != "3" ]; then
    echo "ERROR: Plan A fixes four counterfactuals and three predecessors." >&2
    exit 2
fi
if [ -n "${LLM_COST_HARD_CAP_USD:-}" ]; then
    echo "ERROR: Plan A records usage only; unset LLM_COST_HARD_CAP_USD." >&2
    exit 2
fi
if ! command -v modal >/dev/null 2>&1; then
    echo "ERROR: Modal CLI is required. Install reproduction/requirements-browsecomp.txt." >&2
    exit 2
fi

echo "============================================================"
echo "BrowseComp-Plus Pipeline | model=${MODEL} workers=${WORKERS}"
echo "============================================================"

modal run -m reproduction.browsecomp_construction_modal::main --workers "$WORKERS"

echo "============================================================"
echo "Pipeline complete. Final dataset: final_dataset/browsecomp_plus_final.json"
echo "============================================================"
