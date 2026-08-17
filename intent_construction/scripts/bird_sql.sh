#!/bin/bash
# Reproduce the fixed published BIRD-SQL construction subset.

set -euo pipefail

N="${1:-100}"
MODEL="${2:-kimi-k2.6}"
WORKERS="${3:-4}"
COUNTERFACTUALS="${4:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRUCT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CONSTRUCT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BIRD_MODULE="intent_construction.intent_extraction.dataset_impl.bird_sql.run_published_construction"
TASK_IDS_FILE="${REPO_ROOT}/intent_construction/eval_indices/bird_sql_task_ids.json"
EVAL_MANIFEST="${REPO_ROOT}/intent_construction/eval_indices/bird_sql_eval_ids.json"
RUN_DIR="${BIRD_RUN_DIR:-${REPO_ROOT}/reproduction/runs/bird-sql-kimi-k2.6}"
DATA_DIR="${BIRD_DATA_DIR:-${RUN_DIR}/bird_data}"
EXTRACTED="${REPO_ROOT}/intent_construction/intent_extraction/output/bird_sql/extracted.json"
COND_OUT="${REPO_ROOT}/intent_construction/retrospective_expansion/counterfactual/output/bird_sql/argument_counterfactual.json"
STAGE3_OUTPUT="${REPO_ROOT}/intent_construction/retrospective_expansion/predecessor/output/bird_sql/predecessor.json"
FINAL_OUTPUT="${REPO_ROOT}/final_dataset/bird_sql_final.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ONLY_DATABASE="${BIRD_ONLY_DATABASE:-}"

if [[ "${N}" != "100" ]]; then
    echo "ERROR: the published BIRD-SQL reproduction is fixed at N=100." >&2
    exit 2
fi
if [[ "${MODEL}" != "kimi-k2.6" ]]; then
    echo "ERROR: construction must use exactly kimi-k2.6." >&2
    exit 2
fi
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    echo "ERROR: use the locked LLM_REASONING_EFFORT policy, not REASONING_EFFORT." >&2
    exit 2
fi
if [[ -n "${LLM_REASONING_EFFORT:-}" && "${LLM_REASONING_EFFORT}" != "medium" ]]; then
    echo "ERROR: BIRD construction requires LLM_REASONING_EFFORT=medium." >&2
    exit 2
fi
if [[ -n "${LLM_COST_HARD_CAP_USD:-}" ]]; then
    echo "ERROR: this run records usage but must not enable LLM_COST_HARD_CAP_USD." >&2
    exit 2
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] &&
   { [[ -z "${AZURE_OPENAI_API_KEY:-}" ]] || [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; } &&
   { [[ -z "${LLM_API_KEY:-}" ]] || [[ -z "${LLM_BASE_URL:-}" ]]; }; then
    echo "ERROR: configure OpenAI, Azure, or LLM_API_KEY + LLM_BASE_URL credentials." >&2
    exit 2
fi

export LLM_DISABLE_OUTPUT_LIMITS=1
export LLM_LOCKED_MODEL="${MODEL}"
export LLM_REASONING_EFFORT=medium
export LLM_REQUIRE_USAGE_ACCOUNTING=1
unset LLM_DEFAULT_MAX_OUTPUT_TOKENS
unset LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS
unset LLM_COST_HARD_CAP_USD
export LLM_USAGE_LEDGER_PATH="${LLM_USAGE_LEDGER_PATH:-${RUN_DIR}/usage.jsonl}"
if [[ -z "${LLM_PRICE_MAP:-}" ]]; then
    export LLM_PRICE_MAP='{"kimi-k2.6":{"input":0.95,"cached_input":0.16,"output":4.0,"reasoning":4.0}}'
fi

mkdir -p \
    "${RUN_DIR}" \
    "${DATA_DIR}" \
    "$(dirname "${EXTRACTED}")" \
    "$(dirname "${COND_OUT}")" \
    "$(dirname "${STAGE3_OUTPUT}")" \
    "$(dirname "${FINAL_OUTPUT}")" \
    "$(dirname "${LLM_USAGE_LEDGER_PATH}")"

echo "BIRD-SQL construction: one database at a time, fixed 100 published IDs."
DATABASE_ARGS=()
if [[ -n "${ONLY_DATABASE}" ]]; then
    DATABASE_ARGS=(--database "${ONLY_DATABASE}")
fi
"${PYTHON_BIN}" -m "${BIRD_MODULE}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    --counterfactuals "${COUNTERFACTUALS}" \
    --task-ids-file "${TASK_IDS_FILE}" \
    --manifest "${EVAL_MANIFEST}" \
    --run-dir "${RUN_DIR}" \
    --data-dir "${DATA_DIR}" \
    --extracted-output "${EXTRACTED}" \
    --counterfactual-output "${COND_OUT}" \
    --predecessor-output "${STAGE3_OUTPUT}" \
    --final-output "${FINAL_OUTPUT}" \
    ${DATABASE_ARGS[@]+"${DATABASE_ARGS[@]}"} \
    --resume

if [[ -n "${ONLY_DATABASE}" ]]; then
    echo "BIRD-SQL one-database smoke checkpointed: ${ONLY_DATABASE}"
    echo "Usage ledger: ${LLM_USAGE_LEDGER_PATH}"
    exit 0
fi

echo "BIRD-SQL construction complete: ${FINAL_OUTPUT}"
echo "Usage ledger: ${LLM_USAGE_LEDGER_PATH}"
