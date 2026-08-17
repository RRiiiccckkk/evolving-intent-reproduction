#!/bin/bash
# Evaluate the paper's fixed BIRD-SQL single and evolve settings.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL="${1:-kimi-k2.6}"
if [[ "$#" -gt 1 ]]; then
    echo "ERROR: the strict reproduction accepts one model only." >&2
    exit 2
fi
if [[ "${MODEL}" != "kimi-k2.6" ]]; then
    echo "ERROR: BIRD evaluation must use exactly kimi-k2.6." >&2
    exit 2
fi
if [[ -n "${REASONING_EFFORT:-}" ]]; then
    echo "ERROR: use the locked LLM_REASONING_EFFORT policy, not REASONING_EFFORT." >&2
    exit 2
fi
if [[ -n "${LLM_REASONING_EFFORT:-}" && "${LLM_REASONING_EFFORT}" != "medium" ]]; then
    echo "ERROR: BIRD evaluation requires LLM_REASONING_EFFORT=medium." >&2
    exit 2
fi
if [[ -n "${LLM_COST_HARD_CAP_USD:-}" ]]; then
    echo "ERROR: usage is recorded, but LLM_COST_HARD_CAP_USD must be unset." >&2
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]] &&
   { [[ -z "${AZURE_OPENAI_API_KEY:-}" ]] || [[ -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; } &&
   { [[ -z "${LLM_API_KEY:-}" ]] || [[ -z "${LLM_BASE_URL:-}" ]]; }; then
    echo "ERROR: configure OpenAI, Azure, or LLM_API_KEY + LLM_BASE_URL credentials." >&2
    exit 2
fi

DATA_PATH="${DATA_PATH:-final_dataset/bird_sql_final.json}"
DATASET_NAME="${DATASET_NAME:-bird_sql_n100}"
TASK_IDS_FILE="${TASK_IDS_FILE:-intent_construction/eval_indices/bird_sql_task_ids.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-intent_construction/eval_indices/bird_sql_eval_ids.json}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-evaluation/logs/bird}"
RUN_DIR="${BIRD_RUN_DIR:-${REPO_ROOT}/reproduction/runs/bird-sql-kimi-k2.6}"
BIRD_DATA_DIR="${BIRD_DATA_DIR:-${RUN_DIR}/bird_data}"
BIRD_DOWNLOADER="intent_construction.intent_extraction.dataset_impl.bird_sql.download_required"
BIRD_RUNNER="evaluation.runners.run_bird_experiment"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ONLY_DATABASE="${BIRD_ONLY_DATABASE:-}"

export LLM_DISABLE_OUTPUT_LIMITS=1
export LLM_LOCKED_MODEL="${MODEL}"
export LLM_REASONING_EFFORT=medium
export LLM_REQUIRE_USAGE_ACCOUNTING=1
unset LLM_DEFAULT_MAX_OUTPUT_TOKENS
unset LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS
unset LLM_COST_HARD_CAP_USD
export LLM_USAGE_LEDGER_PATH="${LLM_USAGE_LEDGER_PATH:-${RUN_DIR}/evaluation_usage.jsonl}"
if [[ -z "${LLM_PRICE_MAP:-}" ]]; then
    export LLM_PRICE_MAP='{"kimi-k2.6":{"input":0.95,"cached_input":0.16,"output":4.0,"reasoning":4.0}}'
fi

mkdir -p "${LOG_DIR}" "${BIRD_DATA_DIR}" "$(dirname "${LLM_USAGE_LEDGER_PATH}")"

run_setting() {
    local name="$1"
    local turns="$2"
    local revisions="$3"
    local switches="$4"
    local db_id="$5"
    local log="${LOG_DIR}/bird__kimi-k2.6__${name}.log"
    "${PYTHON_BIN}" -u -m "${BIRD_RUNNER}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --model "${MODEL}" \
        --task_ids_file "${TASK_IDS_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --num_turns "${turns}" \
        --num_revisions "${revisions}" \
        --num_switches "${switches}" \
        --ordering interleaved \
        --db_id "${db_id}" \
        --resume \
        2>&1 | tee -a "${log}"
}

list_pending() {
    local turns="$1"
    local revisions="$2"
    local switches="$3"
    "${PYTHON_BIN}" -m "${BIRD_RUNNER}" \
        --data_path "${DATA_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --model "${MODEL}" \
        --task_ids_file "${TASK_IDS_FILE}" \
        --num_workers "${NUM_WORKERS}" \
        --num_turns "${turns}" \
        --num_revisions "${revisions}" \
        --num_switches "${switches}" \
        --ordering interleaved \
        --list_pending_databases \
        --resume
}

is_listed() {
    local target="$1"
    local values="$2"
    local value
    while IFS= read -r value; do
        if [[ "${value}" == "${target}" ]]; then
            return 0
        fi
    done <<< "${values}"
    return 1
}

SINGLE_PENDING="$(list_pending 1 0 0)"
EVOLVE_PENDING="$(list_pending 7 2 2)"
DATABASE_ORDER="$(
    "${PYTHON_BIN}" -m "${BIRD_DOWNLOADER}" \
        --manifest "${EVAL_MANIFEST}" \
        --data-dir "${BIRD_DATA_DIR}" \
        --ordered-databases
)"
if [[ -n "${ONLY_DATABASE}" ]]; then
    if ! is_listed "${ONLY_DATABASE}" "${DATABASE_ORDER}"; then
        echo "ERROR: ${ONLY_DATABASE} is not a published BIRD database." >&2
        exit 2
    fi
    DATABASE_ORDER="${ONLY_DATABASE}"
fi

while IFS= read -r db_id; do
    [[ -z "${db_id}" ]] && continue
    needs_single=false
    needs_evolve=false
    is_listed "${db_id}" "${SINGLE_PENDING}" && needs_single=true
    is_listed "${db_id}" "${EVOLVE_PENDING}" && needs_evolve=true

    if [[ "${needs_single}" == false && "${needs_evolve}" == false ]]; then
        "${PYTHON_BIN}" -m "${BIRD_DOWNLOADER}" \
            --manifest "${EVAL_MANIFEST}" \
            --data-dir "${BIRD_DATA_DIR}" \
            --evict-database "${db_id}" >/dev/null
        continue
    fi

    "${PYTHON_BIN}" -m "${BIRD_DOWNLOADER}" \
        --manifest "${EVAL_MANIFEST}" \
        --data-dir "${BIRD_DATA_DIR}" \
        --database "${db_id}" \
        --max-file-gib 8.0

    if [[ "${needs_single}" == true ]]; then
        run_setting single 1 0 0 "${db_id}"
    fi
    if [[ "${needs_evolve}" == true ]]; then
        run_setting evolve 7 2 2 "${db_id}"
    fi

    "${PYTHON_BIN}" -m "${BIRD_DOWNLOADER}" \
        --manifest "${EVAL_MANIFEST}" \
        --data-dir "${BIRD_DATA_DIR}" \
        --evict-database "${db_id}"
done <<< "${DATABASE_ORDER}"

if [[ -n "${ONLY_DATABASE}" ]]; then
    echo "BIRD-SQL one-database evaluation smoke checkpointed: ${ONLY_DATABASE}"
    echo "Usage ledger: ${LLM_USAGE_LEDGER_PATH}"
    exit 0
fi

if [[ -n "$(list_pending 1 0 0)" || -n "$(list_pending 7 2 2)" ]]; then
    echo "ERROR: BIRD evaluation ended without exact 100/100 coverage." >&2
    exit 2
fi

echo "BIRD-SQL evaluation complete. Usage ledger: ${LLM_USAGE_LEDGER_PATH}"
