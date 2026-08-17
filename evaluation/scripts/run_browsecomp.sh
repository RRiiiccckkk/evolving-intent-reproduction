#!/bin/bash
# BrowseComp+ evaluation — two scenarios per model.
#
#   single : t=1, p=0, g=0   Fully-specified (single-turn baseline)
#   evolve : t=7, p=2, g=2   Evolving intent scenario
#
# Usage:
#   ./run_browsecomp.sh [kimi-k2.6]
#
# Environment overrides:
#   NUM_WORKERS        parallel workers           (default 8)
#   RETRIEVER_URL      deployed Modal endpoint (required)
#   NUM_SAMPLES        belt-and-suspenders cap    (default 100)
#   DATA_PATH          generated dataset pool     (default final_dataset/browsecomp_plus_final.json)
#   TASK_IDS_FILE      eval-subset selection      (default intent_construction/eval_indices/browsecomp_plus_task_ids.json)
#   DATASET_NAME       experiments/ output label  (default browsecomp_plus_n100)
#
# Requires a deployed ``reproduction.browsecomp_modal`` endpoint. The host does
# not load the embedding model, corpus, or FAISS index.
#
# NOTE: each task is checkpointed atomically. Re-runs resume only missing or
# incomplete task IDs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-final_dataset/browsecomp_plus_final.json}"
DATASET_NAME="${DATASET_NAME:-browsecomp_plus_n100}"
TASK_IDS_FILE="${TASK_IDS_FILE:-intent_construction/eval_indices/browsecomp_plus_task_ids.json}"
PLAN_A_MODEL="kimi-k2.6"
NATURALIZER_MODEL="$PLAN_A_MODEL"
JUDGE_MODEL="$PLAN_A_MODEL"
RETRIEVER_URL="${RETRIEVER_URL:-}"
RETRIEVER_REVISION="browsecomp-retriever-e10361ce3ec95089dd79d3058cb33b73cb3b1da043b239c3525a6c7a7abd5ece"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
export LLM_DISABLE_OUTPUT_LIMITS=1
export LLM_LOCKED_MODEL="$PLAN_A_MODEL"
export LLM_REQUIRE_USAGE_ACCOUNTING=1
export LLM_REASONING_EFFORT="medium"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("$PLAN_A_MODEL")
if [ ${#MODELS[@]} -ne 1 ] || [ "${MODELS[0]}" != "$PLAN_A_MODEL" ]; then
    echo "ERROR: Plan A requires exactly one model: $PLAN_A_MODEL" >&2
    exit 2
fi
if [ "$NUM_SAMPLES" != "100" ]; then
    echo "ERROR: Plan A requires the fixed 100-task set." >&2
    exit 2
fi
if [ -z "$RETRIEVER_URL" ]; then
    echo "ERROR: RETRIEVER_URL must point to the deployed Modal retriever." >&2
    exit 2
fi
if [ -z "${LLM_USAGE_LEDGER_PATH:-}" ]; then
    echo "ERROR: LLM_USAGE_LEDGER_PATH is required." >&2
    exit 2
fi
if [ -n "${LLM_COST_HARD_CAP_USD:-}" ]; then
    echo "ERROR: Plan A records usage only; unset LLM_COST_HARD_CAP_USD." >&2
    exit 2
fi
if [ "$REASONING_EFFORT" != "medium" ]; then
    echo "ERROR: Plan A requires REASONING_EFFORT=medium." >&2
    exit 2
fi

LOG_DIR="evaluation/logs/browsecomp"
mkdir -p "$LOG_DIR"

SCENARIOS=(
  "single 1 0 0"
  "evolve 7 2 2"
)

run_one() {
    local model=$1 sname=$2 nt=$3 p=$4 g=$5
    local model_safe="${model//\//_}"
    local log="$LOG_DIR/browse__${model_safe}__${sname}.log"
    local extra_args=()
    [ -n "$REASONING_EFFORT" ] && extra_args+=(--reasoning_effort "$REASONING_EFFORT")
    [ -n "$NATURALIZER_MODEL" ] && extra_args+=(--naturalizer_model "$NATURALIZER_MODEL")
    echo "  [$(date +%H:%M:%S)] $model on $sname (t=$nt p=$p g=$g, Modal retriever)  ->  $log"
    python -u evaluation/runners/run_browsecomp_experiment.py \
        --data_path "$DATA_PATH" \
        --dataset_name "$DATASET_NAME" \
        --models "$model" \
        --task_ids_file "$TASK_IDS_FILE" \
        --num_samples "$NUM_SAMPLES" \
        --num_workers "$NUM_WORKERS" \
        --retriever_url "$RETRIEVER_URL" \
        --retriever_revision "$RETRIEVER_REVISION" \
        --num_turns "$nt" \
        --num_revisions "$p" \
        --num_switches "$g" \
        --max_search_iterations 51 \
        --max_tool_calls 50 \
        --judge_model "$JUDGE_MODEL" \
        --locked_model "$PLAN_A_MODEL" \
        --expected_samples 100 \
        --force_final_answer \
        --fail_fast \
        --require_usage_ledger \
        --usage_only_accounting \
        "${extra_args[@]}" \
        > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  [$(date +%H:%M:%S)] OK  $model on $sname"
    else
        echo "  [$(date +%H:%M:%S)] FAIL $model on $sname (rc=$rc); see $log"
    fi
    return $rc
}

echo "============================================================"
echo "[$(date +'%F %T')] BrowseComp+ Plan A (workers=$NUM_WORKERS, model=$PLAN_A_MODEL, Modal retriever)"
echo "  data=$DATA_PATH  task_ids=$TASK_IDS_FILE  dataset=$DATASET_NAME"
echo "  models: ${MODELS[*]}"
echo "============================================================"

if [ -z "${OPENAI_API_KEY:-}" ] \
   && { [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; } \
   && { [ -z "${LLM_API_KEY:-}" ] || [ -z "${LLM_BASE_URL:-}" ]; }; then
    echo "ERROR: No OpenAI, Azure, or OpenAI-compatible credentials are configured."
    exit 1
fi

START_TS=$(date +%s)
for model in "${MODELS[@]}"; do
    for spec in "${SCENARIOS[@]}"; do
        read -r sname nt p g <<< "$spec"
        run_one "$model" "$sname" "$nt" "$p" "$g"
    done
done

elapsed=$(( $(date +%s) - START_TS ))
printf "[%s] BrowseComp+ eval complete (wall %dh%dm)\n" \
    "$(date +%H:%M:%S)" $((elapsed/3600)) $(( (elapsed%3600)/60 ))
