#!/bin/bash
# BIRD-SQL evaluation — two scenarios per model.
#
#   single : t=1, p=0, g=0   Fully-specified (single-turn baseline)
#   evolve : t=7, p=2, g=2   Evolving intent scenario
#
# Usage:
#   ./run_bird.sh <model> [model2 ...]
#
# Environment overrides:
#   NUM_WORKERS        parallel workers           (default 8)
#   REASONING_EFFORT   for reasoning models only  (default "medium"; set "" to skip)
#   NATURALIZER_MODEL  online user-turn naturalizer  (BIRD-SQL has no online naturalizer; turns are always rule-based)
#   DATA_PATH          generated dataset pool     (default final_dataset/bird_sql_final.json)
#   TASK_IDS_FILE      eval-subset selection      (default intent_construction/eval_indices/bird_sql_task_ids.json)
#   DATASET_NAME       experiments/ output label  (default bird_sql_n100; must start with "bird_sql" to enable the LLM judge)
#
# Outputs land in evaluation/experiments/{fully_specified,combined_independent}/${DATASET_NAME}/.
# The runner skips a (model, scenario) whose output already exists, so re-runs are safe.

set -u  # error on unset vars; intentionally NOT set -e so one failing job doesn't abort the rest

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-final_dataset/bird_sql_final.json}"
DATASET_NAME="${DATASET_NAME:-bird_sql_n100}"
TASK_IDS_FILE="${TASK_IDS_FILE:-intent_construction/eval_indices/bird_sql_task_ids.json}"
NATURALIZER_MODEL="${NATURALIZER_MODEL:-}"  # SQL has no online naturalizer; rule-based turns only
NUM_WORKERS="${NUM_WORKERS:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("gpt-5.1")

LOG_DIR="evaluation/logs/bird"
mkdir -p "$LOG_DIR"

SCENARIOS=(
  "single 1 0 0"
  "evolve 7 2 2"
)

run_one() {
    local model=$1 sname=$2 nt=$3 p=$4 g=$5
    local model_safe="${model//\//_}"
    local log="$LOG_DIR/bird__${model_safe}__${sname}.log"
    local extra_args=()
    [ -n "$REASONING_EFFORT" ] && extra_args+=(--reasoning_effort "$REASONING_EFFORT")
    [ -n "$NATURALIZER_MODEL" ] && extra_args+=(--naturalizer_model "$NATURALIZER_MODEL")
    echo "  [$(date +%H:%M:%S)] $model on $sname (t=$nt p=$p g=$g, nat=${NATURALIZER_MODEL:-rule-based})  ->  $log"
    python -u evaluation/runners/run_experiment.py \
        --data_path "$DATA_PATH" \
        --dataset_name "$DATASET_NAME" \
        --models "$model" \
        --task_ids_file "$TASK_IDS_FILE" \
        --num_workers "$NUM_WORKERS" \
        --num_turns "$nt" \
        --num_revisions "$p" \
        --num_switches "$g" \
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
echo "[$(date +'%F %T')] BIRD-SQL eval (workers=$NUM_WORKERS, effort=${REASONING_EFFORT:-none}, nat=${NATURALIZER_MODEL:-rule-based})"
echo "  data=$DATA_PATH  task_ids=$TASK_IDS_FILE  dataset=$DATASET_NAME"
echo "  models: ${MODELS[*]}"
echo "============================================================"

if [ -z "${OPENAI_API_KEY:-}" ] && { [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; }; then
    echo "ERROR: No LLM credentials; set OPENAI_API_KEY or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT."
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
printf "[%s] BIRD-SQL eval complete (wall %dh%dm)\n" \
    "$(date +%H:%M:%S)" $((elapsed/3600)) $(( (elapsed%3600)/60 ))
