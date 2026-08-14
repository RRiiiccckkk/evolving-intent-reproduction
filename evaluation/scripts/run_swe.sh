#!/bin/bash
# SWE-bench Verified evaluation — two scenarios per model, via mini-swe-agent v2.
#
#   single : t=1, p=0, g=0   Fully-specified (single-turn baseline; step_limit=100)
#   evolve : t=7, p=2, g=2   Evolving intent scenario   (step_limit=200, output_suffix=native_tool200)
#
# The evolve scenario gets a larger per-turn step cap and a separate output suffix
# so its files don't overwrite the single-turn baseline.
#
# Usage:
#   ./run_swe.sh <model> [model2 ...]
#
# Environment overrides:
#   NUM_WORKERS         parallel workers          (default 8)
#   REASONING_EFFORT    for reasoning models only (default "medium"; set "" to skip)
#   STEP_LIMIT_DEFAULT  steps/turn for single     (default 100)
#   STEP_LIMIT_HARD     steps/turn for evolve     (default 200)
#   HARD_OUTPUT_SUFFIX  evolve output suffix      (default native_tool200)
#   DATA_PATH           generated dataset pool    (default final_dataset/swe_bench_verified_final.json)
#   TASK_IDS_FILE       eval-subset selection     (default intent_construction/eval_indices/swe_bench_verified_task_ids.json)
#
# Requires Docker (the official SWE-bench harness builds per-instance images).
# Outputs land in evaluation/experiments/{fully_specified,combined_independent}/swe_bench_verified/.
# The runner skips existing outputs, so re-runs are safe.

set -u  # error on unset vars; intentionally NOT set -e so one failing job doesn't abort the rest

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_PATH="${DATA_PATH:-final_dataset/swe_bench_verified_final.json}"
TASK_IDS_FILE="${TASK_IDS_FILE:-intent_construction/eval_indices/swe_bench_verified_task_ids.json}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
STEP_LIMIT_DEFAULT="${STEP_LIMIT_DEFAULT:-100}"
STEP_LIMIT_HARD="${STEP_LIMIT_HARD:-200}"
HARD_OUTPUT_SUFFIX="${HARD_OUTPUT_SUFFIX:-native_tool200}"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("gpt-5.1")

LOG_DIR="evaluation/logs/swe"
mkdir -p "$LOG_DIR"

# scenario specs: <name> <num_turns> <num_revisions> <num_switches> <is_hard>
SCENARIOS=(
  "single 1 0 0 0"
  "evolve 7 2 2 1"
)

run_one() {
    local model=$1 sname=$2 nt=$3 p=$4 g=$5 hard=$6
    local model_safe="${model//\//_}"
    local step_cap="$STEP_LIMIT_DEFAULT"
    local extra_args=()
    local log_tag=""
    [ -n "$REASONING_EFFORT" ] && extra_args+=(--reasoning_effort "$REASONING_EFFORT")
    if [ "$hard" = "1" ]; then
        step_cap="$STEP_LIMIT_HARD"
        extra_args+=(--output_suffix "$HARD_OUTPUT_SUFFIX")
        log_tag="__${HARD_OUTPUT_SUFFIX}"
    fi
    local log="$LOG_DIR/swe__${model_safe}__${sname}${log_tag}.log"
    echo "  [$(date +%H:%M:%S)] $model on $sname (t=$nt p=$p g=$g, step_cap=$step_cap)  ->  $log"
    python -u evaluation/runners/run_swe_mini_agent.py \
        --data_path "$DATA_PATH" \
        --models "$model" \
        --task_ids_file "$TASK_IDS_FILE" \
        --num_workers "$NUM_WORKERS" \
        --use_tool_calling \
        --num_turns "$nt" \
        --num_revisions "$p" \
        --num_switches "$g" \
        --step_limit_per_turn "$step_cap" \
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
echo "[$(date +'%F %T')] SWE-bench Verified eval (workers=$NUM_WORKERS, effort=${REASONING_EFFORT:-none})"
echo "  step cap: single=$STEP_LIMIT_DEFAULT, evolve=$STEP_LIMIT_HARD (suffix=$HARD_OUTPUT_SUFFIX)"
echo "  data=$DATA_PATH  task_ids=$TASK_IDS_FILE"
echo "  models: ${MODELS[*]}"
echo "============================================================"

if [ -z "${OPENAI_API_KEY:-}" ] && { [ -z "${AZURE_OPENAI_API_KEY:-}" ] || [ -z "${AZURE_OPENAI_ENDPOINT:-}" ]; }; then
    echo "ERROR: No LLM credentials; set OPENAI_API_KEY or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT."
    exit 1
fi

START_TS=$(date +%s)
for model in "${MODELS[@]}"; do
    for spec in "${SCENARIOS[@]}"; do
        read -r sname nt p g hard <<< "$spec"
        run_one "$model" "$sname" "$nt" "$p" "$g" "$hard"
    done
done

elapsed=$(( $(date +%s) - START_TS ))
printf "[%s] SWE-bench eval complete (wall %dh%dm)\n" \
    "$(date +%H:%M:%S)" $((elapsed/3600)) $(( (elapsed%3600)/60 ))
