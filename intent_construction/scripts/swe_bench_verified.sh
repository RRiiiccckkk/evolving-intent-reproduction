#!/bin/bash
# Fixed SWE-bench Verified construction: local 500-row parquet -> published 50.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export LLM_REASONING_EFFORT="medium"
export LLM_DISABLE_OUTPUT_LIMITS="1"
export LLM_REQUIRE_USAGE_ACCOUNTING="1"

if [ -n "${LLM_DEFAULT_MAX_OUTPUT_TOKENS:-}" ] || \
   [ -n "${LLM_BUDGET_DEFAULT_MAX_OUTPUT_TOKENS:-}" ] || \
   [ -n "${LLM_COST_HARD_CAP_USD:-}" ]; then
    echo "ERROR: SWE construction forbids output-token limits and cost gates." >&2
    exit 2
fi

"${PYTHON_BIN:-python}" -m \
  intent_construction.intent_extraction.dataset_impl.swe_bench_verified.run_published_construction \
  "$@"
