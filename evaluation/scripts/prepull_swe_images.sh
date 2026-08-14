#!/bin/bash
# Sequential docker pull for the SWE-bench instance images in the eval subset.
# Avoids the 120s timeout in mini-agent's `docker run` by pre-pulling.
#
# Usage:
#   bash evaluation/scripts/prepull_swe_images.sh [TASK_IDS_FILE]
#
# Defaults to the canonical eval subset (intent_construction/eval_indices/
# swe_bench_verified_task_ids.json). Idempotent: skips images already present.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TASK_IDS_FILE="${1:-intent_construction/eval_indices/swe_bench_verified_task_ids.json}"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

TASK_IDS_FILE="$TASK_IDS_FILE" LIST="$LIST" python << 'PY'
import json, os
task_ids_file = os.environ["TASK_IDS_FILE"]
data = json.load(open(task_ids_file))
task_ids = data.get("task_ids", data) if isinstance(data, dict) else data
PREFIX = "extracted-swe_bench_verified-test-"
imgs = []
for tid in task_ids:
    assert tid.startswith(PREFIX), f"Unexpected task_id format: {tid}"
    iid = tid[len(PREFIX):]
    img_id = iid.replace("__", "_1776_")
    imgs.append(f"docker.io/swebench/sweb.eval.x86_64.{img_id}:latest")
with open(os.environ["LIST"], "w") as f:
    f.write("\n".join(imgs) + "\n")
print(f"Wrote {len(imgs)} images from {task_ids_file}")
PY

LOCAL=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep swebench | sed 's|^|docker.io/|')
N_TOTAL=$(wc -l < "$LIST")
N=0
SUCCESS=0
FAIL=0
SKIP=0
while IFS= read -r img; do
    N=$((N+1))
    if echo "$LOCAL" | grep -qx "$img"; then
        SKIP=$((SKIP+1))
        echo "[$N/$N_TOTAL] SKIP (local): $img"
        continue
    fi
    echo "[$N/$N_TOTAL] PULL: $img"
    if docker pull "$img" 2>&1 | tail -3; then
        SUCCESS=$((SUCCESS+1))
    else
        FAIL=$((FAIL+1))
        echo "  !!! FAILED"
    fi
done < "$LIST"

echo ""
echo "=== Summary ==="
echo "  Total:   $N_TOTAL"
echo "  Skipped: $SKIP (already local)"
echo "  Pulled:  $SUCCESS"
echo "  Failed:  $FAIL"
