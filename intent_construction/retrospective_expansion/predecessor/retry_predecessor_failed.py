#!/usr/bin/env python3
"""
Retry failed predecessor inference samples with a stronger model.

Compares stage 2 input vs stage 3 output to find missing task_ids,
filters the input to those samples, runs the predecessor pipeline,
and merges results into the existing output.

Usage:
    # Retry failed samples with a stronger model
    python retry_predecessor_failed.py \
        --input ../counterfactual/output/gsm8k/argument_counterfactual.json \
        --output output/gsm8k/predecessor.json \
        --model gpt-5.1 \
        --dataset_type gsm8k \
        --num_predecessors 3 \
        --parallel 100
"""

import json
import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Retry failed predecessor inference samples with a stronger model"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Stage 2 input (all samples)")
    parser.add_argument("--output", type=str, required=True,
                        help="Stage 3 output (existing results to merge into)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use for retries (e.g., gpt-5.1)")
    parser.add_argument("--dataset_type", type=str, default="gsm8k")
    parser.add_argument("--num_predecessors", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=100)
    parser.add_argument("--fallback_model", type=str, default=None)
    parser.add_argument("--no_independence_test", action="store_true")
    parser.add_argument("--independence_runs", type=int, default=3)
    parser.add_argument("--max_independence_retries", type=int, default=2)
    parser.add_argument("--chain_types", nargs="+", default=None)
    args = parser.parse_args()

    # Load input (all samples from stage 2)
    with open(args.input) as f:
        all_samples = json.load(f)
    all_ids = {s.get("task_id") for s in all_samples}
    print(f"Stage 2 input: {len(all_samples)} samples")

    # Load existing output
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing_results = json.load(f)
        done_ids = {s.get("task_id") for s in existing_results}
    else:
        existing_results = []
        done_ids = set()
    print(f"Stage 3 output: {len(existing_results)} samples already done")

    # Find missing
    missing_ids = all_ids - done_ids
    print(f"Missing (to retry): {len(missing_ids)} samples")

    if not missing_ids:
        print("Nothing to retry!")
        return

    # Filter input to missing samples
    retry_samples = [s for s in all_samples if s.get("task_id") in missing_ids]

    # Write to temp file
    retry_input = args.output.replace(".json", f"_retry_input_{args.model}.json")
    retry_output = args.output.replace(".json", f"_retry_output_{args.model}.json")
    with open(retry_input, "w") as f:
        json.dump(retry_samples, f, indent=2)
    print(f"Wrote {len(retry_samples)} retry samples to: {retry_input}")

    # Build command
    script_dir = Path(__file__).parent
    cmd = [
        sys.executable, str(script_dir / "generate_predecessors.py"),
        "--input", retry_input,
        "--output", retry_output,
        "--model", args.model,
        "--dataset_type", args.dataset_type,
        "--num_predecessors", str(args.num_predecessors),
        "--parallel", str(args.parallel),
        "--checkpoint_interval", "50",
        "--independence_runs", str(args.independence_runs),
        "--max_independence_retries", str(args.max_independence_retries),
    ]
    if args.fallback_model:
        cmd += ["--fallback_model", args.fallback_model]
    if args.no_independence_test:
        cmd += ["--no_independence_test"]
    if args.chain_types:
        cmd += ["--chain_types"] + args.chain_types

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n⚠️  Retry run exited with code {result.returncode}")
        print("Check retry output for partial results.")

    # Merge results
    if os.path.exists(retry_output):
        with open(retry_output) as f:
            retry_results = json.load(f)
        print(f"\nRetry produced: {len(retry_results)} results")

        merged = existing_results + retry_results
        # Deduplicate by task_id (keep latest)
        seen = {}
        for s in merged:
            seen[s.get("task_id")] = s
        merged = list(seen.values())

        with open(args.output, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged output: {len(merged)} total samples → {args.output}")

        # Stats
        still_missing = all_ids - {s.get("task_id") for s in merged}
        print(f"Still missing: {len(still_missing)} samples")
    else:
        print("No retry output produced.")

    # Cleanup temp input
    if os.path.exists(retry_input):
        os.remove(retry_input)
        print(f"Cleaned up: {retry_input}")


if __name__ == "__main__":
    main()
