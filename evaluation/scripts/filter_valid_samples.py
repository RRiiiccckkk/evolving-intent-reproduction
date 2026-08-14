#!/usr/bin/env python3
"""Filter valid samples from predecessor generation output(s).

A sample is "valid" when it passes BOTH:
  1. Quality checks  – verification_passed AND independence_passed
  2. Simulation viability   – survives all 23 eval configs (enough arguments/functions)

Supports multiple input files (e.g. initial run + retries); later files
override earlier ones so retry results take precedence.

Usage:
    python evaluation/scripts/filter_valid_samples.py \
        --data_paths intent_construction/retrospective_expansion/predecessor/output/browsecomp_plus/predecessor.json \
                     intent_construction/retrospective_expansion/predecessor/output/browsecomp_plus/predecessor_retry.json

Outputs:
    - Prints per-config and intersection counts
    - Saves filtered task_ids JSON to the path given by --output
"""

import argparse
import json
from pathlib import Path

from situated_simulation.user_simulation import EvolvingIntent, SUPPORTED_DOMAINS

# All 23 configs from PLAN_CONFIGS (excluding single-turn for LLM user)
# Format: (num_switches, num_revisions, num_turns)
PLAN_CONFIGS = [
    # under_specified: t=2,4,8
    (0, 0, 2), (0, 0, 4), (0, 0, 8),
    # argument_revision: (t,p) = (4,1),(4,2),(8,1),(8,2),(8,4)
    (0, 1, 4), (0, 2, 4), (0, 1, 8), (0, 2, 8), (0, 4, 8),
    # function_switch: t=4,8 x g=1,2,3
    (1, 0, 4), (2, 0, 4), (3, 0, 4),
    (1, 0, 8), (2, 0, 8), (3, 0, 8),
    # combined: g=1..3 x p=1..3, t=1+g+p
    (1, 1, 3), (1, 2, 4), (1, 3, 5),
    (2, 1, 4), (2, 2, 5), (2, 3, 6),
    (3, 1, 5), (3, 2, 6), (3, 3, 7),
]


def merge_data_files(data_paths: list[str]) -> list[dict]:
    """Merge multiple JSON files; later files override earlier by task_id."""
    merged: dict[str, dict] = {}
    for path in data_paths:
        with open(path) as f:
            for sample in json.load(f):
                merged[sample["task_id"]] = sample
    return list(merged.values())


def filter_quality(samples: list[dict]) -> set[str]:
    """Return task_ids that pass both verification and independence checks."""
    return {
        s["task_id"] for s in samples
        if s.get("verification_passed") and s.get("independence_passed")
    }


def get_valid_task_ids(data_path: str, config: tuple[int, int, int], domain: str = "search") -> set[str]:
    """Get task_ids that survive simulation filtering for a given config."""
    g, p, t = config
    sim = EvolvingIntent(
        data_path=data_path,
        mode="eval",
        domain=domain,
        num_turns=t,
        num_revisions=p,
        num_switches=g,
        ordering="interleaved",
    )
    return {sample.task_id for sample in sim}


def main():
    parser = argparse.ArgumentParser(description="Filter valid samples across all eval configs")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                        help="Path(s) to predecessor JSON files (later overrides earlier)")
    parser.add_argument("--domain", type=str, default="search", choices=SUPPORTED_DOMAINS)
    parser.add_argument("--output", type=str, default=None, help="Output path for filtered task_ids JSON")
    parser.add_argument("--save_merged", type=str, default=None,
                        help="If set, save the merged dataset to this path")
    args = parser.parse_args()

    data_paths = [str(Path(p).resolve()) for p in args.data_paths]

    # Merge input files (later overrides earlier)
    merged_data = merge_data_files(data_paths)
    total = len(merged_data)
    all_task_ids = {s["task_id"] for s in merged_data}
    print(f"Merged {len(data_paths)} file(s) → {total} unique samples\n")

    # Requirement 1: Quality checks (verification + independence)
    quality_passed = filter_quality(merged_data)
    print(f"Quality checks (verification ✓ + independence ✓): {len(quality_passed)} / {total}")

    # Save merged data to a temp file for simulator loading
    if args.save_merged:
        merged_path = str(Path(args.save_merged).resolve())
    else:
        merged_path = str(Path(data_paths[0]).parent / "_merged_tmp.json")
    with open(merged_path, "w") as f:
        json.dump(merged_data, f)

    # Requirement 2: Simulation config viability across all 23 configs
    print(f"\nChecking {len(PLAN_CONFIGS)} eval configs:")
    valid_per_config: dict[str, set[str]] = {}
    for g, p, t in PLAN_CONFIGS:
        config_name = f"t{t}_g{g}_p{p}"
        valid_ids = get_valid_task_ids(merged_path, (g, p, t), args.domain)
        valid_per_config[config_name] = valid_ids
        print(f"  {config_name:20s} → {len(valid_ids):4d} / {total}")

    config_survived = set.intersection(*valid_per_config.values()) & all_task_ids
    print(f"\nSurvived all configs: {len(config_survived)} / {total}")

    # Final: intersection of BOTH requirements
    valid_all = quality_passed & config_survived
    print(f"\n{'='*50}")
    print(f"VALID (quality ✓ + all configs ✓): {len(valid_all)} / {total}")
    print(f"{'='*50}")

    # Bottleneck configs
    print("\nBottleneck configs (fewest valid samples):")
    sorted_configs = sorted(valid_per_config.items(), key=lambda x: len(x[1]))
    for name, ids in sorted_configs[:5]:
        print(f"  {name:20s} → {len(ids):4d}")

    # Clean up temp file if we created one
    if not args.save_merged:
        Path(merged_path).unlink(missing_ok=True)

    # Save filtered task_ids
    if args.output:
        output_path = args.output
    else:
        dataset_name = Path(data_paths[0]).stem
        output_path = str(Path(__file__).resolve().parent.parent / "data" / f"{dataset_name}_filtered_task_ids.json")

    sorted_ids = sorted(valid_all)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sorted_ids, f, indent=2)
    print(f"\nSaved {len(sorted_ids)} valid task_ids → {output_path}")


if __name__ == "__main__":
    main()
