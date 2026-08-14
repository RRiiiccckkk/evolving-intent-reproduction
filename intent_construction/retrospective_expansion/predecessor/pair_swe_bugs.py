"""
SWE-bench Verified — Stage 3: Real-bug pairing.

For each target sample (G3) in the curated 251-sample pool, find another
real SWE-bench Verified instance from the same repo / area / folder to use
as a predecessor function (G2). The predecessor is *just discussed* in the
conversation — never patched, never predecessor — so the only thing we
borrow from it is its function + source arguments.

Matching cascade (most specific -> broadest):
  1. Same repo + same area (first 2 path segments of affected_files[0])
  2. Same repo + same top-level folder (first 1 segment)
  3. Same repo (any file)
  4. Drop (no function-switch scenario for this sample)

Among the candidates at the chosen level, a single G2 is selected by
seeded random.

Predecessor entries follow the existing ``predecessor_functions`` schema used by
other domains' predecessor-inference output:
  {
    "predecessor_function": <G2 source function text>,
    "counterfactual_arguments": [<G2's source arguments, cid offset by +1000>],
    "is_predecessor": false,
    "transition_type": "real_bug_pair",
    "taxonomy_type": null,
    "transition_reason": "User noticed another bug in the same area before settling on the target",
    "entity_sought": "code patch",
    "_pair_metadata": {
        "match_level": "area" | "folder" | "repo",
        "paired_task_id": ...,
        "paired_repo": ...,
        "cid_offset": 1000,
    }
  }

CID offset (default 1000) prevents collisions with the target's
argument_ids when the scheduler walks both phases.

Usage:
    python pair_swe_bugs.py \
        --target_input ../counterfactual/output/swe_bench_verified/argument_counterfactual_n251.json \
        --pool_input  ../../intent_extraction/output/swe_bench_verified/extracted_test.json \
        --output      output/swe_bench_verified/paired_n251.json \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

CID_OFFSET = 1000  # added to predecessor argument_ids to avoid target collisions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_segments(sample: dict[str, Any]) -> list[str]:
    """Return path segments of the sample's first affected file."""
    files = sample.get("swe_bench_metadata", {}).get("affected_files", [])
    if not files:
        return []
    return files[0].split("/")


def _repo(sample: dict[str, Any]) -> str:
    return sample.get("swe_bench_metadata", {}).get("repo", "") or ""


def _key_at_level(sample: dict[str, Any], level: int) -> tuple[str, str]:
    """Return a hashable matching key.

    level=2 -> (repo, file_segments[:2] joined)   # area
    level=1 -> (repo, file_segments[:1] joined)   # top folder
    level=0 -> (repo, "")                         # repo only
    """
    repo = _repo(sample)
    if level <= 0:
        return (repo, "")
    segs = _file_segments(sample)
    if not segs:
        return (repo, "")
    return (repo, "/".join(segs[:level]))


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def build_pair_index(
    pool: list[dict[str, Any]],
    levels: tuple[int, ...] = (2, 1, 0),
) -> dict[int, dict[tuple, list[dict[str, Any]]]]:
    """Return {level -> {key -> list of pool samples}} for fast lookup."""
    index: dict[int, dict[tuple, list]] = {lvl: defaultdict(list) for lvl in levels}
    for s in pool:
        for lvl in levels:
            index[lvl][_key_at_level(s, lvl)].append(s)
    return index


def find_pair(
    target: dict[str, Any],
    index: dict[int, dict[tuple, list[dict[str, Any]]]],
    rng: random.Random,
    levels: tuple[int, ...] = (2, 1, 0),
) -> tuple[dict[str, Any] | None, str | None]:
    """Find one paired sample, walking levels from most specific to broadest.

    Returns (paired_sample, match_level_label) or (None, None).
    """
    target_id = target.get("task_id")
    level_labels = {2: "area", 1: "folder", 0: "repo"}
    for lvl in levels:
        key = _key_at_level(target, lvl)
        candidates = [s for s in index[lvl][key] if s.get("task_id") != target_id]
        if candidates:
            return rng.choice(candidates), level_labels[lvl]
    return None, None


def make_predecessor_entry(
    paired: dict[str, Any],
    cid_offset: int = CID_OFFSET,
) -> dict[str, Any]:
    """Build a predecessor_functions[i] entry from a paired SWE-bench instance."""
    pg_conds = []
    for c in paired.get("arguments", []) or []:
        new_c = dict(c)
        cid = c.get("argument_id")
        if cid is not None:
            new_c["argument_id"] = cid + cid_offset
        new_c["is_shared"] = False
        # Drop counterfactual_arguments: predecessor is not predecessor
        new_c.pop("counterfactual_arguments", None)
        pg_conds.append(new_c)
    return {
        "predecessor_function": paired.get("function", ""),
        "counterfactual_arguments": pg_conds,
        "is_predecessor": False,
        "transition_type": "real_bug_pair",
        "taxonomy_type": None,
        "transition_reason": (
            "User noticed another bug in the same area before settling on "
            "the target."
        ),
        "entity_sought": "code patch",
    }


def attach_pair(
    target: dict[str, Any],
    paired: dict[str, Any],
    match_level: str,
    cid_offset: int = CID_OFFSET,
) -> dict[str, Any]:
    """Return a new sample dict (deep-copied) with the predecessor attached."""
    out = deepcopy(target)
    entry = make_predecessor_entry(paired, cid_offset=cid_offset)
    entry["_pair_metadata"] = {
        "match_level": match_level,
        "paired_task_id": paired.get("task_id"),
        "paired_repo": _repo(paired),
        "paired_files": paired.get("swe_bench_metadata", {}).get("affected_files", []),
        "cid_offset": cid_offset,
    }
    out.setdefault("predecessor_functions", []).append(entry)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified real-bug pairing")
    parser.add_argument(
        "--target_input", required=True,
        help="Path to G3 pool JSON (e.g., argument_counterfactual_n251.json)",
    )
    parser.add_argument(
        "--pool_input", required=True,
        help="Path to G2 candidate pool JSON (e.g., extracted_test.json)",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cid_offset", type=int, default=CID_OFFSET,
        help="Offset added to predecessor argument_ids (default 1000)",
    )
    args = parser.parse_args()

    targets = json.load(open(args.target_input))
    pool = json.load(open(args.pool_input))
    print(f"Targets: {len(targets)}; pool: {len(pool)}")

    rng = random.Random(args.seed)
    index = build_pair_index(pool)

    paired_samples: list[dict[str, Any]] = []
    level_counts: Counter = Counter()
    drop_count = 0

    for target in targets:
        match, level = find_pair(target, index, rng)
        if match is None:
            drop_count += 1
            paired_samples.append(deepcopy(target))  # keep but no predecessor
            continue
        level_counts[level] += 1
        paired_samples.append(attach_pair(target, match, level, args.cid_offset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(paired_samples, f, indent=2)

    print(f"\nMatch level distribution:")
    for lvl in ("area", "folder", "repo"):
        print(f"  {lvl:<8} {level_counts[lvl]}")
    print(f"  dropped (no pair) {drop_count}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
