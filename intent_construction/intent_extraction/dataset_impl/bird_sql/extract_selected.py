"""
Extract function+arguments for a pre-selected subset of BIRD-SQL samples.

Wraps ``BirdSqlExtractor`` so the BIRD complex pipeline does NOT have to
run extraction over all 10k+ samples just to cover ~120 selected ones.

Input: a JSON file in the shape of ``select_complex.py``'s output (each
sample carries ``gold_sql``, ``db_id``, ``source``, ``index``, ``question``,
``evidence``).
Output: extracted JSON consumable by ``generate_counterfactuals_sql.py``.

Usage:
    python extract_selected.py \
        --input  data/selected_complex_v1.json \
        --output ../../output/bird_sql/extracted.json \
        --model  gpt-5.1 \
        --num_workers 8
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from intent_construction.intent_extraction.dataset_impl.bird_sql.extractor import BirdSqlExtractor


def _process_one(extractor: BirdSqlExtractor, sample: dict) -> dict | None:
    try:
        extracted = extractor.extract(sample)
    except Exception as e:
        sid = sample.get("index", "unknown")
        print(f"  ✗ extract failed for {sid}: {e}")
        return None
    if extracted is None:
        return None
    return extractor.build_output(sample, extracted)


def main():
    parser = argparse.ArgumentParser(description="Extract BIRD-SQL function+arguments for a pre-selected subset.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    with open(args.input) as f:
        samples = json.load(f)
    if args.num_samples:
        samples = samples[: args.num_samples]
    print(f"Extracting {len(samples)} samples (model={args.model}, workers={args.num_workers})")

    extractor = BirdSqlExtractor(model=args.model)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(_process_one, extractor, s): s for s in samples}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
            r = fut.result()
            if r is not None:
                results.append(r)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Extracted {len(results)}/{len(samples)} → {args.output}")


if __name__ == "__main__":
    main()
