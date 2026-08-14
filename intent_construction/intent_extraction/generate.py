"""
Unified CLI for function + argument extraction across different datasets.

Usage:
    # Sequential processing
    python generate.py --dataset gsm8k --num_samples 10
    
    # Batch (parallel) processing
    python generate.py --dataset gsm8k --num_samples 10 --batch --batch_size 5
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from intent_construction.intent_extraction.registry import get_extractor


def load_gsm8k_samples(split: str, num_samples: int = None, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    """Load GSM8k dataset samples."""
    dataset = load_dataset("gsm8k", "main")
    samples = list(dataset[split])
    
    # Add IDs and metadata
    for i, sample in enumerate(samples):
        sample["id"] = i
        sample["task"] = "math"
        sample["split"] = split
    
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    
    if num_samples:
        samples = samples[:num_samples]
    
    return samples


def load_browsecomp_plus_samples(split: str, num_samples: int = None, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    """Load BrowseComp-Plus dataset from pre-downloaded decrypted JSONL."""
    raw_file = Path(__file__).parent / "output" / "browsecomp_plus" / "raw_data.jsonl"
    
    if not raw_file.exists():
        raise FileNotFoundError(
            f"BrowseComp-Plus data not found: {raw_file}\n"
            "Download and decrypt first. See intent_extraction/dataset_impl/browsecomp_plus/README.md"
        )
    
    print(f"Loading BrowseComp-Plus from {raw_file}...")
    samples = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sample = {
                "id": item["query_id"],
                "query_id": item["query_id"],
                "question": item["query"],
                "answer": item["answer"],
                "task": "search",
                "split": "test",
                "evidence_docs": item.get("evidence_docs", []),
                "gold_docs": item.get("gold_docs", []),
            }
            samples.append(sample)
    
    print(f"Loaded {len(samples)} samples")
    
    if shuffle:
        random.seed(seed)
        random.shuffle(samples)
    
    if num_samples:
        samples = samples[:num_samples]
    
    return samples


def load_swe_bench_verified_samples(split: str, num_samples: int = None, shuffle: bool = False, seed: int = 42) -> List[Dict[str, Any]]:
    """Load SWE-bench Verified dataset from HuggingFace."""
    print(f"Loading SWE-bench Verified dataset (split: {split})...")
    dataset = load_dataset("SWE-bench/SWE-bench_Verified", split=split)

    samples = []
    for item in dataset:
        sample = {
            "id": item["instance_id"],
            "instance_id": item["instance_id"],
            "question": item["problem_statement"],
            "answer": "",  # SWE-bench uses test-suite verification
            "task": "swe_bench",
            "split": split,
            # SWE-bench specific fields
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "patch": item["patch"],
            "test_patch": item.get("test_patch", ""),
            "version": item.get("version", ""),
            "difficulty": item.get("difficulty", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", "[]"),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", "[]"),
            "hints_text": item.get("hints_text", ""),
            "environment_setup_commit": item.get("environment_setup_commit", ""),
        }
        samples.append(sample)

    print(f"Loaded {len(samples)} samples")

    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    if num_samples:
        samples = samples[:num_samples]

    return samples


# Dataset loaders registry
DATASET_LOADERS = {
    "gsm8k": load_gsm8k_samples,
    "browsecomp_plus": load_browsecomp_plus_samples,
    "swe_bench_verified": load_swe_bench_verified_samples,
}


def main():
    parser = argparse.ArgumentParser(description="Extract function + arguments from various datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_LOADERS.keys()),
        help=(
            f"Dataset to process. Available: {sorted(DATASET_LOADERS.keys())}. "
            "BIRD-SQL is not loaded here; use "
            "dataset_impl/bird_sql/extract_selected.py instead."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model for extraction (default: gpt-5.1)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: output/{dataset}_extracted.json)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--num_arguments",
        type=int,
        default=4,
        help="[DEPRECATED] Target number of arguments (actual count is determined by problem complexity, typically 2-6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--verif_model",
        type=str,
        default="gpt-5.1",
        help="Model for verification (default: gpt-5.1)"
    )
    parser.add_argument(
        "--disable_model_verification",
        action="store_true",
        help="Disable Step 4 model performance verification"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch (parallel) processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of parallel workers for batch processing (default: 5)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for verification failures (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Set default output path: output/{dataset}/extracted_{split}.json
    if args.output is None:
        args.output = f"output/{args.dataset}/extracted_{args.split}.json"
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset (split: {args.split})...")
    if args.dataset not in DATASET_LOADERS:
        raise ValueError(f"No loader for dataset: {args.dataset}")
    
    samples = DATASET_LOADERS[args.dataset](
        split=args.split,
        num_samples=args.num_samples,
        shuffle=args.shuffle,
        seed=args.seed
    )
    print(f"Processing {len(samples)} samples...")
    
    # Initialize extractor
    extractor = get_extractor(
        args.dataset,
        model=args.model,
        num_arguments=args.num_arguments,
        verif_model=args.verif_model,
        enable_model_verification=not args.disable_model_verification,
        max_verification_attempts=args.max_retries
    )
    
    # Process samples with checkpointing
    checkpoint_dir = output_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    all_results = []
    checkpoint_counter = 0
    
    if args.batch:
        # Batch (parallel) processing
        print(f"Using batch processing with {args.batch_size} workers...")
        
        def process_sample(sample):
            """Process a single sample (for parallel execution)."""
            try:
                return extractor.extract(sample)
            except Exception as e:
                print(f"Failed to process sample {sample.get('id')}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {executor.submit(process_sample, sample): sample for sample in samples}
            
            for future in tqdm(as_completed(futures), total=len(samples), desc=f"Extracting {args.dataset}"):
                result = future.result()
                if result is not None:
                    results.append(result)
                
                # Checkpoint
                if len(results) >= args.checkpoint_interval:
                    checkpoint_counter += 1
                    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_counter:04d}.json"
                    with open(checkpoint_file, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"\nCheckpoint saved: {checkpoint_file}")
                    
                    all_results.extend(results)
                    with open(output_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    results = []
    else:
        # Sequential processing
        for sample in tqdm(samples, desc=f"Extracting {args.dataset}"):
            try:
                result = extractor.extract(sample)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Failed to process sample {sample.get('id')}: {e}")
                continue
            
            # Checkpoint
            if len(results) >= args.checkpoint_interval:
                checkpoint_counter += 1
                checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_counter:04d}.json"
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nCheckpoint saved: {checkpoint_file}")
                
                all_results.extend(results)
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                results = []
    
    # Final save
    if results:
        all_results.extend(results)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {'Batch' if args.batch else 'Sequential'}" + (f" (workers: {args.batch_size})" if args.batch else ""))
    print(f"Total: {len(samples)} | Success: {len(all_results)} | Failed: {len(samples) - len(all_results)}")
    print(f"Output: {output_path}")
    print(f"{'='*50}")
    
    if all_results:
        example = all_results[0]
        print("\nExample output:")
        if "function" in example:
            print(f"  Function: {example['function'][:60]}...")
            print(f"  Arguments: {len(example.get('arguments', []))}")
            for cond in example.get('arguments', [])[:3]:
                print(f"    [{cond['argument_id']}] {cond['argument'][:50]}...")


if __name__ == "__main__":
    main()
