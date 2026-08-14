"""
Retry failed samples from argument counterfactual (Stage 2).
Automatically detects missing samples and merges results.

Usage:
    # Auto-detect failed samples and retry
    python retry_failed.py \
        --input ../../intent_extraction/output/gsm8k/extracted_test.json \
        --output output/gsm8k/argument_counterfactual.json \
        --model gpt-5.1
    
    # Specify specific task_ids to retry
    python retry_failed.py \
        --input ../../intent_extraction/output/gsm8k/extracted_test.json \
        --output output/gsm8k/argument_counterfactual.json \
        --model gpt-5.1 \
        --task_ids extracted-gsm8k-test-135 extracted-gsm8k-test-200
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from intent_construction.retrospective_expansion.counterfactual.generate_counterfactuals import CounterfactualGenerator


def get_task_ids_from_file(filepath: Path) -> set:
    """Get set of task_ids from a JSON file."""
    if not filepath.exists():
        return set()
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {item.get('task_id') for item in data if item.get('task_id')}


def load_samples_by_task_ids(input_file: Path, task_ids: set) -> list:
    """Load specific samples by their task_ids from input file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return [s for s in data if s.get('task_id') in task_ids]


def merge_results(original_file: Path, retry_results: list) -> int:
    """Merge retry results into original file. Returns new total count."""
    # Load original
    if original_file.exists():
        with open(original_file, 'r') as f:
            original = json.load(f)
    else:
        original = []
    
    if not retry_results:
        print("No new samples to merge.")
        return len(original)
    
    # Get existing task_ids
    existing_ids = {s.get('task_id') for s in original}
    
    # Add only new samples
    new_samples = [s for s in retry_results if s.get('task_id') not in existing_ids]
    merged = original + new_samples
    
    # Sort by task_id
    merged.sort(key=lambda x: x.get('task_id', ''))
    
    # Save
    with open(original_file, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Retry failed samples from argument counterfactual")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to Stage 1 extracted JSON (source)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to Stage 2 output JSON (to check & merge)")
    parser.add_argument("--model", type=str, default="gpt-5.1",
                        help="Model to use (default: gpt-5.1)")
    parser.add_argument("--dataset_type", type=str, default="math",
                        choices=["math", "search", "default"],
                        help="Dataset type for prompt selection")
    parser.add_argument("--num_counterfactuals", type=int, default=2,
                        help="Number of counterfactuals per argument (default: 2)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Parallel workers (default: 10)")
    parser.add_argument("--task_ids", nargs='+', default=None,
                        help="Specific task_ids to retry (auto-detect if not provided)")
    parser.add_argument("--no_merge", action="store_true",
                        help="Don't auto-merge results")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    retry_path = output_path.parent / f"{output_path.stem}_retry.json"
    
    print(f"{'='*60}")
    print(f"Retry Failed Samples - Stage 2 (Argument Counterfactual)")
    print(f"{'='*60}")
    print(f"Input (Stage 1): {input_path}")
    print(f"Output (Stage 2): {output_path}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")
    
    # Determine which task_ids to retry
    if args.task_ids:
        failed_ids = set(args.task_ids)
        print(f"\nUsing specified task_ids: {failed_ids}")
    else:
        # Auto-detect missing samples
        print(f"\nAuto-detecting failed samples...")
        input_ids = get_task_ids_from_file(input_path)
        output_ids = get_task_ids_from_file(output_path)
        failed_ids = input_ids - output_ids
        
        if not failed_ids:
            print("✅ No failed samples found! All samples already processed.")
            return
        
        print(f"Found {len(failed_ids)} missing samples: {sorted(failed_ids)}")
    
    # Load samples to retry
    samples = load_samples_by_task_ids(input_path, failed_ids)
    print(f"Loaded {len(samples)} samples to retry")
    
    if not samples:
        print("❌ No samples found with the specified task_ids")
        return
    
    # Initialize generator
    generator = CounterfactualGenerator(
        model=args.model,
        dataset_type=args.dataset_type,
        temperature=args.temperature
    )
    
    # Process samples
    results = []
    
    def process_sample(sample):
        try:
            return generator.generate_counterfactuals(sample, num_counterfactuals=args.num_counterfactuals)
        except Exception as e:
            print(f"Failed sample {sample.get('task_id')}: {e}")
            return None
    
    print(f"\nProcessing with {args.batch_size} workers...")
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in samples}
        
        for future in tqdm(as_completed(futures), total=len(samples), desc="Retrying"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Save retry results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(retry_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Retry Results")
    print(f"{'='*60}")
    print(f"Attempted: {len(samples)}")
    print(f"Succeeded: {len(results)}")
    print(f"Failed: {len(samples) - len(results)}")
    print(f"Retry output: {retry_path}")
    
    # Show still-failed samples
    success_ids = {r.get('task_id') for r in results}
    still_failed = [i for i in failed_ids if i not in success_ids]
    if still_failed:
        print(f"\nStill failed: {sorted(still_failed)}")
    
    # Auto-merge if requested
    if not args.no_merge and results:
        print(f"\n{'='*60}")
        print("Merging results...")
        total = merge_results(output_path, results)
        print(f"✅ Merged! Total samples: {total}")
        
        # Clean up retry file after successful merge
        retry_path.unlink()
        print(f"Cleaned up: {retry_path}")
    
    print(f"{'='*60}")
    
    if not still_failed:
        print("\n🎉 All samples recovered!")
    else:
        print(f"\n⚠️  {len(still_failed)} samples still need attention: {sorted(still_failed)}")


if __name__ == "__main__":
    main()
