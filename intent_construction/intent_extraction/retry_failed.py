"""
Retry failed samples from any dataset extraction.
Automatically detects missing samples and merges results.

Usage:
    # Auto-detect failed samples and retry
    python retry_failed.py --dataset gsm8k --model gpt-5.1
    
    # Specify specific IDs to retry
    python retry_failed.py --dataset gsm8k --model gpt-5.1 --ids 267 369 400
    
    # Skip auto-merge
    python retry_failed.py --dataset gsm8k --model gpt-5.1 --no_merge
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from intent_construction.intent_extraction.registry import get_extractor
from intent_construction.intent_extraction.generate import DATASET_LOADERS


def get_extracted_ids(extracted_file: Path) -> set:
    """Get set of original_ids (as strings) from extracted file."""
    if not extracted_file.exists():
        return set()
    
    with open(extracted_file, 'r') as f:
        data = json.load(f)
    
    return {str(item['original_id']) for item in data}


def get_all_ids(dataset_name: str, split: str = "test") -> set:
    """Get all sample IDs (as strings) from the original dataset."""
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load all samples
    samples = DATASET_LOADERS[dataset_name](split=split, num_samples=None)
    return {str(sample['id']) for sample in samples}


def load_samples_by_ids(dataset_name: str, ids: list, split: str = "test") -> list:
    """Load specific samples by their IDs."""
    # Load all samples first
    all_samples = DATASET_LOADERS[dataset_name](split=split, num_samples=None)
    
    # Normalize to strings for comparison (IDs may be int or str depending on dataset)
    id_set = {str(i) for i in ids}
    return [s for s in all_samples if str(s['id']) in id_set]


def merge_results(original_file: Path, retry_file: Path) -> int:
    """Merge retry results into original file. Returns new total count."""
    # Load original
    if original_file.exists():
        with open(original_file, 'r') as f:
            original = json.load(f)
    else:
        original = []
    
    # Load retry
    with open(retry_file, 'r') as f:
        retry = json.load(f)
    
    if not retry:
        print("No new samples to merge.")
        return len(original)
    
    # Merge (retry samples are new, no duplicates expected)
    merged = original + retry
    
    # Sort by original_id (numeric sort when possible)
    def _sort_key(x):
        oid = x['original_id']
        try:
            return (0, int(oid))
        except (ValueError, TypeError):
            return (1, str(oid))
    merged.sort(key=_sort_key)
    
    # Save
    with open(original_file, 'w') as f:
        json.dump(merged, f, indent=2)
    
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Retry failed samples from extraction")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., gsm8k)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model to use (default: gpt-5.1)")
    parser.add_argument("--batch_size", type=int, default=10, help="Parallel workers (default: 10)")
    parser.add_argument("--ids", type=str, nargs='+', default=None, help="Specific sample IDs to retry (auto-detect if not provided)")
    parser.add_argument("--max_retries", type=int, default=5, help="Max verification attempts (default: 5)")
    parser.add_argument("--no_merge", action="store_true", help="Don't auto-merge results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: output/{dataset}/)")
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"output/{args.dataset}")
    extracted_file = output_dir / f"extracted_{args.split}.json"
    retry_file = output_dir / f"extracted_{args.split}_retry.json"
    
    print(f"{'='*60}")
    print(f"Retry Failed Samples")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Extracted file: {extracted_file}")
    print(f"{'='*60}")
    
    # Determine which IDs to retry
    if args.ids:
        failed_ids = args.ids
        print(f"\nUsing specified IDs: {failed_ids}")
    else:
        # Auto-detect missing samples
        print(f"\nAuto-detecting failed samples...")
        extracted_ids = get_extracted_ids(extracted_file)
        all_ids = get_all_ids(args.dataset, args.split)
        failed_ids = sorted(all_ids - extracted_ids)
        
        if not failed_ids:
            print("✅ No failed samples found! All samples already extracted.")
            return
        
        print(f"Found {len(failed_ids)} missing samples: {failed_ids}")
    
    # Load samples to retry
    samples = load_samples_by_ids(args.dataset, failed_ids, args.split)
    print(f"Loaded {len(samples)} samples to retry")
    
    # Initialize extractor
    extractor = get_extractor(
        args.dataset,
        model=args.model,
        verif_model=args.model,
        enable_model_verification=True,
        max_verification_attempts=args.max_retries
    )
    
    # Process in parallel
    results = []
    
    def process_sample(sample):
        try:
            return extractor.extract(sample)
        except Exception as e:
            print(f"Failed sample {sample.get('id')}: {e}")
            return None
    
    print(f"\nProcessing with {args.batch_size} workers...")
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in samples}
        
        for future in tqdm(as_completed(futures), total=len(samples), desc="Retrying"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Save retry results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(retry_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Retry Results")
    print(f"{'='*60}")
    print(f"Attempted: {len(samples)}")
    print(f"Succeeded: {len(results)}")
    print(f"Failed: {len(samples) - len(results)}")
    print(f"Retry output: {retry_file}")
    
    # Show still-failed samples (normalize to str for comparison)
    success_ids = {str(r['original_id']) for r in results}
    still_failed = [i for i in failed_ids if str(i) not in success_ids]
    if still_failed:
        print(f"\nStill failed: {still_failed}")
    
    # Auto-merge if requested
    if not args.no_merge and results:
        print(f"\n{'='*60}")
        print("Merging results...")
        total = merge_results(extracted_file, retry_file)
        print(f"✅ Merged! Total samples: {total}")
        
        # Clean up retry file after successful merge
        retry_file.unlink()
        print(f"Cleaned up: {retry_file}")
    
    print(f"{'='*60}")
    
    if not still_failed:
        print("\n🎉 All samples recovered!")
    else:
        print(f"\n⚠️  {len(still_failed)} samples still need attention: {still_failed}")


if __name__ == "__main__":
    main()
