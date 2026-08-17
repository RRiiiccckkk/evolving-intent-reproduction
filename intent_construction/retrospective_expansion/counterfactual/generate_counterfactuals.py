"""
Argument Counterfactual Script

Generates counterfactual (similar but different) versions of arguments in extracted data.
The original arguments are preserved, and counterfactual versions are added alongside them.

Usage:
    # Basic usage with default settings (2 counterfactuals per argument)
    python generate_counterfactuals.py \
        --input ../../intent_extraction/output/gsm8k/extracted_test.json \
        --output output/gsm8k/counterfactual.json

    # Specify number of counterfactuals per argument
    python generate_counterfactuals.py \
        --input ../../intent_extraction/output/gsm8k/extracted_test.json \
        --output output/gsm8k/counterfactual.json \
        --num_counterfactuals 3

    # Use specific dataset type for appropriate prompts
    python generate_counterfactuals.py \
        --input ../../intent_extraction/output/browsecomp_plus/extracted_test.json \
        --output output/browsecomp_plus/counterfactual.json \
        --dataset_type search
"""

import json
import os
import re
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from intent_construction.intent_extraction.core.llm_utils import (
    LLMAccountingError,
    LLMIncompleteResponse,
    generate_json,
    load_prompt,
    populate_prompt,
)


class CounterfactualGenerator:
    """Handles generation of counterfactual arguments for extracted data."""
    
    def __init__(
        self,
        model: str = "gpt-5.1",
        prompts_dir: str = "prompts",
        dataset_type: str = "default",
        max_attempts: int = 5,
        temperature: float = 1.0
    ):
        """
        Initialize the generator.
        
        Args:
            model: Model identifier for generation
            prompts_dir: Directory containing prompt templates
            dataset_type: Type of dataset (math, search, default) for prompt selection
            max_attempts: Max retries for LLM generation
            temperature: Sampling temperature (higher = more diverse, default: 1.0)
        """
        self.model = model
        self.prompts_dir = Path(prompts_dir)
        self.dataset_type = dataset_type
        self.max_attempts = max_attempts
        self.temperature = temperature
        
        # Load appropriate prompt based on dataset type
        prompt_file = self.prompts_dir / f"generate_counterfactual_{dataset_type}.txt"
        if not prompt_file.exists():
            print(f"Warning: Prompt for '{dataset_type}' not found, using default")
            prompt_file = self.prompts_dir / "generate_counterfactual_default.txt"
        
        self.prompt_template = load_prompt(prompt_file)
    
    @staticmethod
    def validate_counterfactual(
        original_argument: str,
        result: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Validate that a counterfactual is a clean value swap.
        
        Checks:
        1. Forward reconstruction: orig.replace(orig_val, pert_val) == counterfactual
        2. orig_val exists in original argument
        3. pert_val exists in counterfactual argument
        4. No containment: orig_val not in pert_val and vice versa
        
        Returns:
            (is_valid, reason) tuple
        """
        counterfactual = result.get("counterfactual_argument", "")
        orig_val = result.get("original_value", "")
        pert_val = result.get("counterfactual_value", "")
        
        if not orig_val or not pert_val:
            return False, "Missing original_value or counterfactual_value"
        
        # Check orig_val exists in original
        if orig_val not in original_argument:
            return False, f"original_value '{orig_val}' not found in original argument"
        
        # Check pert_val exists in counterfactual
        if pert_val not in counterfactual:
            return False, f"counterfactual_value '{pert_val}' not found in counterfactual argument"
        
        # Forward reconstruction: replace orig_val with pert_val in original
        reconstructed = original_argument.replace(orig_val, pert_val, 1)
        if reconstructed != counterfactual:
            # Check if the only difference is a/an article mismatch
            # or minor apostrophe differences
            def normalize_articles(s):
                """Collapse a/an differences for comparison."""
                s = s.replace(" an ", " a ").replace(" An ", " A ")
                if s.startswith("An "): s = "A " + s[3:]
                if s.startswith("an "): s = "a " + s[3:]
                return s
            
            def normalize_apostrophes(s):
                """Collapse apostrophe differences (e.g., artist's vs artists)."""
                # Handle both ASCII (') and Unicode (\u2019) apostrophes
                s = s.replace("\u2019s ", "s ").replace("\u2019s.", "s.")
                s = s.replace("'s ", "s ").replace("'s.", "s.")
                s = s.replace("\u2019", "").replace("'", "")
                return s
            
            def normalize_text(s):
                return normalize_apostrophes(normalize_articles(s))
            
            if normalize_text(reconstructed) == normalize_text(counterfactual):
                # Only a/an or apostrophe difference — accept the model's version
                result["_article_grammar_adjusted"] = True
            else:
                return False, (
                    f"Forward reconstruction failed — replacing '{orig_val}' with '{pert_val}' "
                    f"in original does not produce the counterfactual argument. "
                    f"Expected: '{counterfactual}' | Got: '{reconstructed}'"
                )
        
        # Containment check: orig_val must not be substring of pert_val or vice versa
        # Exception: small prefix/suffix differences (≤20% length) like negation "not"
        if orig_val in pert_val and orig_val != pert_val:
            ratio = len(orig_val) / len(pert_val)
            if ratio < 0.80:
                return False, (
                    f"ADDITIVE: counterfactual_value '{pert_val}' contains original_value '{orig_val}'. "
                    f"This is an elaboration/subset, not a value swap. "
                    f"The original will not contradict the counterfactual version."
                )
        if pert_val in orig_val and orig_val != pert_val:
            ratio = len(pert_val) / len(orig_val)
            if ratio < 0.80:
                return False, (
                    f"DELETION: original_value '{orig_val}' contains counterfactual_value '{pert_val}'. "
                    f"This is a deletion/generalization, not a value swap. "
                    f"The counterfactual version is a subset of the original."
                )
        
        # Length ratio check: counterfactual_value shouldn't be wildly longer than original
        if len(pert_val) > max(len(orig_val) * 2, len(orig_val) + 10):
            return False, (
                f"LENGTH: counterfactual_value '{pert_val}' is much longer than original_value "
                f"'{orig_val}' ({len(pert_val)} vs {len(orig_val)} chars). "
                f"This looks like an elaboration, not a value swap."
            )
        
        return True, "OK"
    
    def _get_arguments_from_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get arguments list from sample (function + arguments format)."""
        result = [{"argument_id": 0, "argument": sample["function"]}]
        for cond in sample["arguments"]:
            result.append({
                "argument_id": cond["argument_id"],
                "argument": cond["argument"]
            })
        return result
    
    def generate_counterfactual_argument(
        self,
        question: str,
        arguments: List[Dict[str, Any]],
        target_argument: Dict[str, Any],
        previous_counterfactuals: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single counterfactual version of a argument.
        
        Args:
            question: The original question
            arguments: All arguments for context (function + arguments)
            target_argument: The argument to counterfactual
            previous_counterfactuals: List of previously generated counterfactual arguments to avoid duplicates
            
        Returns:
            Dict with counterfactual_argument, original_value, counterfactual_value, reasoning
            or None if generation failed
        """
        validation_feedback = ""
        for attempt in range(self.max_attempts):
            try:
                prompt = populate_prompt(
                    self.prompt_template,
                    {
                        "QUESTION": question,
                        "CONDITIONS": json.dumps(arguments, indent=2),
                        "TARGET_CONDITION": json.dumps(target_argument)
                    }
                )
                
                # Add previous counterfactuals to prompt to avoid duplicates
                if previous_counterfactuals:
                    avoid_text = "\n\nIMPORTANT: The following counterfactuals have ALREADY been generated. You MUST generate a DIFFERENT counterfactual that changes a DIFFERENT value or uses a DIFFERENT number. Do NOT repeat any of these:\n"
                    for i, prev in enumerate(previous_counterfactuals, 1):
                        avoid_text += f"  {i}. \"{prev}\"\n"
                    avoid_text += "\nGenerate a NEW and DIFFERENT counterfactual:\n"
                    prompt += avoid_text
                
                # Add validation feedback from previous failed attempt
                if validation_feedback:
                    prompt += f"\n\nWARNING — your previous attempt FAILED validation: {validation_feedback}\nFix this issue. Remember: the counterfactual argument must be obtainable by a SINGLE find-and-replace of original_value → counterfactual_value in the original. No other characters may change (no article a/an changes, no apostrophe changes, no rephrasing).\n"
                
                result = generate_json(
                    [{"role": "user", "content": prompt}],
                    model=self.model,
                    step="generate-counterfactual-argument",
                    temperature=self.temperature
                )
                
                counterfactual_argument = result.get("counterfactual_argument", "")
                
                if not counterfactual_argument:
                    print(f"    Empty counterfactual argument, retrying... (attempt {attempt + 1})")
                    validation_feedback = ""
                    continue
                
                # Validate that the counterfactual argument is different from original
                if counterfactual_argument.strip() == target_argument["argument"].strip():
                    print(f"    Counterfactual argument is same as original, retrying... (attempt {attempt + 1})")
                    validation_feedback = ""
                    continue
                
                # Programmatic validation: forward reconstruction check
                # (may auto-correct a/an articles in result)
                is_valid, reason = self.validate_counterfactual(
                    target_argument["argument"], result
                )
                
                if not is_valid:
                    print(f"    Validation failed: {reason} (attempt {attempt + 1})")
                    validation_feedback = reason
                    continue
                
                # Re-read counterfactual_argument from result (may have been auto-corrected)
                final_result = {
                    "counterfactual_argument": result.get("counterfactual_argument", ""),
                    "original_value": result.get("original_value", ""),
                    "counterfactual_value": result.get("counterfactual_value", ""),
                    "reasoning": result.get("reasoning", "")
                }
                if result.get("_auto_corrected_article") or result.get("_article_grammar_adjusted"):
                    final_result["_article_grammar_adjusted"] = True
                return final_result
                
            except (LLMAccountingError, LLMIncompleteResponse):
                raise
            except Exception as e:
                print(f"    Error generating counterfactual argument: {e} (attempt {attempt + 1})")
                validation_feedback = ""
                continue
        
        return None
    
    def generate_counterfactuals(
        self,
        sample: Dict[str, Any],
        num_counterfactuals: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate counterfactual arguments for all arguments (non-function) in a sample.
        
        Args:
            sample: Original extracted sample (supports both new and legacy formats)
            num_counterfactuals: Number of counterfactual versions to generate per argument
            
        Returns:
            Modified sample with counterfactual_arguments added to each argument,
            or None if all counterfactuals failed
        """
        question = sample["question"]
        task_id = sample.get("task_id", "unknown")
        
        # Create a copy of the sample
        new_sample = deepcopy(sample)
        
        # Get arguments (new format)
        if "arguments" not in sample:
            print(f"  ✗ Sample {task_id} has no arguments, skipping...")
            return None
            
        arguments = sample["arguments"]
        total_arguments = len(arguments)
        successful_counterfactuals = 0
        
        # Build context for LLM once (shared across all arguments, read-only)
        context_for_llm = [{"argument_id": 0, "argument": sample.get("function", "")}]
        for c_idx, c in enumerate(arguments, 1):
            context_for_llm.append({
                "argument_id": c.get("argument_id", c_idx),
                "argument": c["argument"]
            })
        
        def _generate_single_counterfactual(cond_idx, cond):
            """Counterfactual a single argument (thread-safe). Returns (cond_idx, new_cond, num_success)."""
            new_cond = deepcopy(cond)
            
            # Skip already injected arguments
            if cond.get("is_injected", False):
                return (cond_idx, new_cond, 0)
            
            # === LLM-based counterfactual ===
            
            target = {
                "argument_id": cond.get("argument_id", cond_idx + 1),
                "argument": cond["argument"]
            }
            
            counterfactual_arguments = []
            max_duplicate_retries = 2
            num_success = 0
            
            for p_idx in range(num_counterfactuals):
                prev_texts = [pc["counterfactual_argument"] for pc in counterfactual_arguments]
                
                for dup_retry in range(max_duplicate_retries + 1):
                    result = self.generate_counterfactual_argument(
                        question=question,
                        arguments=context_for_llm,
                        target_argument=target,
                        previous_counterfactuals=prev_texts if prev_texts else None
                    )
                    
                    if result is not None:
                        is_duplicate = any(
                            pc["counterfactual_argument"].strip() == result["counterfactual_argument"].strip()
                            for pc in counterfactual_arguments
                        )
                        
                        if not is_duplicate:
                            counterfactual_arguments.append({
                                "counterfactual_argument": result["counterfactual_argument"],
                                "original_value": result.get("original_value", ""),
                                "counterfactual_value": result.get("counterfactual_value", ""),
                                "reasoning": result.get("reasoning", "")
                            })
                            num_success += 1
                            break
                        else:
                            if dup_retry < max_duplicate_retries:
                                if result["counterfactual_argument"] not in prev_texts:
                                    prev_texts.append(result["counterfactual_argument"])
                            else:
                                print(f"    Duplicate counterfactual argument for argument {cond['argument_id']} after {max_duplicate_retries} retries, skipping...")
                    else:
                        break
            
            if counterfactual_arguments:
                new_cond["counterfactual_arguments"] = counterfactual_arguments
            
            return (cond_idx, new_cond, num_success)
        
        # Parallelize across arguments within the sample
        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=min(len(arguments), 20)) as cond_executor:
            futures = {
                cond_executor.submit(_generate_single_counterfactual, i, cond): i
                for i, cond in enumerate(arguments)
            }
            for future in as_completed(futures):
                cond_idx, new_cond, num_success = future.result()
                results_by_idx[cond_idx] = new_cond
                successful_counterfactuals += num_success
        
        # Reassemble in original order
        new_arguments = [results_by_idx[i] for i in range(len(arguments))]
        
        # Check if we got at least some counterfactuals
        if successful_counterfactuals == 0:
            print(f"  ✗ No counterfactuals generated for {task_id}")
            return None
        
        new_sample["arguments"] = new_arguments
        new_sample["counterfactual_info"] = {
            "num_counterfactuals_requested": num_counterfactuals,
            "total_arguments": total_arguments,
            "successful_counterfactuals": successful_counterfactuals,
            "dataset_type": self.dataset_type
        }
        
        print(f"  ✓ Generated {successful_counterfactuals}/{total_arguments * num_counterfactuals} counterfactuals for {task_id}")
        return new_sample


def main():
    parser = argparse.ArgumentParser(
        description="Generate counterfactual arguments for extracted data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input extracted JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--num_counterfactuals",
        type=int,
        default=2,
        help="Number of counterfactual arguments to generate per original argument (default: 2)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="math",
        choices=["math", "search", "default"],
        help="Type of dataset for appropriate prompt selection (default: math)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="Model to use (default: gpt-5.1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for LLM (higher = more diverse, default: 1.0)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable parallel batch processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of parallel workers for batch processing (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = args.output.replace(".json", "_checkpoint.json")
    
    # Load input data
    print(f"Loading input data from: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Limit samples if specified
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Processing first {len(data)} samples")
    
    print(f"Dataset type: {args.dataset_type}")
    print(f"Counterfactuals per argument: {args.num_counterfactuals}")
    
    # Resume from checkpoint if requested
    results = []
    failed = 0
    start_idx = 0
    processed_ids = set()
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get("results", [])
        failed = checkpoint_data.get("failed", 0)
        start_idx = checkpoint_data.get("next_idx", 0)
        processed_ids = set(checkpoint_data.get("processed_ids", []))
        print(f"  Loaded {len(results)} completed results, starting from index {start_idx}")
    
    # Initialize generator
    generator = CounterfactualGenerator(
        model=args.model,
        prompts_dir="prompts",
        dataset_type=args.dataset_type,
        temperature=args.temperature
    )
    
    # Helper function for batch processing
    def process_single_sample(sample_with_idx):
        idx, sample = sample_with_idx
        sample_id = sample.get("task_id", f"sample-{idx}")
        
        # Skip if already processed
        if sample_id in processed_ids:
            return None, sample_id, True  # skipped
        
        result = generator.generate_counterfactuals(sample, args.num_counterfactuals)
        return result, sample_id, False  # not skipped
    
    # Process samples
    actual_idx = start_idx
    samples_to_process = [(start_idx + i, sample) for i, sample in enumerate(data[start_idx:])]
    
    try:
        if args.batch:
            # Batch (parallel) processing
            print(f"🚀 Batch processing enabled with {args.batch_size} workers")
            
            with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
                # Process in chunks for better checkpoint control
                chunk_size = args.checkpoint_interval
                
                for chunk_start in range(0, len(samples_to_process), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(samples_to_process))
                    chunk = samples_to_process[chunk_start:chunk_end]
                    
                    # Submit all tasks in chunk
                    future_to_sample = {
                        executor.submit(process_single_sample, s): s 
                        for s in chunk
                    }
                    
                    # Collect results with progress bar
                    for future in tqdm(
                        as_completed(future_to_sample),
                        desc=f"Chunk {chunk_start//chunk_size + 1}",
                        total=len(chunk)
                    ):
                        result, sample_id, skipped = future.result()
                        
                        if skipped:
                            continue
                        
                        if result is not None:
                            results.append(result)
                            processed_ids.add(sample_id)
                        else:
                            failed += 1
                    
                    actual_idx = start_idx + chunk_end - 1
                    
                    # Save checkpoint after each chunk
                    checkpoint_data = {
                        "results": results,
                        "failed": failed,
                        "next_idx": actual_idx + 1,
                        "processed_ids": list(processed_ids),
                        "total_samples": len(data),
                        "num_counterfactuals": args.num_counterfactuals,
                        "dataset_type": args.dataset_type
                    }
                    with open(checkpoint_path, "w") as f:
                        json.dump(checkpoint_data, f)
                    print(f"\n  💾 Checkpoint saved at index {actual_idx + 1} ({len(results)} results)")
        else:
            # Sequential processing (original behavior)
            for idx, sample in enumerate(tqdm(
                data[start_idx:],
                desc="Generating counterfactuals",
                initial=start_idx,
                total=len(data)
            )):
                actual_idx = start_idx + idx
                sample_id = sample.get("task_id", f"sample-{actual_idx}")
                
                # Skip if already processed
                if sample_id in processed_ids:
                    continue
                
                result = generator.generate_counterfactuals(sample, args.num_counterfactuals)
                
                if result is not None:
                    results.append(result)
                    processed_ids.add(sample_id)
                else:
                    failed += 1
                
                # Save checkpoint periodically
                if (actual_idx + 1) % args.checkpoint_interval == 0:
                    checkpoint_data = {
                        "results": results,
                        "failed": failed,
                        "next_idx": actual_idx + 1,
                        "processed_ids": list(processed_ids),
                        "total_samples": len(data),
                        "num_counterfactuals": args.num_counterfactuals,
                        "dataset_type": args.dataset_type
                    }
                    with open(checkpoint_path, "w") as f:
                        json.dump(checkpoint_data, f)
                    print(f"\n  💾 Checkpoint saved at index {actual_idx + 1} ({len(results)} results)")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Saving checkpoint...")
        checkpoint_data = {
            "results": results,
            "failed": failed,
            "next_idx": actual_idx + 1,
            "processed_ids": list(processed_ids),
            "total_samples": len(data),
            "num_counterfactuals": args.num_counterfactuals,
            "dataset_type": args.dataset_type
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)
        print(f"  💾 Checkpoint saved to: {checkpoint_path}")
        print(f"  To resume, run with --resume flag")
        return
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print(f"  Saving emergency checkpoint...")
        checkpoint_data = {
            "results": results,
            "failed": failed,
            "next_idx": actual_idx + 1,
            "processed_ids": list(processed_ids),
            "total_samples": len(data),
            "num_counterfactuals": args.num_counterfactuals,
            "dataset_type": args.dataset_type
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)
        print(f"  💾 Emergency checkpoint saved to: {checkpoint_path}")
        raise
    
    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✓ Successfully processed: {len(results)}/{len(data)} samples")
    print(f"✗ Failed: {failed}/{len(data)} samples")
    print(f"✓ Saved to: {args.output}")
    
    # Save example for inspection
    if results:
        example_path = args.output.replace(".json", "_example.json")
        with open(example_path, "w") as f:
            json.dump(results[0], f, indent=2)
        print(f"✓ Example saved to: {example_path}")
    
    # Remove checkpoint file on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint file removed (completed successfully)")


if __name__ == "__main__":
    main()
