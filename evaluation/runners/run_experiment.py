#!/usr/bin/env python3
"""
Run systematic evaluation experiments for EvolvingIntent.

Scenario is automatically inferred from parameters:
- num_turns=1, no changes → fully-specified (baseline)
- num_turns>1, no changes → under-specified
- num_revisions>0 → argument-revision
- num_switches>0 → function-switch

Usage:
    # Fully-specified baseline (num_turns=1, no changes)
    python run_experiment.py \
        --data_path ../intent_construction/retrospective_expansion/predecessor/output/gsm8k/predecessor.json \
        --models gpt-5.1 \
        --dataset_name gsm8k \
        --num_turns 1

    # Under-specified with 3 turns (no changes)
    python run_experiment.py \
        --data_path ../intent_construction/retrospective_expansion/predecessor/output/gsm8k/predecessor.json \
        --models gpt-5.1 \
        --dataset_name gsm8k \
        --num_turns 3

    # Argument-change with 2 revisions
    python run_experiment.py \
        --data_path ../intent_construction/retrospective_expansion/predecessor/output/gsm8k/predecessor.json \
        --models gpt-5.1 \
        --dataset_name gsm8k \
        --num_turns 3 \
        --num_revisions 2

    # Parallel execution
    python run_experiment.py \
        --data_path ../intent_construction/retrospective_expansion/predecessor/output/gsm8k/predecessor.json \
        --models gpt-5.1 \
        --dataset_name gsm8k \
        --num_turns 1 \
        --num_workers 8
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Math verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: math-verify not installed. Using fallback answer checking.")

from intent_construction.intent_extraction.core.llm_utils import (
    generate_text,
    generate_multi_turn,
    clean_model_name,
)
from situated_simulation.user_simulation import EvolvingIntent
from situated_simulation.naturalizer import supports_llm_naturalization

# IF (Instruction Following) evaluation removed — IFEval/IFBench no longer supported
from evaluation.common.sql_evaluator import is_sql_dataset, evaluate_sql_response, extract_sql_from_response
# SWE-bench evaluation (lazy: imports swebench library + Docker only when SWE samples seen)
from evaluation.common.swe_evaluator import evaluate_swe_response, is_swe_dataset, instance_id_from_metadata


# =============================================================================
# Constants
# =============================================================================

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


# =============================================================================
# Scenario Inference
# =============================================================================

def infer_scenario(num_turns: int, num_revisions: int, num_switches: int) -> str:
    """Infer scenario from parameters."""
    if num_revisions > 0 and num_switches > 0:
        return "combined"
    
    if num_revisions > 0:
        return "argument_revision"
    elif num_switches > 0:
        return "function_switch"
    elif num_turns == 1:
        return "fully_specified"
    else:
        return "under_specified"


# =============================================================================
# Answer Extraction & Checking
# =============================================================================

def extract_answer(response: str) -> str | None:
    """Extract answer from model response (prioritize \\boxed{} format)."""
    if not response:
        return None
    
    # Priority 1: Find \boxed{} - handle nested braces
    # This regex handles nested braces like \boxed{\frac{1}{2}}
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    boxed_matches = re.findall(boxed_pattern, response)
    if boxed_matches:
        # Return the last \boxed{} (final answer)
        return boxed_matches[-1].strip()
    
    # Priority 2: Simple \boxed{} without nested braces
    simple_boxed = re.search(r"\\boxed\{([^}]+)\}", response)
    if simple_boxed:
        return simple_boxed.group(1).strip()
    
    # Priority 3: Answer: format variants
    answer_patterns = [
        r"[Aa]nswer:\s*\$?\\?boxed\{([^}]+)\}",  # Answer: \boxed{...}
        r"[Aa]nswer:\s*\$([^$\n]+)\$",             # Answer: $...$
        r"[Aa]nswer:\s*([^\n]+)",                   # Answer: ...
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, response)
        if match:
            answer = match.group(1).strip()
            answer = answer.rstrip('.')
            return answer
    
    # Priority 4: GSM8K format
    gsm8k_match = re.search(r"####\s*(\-?[0-9\.\,]+)", response)
    if gsm8k_match:
        return gsm8k_match.group(1).strip()
    
    # Last resort: find last number
    numbers = re.findall(r"\-?[0-9]+\.?[0-9]*", response)
    if numbers:
        return numbers[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    answer = answer.strip()
    
    # Remove LaTeX formatting
    answer = re.sub(r"^\\\((.+)\\\)$", r"\1", answer)
    answer = re.sub(r"^\\\[(.+)\\\]$", r"\1", answer)
    answer = re.sub(r"\\boxed\{([^}]+)\}", r"\1", answer)
    answer = re.sub(r"\\text\{[^}]*\}", "", answer)  # Strip units entirely
    answer = re.sub(r"\\mathrm\{[^}]*\}", "", answer)  # Strip units entirely
    answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\^\s*\\circ", "", answer)  # Strip degree symbol
    
    # Remove common formatting
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.replace("\\!", "")  # LaTeX thin space
    answer = re.sub(r"\{,\}", "", answer)  # LaTeX comma in numbers
    answer = answer.replace("{", "").replace("}", "")
    answer = answer.replace("\\", "")
    answer = answer.strip()
    
    # Try to convert to float for numeric comparison
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer.lower()


def check_answer(predicted: str | None, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth using math-verify.
    Falls back to string comparison if math-verify fails.
    """
    if predicted is None:
        return False
    
    # Try math-verify first (handles mathematical equivalence)
    if MATH_VERIFY_AVAILABLE:
        try:
            # Wrap in $ for LaTeX parsing
            parsed_gold = parse(f"${ground_truth}$")
            parsed_pred = parse(f"${predicted}$")
            if parsed_gold and parsed_pred:
                return verify(parsed_gold, parsed_pred)
        except Exception:
            pass  # Fall back to string comparison
    
    # Fallback: normalized string comparison
    return normalize_answer(predicted) == normalize_answer(ground_truth)


# =============================================================================
# API Call with Retry
# =============================================================================

def call_with_retry(
    messages: list[dict],
    model: str,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    max_retries: int = 15,
    reasoning_effort: str | None = None,
) -> tuple[str, list[dict]]:
    """Call LLM API with retry logic for multi-turn conversations.

    Returns ``(text, output_items)``. ``output_items`` should be appended to
    the conversation by the caller so reasoning trace (if any) is preserved on
    the next call. ``model`` should be a quota-resolved identifier (e.g.
    ``gpt-5.1`` or ``gpt-5.1``); do not pass a base name and rely on
    quota rotation here, because encrypted reasoning blobs are tied to the
    deployment that issued them.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return generate_multi_turn(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:
            last_error = e
            error_str = str(e)

            if attempt < max_retries - 1:
                # Parse rate limit delay
                match = re.search(r"[Tt]ry again in (\d+) seconds", error_str)
                if match:
                    delay = int(match.group(1)) + 1
                else:
                    delay = min(2 ** attempt, 60)

                time.sleep(delay)

    raise last_error


# =============================================================================
# Multi-turn Conversation Runner
# =============================================================================

def run_multi_turn_conversation(
    sample,  # IntentSample
    model: str,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Run multi-turn conversation and collect all responses.

    Args:
        sample: IntentSample from EvolvingIntent
        model: Model identifier
        temperature: Sampling temperature (0 for greedy)
        max_tokens: Max tokens per response

    Returns:
        Result dict with prediction, responses, etc.
    """
    messages: list[dict] = []
    all_responses: list[str] = []

    # Use the model as given for the whole conversation. Encrypted reasoning
    # blobs from the Responses API are tied to a specific deployment, so the
    # model must stay fixed between turns.
    effective_model = model

    # Initialize: get first turn(s) from sample
    initial_turns = sample.reset()
    for turn in initial_turns:
        messages.append(turn)

    # Conversation loop
    while True:
        # Get model response for the latest user turn
        last_user = next(
            (t for t in reversed(messages) if t.get("role") == "user"), None
        )
        if last_user is None:
            break

        try:
            response, output_items = call_with_retry(
                messages=messages,
                model=effective_model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            all_responses.append(response)
            messages.extend(output_items)
        except Exception as e:
            user_messages = [
                m["content"] for m in messages if m.get("role") == "user"
            ]
            return {
                "success": False,
                "error": str(e),
                "responses": all_responses,
                "prediction": None,
                "user_messages": user_messages,
            }

        # Check if sample has more turns
        if sample.is_done():
            break

        # Get next user turn(s) from sample
        next_turns = sample.step(response)
        for turn in next_turns:
            messages.append(turn)

    # Extract answer from final response
    final_response = all_responses[-1] if all_responses else None
    prediction = extract_answer(final_response) if final_response else None

    # Collect user messages for display
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]

    return {
        "success": True,
        "error": None,
        "responses": all_responses,
        "prediction": prediction,
        "final_response": final_response,  # Full text for IF evaluation
        "user_messages": user_messages,
    }


# =============================================================================
# Single Sample Evaluation
# =============================================================================

def evaluate_sample(
    sample,  # IntentSample
    model: str,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Evaluate a single sample and return result."""

    # Run conversation
    result = run_multi_turn_conversation(
        sample=sample,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    
    # IF datasets removed — see Refactor for paper
    sample_is_sql = is_sql_dataset(sample.metadata.get("data_source", ""))
    # SWE detection: metadata.task is currently stripped by the simulator (None for
    # SWE samples in eval mode). Fall back to the task_id prefix, which is
    # always present and unambiguous: 'extracted-swe_bench_verified-test-...'.
    sample_is_swe = (
        is_swe_dataset(sample.metadata)
        or (isinstance(sample.task_id, str)
            and sample.task_id.startswith("extracted-swe_bench_verified-"))
    )

    if sample_is_swe:
        # Resolve SWE-bench instance_id. Prefer metadata, fall back to
        # parsing task_id (verified 251/251 parsable for the canonical
        # data file).
        instance_id = instance_id_from_metadata(sample.metadata)
        if instance_id is None:
            prefix = "extracted-swe_bench_verified-test-"
            if isinstance(sample.task_id, str) and sample.task_id.startswith(prefix):
                instance_id = sample.task_id[len(prefix):]

        final_response = result.get("final_response", "") or ""

        if instance_id is None:
            # Refuse to fabricate an answer. Log and mark as failed sample.
            return {
                "task_id": sample.task_id,
                "prediction": None,
                "correct": False,
                "ground_truth": "",
                "decoding": result["responses"],
                "user_messages": result.get("user_messages", []),
                "success": False,
                "error": "swe_instance_id_unresolvable",
                "metadata": sample.metadata,
                "swe_eval": {
                    "resolved": False,
                    "patch_extracted": False,
                    "patch_apply_ok": False,
                    "harness_error": "instance_id_unresolvable",
                    "ftp_pass": [], "ftp_fail": [],
                    "ptp_pass": [], "ptp_fail": [],
                    "duration_s": 0.0,
                    "from_cache": False,
                },
            }

        swe_result = evaluate_swe_response(
            response=final_response,
            instance_id=instance_id,
            model_name=model,
        )

        return {
            "task_id": sample.task_id,
            "prediction": swe_result.patch,
            "correct": swe_result.correct,
            "ground_truth": "",  # SWE has no answer string; correctness lives in swe_eval
            "decoding": result["responses"],
            "user_messages": result.get("user_messages", []),
            "success": result["success"],
            "error": result["error"],
            "metadata": sample.metadata,
            "swe_eval": {
                "resolved": swe_result.correct,
                "patch_extracted": swe_result.patch_extracted,
                "patch_apply_ok": swe_result.patch_apply_ok,
                "ftp_pass": swe_result.ftp_pass,
                "ftp_fail": swe_result.ftp_fail,
                "ptp_pass": swe_result.ptp_pass,
                "ptp_fail": swe_result.ptp_fail,
                "harness_error": swe_result.harness_error,
                "duration_s": swe_result.duration_s,
                "from_cache": swe_result.from_cache,
                "instance_id": instance_id,
            },
        }

    if sample_is_sql:
        # SQL evaluation: extract SQL from response, execute, compare
        db_path = sample.metadata.get("db_path", "")
        gold_sql = sample.metadata.get("gold_sql", "")
        
        if result.get("final_response"):
            sql_result = evaluate_sql_response(result["final_response"], gold_sql, db_path)
            correct = sql_result.execution_match
            prediction = sql_result.model_sql or ""
        else:
            correct = False
            prediction = ""
        
        # Per-turn evaluation (mid-turn)
        per_turn_results = []
        per_turn_gold = sample.metadata.get("per_turn_gold", [])
        responses = result.get("responses", [])
        for turn_idx, (response, gold) in enumerate(zip(responses, per_turn_gold)):
            if gold and gold.get("sql"):
                turn_result = evaluate_sql_response(response, gold["sql"], db_path)
                per_turn_results.append({
                    "turn": turn_idx,
                    "correct": turn_result.execution_match,
                    "gold_sql": gold["sql"],
                    "gold_answer": gold.get("answer", ""),
                    "model_sql": turn_result.model_sql,
                    "model_answer": turn_result.model_answer,
                })
        
        return {
            "task_id": sample.task_id,
            "prediction": prediction,
            "correct": correct,
            "ground_truth": sample.label,
            "per_turn_results": per_turn_results,
            "decoding": result["responses"],
            "user_messages": result.get("user_messages", []),
            "success": result["success"],
            "error": result["error"],
            "metadata": sample.metadata,
        }
    else:
        # Math/Search evaluation: answer comparison
        correct = check_answer(result["prediction"], sample.label)
        
        return {
            "task_id": sample.task_id,
            "prediction": result["prediction"],
            "correct": correct,
            "ground_truth": sample.label,
            "decoding": result["responses"],
            "user_messages": result.get("user_messages", []),
            "success": result["success"],
            "error": result["error"],
            "metadata": sample.metadata,
        }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    data_path: str,
    dataset_name: str,
    model: str,
    num_turns: int = 1,
    num_revisions: int = 0,
    num_switches: int = 0,
    ordering: str = "sequential",
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    num_samples: int | None = None,
    num_workers: int = 1,
    instruction: str | None = None,
    task_ids_file: str | None = None,
    naturalizer_model: str | None = None,
    reasoning_effort: str | None = None,
    rerun_failed: bool = False,
    recap_method: str | None = None,
    output_suffix: str | None = None,
    prefix_style: str | None = None,
    skip_llm_judge: bool = False,
    llm_judge_model: str = "gpt-5.1",
    include_evidence: bool = True,
) -> dict[str, Any]:
    """
    Run a single experiment configuration.

    Returns:
        Summary dict with accuracy and paths
    """
    # Infer scenario from parameters
    scenario = infer_scenario(num_turns, num_revisions, num_switches)

    # Auto-detect domain from dataset (needed for filename + simulator)
    domain = "math"  # default
    dataset_type = None
    try:
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        if raw_data and isinstance(raw_data, list):
            first_sample = raw_data[0]
            dataset_type = first_sample.get("data_source", first_sample.get("task", ""))
            if dataset_type == "search":
                raise NotImplementedError(
                    "Search/BrowseComp datasets are not supported by run_experiment.py. "
                    "Use run_browsecomp_experiment.py instead."
                )
            elif dataset_type == "bird_sql":
                domain = "sql"
            elif dataset_type in ("swe_bench", "swe_bench_verified"):
                domain = "swe_bench_verified"
    except NotImplementedError:
        raise
    except Exception:
        pass

    # Determine output filename based on scenario (strip quota prefix)
    save_name = clean_model_name(model)
    if reasoning_effort:
        save_name = f"{save_name}-reasoning-{reasoning_effort}"
    if scenario == "fully_specified":
        output_filename = f"{save_name}.json"
    elif scenario == "under_specified":
        output_filename = f"{save_name}_t{num_turns}.json"
    elif scenario == "argument_revision":
        output_filename = f"{save_name}_t{num_turns}_p{num_revisions}_{ordering}.json"
    elif scenario == "function_switch":
        output_filename = f"{save_name}_t{num_turns}_g{num_switches}.json"
    elif scenario == "combined":
        output_filename = f"{save_name}_t{num_turns}_g{num_switches}_p{num_revisions}.json"
    else:
        output_filename = f"{save_name}.json"
    
    # Append _naturalized suffix only when online naturalization actually runs
    # for this domain (other domains fall back to the rule-based naturalizer).
    if supports_llm_naturalization(domain, naturalizer_model):
        output_filename = output_filename.replace(".json", "_naturalized.json")

    # Append recap method suffix
    if recap_method:
        output_filename = output_filename.replace(".json", f"_recap-{recap_method}.json")

    # Append custom output suffix
    if output_suffix:
        output_filename = output_filename.replace(".json", f"_{output_suffix}.json")

    # Append prefix_style suffix when non-base
    if prefix_style and prefix_style != "base":
        output_filename = output_filename.replace(".json", f"_{prefix_style}.json")

    # Append no-evidence suffix when evidence is disabled (SQL-only effect)
    if not include_evidence:
        output_filename = output_filename.replace(".json", "_no-evidence.json")
    
    # Setup output directory
    # Use "combined_independent" folder to separate from legacy coupled results
    dir_scenario = "combined_independent" if scenario == "combined" else scenario
    exp_dir = EXPERIMENTS_DIR / dir_scenario / dataset_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / output_filename
    
    # Skip if results already exist (or detect failed samples for rerun)
    existing_results = None
    failed_task_ids = None

    if output_path.exists():
        if rerun_failed:
            with open(output_path, 'r') as f:
                existing_results = json.load(f)

            failed_task_ids = [
                task_id for task_id, result in existing_results.items()
                if not result.get("success", True) or result.get("error") is not None
            ]

            if not failed_task_ids:
                total = len(existing_results)
                correct = sum(1 for r in existing_results.values() if r.get("correct", False))
                print(f"✅ No failed samples in {output_path} ({correct}/{total} correct). Skipping.")
                return {"model": model, "status": "skipped", "path": str(output_path)}

            print(f"🔄 Found {len(failed_task_ids)} failed samples in {output_path}. Rerunning...")
        else:
            print(f"⏭️  Skipping (already exists): {output_path}")
            return {"model": model, "status": "skipped", "path": str(output_path)}
    
    # Load task_ids filter if provided
    task_ids = None
    if task_ids_file:
        with open(task_ids_file, 'r') as f:
            task_ids_data = json.load(f)
            task_ids = task_ids_data.get('task_ids', task_ids_data)

    # Override task_ids with failed samples for rerun
    if failed_task_ids is not None:
        task_ids = failed_task_ids
    
    # Load simulator (scenario is auto-inferred inside EvolvingIntent;
    # domain was detected above)
    sim = EvolvingIntent(
        data_path=data_path,
        mode="eval",
        domain=domain,
        num_turns=num_turns,
        num_revisions=num_revisions,
        num_switches=num_switches,
        ordering=ordering,
        instruction=instruction,  # None → auto-detected by EvolvingIntent
        task_ids=task_ids,
        naturalizer_model=naturalizer_model,
        recap_method=recap_method,
        prefix_style=prefix_style,
        include_evidence=include_evidence,
    )

    samples = list(sim)
    if num_samples and failed_task_ids is None:
        samples = samples[:num_samples]

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario} (auto-inferred)")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model}")
    print(f"Samples: {len(samples)}")
    print(f"Num turns: {num_turns}")
    if num_revisions > 0:
        print(f"Num revisions: {num_revisions}")
        print(f"Ordering: {ordering}")
    if num_switches > 0:
        print(f"Num switches: {num_switches}")
    print(f"Temperature: {temperature}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    if recap_method:
        print(f"Recap method: {recap_method}")
    print(f"Output: {output_path}")
    if failed_task_ids is not None:
        print(f"🔄 RERUN MODE: {len(failed_task_ids)} failed samples")
    print(f"{'='*60}")
    
    # Run evaluation
    results = {}
    
    if num_workers > 1:
        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_sample, sample, model, temperature, max_tokens,
                    reasoning_effort
                ): sample
                for sample in samples
            }
            
            for future in tqdm(as_completed(futures), total=len(samples), desc=f"{model}"):
                try:
                    result = future.result()
                    results[result["task_id"]] = result
                except Exception as e:
                    sample = futures[future]
                    print(f"\n❌ Error: {sample.task_id}: {e}")
    else:
        # Sequential evaluation
        for sample in tqdm(samples, desc=f"{model}"):
            try:
                result = evaluate_sample(sample, model, temperature, max_tokens,
                                        reasoning_effort)
                results[result["task_id"]] = result
            except Exception as e:
                print(f"\n❌ Error: {sample.task_id}: {e}")
    
    # Merge rerun results into existing results
    if existing_results is not None:
        existing_results.update(results)
        results = existing_results

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # SQL: post-grade with LLM judge to catch semantically-equivalent answers
    # (e.g. 'NY' vs 'New York', split vs concatenated names). Updates the
    # `correct` field in-place and writes the file back.
    if dataset_name and dataset_name.lower().startswith("bird_sql") and not skip_llm_judge:
        try:
            import importlib.util
            judge_path = (
                Path(__file__).parent.parent / "scripts" / "llm_judge_bird_sql.py"
            )
            spec = importlib.util.spec_from_file_location(
                "llm_judge_bird_sql", judge_path
            )
            judge_mod = importlib.util.module_from_spec(spec)  # type: ignore
            spec.loader.exec_module(judge_mod)  # type: ignore

            print(f"\n🧑‍⚖️  LLM judge re-grading SQL results ({llm_judge_model})...")
            flipped, jc, sk, strict_n, lenient, n = judge_mod.run_judge_on_file(
                output_path,
                judge_model=llm_judge_model,
                workers=10,
                update_correct=True,
                verbose=False,
            )
            if n:
                print(
                    f"   strict={strict_n}/{n} ({strict_n/n:.1%}) "
                    f"→ lenient={lenient}/{n} ({lenient/n:.1%})  "
                    f"flipped={flipped}, skipped={sk}, calls={jc}"
                )
            # Reload updated results so the summary below reflects lenient grades
            with open(output_path) as f:
                results = json.load(f)
        except Exception as e:
            print(f"   ⚠️  LLM judge failed: {e} — keeping strict grades")
    
    # Calculate summary
    total = len(results)
    correct = sum(1 for r in results.values() if r.get("correct", False))
    failed = sum(1 for r in results.values() if not r.get("success", True))
    accuracy = correct / total if total > 0 else 0.0
    
    summary = {
        "scenario": scenario,
        "dataset": dataset_name,
        "model": model,
        "num_turns": num_turns,
        "num_revisions": num_revisions,
        "num_switches": num_switches,
        "ordering": ordering,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "total_samples": total,
        "correct": correct,
        "failed": failed,
        "accuracy": accuracy,
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Add per-turn metrics (SQL datasets with per_turn_results)
    per_turn_samples = [r for r in results.values() if r.get("per_turn_results")]
    if per_turn_samples:
        per_turn_agg: dict[int, dict[str, int]] = {}
        for r in per_turn_samples:
            for tr in r["per_turn_results"]:
                t = tr["turn"]
                if t not in per_turn_agg:
                    per_turn_agg[t] = {"correct": 0, "total": 0}
                per_turn_agg[t]["total"] += 1
                if tr["correct"]:
                    per_turn_agg[t]["correct"] += 1

        per_turn_metrics = {}
        for t in sorted(per_turn_agg):
            agg = per_turn_agg[t]
            acc = agg["correct"] / agg["total"] if agg["total"] else 0.0
            per_turn_metrics[f"turn_{t}"] = {
                "correct": agg["correct"],
                "total": agg["total"],
                "accuracy": round(acc, 4),
            }
        summary["per_turn_metrics"] = per_turn_metrics

    # IF-specific summary metrics removed (IFEval/IFBench no longer supported)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    if "per_turn_metrics" in summary:
        print(f"  Per-turn accuracy:")
        for turn_key in sorted(summary["per_turn_metrics"], key=lambda k: int(k.split("_")[1])):
            m = summary["per_turn_metrics"][turn_key]
            print(f"    {turn_key}: {m['accuracy']:.2%} ({m['correct']}/{m['total']})")
    if failed > 0:
        print(f"  ⚠ Failed: {failed}")
    
    return summary


# =============================================================================
# Config Management
# =============================================================================

def save_experiment_config(scenario: str, config: dict):
    """Save experiment configuration."""
    exp_dir = EXPERIMENTS_DIR / scenario
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_dir / "config.json"
    
    # Load existing config or create new
    if config_path.exists():
        with open(config_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {"runs": []}
    
    # Add new run
    existing["runs"].append(config)
    existing["last_updated"] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(existing, f, indent=2)


def save_results_summary(scenario: str, summaries: list[dict]):
    """Save results summary for experiment."""
    exp_dir = EXPERIMENTS_DIR / scenario
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = exp_dir / "results.json"
    
    # Load existing or create new
    if results_path.exists():
        with open(results_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {"summaries": []}
    
    # Add new summaries (avoid duplicates by checking key fields)
    for summary in summaries:
        key = (summary["model"], summary["dataset"], summary.get("num_turns"), 
               summary.get("num_revisions"), summary.get("ordering"))
        
        # Remove existing entry with same key
        existing["summaries"] = [
            s for s in existing["summaries"]
            if (s["model"], s["dataset"], s.get("num_turns"),
                s.get("num_revisions"), s.get("ordering")) != key
        ]
        
        existing["summaries"].append(summary)
    
    existing["last_updated"] = datetime.now().isoformat()
    
    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run EvolvingIntent experiments")
    
    # Required arguments
    parser.add_argument("--data_path", required=True,
                        help="Path to Stage 3 output JSON")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Models to evaluate")
    parser.add_argument("--dataset_name", required=True,
                        help="Dataset name for output folder")
    
    # Scenario parameters (scenario is auto-inferred)
    parser.add_argument("--num_turns", type=int, default=1,
                        help="Number of turns (1=fully-specified, >1=multi-turn)")
    parser.add_argument("--num_revisions", type=int, default=0,
                        help="Number of revisions (argument-value changes; 0 = none)")
    parser.add_argument("--num_switches", type=int, default=0,
                        help="Number of function switches (0 = no switch)")
    parser.add_argument("--ordering", default="interleaved",
                        choices=["sequential", "interleaved", "mixed", "random"],
                        help="Ordering strategy for argument-revision")
    
    # Execution parameters
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max tokens per response")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit samples (for testing)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Custom instruction (use {content} placeholder)")
    parser.add_argument("--task_ids_file", type=str, default=None,
                        help="JSON file with task_ids to filter (for fair comparison)")
    parser.add_argument("--naturalizer_model", type=str, default=None,
                        help="Model for online naturalizer (default: None = use pre-built turns)")
    parser.add_argument("--run_plan", action="store_true",
                        help="Run planned 24 configs (log-scale sweep)")
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort for GPT-5 models (default: None = no reasoning)")
    parser.add_argument("--rerun_failed", action="store_true",
                        help="Rerun only failed samples (success=false or error!=null) from existing results")
    parser.add_argument("--recap_method", type=str, default=None,
                        choices=["prompt", "dump", "ground_truth"],
                        help="Recap method: prompt (CoT nudge), dump (paste prior turns), ground_truth (exact state)")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Custom suffix appended to output filename (e.g., 'safe-prefix' → *_safe-prefix.json)")
    parser.add_argument("--prefix_style", type=str, default=None,
                        choices=["base", "function-naturalized", "function-naturalized-v2"],
                        help="Prefix style for function changes. 'function-naturalized' uses short correction-style "
                             "prefixes for SQL (e.g., 'How about the average instead of the count?'). "
                             "Auto-appends suffix to output filename when non-base.")
    parser.add_argument("--skip_llm_judge", action="store_true",
                        help="Skip LLM-judge post-grading for SQL results "
                             "(default: run judge to catch semantically-equivalent answers).")
    parser.add_argument("--llm_judge_model", type=str, default="gpt-5.1",
                        help="Model used for the SQL LLM judge. Default gpt-5.1.")
    parser.add_argument("--no_evidence", action="store_true",
                        help="SQL only: drop the BIRD evidence field from user turn 1. "
                             "Auto-appends '_no-evidence' suffix to output filename.")
    
    args = parser.parse_args()
    
    # Validate models
    for model in args.models:
        if not model:
            print(f"❌ Unsupported model: {model!r}")
            print("   Set a model id served by your OpenAI / Azure OpenAI account "
                  "(e.g. gpt-5.1).")
            return
    
    # Run experiments
    all_summaries = []
    
    if args.run_plan:
        # Planned experiment configs (log-scale sweep):
        # (g, p, t)
        PLAN_CONFIGS = [
            # fully_specified
            (0, 0, 1),
            # under_specified: t=2,4,8
            (0, 0, 2), (0, 0, 4), (0, 0, 8),
            # argument_revision: (t,p) = (4,1),(4,2),(4,3),(8,1),(8,2),(8,4)
            (0, 1, 4), (0, 2, 4), (0, 3, 4), (0, 1, 8), (0, 2, 8), (0, 4, 8),
            # function_switch: t=4,8 x g=1,2,3
            (1, 0, 4), (2, 0, 4), (3, 0, 4),
            (1, 0, 8), (2, 0, 8), (3, 0, 8),
            # combined: g=1..3 x p=1..3, t=1+g+p
            (1, 1, 3), (1, 2, 4), (1, 3, 5),
            (2, 1, 4), (2, 2, 5), (2, 3, 6),
            (3, 1, 5), (3, 2, 6), (3, 3, 7),
        ]
        total = len(PLAN_CONFIGS)
        for model in args.models:
            print(f"\n{'='*60}")
            print(f"Plan evaluation for: {model} ({total} configs)")
            print(f"{'='*60}")
            for i, (g, p, t) in enumerate(PLAN_CONFIGS):
                config_name = f"t{t}_g{g}_p{p}"
                print(f"\n[{i+1}/{total}] {config_name} (turns={t}, functions={g}, conds={p})")
                try:
                    summary = run_experiment(
                        data_path=args.data_path,
                        dataset_name=args.dataset_name,
                        model=model,
                        num_turns=t,
                        num_revisions=p,
                        num_switches=g,
                        ordering=args.ordering,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        num_samples=args.num_samples,
                        num_workers=args.num_workers,
                        instruction=args.instruction,
                        task_ids_file=args.task_ids_file,
                        naturalizer_model=args.naturalizer_model,
                        reasoning_effort=args.reasoning_effort,
                        rerun_failed=args.rerun_failed,
                        recap_method=args.recap_method,
                        output_suffix=args.output_suffix,
                        prefix_style=args.prefix_style,
                        skip_llm_judge=args.skip_llm_judge,
                        llm_judge_model=args.llm_judge_model,
                        include_evidence=not args.no_evidence,
                    )
                    if summary.get("status") != "skipped":
                        all_summaries.append(summary)
                except Exception as e:
                    print(f"\n❌ FAILED: {config_name}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # Infer scenario for display
        scenario = infer_scenario(args.num_turns, args.num_revisions, args.num_switches)
        dir_scenario = "combined_independent" if scenario == "combined" else scenario
        print(f"🎯 Scenario: {scenario} (saving to {dir_scenario}/)")
        print(f"   num_turns={args.num_turns}, revisions={args.num_revisions}, switches={args.num_switches}")
        
        for model in args.models:
            try:
                summary = run_experiment(
                    data_path=args.data_path,
                    dataset_name=args.dataset_name,
                    model=model,
                    num_turns=args.num_turns,
                    num_revisions=args.num_revisions,
                    num_switches=args.num_switches,
                    ordering=args.ordering,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    num_samples=args.num_samples,
                    num_workers=args.num_workers,
                    instruction=args.instruction,
                    task_ids_file=args.task_ids_file,
                    naturalizer_model=args.naturalizer_model,
                    reasoning_effort=args.reasoning_effort,
                    rerun_failed=args.rerun_failed,
                    recap_method=args.recap_method,
                    output_suffix=args.output_suffix,
                    prefix_style=args.prefix_style,
                    skip_llm_judge=args.skip_llm_judge,
                    llm_judge_model=args.llm_judge_model,
                    include_evidence=not args.no_evidence,
                )
                if summary.get("status") == "skipped":
                    continue
                all_summaries.append(summary)
                
                # Save config
                save_experiment_config(dir_scenario, {
                    "data_path": args.data_path,
                    "dataset_name": args.dataset_name,
                    "model": model,
                    "num_turns": args.num_turns,
                    "num_revisions": args.num_revisions,
                    "num_switches": args.num_switches,
                    "ordering": args.ordering,
                    "temperature": args.temperature,
                    "reasoning_effort": args.reasoning_effort,
                    "timestamp": datetime.now().isoformat(),
                })
                
            except Exception as e:
                print(f"\n❌ Error with {model}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results summary
    if all_summaries:
        if not args.run_plan:
            save_results_summary(dir_scenario, all_summaries)

        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*60}")

        # Print summary table
        print(f"\n{'Model':<20} {'Accuracy':<15} {'Samples':<10}")
        print("-"*50)
        for s in all_summaries:
            print(f"{s['model']:<20} {s['accuracy']:.2%} ({s['correct']}/{s['total_samples']})")


if __name__ == "__main__":
    main()
