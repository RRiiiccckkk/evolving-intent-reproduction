"""
SWE-bench Verified specific argument counterfactual.

Differences from generic generate_counterfactuals.py:
- 4 category-specific prompts (constraint/scope/approach/location)
- Only counterfactual_eligible categories are processed; symptom/trigger keep empty counterfactual_arguments list
- No programmatic forward-reconstruction validation (SWE counterfactuals are not clean value swaps).
  Only checks that counterfactual != original.
- Function and full problem_statement are passed to the prompt for context.

Usage:
    python generate_counterfactuals_swe.py \
        --input ../../intent_extraction/output/swe_bench_verified/extracted_filtered_n251.json \
        --output output/swe_bench_verified/argument_counterfactual.json \
        --num_counterfactuals 2 \
        --num_workers 8

Resume:
    Add --resume to continue from a saved checkpoint.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

# Reuse llm utilities from the extraction pipeline (canonical location).
from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt  # noqa: E402

COUNTERFACTUAL_ELIGIBLE_CATEGORIES = ("constraint", "scope", "approach", "location")
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = {cat: PROMPTS_DIR / f"generate_counterfactual_swe_{cat}.txt" for cat in COUNTERFACTUAL_ELIGIBLE_CATEGORIES}


def _build_prompt(template: str, sample: dict, target_cond: dict) -> str:
    arguments_payload = [
        {
            "id": c["argument_id"],
            "argument": c["argument"],
            "category": c.get("category"),
        }
        for c in sample.get("arguments", [])
    ]
    return populate_prompt(
        template,
        {
            "GOAL": sample.get("function", ""),
            "PROBLEM_STATEMENT": (sample.get("question") or "")[:1500],
            "CONDITIONS": json.dumps(arguments_payload, indent=2),
            "TARGET_CONDITION": target_cond["argument"],
        },
    )


def generate_one_counterfactual(
    sample: dict,
    cond: dict,
    template: str,
    model: str,
    num_counterfactuals: int,
    max_attempts: int = 3,
    reasoning_effort: str | None = None,
) -> list[dict]:
    """Generate `num_counterfactuals` distinct counterfactuals for one argument."""
    out: list[dict] = []
    seen: set[str] = {cond["argument"].strip()}

    for p_idx in range(num_counterfactuals):
        for attempt in range(max_attempts):
            prompt = _build_prompt(template, sample, cond)
            if out:
                avoid = "\n\nIMPORTANT: Already-generated counterfactuals to AVOID duplicating:\n"
                for i, prev in enumerate(out, 1):
                    avoid += f"  {i}. \"{prev['counterfactual_argument']}\"\n"
                avoid += "\nGenerate a DIFFERENT counterfactual:\n"
                prompt = prompt + avoid
            try:
                result = generate_json(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    step=f"swe-counterfactual-{cond.get('category')}",
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                print(f"      ! attempt {attempt+1} error: {exc}")
                continue

            counterfactual_text = (result.get("counterfactual_argument") or "").strip()
            if not counterfactual_text:
                continue
            if counterfactual_text in seen:
                continue

            seen.add(counterfactual_text)
            out.append(
                {
                    "counterfactual_argument": counterfactual_text,
                    "counterfactual_type": result.get("counterfactual_type", ""),
                    "what_changed": result.get("what_changed", ""),
                    "reasoning": result.get("reasoning", ""),
                }
            )
            break  # got a valid one; move to next p_idx

    return out


def generate_counterfactuals(
    sample: dict,
    templates: dict[str, str],
    model: str,
    num_counterfactuals: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Add a `counterfactual_arguments` list to each argument (empty for non-counterfactual_eligible)."""
    new_arguments: list[dict] = []
    for cond in sample.get("arguments", []):
        cat = cond.get("category")
        cond_out = dict(cond)
        if cat in COUNTERFACTUAL_ELIGIBLE_CATEGORIES:
            cond_out["counterfactual_arguments"] = generate_one_counterfactual(
                sample,
                cond,
                templates[cat],
                model,
                num_counterfactuals,
                reasoning_effort=reasoning_effort,
            )
        else:
            cond_out["counterfactual_arguments"] = []
        new_arguments.append(cond_out)
    out = dict(sample)
    out["arguments"] = new_arguments
    out["counterfactual_info"] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "num_counterfactuals_requested": num_counterfactuals,
        "counterfactual_eligible_categories": list(COUNTERFACTUAL_ELIGIBLE_CATEGORIES),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified argument counterfactual")
    parser.add_argument("--input", required=True, help="Path to extracted_filtered_n251.json")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--num_counterfactuals", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    with open(args.input) as f:
        data = json.load(f)
    if args.num_samples:
        data = data[: args.num_samples]
    print(f"Loaded {len(data)} samples from {args.input}")

    templates = {cat: load_prompt(str(PROMPT_FILES[cat])) for cat in COUNTERFACTUAL_ELIGIBLE_CATEGORIES}

    results: list[dict] = []
    done_ids: set[str] = set()
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        done_ids = {r["task_id"] for r in results}
        print(f"Resumed: {len(results)} samples already done.")

    pending = [s for s in data if s["task_id"] not in done_ids]
    print(f"Pending: {len(pending)} samples × ~{args.num_counterfactuals} counterfactuals each (×counterfactual_eligible arguments).")

    write_lock = Lock()
    start = time.time()

    def task(sample):
        try:
            return generate_counterfactuals(
                sample,
                templates,
                args.model,
                args.num_counterfactuals,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as exc:
            return {"task_id": sample.get("task_id"), "_error": str(exc)}

    completed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(task, s): s for s in pending}
        for fut in as_completed(futures):
            res = fut.result()
            with write_lock:
                results.append(res)
                completed += 1
                if completed % args.checkpoint_interval == 0:
                    with open(checkpoint_path, "w") as f:
                        json.dump({"results": results}, f)
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1e-9)
                    eta = (len(pending) - completed) / max(rate, 1e-9)
                    print(f"  [{completed}/{len(pending)}] checkpoint saved. rate={rate:.2f}/s eta={eta/60:.1f}min")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    if checkpoint_path.exists():
        os.remove(checkpoint_path)

    n_err = sum(1 for r in results if "_error" in r)
    print(f"\nDone. {len(results) - n_err}/{len(results)} samples succeeded ({n_err} errors).")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
