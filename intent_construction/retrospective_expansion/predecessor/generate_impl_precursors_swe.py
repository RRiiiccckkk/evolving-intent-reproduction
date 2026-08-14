"""SWE-bench Verified — Stage 3c: implementation-precursor function.

Replaces the existing real_bug_pair G2 entry in ``predecessor_functions[0]``
with a generated implementation-planning request that depends on the
buggy API. The G1 exploration entry (predecessor_functions[1]) is preserved.

Result chain rendered by the simulator (storage is nearest-first):
    Turn 1: G1 (orientation question, from existing pg[1])
    Turn 2: G2 (implementation planning, NEW — replaces real_bug_pair)
    Turn 3: G3 (target bug fix, from raw["function"])

Arguments for the new G2 are stored with ``is_shared=False`` and
``argument_id`` in the 2000+ range so:
  - they live entirely in G2's phase (turn 2 only)
  - argument predecessors during the conversation auto-anchor to the
    final target's source arguments (the bug fix), NOT to the impl plan

Usage:
    python generate_impl_precursors_swe.py \\
        --input  output/swe_bench_verified/paired_g1_n251.json \\
        --output output/swe_bench_verified/paired_g1_n251_implprec.json \\
        --num_workers 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

# Reuse extraction's llm utilities.
from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt  # noqa: E402

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generate_impl_precursor_swe.txt"

# argument_id base for the implementation plan's specs. cid_offset=2000
# keeps these out of the 0-999 (source) and 1000-1999 (real_bug_pair, if
# we ever co-exist) ranges.
IMPL_CID_OFFSET = 2000


def derive_module_focus(sample: dict[str, Any]) -> str:
    """Module focus = first 2 path segments of the first affected file
    (e.g. ``django/utils``). Falls back to repo identifier."""
    files = sample.get("swe_bench_metadata", {}).get("affected_files", []) or []
    repo = sample.get("swe_bench_metadata", {}).get("repo", "")
    if files:
        parts = files[0].split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0]
    return repo


def build_impl_pg_entry(
    llm_output: dict[str, Any],
    cid_offset: int = IMPL_CID_OFFSET,
) -> dict[str, Any]:
    """Convert LLM output to a predecessor_functions entry compatible with the simulator."""
    impl_text = (llm_output.get("implementation_request") or "").strip()
    if not impl_text:
        raise ValueError("empty implementation_request")

    raw_conds = llm_output.get("implementation_arguments") or []
    impl_conds: list[dict[str, Any]] = []
    for i, c in enumerate(raw_conds):
        if not isinstance(c, dict):
            continue
        text = (c.get("argument") or "").strip()
        if not text:
            continue
        impl_conds.append({
            "argument_id": cid_offset + 1 + i,  # 2001, 2002, ...
            "argument": text,
            "category": c.get("category") or "constraint",
            "predecessor_eligible": bool(c.get("predecessor_eligible", False)),
            # CRITICAL: is_shared=False so cond predecessor in combined
            # scenarios does not pick from impl-spec (anchors to source
            # arguments of the target bug instead).
            "is_shared": False,
        })

    return {
        "predecessor_function": impl_text,
        "counterfactual_arguments": impl_conds,
        # is_predecessor=True since we're generating predecessor from target.
        "is_predecessor": True,
        "transition_type": "impl_precursor",
        "taxonomy_type": llm_output.get("implementation_taxonomy"),
        "transition_reason": (
            "User considered planning a related new feature that depends on "
            "the buggy API, then realized the prerequisite bug needed fixing."
        ),
        "transition_phrase": (llm_output.get("transition_phrase") or "").strip(),
        "buggy_api_dependency": (llm_output.get("buggy_api_dependency") or "").strip(),
        "topical_relation": (llm_output.get("topical_relation") or "").strip(),
        "entity_sought": "implementation_plan",
        # Keep the LLM's reasoning around for audit / debugging.
        "_generation_reasoning": (llm_output.get("reasoning") or "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified implementation-precursor function generation")
    parser.add_argument("--input", required=True,
                        help="Input JSON (e.g. paired_g1_n251.json) with existing predecessor_functions.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--reasoning_effort", default="medium")
    args = parser.parse_args()

    template = load_prompt(str(PROMPT_PATH))
    with open(args.input) as f:
        data = json.load(f)
    if args.num_samples:
        data = data[: args.num_samples]
    print(f"Loaded {len(data)} samples from {args.input}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = Lock()
    results: list[dict[str, Any] | None] = [None] * len(data)
    counts: Counter = Counter()
    errors: list[str] = []
    t0 = time.time()
    completed = 0

    def task(idx: int) -> tuple[int, dict[str, Any] | None, str | None]:
        sample = data[idx]
        out = deepcopy(sample)
        repo = sample.get("swe_bench_metadata", {}).get("repo", "")
        mod_focus = derive_module_focus(sample)
        target_function = sample.get("function", "")
        affected_files = sample.get("swe_bench_metadata", {}).get("affected_files", []) or []

        # Validate input shape: pg[0] must be a real_bug_pair (replaceable),
        # pg[1] must be an exploration (preserved). Skip samples that don't
        # match — the alternative (silent fallback to original) produces a
        # heterogeneous output file.
        existing_pgs = sample.get("predecessor_functions") or []
        if len(existing_pgs) < 2:
            return idx, None, "input has <2 predecessor_functions"
        if existing_pgs[0].get("transition_type") != "real_bug_pair":
            return idx, None, (
                f"pg[0].transition_type={existing_pgs[0].get('transition_type')!r} "
                "(expected 'real_bug_pair')"
            )
        if existing_pgs[1].get("transition_type") != "exploration":
            return idx, None, (
                f"pg[1].transition_type={existing_pgs[1].get('transition_type')!r} "
                "(expected 'exploration')"
            )

        prompt = populate_prompt(template, {
            "REPO": repo,
            "MODULE_FOCUS": mod_focus,
            "TARGET_GOAL": target_function,
            "AFFECTED_FILES": json.dumps(affected_files),
        })
        try:
            r = generate_json(
                [{"role": "user", "content": prompt}],
                model=args.model,
                step="swe-impl-precursor",
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as exc:
            return idx, None, f"{type(exc).__name__}: {exc}"
        try:
            new_pg0 = build_impl_pg_entry(r)
        except Exception as exc:
            return idx, None, f"build_pg_entry: {type(exc).__name__}: {exc}"

        # Replace pg[0] (the real_bug_pair G2) with our new impl-precursor.
        # pg[1] (the G1 exploration) is preserved.
        existing = list(existing_pgs)
        existing[0] = new_pg0
        out["predecessor_functions"] = existing
        return idx, out, None

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(task, i): i for i in range(len(data))}
        for fut in as_completed(futures):
            idx, out, err = fut.result()
            with write_lock:
                if out is None:
                    # Drop failed samples instead of silently substituting
                    # the original real_bug_pair entry. The output file
                    # must contain ONLY impl_precursor pg[0]s — a mix of
                    # impl_precursor and real_bug_pair would break any
                    # downstream code that branches on transition_type.
                    errors.append(f"sample {idx} (task_id={data[idx].get('task_id')}): {err}")
                    results[idx] = None
                else:
                    results[idx] = out
                    tax = (out.get("predecessor_functions") or [{}])[0].get("taxonomy_type") or "unknown"
                    counts[tax] += 1
                completed += 1
                if completed % args.checkpoint_interval == 0:
                    with open(out_path, "w") as f:
                        json.dump([r for r in results if r is not None], f, indent=2)
                    rate = completed / max(time.time() - t0, 1e-9)
                    eta = (len(data) - completed) / max(rate, 1e-9)
                    print(f"  [{completed}/{len(data)}] rate={rate:.2f}/s eta={eta/60:.1f}min")

    successes = [r for r in results if r is not None]
    with open(out_path, "w") as f:
        json.dump(successes, f, indent=2)

    # Sidecar manifest of which task_ids succeeded vs failed, so downstream
    # consumers can audit gaps explicitly without re-parsing the main file.
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest = {
        "input": str(args.input),
        "output": str(out_path),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "total_input": len(data),
        "successful": len(successes),
        "failed": len(errors),
        "succeeded_task_ids": [s.get("task_id") for s in successes],
        "failed_samples": errors,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote sidecar manifest to {manifest_path}")

    print(f"\nDone. Wrote {sum(1 for r in results if r is not None)} samples to {out_path}")
    print("Implementation taxonomy distribution:")
    for k, v in counts.most_common():
        print(f"  {k:<22} {v}")
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
