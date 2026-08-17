"""SWE-bench Verified — Stage 3b: optional G1 exploration function.

Generates a generic repo / module orientation question and prepends it to
``predecessor_functions`` for each target sample, so the resulting function-switch
chain becomes G1 (orientation) -> G2 (paired real bug) -> G3 (target).

The 7 archetypes are pre-assigned in round-robin order (seeded shuffle for
deterministic but varied output) so the dataset has uniform category
coverage. The LLM is told which archetype to use, and the prompt's leak
rules forbid mentioning specific symbols / errors from the eventual fix.

Usage:
    python generate_g1_swe.py \
        --input  output/swe_bench_verified/paired_n251.json \
        --output output/swe_bench_verified/paired_g1_n251.json \
        --num_workers 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

# Reuse extraction's llm utilities.
from intent_construction.intent_extraction.core.llm_utils import generate_json, load_prompt, populate_prompt  # noqa: E402

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "exploration_function_swe.txt"

ARCHETYPES = (
    "repo_layout",
    "module_overview",
    "public_api",
    "io_shape",
    "testing_convention",
    "dependency_map",
    "coding_convention",
)


def derive_module_focus(sample: dict[str, Any]) -> str:
    """Longest common prefix (path segments) between target's affected file
    and paired G2's affected file. Falls back to repo identifier."""
    g3_files = sample.get("swe_bench_metadata", {}).get("affected_files", [])
    pg = (sample.get("predecessor_functions") or [{}])[0]
    g2_files = (pg.get("_pair_metadata") or {}).get("paired_files", [])
    repo = sample.get("swe_bench_metadata", {}).get("repo", "")
    if g3_files and g2_files:
        a = g3_files[0].split("/")
        b = g2_files[0].split("/")
        common: list[str] = []
        for x, y in zip(a, b):
            if x == y:
                common.append(x)
            else:
                break
        if len(common) >= 2:
            return "/".join(common[:2])
        if common:
            return common[0]
    if g3_files:
        return "/".join(g3_files[0].split("/")[:2])
    return repo


def make_g1_entry(g1_text: str, archetype: str) -> dict[str, Any]:
    return {
        "predecessor_function": g1_text,
        "counterfactual_arguments": [],   # G1 has no per-argument reveals
        "is_predecessor": False,
        "transition_type": "exploration",
        "taxonomy_type": archetype,
        "transition_reason": (
            "User asked an orientation question about the repo/module before "
            "diving into a specific bug."
        ),
        "entity_sought": "explanation",
    }


def generate_g1_for_sample(
    sample: dict[str, Any],
    *,
    template: str,
    model: str,
    archetype: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generate one G1 row; strict orchestrators validate the returned row."""
    out = deepcopy(sample)
    repo = sample.get("swe_bench_metadata", {}).get("repo", "")
    prompt = populate_prompt(
        template,
        {
            "REPO": repo,
            "MODULE_FOCUS": derive_module_focus(sample),
            "TARGET_GOAL": sample.get("function", ""),
            "ARCHETYPE": archetype,
        },
    )
    response = generate_json(
        [{"role": "user", "content": prompt}],
        model=model,
        step=f"swe-g1-{archetype}",
        reasoning_effort=reasoning_effort,
    )
    g1_text = (response.get("exploration_function") or "").strip()
    if not g1_text:
        raise ValueError("empty exploration_function")
    existing = out.get("predecessor_functions") or []
    out["predecessor_functions"] = list(existing) + [make_g1_entry(g1_text, archetype)]
    out["g1_info"] = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "archetype": archetype,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified G1 exploration function generation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--reasoning_effort", default=None)
    args = parser.parse_args()

    template = load_prompt(str(PROMPT_PATH))
    with open(args.input) as f:
        data = json.load(f)
    if args.num_samples:
        data = data[: args.num_samples]
    print(f"Loaded {len(data)} samples.")

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    archetype_assignments: dict[int, str] = {
        idx: ARCHETYPES[pos % len(ARCHETYPES)] for pos, idx in enumerate(indices)
    }

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
        archetype = archetype_assignments[idx]
        try:
            out = generate_g1_for_sample(
                sample,
                template=template,
                model=args.model,
                archetype=archetype,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as exc:
            return idx, None, f"{type(exc).__name__}: {exc}"
        return idx, out, None

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(task, i): i for i in range(len(data))}
        for fut in as_completed(futures):
            idx, out, err = fut.result()
            with write_lock:
                if out is None:
                    errors.append(f"sample {idx}: {err}")
                    results[idx] = data[idx]  # keep original on failure
                else:
                    results[idx] = out
                    counts[archetype_assignments[idx]] += 1
                completed += 1
                if completed % args.checkpoint_interval == 0:
                    with open(out_path, "w") as f:
                        json.dump([r for r in results if r is not None], f, indent=2)
                    rate = completed / max(time.time() - t0, 1e-9)
                    eta = (len(data) - completed) / max(rate, 1e-9)
                    print(f"  [{completed}/{len(data)}] rate={rate:.2f}/s eta={eta/60:.1f}min")

    with open(out_path, "w") as f:
        json.dump([r for r in results if r is not None], f, indent=2)

    print("\nArchetype distribution:")
    for a in ARCHETYPES:
        print(f"  {a:<22} {counts[a]}")
    print(f"\nFailed: {len(errors)}")
    for e in errors[:10]:
        print(f"  {e}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
