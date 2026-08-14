#!/usr/bin/env python3
"""
Mini-SWE-agent v2 baseline runner for SWE-bench Verified.

Runs each sample as a single user-turn conversation through mini-swe-agent v2
(text-based action parsing, shared-client-routed model). The agent's final patch is
verified against the official SWE-bench harness via our existing
``swe_harness.py``. With ``--num_turns 1`` this is the single-shot SWE-bench
baseline, but with a real coding agent under the hood instead of one LLM
completion; higher ``--num_turns`` enables the multi-turn simulation scenarios.

Usage
-----

    # Single-turn baseline on 50 stratified samples
    python evaluation/runners/run_swe_mini_agent.py \
        --data_path intent_construction/retrospective_expansion/predecessor/output/swe_bench_verified/paired_g1_n251.json \
        --models gpt-5.1 \
        --task_ids_file intent_construction/eval_indices/swe_bench_verified_task_ids.json \
        --num_workers 4 \
        --step_limit 60

Multi-turn SWE scenarios (under-specified / argument-revision / function-switch /
combined) are driven by *this same runner* via ``--num_turns``, ``--num_switches``
and ``--num_revisions``. With those at their defaults the runner produces the
*single-turn* paraphrase / raw baselines that demand the higher-fidelity scaffold.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent

from intent_construction.intent_extraction.core.llm_utils import clean_model_name  # noqa: E402
from evaluation.common.swe_harness import SWEHarness  # noqa: E402
from evaluation.common.swe_minisweagent_scaffold import (  # noqa: E402
    MiniAgentResult,
    run_evolvingintent_with_mini_agent,
)
from situated_simulation.user_simulation import EvolvingIntent  # noqa: E402

EXPERIMENTS_DIR = _REPO / "evaluation" / "experiments"


_VALID_SOURCES = ("raw", "paraphrase")


def _question_from_sample(sample: dict[str, Any], source: str) -> str:
    if source == "raw":
        return sample.get("question") or ""
    if source == "paraphrase":
        return sample.get("fully_specified_question") or sample.get("question") or ""
    raise ValueError(f"unknown source: {source!r}; expected one of {_VALID_SOURCES}")


# Map EvolvingIntent `_infer_scenario` outputs to mini-agent output subdirectories.
# fully-specified keeps the legacy raw/paraphrase naming for back-compat.
_SCENARIO_TO_SUBDIR: dict[str, str] = {
    "under-specified": "swe_under_specified_mini_agent",
    "argument-revision": "swe_argument_revision_mini_agent",
    "function-switch": "swe_function_switch_mini_agent",
    "combined": "swe_combined_independent_mini_agent",
}


def _experiments_subdir(source: str, scenario: str = "fully-specified") -> str:
    if scenario == "fully-specified":
        return {
            "raw": "swe_original_mini_agent",
            "paraphrase": "swe_paraphrased_mini_agent",
        }[source]
    if scenario in _SCENARIO_TO_SUBDIR:
        return _SCENARIO_TO_SUBDIR[scenario]
    raise ValueError(f"unknown scenario: {scenario!r}")


def _sample_to_dict(gs) -> dict[str, Any]:
    """Adapter: flatten a situated_simulation.IntentSample into the dict shape evaluate_one expects.

    Pre-builds ``user_turns`` from the simulator's user-role messages so the mini-agent
    receives the multi-turn script verbatim. The original raw fields (question,
    fully_specified_question) are NOT carried through — for multi-turn mode the
    ``_user_turns`` key is the source of truth and ``--source`` is ignored.
    """
    md = gs.metadata or {}
    return {
        "task_id": gs.task_id,
        "original_id": md.get("original_id"),
        "swe_bench_metadata": md.get("swe_bench_metadata"),
        # ``_user_turns`` is the new key: when present, evaluate_one uses it
        # directly as the multi-turn script. When absent, evaluate_one falls
        # back to the legacy single-turn ``_question_from_sample(source)``.
        "_user_turns": [t["content"] for t in gs.turns if t.get("role") == "user"],
        # Carry simulation scenario so output dir routing can read it back.
        "_scenario": md.get("scenario", "fully-specified"),
    }


def evaluate_one(
    sample: dict[str, Any],
    model: str,
    source: str,
    harness: SWEHarness,
    *,
    cost_limit: float,
    step_limit: int,
    step_limit_per_turn: int | None = None,
    use_tool_calling: bool = False,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    instance_id = sample.get("original_id") or ""
    task_id = sample.get("task_id") or instance_id

    # Multi-turn path: pre-built simulator user_turns list is the source of truth.
    # Single-turn path: build [question] from --source as before.
    pre_built_turns = sample.get("_user_turns")
    if pre_built_turns:
        user_turns = list(pre_built_turns)
        first_turn = user_turns[0] if user_turns else ""
    else:
        first_turn = _question_from_sample(sample, source)
        user_turns = [first_turn]

    if not instance_id or not first_turn:
        return {
            "task_id": task_id, "prediction": None, "correct": False,
            "ground_truth": "", "decoding": [], "user_messages": list(user_turns),
            "success": False, "error": "missing_instance_id_or_question",
            "metadata": {"task_id": task_id, "original_id": instance_id, "source": source},
            "swe_eval": None,
        }

    instance = {"instance_id": instance_id}

    # Inject reasoning_effort into the mini config via overrides → model_kwargs.
    overrides: dict[str, Any] = {}
    if reasoning_effort:
        overrides["model"] = {"model_kwargs": {"reasoning_effort": reasoning_effort}}

    error: str | None = None
    success = True
    mini_result: MiniAgentResult | None = None
    try:
        mini_result = run_evolvingintent_with_mini_agent(
            user_turns=user_turns,
            instance=instance,
            model_name=model,
            cost_limit=cost_limit,
            step_limit=step_limit,
            step_limit_per_turn=step_limit_per_turn,
            use_tool_calling=use_tool_calling,
            overrides=overrides if overrides else None,
        )
    except Exception as e:
        success = False
        error = f"mini_agent_failed: {type(e).__name__}: {e}"

    if not success or mini_result is None:
        return {
            "task_id": task_id, "prediction": None, "correct": False,
            "ground_truth": "", "decoding": [], "user_messages": list(user_turns),
            "success": False, "error": error,
            "metadata": {"task_id": task_id, "original_id": instance_id, "source": source,
                         "scaffold_mode": "native" if use_tool_calling else "textbased"},
            "swe_eval": None,
        }

    # Score the patch via official harness.
    if mini_result.submission:
        hr = harness.verify_patch(
            instance_id=instance_id,
            patch=mini_result.submission,
            model_name=f"{model}-mini-agent",
        )
        correct = hr.resolved
        swe_eval = {
            "resolved": hr.resolved,
            "patch_extracted": hr.patch_extracted,
            "patch_apply_ok": hr.patch_apply_ok,
            "ftp_pass": hr.ftp_pass, "ftp_fail": hr.ftp_fail,
            "ptp_pass": hr.ptp_pass, "ptp_fail": hr.ptp_fail,
            "harness_error": hr.harness_error,
            "duration_s": hr.duration_s,
            "from_cache": hr.from_cache,
            "instance_id": instance_id,
        }
    else:
        correct = False
        swe_eval = {
            "resolved": False, "patch_extracted": False, "patch_apply_ok": False,
            "ftp_pass": [], "ftp_fail": [], "ptp_pass": [], "ptp_fail": [],
            "harness_error": "no_patch_extracted",
            "duration_s": 0.0, "from_cache": False, "instance_id": instance_id,
        }

    return {
        "task_id": task_id,
        "prediction": mini_result.submission,
        "correct": correct,
        "ground_truth": "",
        "decoding": [],  # full trajectory stored separately if needed
        "user_messages": list(user_turns),
        "success": True,
        "error": mini_result.error,
        "metadata": {
            "task_id": task_id,
            "original_id": instance_id,
            "source": source,
            "repo": (sample.get("swe_bench_metadata") or {}).get("repo"),
            "scaffold": "mini-swe-agent-v2",
            "scaffold_mode": mini_result.scaffold_mode,
            "n_steps": mini_result.n_steps,
            "n_user_turns_delivered": mini_result.n_user_turns_delivered,
            "exit_status": mini_result.exit_status,
            "step_limit_per_turn": mini_result.step_limit_per_turn,
            "per_turn_steps": list(mini_result.per_turn_steps),
            "scenario": sample.get("_scenario", "fully-specified"),
        },
        "swe_eval": swe_eval,
    }


def run(
    data_path: str,
    model: str,
    source: str,
    *,
    num_samples: int | None = None,
    num_workers: int = 1,
    task_ids_file: str | None = None,
    cost_limit: float = 3.0,
    step_limit: int = 250,
    step_limit_per_turn: int | None = None,
    rerun_failed: bool = False,
    extend: bool = False,
    output_suffix: str | None = None,
    use_tool_calling: bool = False,
    reasoning_effort: str | None = None,
    # Multi-turn flags. When all defaults (num_turns=1, no changes), the
    # legacy single-turn JSON-loading path is used (preserves the validated
    # 64% native baseline byte-for-byte). Otherwise samples are built via
    # EvolvingIntent.
    num_turns: int = 1,
    num_revisions: int = 0,
    num_switches: int = 0,
    ordering: str = "interleaved",
    prefix_style: str = "base",
    naturalizer_model: str | None = None,
    include_evidence: bool = True,
) -> dict[str, Any]:
    if source not in _VALID_SOURCES:
        raise ValueError(f"--source must be one of {_VALID_SOURCES}")

    # Explicit AND gate: only the legacy JSON path counts as fully-specified.
    # Single-turn fully-specified MUST use the JSON path to preserve the
    # validated 64% baseline.
    is_single_turn = (
        num_turns == 1
        and num_revisions == 0
        and num_switches == 0
    )

    if is_single_turn:
        scenario = "fully-specified"
        with open(data_path) as f:
            data: list[dict[str, Any]] = json.load(f)
        if task_ids_file:
            with open(task_ids_file) as f:
                payload = json.load(f)
                allowed = set(payload.get("task_ids", payload))
            data = [d for d in data if d.get("task_id") in allowed]
        if num_samples is not None and not rerun_failed and not extend:
            data = data[:num_samples]
    else:
        # Multi-turn path: build samples through EvolvingIntent.
        task_ids_filter: set[str] | None = None
        if task_ids_file:
            with open(task_ids_file) as f:
                payload = json.load(f)
                task_ids_filter = set(payload.get("task_ids", payload))
        sim = EvolvingIntent(
            data_path=data_path,
            mode="eval",
            domain="swe_bench_verified",
            num_turns=num_turns,
            num_revisions=num_revisions,
            num_switches=num_switches,
            ordering=ordering,
            task_ids=task_ids_filter,
            naturalizer_model=naturalizer_model,
            prefix_style=prefix_style,
            include_evidence=include_evidence,
        )
        scenario = sim.scenario
        samples = list(sim)
        if num_samples is not None and not rerun_failed and not extend:
            samples = samples[:num_samples]
        data = [_sample_to_dict(gs) for gs in samples]

    # Default suffix when use_tool_calling=True so on-disk files don't
    # collide with the existing text-based baselines (gpt-5.1.json).
    # Final filename: gpt-5.1_native.json (single underscore).
    if output_suffix is None and use_tool_calling:
        output_suffix = "native"

    save_name = clean_model_name(model)
    output_filename = f"{save_name}.json"
    # For multi-turn scenarios, append _t{T}_g{G}_p{P} so configs that share
    # a scenario (e.g. combined t=5 vs combined t=7) don't collide on disk.
    if not is_single_turn:
        output_suffix = (output_suffix or "") + (
            f"_t{num_turns}_g{num_switches}_p{num_revisions}"
        )
    if output_suffix:
        output_filename = output_filename.replace(".json", f"_{output_suffix}.json")

    exp_dir = EXPERIMENTS_DIR / _experiments_subdir(source, scenario) / "swe_bench_verified"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / output_filename

    existing: dict[str, Any] = {}
    failed_ids: list[str] | None = None
    if output_path.exists():
        if extend:
            with open(output_path) as f:
                existing = json.load(f)
            already_done = set(existing.keys())
            data = [d for d in data if d.get("task_id") not in already_done]
            if not data:
                total = len(existing)
                correct = sum(1 for r in existing.values() if r.get("correct"))
                print(f"✅ All requested task_ids already in {output_path} ({correct}/{total}). Skip.")
                return {"model": model, "status": "skipped", "path": str(output_path)}
            print(f"🧩 Extend mode: keeping {len(already_done)} existing, running {len(data)} new in {output_path}")
        elif rerun_failed:
            with open(output_path) as f:
                existing = json.load(f)
            failed_ids = [
                tid for tid, r in existing.items()
                if not r.get("success", True) or r.get("error") is not None
                or (r.get("swe_eval") or {}).get("harness_error") is not None
            ]
            if not failed_ids:
                total = len(existing)
                correct = sum(1 for r in existing.values() if r.get("correct"))
                print(f"✅ No failed samples in {output_path} ({correct}/{total}). Skip.")
                return {"model": model, "status": "skipped", "path": str(output_path)}
            print(f"🔄 Rerunning {len(failed_ids)} failed samples in {output_path}")
        else:
            print(f"⏭️  Skipping (already exists): {output_path}")
            return {"model": model, "status": "skipped", "path": str(output_path)}

    if failed_ids is not None:
        data = [d for d in data if d.get("task_id") in set(failed_ids)]

    print("=" * 60)
    print(f"SWE-bench Verified mini-agent baseline ({source})")
    print(f"Scenario: {scenario}")
    if not is_single_turn:
        print(f"Multi-turn config: t={num_turns} g={num_switches} p={num_revisions} ordering={ordering}")
    print(f"Model: {model}")
    print(f"Samples: {len(data)}")
    print(f"step_limit={step_limit}, cost_limit={cost_limit}")
    if step_limit_per_turn is not None:
        print(f"step_limit_per_turn={step_limit_per_turn} (per-turn cap; legacy step_limit ignored)")
    print(f"Scaffold mode: {'native (function calling)' if use_tool_calling else 'textbased (regex parsing)'}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Per-model SWEHarness instance + per-model run_id namespace. This
    # gives each concurrent model its own threading.Lock so verify-time
    # contention is per-model rather than global within the launcher
    # process. Only relevant when multiple models share one launcher
    # process (multi-process launching gives the same isolation
    # automatically, since each process has its own SWEHarness).
    harness = SWEHarness()
    harness_run_id = f"swe_{clean_model_name(model)}"

    # Results buffer: starts with whatever extend/rerun_failed loaded so
    # incremental saves always reflect the full merged state. The
    # ``existing`` dict has already been used to filter ``data``; from
    # here on it's just the seed for ``results``.
    results: dict[str, Any] = dict(existing)
    write_lock = threading.Lock()

    def _save_incremental() -> None:
        """Atomically dump the current results dict to ``output_path``.

        Uses a tmp + rename so an interruption (SIGINT, SIGKILL, OOM,
        host reboot) can never leave a half-written JSON on disk: the
        previous file remains intact until rename succeeds.
        """
        with write_lock:
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            tmp.replace(output_path)

    # Make sure the file exists from the start so other tooling can
    # poll it while the run is in progress (and `--extend` on a later
    # invocation finds something even if we crash before the first
    # task completes).
    _save_incremental()

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = {
                ex.submit(
                    evaluate_one, sample, model, source, harness,
                    cost_limit=cost_limit, step_limit=step_limit,
                    step_limit_per_turn=step_limit_per_turn,
                    use_tool_calling=use_tool_calling,
                    reasoning_effort=reasoning_effort,
                ): sample
                for sample in data
            }
            from tqdm import tqdm
            for fut in tqdm(as_completed(futs), total=len(data), desc=model):
                try:
                    r = fut.result()
                    results[r["task_id"]] = r
                    _save_incremental()
                except Exception as e:
                    s = futs[fut]
                    print(f"\n❌ {s.get('task_id')}: {e}")
    else:
        from tqdm import tqdm
        for sample in tqdm(data, desc=model):
            try:
                r = evaluate_one(
                    sample, model, source, harness,
                    cost_limit=cost_limit, step_limit=step_limit,
                    step_limit_per_turn=step_limit_per_turn,
                    use_tool_calling=use_tool_calling,
                    reasoning_effort=reasoning_effort,
                )
                results[r["task_id"]] = r
                _save_incremental()
            except Exception as e:
                print(f"\n❌ {sample.get('task_id')}: {e}")

    # Final save (no-op if we've been saving incrementally, but cheap
    # insurance and writes the same atomic file).
    _save_incremental()

    total = len(results)
    correct = sum(1 for r in results.values() if r.get("correct"))
    failed = sum(1 for r in results.values() if not r.get("success", True))
    accuracy = correct / total if total else 0.0
    print(f"\n📊 {model} ({source}): {correct}/{total} = {accuracy:.1%}  failed={failed}")

    return {
        "model": model, "source": source,
        "total": total, "correct": correct, "failed": failed,
        "accuracy": accuracy, "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--source", choices=_VALID_SOURCES, default="paraphrase",
                   help="'raw' (problem_statement) or 'paraphrase' (fully_specified_question, default).")
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=1,
                   help="Parallel agent runs. Each spins up its own docker container.")
    p.add_argument("--task_ids_file", type=str, default=None)
    p.add_argument("--cost_limit", type=float, default=3.0,
                   help="Per-instance cost cap (USD). Default 3.0 matches mini-swe-agent v2 swebench.yaml. "
                        "Note: the API doesn't surface per-call cost, so this guard never fires; "
                        "step_limit is the binding constraint.")
    p.add_argument("--step_limit", type=int, default=250,
                   help="Legacy per-instance global step cap. Default 250 matches mini-swe-agent v2 swebench.yaml leaderboard config. "
                        "When --step_limit_per_turn is also passed, this value is ignored (per-turn cap wins).")
    p.add_argument("--step_limit_per_turn", type=int, default=50,
                   help="Per-turn step cap for multi-turn EvolvingIntent scenarios. Each intent turn gets a fresh "
                        "budget of N steps (mini's agent.n_calls is reset to 0 at each turn boundary). "
                        "Default 50; empirically fully-specified GPT-5.1 used median 24 / max 61 steps so "
                        "50 covers ~94%% of single-turn trajectories. Pass an explicit value or 0 (= unlimited "
                        "per turn, mirrors mini's step_limit=0) to override. Set --step_limit_per_turn -1 to "
                        "fall back to legacy single-cap mode using --step_limit.")
    p.add_argument("--rerun_failed", action="store_true")
    p.add_argument("--extend", action="store_true",
                   help="Keep existing per-sample results; only run task_ids not yet present.")
    p.add_argument("--output_suffix", type=str, default=None)
    p.add_argument("--reasoning_effort", type=str, default="medium",
                   choices=["minimal", "low", "medium", "high"],
                   help="Reasoning effort for GPT-5.x reasoning models. Default 'medium' "
                        "matches mini-swe-agent v2 leaderboard config (metadata.yaml: "
                        "'GPT-5.1 (2025-11-13) (medium reasoning)').")
    p.add_argument("--use_tool_calling", action="store_true",
                   help="Route LLM calls through native OpenAI function calling "
                        "(LLMToolModel + swebench.yaml). Default off — uses "
                        "the regex-on-fenced-blocks scaffold (LLMTextbasedModel + "
                        "swebench_backticks.yaml). When set, output_suffix defaults to "
                        "'native' so files don't collide with text-based baselines.")
    # Multi-turn EvolvingIntent scenario flags (mirrors run_experiment.py). Default
    # 0/0/0 keeps the legacy single-turn JSON-loading path → preserves the
    # validated 64% native baseline byte-for-byte. Any non-default flips the
    # runner into EvolvingIntent sample-generation mode.
    p.add_argument("--num_turns", type=int, default=1,
                   help="Number of conversation turns. 1 = fully-specified (default).")
    p.add_argument("--num_revisions", type=int, default=0,
                   help="Number of revisions (revision scenario when >0).")
    p.add_argument("--num_switches", type=int, default=0,
                   help="Number of function switches (switch scenario when >0).")
    p.add_argument("--ordering", type=str, default="interleaved",
                   choices=["sequential", "interleaved", "mixed", "random"],
                   help="EvolvingIntent turn-ordering policy.")
    p.add_argument("--prefix_style", type=str, default="base",
                   choices=["base", "function-naturalized", "function-naturalized-v2"])
    p.add_argument("--naturalizer_model", type=str, default=None,
                   help="Optional online naturalizer model.")
    args = p.parse_args()

    summaries = []
    # Sentinel -1 means "fall back to legacy single-cap mode" (use --step_limit
    # as a single global trajectory cap). Any other value (including 0) is
    # treated as the per-turn cap.
    step_limit_per_turn = None if args.step_limit_per_turn == -1 else args.step_limit_per_turn
    for model in args.models:
        s = run(
            data_path=args.data_path,
            model=model,
            source=args.source,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            task_ids_file=args.task_ids_file,
            cost_limit=args.cost_limit,
            step_limit=args.step_limit,
            step_limit_per_turn=step_limit_per_turn,
            rerun_failed=args.rerun_failed,
            extend=args.extend,
            output_suffix=args.output_suffix,
            use_tool_calling=args.use_tool_calling,
            reasoning_effort=args.reasoning_effort,
            num_turns=args.num_turns,
            num_revisions=args.num_revisions,
            num_switches=args.num_switches,
            ordering=args.ordering,
            prefix_style=args.prefix_style,
            naturalizer_model=args.naturalizer_model,
        )
        summaries.append(s)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for s in summaries:
        if s.get("status") == "skipped":
            print(f"  {s['model']:30s}  skipped ({s['path']})")
        else:
            print(f"  {s['model']:30s}  {s['source']:11s}  {s['correct']}/{s['total']} = {s['accuracy']:.1%}")


if __name__ == "__main__":
    main()
