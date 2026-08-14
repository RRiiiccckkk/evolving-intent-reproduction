"""LLM-judge re-grade for BIRD-SQL: catch semantically-equivalent answers.

Identifies samples that strict execution match marks wrong but where the
model and gold results are semantically equivalent (e.g. 'NY' vs 'New York',
('Wayne', 'Gretzky') vs ('Wayne Gretzky',), state code vs state name,
team abbreviation vs full team name).

Only samples currently marked correct=False AND that successfully executed
both gold and model SQL are sent to the judge. Samples already correct
stay correct.

Writes a sibling key `correct_lenient` (and updates `correct` if
--update-correct passed). Default: dry run reporting deltas without
updating the `correct` field.

Usage:
    python llm_judge_bird_sql.py PATH [--update-correct] [--judge-model gpt-5.1]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from intent_construction.intent_extraction.core.llm_utils import generate_text  # type: ignore  # noqa: E402


JUDGE_PROMPT = """You are evaluating whether two SQL query result sets answer \
the same natural-language question equivalently.

Question: {question}
{evidence_block}
Gold result (one row per line, columns separated by " | "):
{gold_text}

Model result:
{model_text}

Decide if the model result is semantically equivalent to the gold result \
*as an answer to the natural-language question*. They count as equivalent if \
they refer to the same real-world entity/value, even when the form differs:

- Code vs full name for the same entity ('NY' vs 'New York', 'MIL' vs \
'Milwaukee Bucks', 'MA' vs 'Massachusetts').
- Split-name columns vs concatenated name (('Wayne','Gretzky') vs \
'Wayne Gretzky' or 'Wayne Douglas Gretzky').
- Different but equivalent column packaging when the answer entities match \
(e.g. extra ID columns next to the name, or counts that match the question).
- Numeric values within a tiny rounding tolerance (1e-3 relative).

They are NOT equivalent if:
- Different real-world entities (different city, person, team).
- Different numeric values beyond rounding.
- Model is missing rows the gold has, or returns extra unrelated rows.

Reply with strict JSON: {{"equivalent": true|false, "reason": "<brief>"}}.
"""


def execute(db: str, sql: str, timeout: int = 10) -> tuple[list | None, str | None]:
    try:
        conn = sqlite3.connect(db, timeout=timeout)
        rows = conn.execute(sql).fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def format_rows(rows: list, max_rows: int = 20) -> str:
    if not rows:
        return "(empty)"
    lines = []
    for r in rows[:max_rows]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    if len(rows) > max_rows:
        lines.append(f"... and {len(rows) - max_rows} more rows")
    return "\n".join(lines)


def judge_one(
    tid: str,
    entry: dict,
    src: dict,
    judge_model: str,
) -> tuple[str, bool | None, str]:
    """Returns (tid, lenient_correct or None, reason).

    None means the judge could not be invoked (e.g. no model SQL, exec error
    on either side); the caller falls back to the strict label.
    """
    db_path = src.get("db_path", "")
    gold_sql = src.get("gold_sql", "")
    if not db_path or not gold_sql:
        return tid, None, "missing db_path or gold_sql"

    msql = (entry.get("per_turn_results") or [{}])[-1].get("model_sql", "")
    if not msql:
        return tid, False, "no model SQL"

    gold_rows, ge = execute(db_path, gold_sql)
    if ge or gold_rows is None:
        return tid, None, f"gold exec error: {ge}"
    model_rows, me = execute(db_path, msql)
    if me or model_rows is None:
        return tid, False, f"model exec error: {me}"

    question = src.get("fully_specified_question") or src.get("function", "")
    evidence = (src.get("evidence") or "").strip()
    evidence_block = f"Evidence: {evidence}\n\n" if evidence else ""

    prompt = JUDGE_PROMPT.format(
        question=question,
        evidence_block=evidence_block,
        gold_text=format_rows(gold_rows),
        model_text=format_rows(model_rows),
    )
    try:
        resp = generate_text(
            messages=[{"role": "user", "content": prompt}],
            model=judge_model,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception as e:
        return tid, None, f"judge call failed: {e}"

    # Parse JSON from response
    s = resp.strip()
    # Strip code fences if present
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        # Find first { ... } span
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(s[start : end + 1])
            return tid, bool(data.get("equivalent")), str(data.get("reason", ""))[:200]
    except Exception as e:
        return tid, None, f"parse error: {e} | raw: {resp[:120]}"
    return tid, None, f"unparseable: {resp[:120]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Result JSON to re-grade")
    p.add_argument(
        "--judge-model", default="gpt-5.1",
        help="LLM judge model. Default gpt-5.1 (cheap).",
    )
    p.add_argument(
        "--workers", type=int, default=10,
        help="Parallel judge calls.",
    )
    p.add_argument(
        "--update-correct", action="store_true",
        help="Overwrite the `correct` field in the result file.",
    )
    args = p.parse_args()

    result_path = Path(args.path).resolve()
    flipped, judge_count, skipped, strict, lenient, n = run_judge_on_file(
        result_path,
        judge_model=args.judge_model,
        workers=args.workers,
        update_correct=args.update_correct,
        verbose=True,
    )
    print(
        f"\nTotal: {n}  strict={strict}/{n} ({strict/n:.1%})  "
        f"lenient={lenient}/{n} ({lenient/n:.1%})  "
        f"flipped={flipped}  judge_calls={judge_count}  skipped={skipped}"
    )


def run_judge_on_file(
    result_path: Path,
    judge_model: str = "gpt-5.1",
    workers: int = 10,
    update_correct: bool = True,
    verbose: bool = False,
) -> tuple[int, int, int, int, int, int]:
    """Run LLM judge over an existing BIRD-SQL result JSON file.

    Returns (flipped, judge_calls, skipped, strict_n, lenient_n, total).
    """
    result_path = Path(result_path)
    data = json.loads(result_path.read_text())

    # Best-effort dataset lookup: prefer the path embedded in the result
    # metadata, fall back to the canonical n100 file.
    ds_path: Path | None = None
    for entry in data.values():
        m = entry.get("metadata") or {}
        if m.get("data_source") == "bird_sql":
            cand = REPO / "final_dataset" / "bird_sql_n100.json"
            if cand.exists():
                ds_path = cand
            break
    if ds_path is None:
        ds_path = REPO / "final_dataset" / "bird_sql_n100.json"
    if not ds_path.exists():
        # Cannot judge without dataset
        n = len(data)
        s = sum(1 for v in data.values() if v.get("correct"))
        return 0, 0, n, s, s, n

    dataset = json.loads(ds_path.read_text())
    by_tid = {s["task_id"]: s for s in dataset}

    targets = [
        (tid, entry) for tid, entry in data.items()
        if not entry.get("correct")
    ]
    if verbose:
        print(f"Judging {len(targets)} incorrect samples in {result_path.name}\n")

    n = len(data)
    strict_n = sum(1 for v in data.values() if v.get("correct"))

    flipped = 0
    skipped = 0
    judge_count = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(judge_one, tid, entry, by_tid.get(tid, {}), judge_model): tid
            for tid, entry in targets
        }
        for fut in as_completed(futures):
            tid, verdict, reason = fut.result()
            entry = data[tid]
            if verdict is None:
                skipped += 1
                continue
            judge_count += 1
            entry["correct_lenient"] = bool(verdict)
            if verdict:
                flipped += 1
                if verbose:
                    print(f"  ✅ FLIP  {tid}  {reason[:80]}")

    for tid, entry in data.items():
        if entry.get("correct"):
            entry.setdefault("correct_lenient", True)

    if update_correct:
        for tid, entry in data.items():
            if "correct_lenient" in entry:
                entry["correct"] = bool(entry["correct_lenient"])

    result_path.write_text(json.dumps(data, indent=2))

    lenient = sum(1 for v in data.values() if v.get("correct_lenient"))
    return flipped, judge_count, skipped, strict_n, lenient, n


if __name__ == "__main__":
    main()
