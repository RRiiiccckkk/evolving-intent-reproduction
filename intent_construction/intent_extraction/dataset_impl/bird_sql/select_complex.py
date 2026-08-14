"""Select complex BIRD-SQL queries for the predecessor/counterfactual generation redesign.

Pipeline Stage 2.

Loads BIRD train (~9428) + dev (~1534) splits, parses each gold SQL, scores by
function-clause + argument-clause richness, applies hard feasibility filters,
and writes the top-N samples as a JSON list ready to be fed into the
existing :class:`BirdSqlExtractor`.

Usage
-----
    python -m intent_extraction.dataset_impl.bird_sql.select_complex \
        --n 120 \
        --out intent_extraction/dataset_impl/bird_sql/data/selected_complex_v1.json

Notes
-----
* Output samples have ``gold_sql`` (renamed from BIRD's ``SQL`` field) plus
  ``source`` ('train'|'dev') and ``index`` (position in the source split).
* Subqueries / CTEs are excluded for v1 (LLM function rewrite is harder there).
* Each sample's gold SQL is executed once to confirm it returns ≥ 1 non-NULL
  row before scoring (drops broken samples). Set ``--no_exec_check`` to skip.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

_THIS_DIR = Path(__file__).parent

from intent_construction.intent_extraction.dataset_impl.bird_sql.db_utils import execute_sql, validate_result  # noqa: E402
from intent_construction.intent_extraction.dataset_impl.bird_sql.sql_parser import parse_sql, ParsedSQL       # noqa: E402


# =============================================================================
# Paths
# =============================================================================

_DATA_DIR = _THIS_DIR / "data"
DEV_JSON = _DATA_DIR / "dev_extracted" / "dev_20240627" / "dev.json"
DEV_DB_DIR = _DATA_DIR / "dev_extracted" / "dev_20240627" / "dev_databases"
TRAIN_JSON = _DATA_DIR / "train_extracted" / "train" / "train.json"
TRAIN_DB_DIR = _DATA_DIR / "train_extracted" / "train" / "train_databases"


# =============================================================================
# Scoring weights (tunable)
# =============================================================================

W_AGG_PRESENT = 2          # SELECT contains an aggregate
W_GROUP_BY_COL = 2         # per GROUP BY column
W_ORDER_BY_COL = 1         # per ORDER BY column
W_JOIN = 2                 # per JOIN
W_WHERE_PRED = 1           # per WHERE predicate
W_HAVING = 3               # any HAVING (binary bonus)
W_LIMIT = 2                # any LIMIT (binary bonus)


def score(parsed: ParsedSQL) -> int:
    """Compute the additive complexity score for a parsed SQL query."""
    return (
        W_AGG_PRESENT * (1 if parsed.goal.aggregate else 0)
        + W_GROUP_BY_COL * len(parsed.group_by_columns)
        + W_ORDER_BY_COL * len(parsed.order_by_columns)
        + W_JOIN * len(parsed.joins)
        + W_WHERE_PRED * len(parsed.conditions)
        + W_HAVING * (1 if parsed.has_having else 0)
        + W_LIMIT * (1 if parsed.limit is not None else 0)
    )


def function_clause_categories(parsed: ParsedSQL) -> int:
    """Count how many of {SELECT-agg, GROUP BY, ORDER BY, JOIN} are non-empty."""
    return sum([
        1 if parsed.goal.aggregate else 0,
        1 if parsed.group_by_columns else 0,
        1 if parsed.order_by_columns else 0,
        1 if parsed.joins else 0,
    ])


def condition_clause_categories(parsed: ParsedSQL) -> int:
    """Count how many of {WHERE, HAVING, LIMIT} are non-empty."""
    return sum([
        1 if parsed.conditions else 0,
        1 if parsed.has_having else 0,
        1 if parsed.limit is not None else 0,
    ])


# =============================================================================
# Data loading
# =============================================================================

def _resolve_db_path(db_id: str, source: str) -> Path:
    base = DEV_DB_DIR if source == "dev" else TRAIN_DB_DIR
    return base / db_id / f"{db_id}.sqlite"


def _normalize_sample(raw: dict[str, Any], source: str, idx: int) -> dict[str, Any]:
    """Convert a raw BIRD sample into the schema expected by BirdSqlExtractor."""
    return {
        "db_id": raw["db_id"],
        "question": raw["question"],
        "evidence": raw.get("evidence", ""),
        "gold_sql": raw["SQL"],
        "difficulty": raw.get("difficulty", ""),
        "source": source,
        "index": idx,
        "question_id": raw.get("question_id", idx),
    }


def load_all_samples() -> list[dict[str, Any]]:
    """Load BIRD train + dev splits and merge into a single list."""
    samples: list[dict[str, Any]] = []
    for path, src in ((TRAIN_JSON, "train"), (DEV_JSON, "dev")):
        if not path.exists():
            print(f"[warn] missing split file: {path}")
            continue
        with open(path) as f:
            rows = json.load(f)
        for i, row in enumerate(rows):
            samples.append(_normalize_sample(row, src, i))
        print(f"  loaded {len(rows)} {src} samples from {path.name}")
    return samples


# =============================================================================
# Per-sample scoring (process-pool worker)
# =============================================================================

def _score_one(sample: dict[str, Any], do_exec_check: bool) -> dict[str, Any] | None:
    """Score a single sample; return ``None`` on hard-filter rejection.

    Returns enriched sample with ``_score`` and per-clause counts when accepted.
    """
    sql = sample["gold_sql"]
    parsed = parse_sql(sql)

    if not parsed.parseable:
        return None
    # v1: skip subqueries / CTEs / UNION
    if parsed.has_subquery or parsed.has_cte or parsed.has_union:
        return None

    db_path = _resolve_db_path(sample["db_id"], sample["source"])
    if not db_path.exists():
        return None

    if do_exec_check:
        result = execute_sql(str(db_path), sql, timeout=10)
        if not validate_result(result):
            return None

    s = score(parsed)
    enriched = dict(sample)
    enriched["_score"] = s
    enriched["_n_function_categories"] = function_clause_categories(parsed)
    enriched["_n_cond_categories"] = condition_clause_categories(parsed)
    enriched["_n_joins"] = len(parsed.joins)
    enriched["_n_where_preds"] = len(parsed.conditions)
    enriched["_n_group_by"] = len(parsed.group_by_columns)
    enriched["_n_order_by"] = len(parsed.order_by_columns)
    enriched["_has_having"] = parsed.has_having
    enriched["_has_limit"] = parsed.limit is not None
    enriched["_aggregate"] = parsed.goal.aggregate
    return enriched


def _worker(args: tuple[dict[str, Any], bool]) -> dict[str, Any] | None:
    sample, do_exec = args
    try:
        return _score_one(sample, do_exec)
    except Exception:
        return None


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120, help="Top-N samples to keep.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(_DATA_DIR / "selected_complex_v1.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        default=None,
        help="Optional CSV with per-sample complexity stats (defaults to <out>.stats.csv).",
    )
    parser.add_argument("--no_exec_check", action="store_true", help="Skip gold SQL execution validation.")
    parser.add_argument("--num_workers", type=int, default=8, help="Process-pool workers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading BIRD splits ...")
    samples = load_all_samples()
    print(f"Total candidates: {len(samples)}")

    do_exec = not args.no_exec_check
    print(f"Scoring ({'with' if do_exec else 'without'} gold-SQL execution check) ...")

    accepted: list[dict[str, Any]] = []
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(_worker, (s, do_exec)) for s in samples]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="score"):
                r = fut.result()
                if r is not None:
                    accepted.append(r)
    else:
        for s in tqdm(samples, desc="score"):
            r = _worker((s, do_exec))
            if r is not None:
                accepted.append(r)

    print(f"After filters: {len(accepted)} candidates remain")

    # Sort by (score desc, multi-category bonus desc, deterministic tie-break)
    accepted.sort(
        key=lambda x: (
            -x["_score"],
            -x["_n_function_categories"],
            -x["_n_cond_categories"],
            x["source"],
            x["index"],
        )
    )

    top = accepted[: args.n]
    print(f"Selected top-{len(top)} (target was {args.n})")

    # ── Audit prints ──
    print("\n=== Selected complexity histogram ===")
    cat_hist: dict[int, int] = {}
    for s in top:
        cat_hist[s["_n_function_categories"]] = cat_hist.get(s["_n_function_categories"], 0) + 1
    print("  #function_clause_categories → count:")
    for k in sorted(cat_hist):
        print(f"    {k}: {cat_hist[k]}")

    n_having = sum(1 for s in top if s["_has_having"])
    n_limit = sum(1 for s in top if s["_has_limit"])
    n_join = sum(1 for s in top if s["_n_joins"] > 0)
    n_group = sum(1 for s in top if s["_n_group_by"] > 0)
    n_order = sum(1 for s in top if s["_n_order_by"] > 0)
    n_agg = sum(1 for s in top if s["_aggregate"])
    n = len(top)
    print(f"\n  HAVING:        {n_having}/{n} ({100*n_having/max(1,n):.1f}%)")
    print(f"  LIMIT:         {n_limit}/{n} ({100*n_limit/max(1,n):.1f}%)")
    print(f"  JOIN:          {n_join}/{n}")
    print(f"  GROUP BY:      {n_group}/{n}")
    print(f"  ORDER BY:      {n_order}/{n}")
    print(f"  SELECT agg:    {n_agg}/{n}")

    multi_function = sum(1 for s in top if s["_n_function_categories"] >= 2)
    print(f"\n  ≥2 function categories:  {multi_function}/{n} ({100*multi_function/max(1,n):.1f}%)")
    rich_cond = sum(1 for s in top if s["_has_having"] or s["_has_limit"])
    print(f"  HAVING or LIMIT:    {rich_cond}/{n} ({100*rich_cond/max(1,n):.1f}%)")

    # Verification thresholds
    if multi_function < 0.8 * n:
        print("\n[WARN] <80% of selected samples have ≥2 function categories — consider raising weights.")
    if rich_cond < 0.5 * n:
        print("[WARN] <50% of selected samples have HAVING or LIMIT — consider raising weights.")

    # Strip internal _* fields before writing output
    out_records = [
        {k: v for k, v in s.items() if not k.startswith("_")} for s in top
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_records, f, indent=2)
    print(f"\n✓ Wrote {len(out_records)} samples → {out_path}")

    stats_csv = args.stats_csv or (str(out_path) + ".stats.csv")
    with open(stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "source", "index", "db_id", "score",
            "n_function_categories", "n_cond_categories",
            "n_joins", "n_where_preds", "n_group_by", "n_order_by",
            "has_having", "has_limit", "aggregate",
        ])
        for s in top:
            w.writerow([
                s["source"], s["index"], s["db_id"], s["_score"],
                s["_n_function_categories"], s["_n_cond_categories"],
                s["_n_joins"], s["_n_where_preds"], s["_n_group_by"], s["_n_order_by"],
                int(s["_has_having"]), int(s["_has_limit"]), s["_aggregate"] or "",
            ])
    print(f"✓ Wrote stats → {stats_csv}")


if __name__ == "__main__":
    main()
