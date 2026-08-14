# Function Predecessor

Generates **related but different functions (questions)** that share arguments with the original problem.

Function predecessor uses **predecessor inference**: causally-linked predecessor functions generated via chained predecessor generation.

---

## Predecessor Inference (`generate_predecessors.py`)

Generates **causally-linked predecessor functions** via chained predecessor generation. Each predecessor's answer naturally enables or motivates the next question.

### Dataset-Specific Prompts

Predecessor inference selects a dataset-specific prompt via `--dataset_type`:

| `--dataset_type` | Prompt File | Datasets |
|------------------|-------------|----------|
| `browsecomp` | `generate_predecessor_browsecomp.txt` | browsecomp_plus |
| `gsm8k` | `generate_predecessor_gsm8k.txt` | gsm8k |
| `default` | `generate_predecessor_default.txt` | other / general |

If a prompt for the given type is not found, it falls back to `default`.

### Key Idea

Given the final function, generate plausible **predecessor functions** predecessor in a chain:

```
Final turn (t)  →  generate t-1 from t  →  generate t-2 from t-1 (not t!)
```

Each predecessor gets a **new function**, **shared arguments** (from successor), **new fabricated arguments**, and a **causal link** to the next turn.

**Why chain from t-1 (not t)?** Like a reverse MDP — chaining avoids independent/overlapping predecessors. When generating t-2, the LLM also receives the full future chain for overlap prevention.

### Chain Archetypes

Archetypes are organized by a four-type taxonomy (T1–T4) and are dataset-specific. The set used for a run depends on `--dataset_type` (and can be restricted via `--chain_types`).

**Search (browsecomp):**

- `identify_then_seek` (T1) — identify entity A, then ask about a different entity B that A's identity helps pin down
- `survey_then_focus` (T2) — broad set question, then narrow to one member of that set
- `trace_then_follow` (T3) — ask for A, then a follow-up that mechanically requires A's answer
- `pivot_inquiry` (T4) — same scenario, but the user changes their mind about what to ask

**Math (gsm8k):**

- `lookup_then_compute` (T1) — look up a rate/price, then compute a total using it
- `total_then_component` (T2) — ask for a total, then for a specific component of it
- `compute_then_extend` (T3) — compute an intermediate value, then use it for a final value
- `reframe_problem` (T4) — compute a quantity under one framing, then under a different framing

### Usage

```bash
# Search (browsecomp_plus)
python generate_predecessors.py \
    --input ../../intent_extraction/output/browsecomp_plus/extracted_test.json \
    --output output/browsecomp_plus/predecessor.json \
    --dataset_type browsecomp \
    --num_predecessors 2 \
    --model gpt-5.1 \
    --fallback_model gpt-5.1 \
    --parallel 20

# Math (gsm8k)
python generate_predecessors.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/predecessor.json \
    --dataset_type gsm8k \
    --num_predecessors 2 \
    --model gpt-5.1 \
    --fallback_model gpt-5.1 \
    --parallel 20

# Resume from checkpoint (any dataset)
python generate_predecessors.py \
    --input ../../intent_extraction/output/browsecomp_plus/extracted_test.json \
    --output output/browsecomp_plus/predecessor.json \
    --dataset_type browsecomp \
    --resume
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | required | Path to extracted data JSON |
| `--output` | str | required | Path to output JSON file |
| `--dataset_type` | str | browsecomp | Dataset type: `browsecomp`, `gsm8k`, `default` |
| `--num_predecessors` | int | 2 | Number of predecessor turns per sample |
| `--model` | str | gpt-5.1 | Base model for generation and verification |
| `--fallback_model` | str | None | Stronger model for escalation on failure |
| `--chain_types` | list | all | Chain archetypes to use (cycles through them) |
| `--share_num` | int | None | Exact number of shared arguments (auto if None) |
| `--num_samples` | int | None | Number of samples to process (default: all) |
| `--parallel` | int | 1 | Number of parallel workers |
| `--checkpoint_interval` | int | 50 | Save checkpoint every N samples |
| `--resume` | flag | False | Resume from checkpoint if exists |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--seed` | int | 42 | Random seed |

### Validation

Five programmatic checks run during generation:
1. **Exact dedup** — no duplicate functions in the chain
2. **Semantic overlap** — word overlap < 60% between any two functions
3. **Dangling references** — no "this author", "that person" without antecedent
4. **Entity type + overlap** — same entity type with > 40% word overlap → reject
5. **Question length** — max 35 words per question

Plus post-hoc **cross-turn relevance verification**: fabricated arguments from one turn must not help answer another turn's question. Failing samples are regenerated (up to 2 attempts) or discarded.

### Output Format

Each sample includes both `predecessors` (detailed) and `predecessor_functions` (simulator-compatible):

```json
{
  "task_id": "extracted-browsecomp_plus-test-629",
  "original_function": "What is the name of the historical landmark?",
  "answer": "Captain John A. Sutter's Landing",
  "arguments": ["..."],
  "chain_type": "discover_and_pivot",
  "predecessors": [
    {
      "predecessor_function": "Which regional historical figure has a major U.S. monument?",
      "entity_sought": "person name",
      "full_arguments": [
        {"argument_id": 1, "argument": "...", "is_shared": true},
        {"argument_id": 105, "argument": "...", "is_shared": false}
      ],
      "shared_argument_ids": [1, 2],
      "new_arguments": ["The monument commemorates a significant regional figure..."],
      "transition_type": "discover_and_pivot",
      "causal_link": "Finding the historical figure reveals the monument..."
    }
  ],
  "predecessor_functions": ["..."],
  "verification_passed": true,
  "verification_details": ["..."]
}
```

The `predecessor_functions` field is compatible with `user_simulation.py`. Each predecessor function has `is_predecessor: true`, which causes the simulator to use **follow-up style prefixes** ("Following up on that,") instead of correction-style prefixes ("Actually,").

---


## SQL-Specific Function Predecessor (BIRD-SQL)

SQL function predecessor modifies the gold SQL query (`SELECT` / `GROUP BY` / `ORDER BY` / `JOIN` clauses) while preserving its argument clauses, validating each rewrite by executing it against the database.

### SQL Function/Argument Classification

Natural language function/argument separation is intuitive (e.g., "Among professors who advise more than 2 students | who has the highest teaching ability?"), but SQL clauses don't map 1:1 — the same clause can be function or argument depending on context. As a practical approximation:

| Category | SQL Clauses | Rationale |
|----------|-------------|-----------|
| **Function** | `SELECT`, `GROUP BY`, `ORDER BY`, `JOIN` | Output shape: what to show, how to group, how to sort, what data to connect |
| **Argument** | `WHERE`, `HAVING`, `LIMIT` | Filters/constraints: row filter, group filter, result count limit |
| **Fixed** | `FROM` | Base entity; changing it creates an entirely different question |

### Implementation

Function changes should modify any combination of function clauses (`SELECT`, `GROUP BY`, `ORDER BY`, `JOIN`) while preserving argument clauses (`WHERE`, `HAVING`, `LIMIT`). Pipeline:

1. Filter BIRD-SQL for complex queries (those with GROUP BY, ORDER BY, multiple JOINs)
2. Modify function clauses at AST level (swap columns, aggregates, sort direction, join targets)
3. Keep argument clauses (WHERE, HAVING, LIMIT) intact
4. Execute modified SQL against DB to validate correctness
5. Generate **episodic** follow-up text via LLM (not full question — references prior context)

Example:
```
Turn 1: "Among professors who advise more than 2 students, who has the highest teaching ability?"
Turn 2: "Among those, who has the most publications instead?"
```

SQL changes substantially (new JOIN, SELECT, ORDER BY), but arguments (HAVING COUNT > 2) are preserved. The model must remember arguments from Turn 1 to answer Turn 2.

See `intent_construction/retrospective_expansion/predecessor/generate_predecessors_sql_llm.py` for the implementation.

---

## Prompts

| Prompt File | Approach | Purpose |
|-------------|----------|---------|
| `generate_predecessor_default.txt` | Predecessor | Predecessor generation (default template) |
| `generate_predecessor_gsm8k.txt` | Predecessor | Predecessor generation (gsm8k) |
| `generate_predecessor_math.txt` | Predecessor | Predecessor generation (math) |
| `generate_predecessor_browsecomp.txt` | Predecessor | Predecessor generation (search/browsecomp_plus) |
| `generate_predecessor_sql.txt` | Predecessor | SQL keep-arguments/change-function follow-up |
| `generate_impl_precursor_swe.txt` | Predecessor | SWE implementation-planning precursor |
| `cross_turn_relevance_check.txt` | Predecessor | Cross-turn argument independence verification |
| `similarity_check_default.txt` | Predecessor | Similarity checking (default) |
| `similarity_check_gsm8k.txt` | Predecessor | Similarity checking (gsm8k) |
| `exploration_function_swe.txt` | Predecessor | SWE orientation (G1) question generation |
| `naturalize_sql_followup.txt` | Predecessor | SQL follow-up naturalization |

## Retry Failed Samples

```bash
python retry_predecessor_failed.py \
    --input ../counterfactual/output/gsm8k/argument_counterfactual.json \
    --output output/gsm8k/predecessor.json \
    --model gpt-5.1
```
