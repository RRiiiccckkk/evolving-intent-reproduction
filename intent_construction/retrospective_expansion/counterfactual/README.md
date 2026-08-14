# Argument Counterfactual

Generates **counterfactual (similar but different)** versions of arguments in extracted data.

## Overview

After extraction, each sample contains:
- **Function**: The main question to answer
- **Arguments**: Key information needed to solve the problem (argument_id: 1, 2, ...)

This module generates **counterfactual versions** of each argument - variations that are similar but slightly different from the original. This is useful for:
- Data augmentation
- Testing model robustness
- Creating contrastive training examples

## Key Concept: Counterfactual Types

| Aspect | Argument Counterfactual | Function Predecessor |
|--------|------------------------|-------------------|
| Purpose | Generate value variations | Generate related questions |
| What changes | Numbers/values in arguments | The function itself |
| Original preserved? | ✅ Yes | ✅ Yes |
| Output field | `counterfactual_arguments` | `predecessor_functions` |
| Example | 48 → 52 friends | "total sold?" → "which month more?" |

See [../predecessor/README.md](../predecessor/README.md) for function predecessor.

## Usage

### Basic Usage

```bash
# Generate 2 counterfactuals per argument (default)
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/counterfactual.json
```

### Custom Number of Counterfactuals

```bash
# Generate 3 counterfactuals per argument
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/counterfactual.json \
    --num_counterfactuals 3
```

### Different Dataset Types

```bash
# For math problems
python generate_counterfactuals.py \
    --input input.json \
    --output output.json \
    --dataset_type math

# For search/retrieval problems (browsecomp_plus)
# Use gpt-5.1 for search — it avoids synonym/rephrase issues
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/browsecomp_plus/extracted_test.json \
    --output output/browsecomp_plus/argument_counterfactual.json \
    --dataset_type search \
    --model gpt-5.1 \
    --batch --batch_size 50
```

## Validation

Counterfactuals are validated programmatically before acceptance:

1. **Forward reconstruction**: `original.replace(orig_val, pert_val)` must exactly equal the counterfactual argument
2. **Containment**: `orig_val` must exist in original, `pert_val` in counterfactual
3. **Length ratio**: Counterfactual value can't be >2× original value length
4. **Article/apostrophe normalization**: Handles a/an and Unicode apostrophe differences

When validation fails, the error message is sent back to the LLM as feedback on retry (up to 5 attempts).

### Model Recommendations

| Dataset Type | Recommended Model | Notes |
|--------------|-------------------|-------|
| Math | gpt-5.1 | Unified OSS default |
| Search | gpt-5.1 | Text attribute swaps need stronger model to avoid synonyms |

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | required | Path to input extracted JSON file |
| `--output` | str | required | Path to output JSON file |
| `--num_counterfactuals` | int | 2 | Number of counterfactual arguments per original |
| `--dataset_type` | str | math | Type of dataset (math, search, default) |
| `--model` | str | gpt-5.1 | Model to use |
| `--num_samples` | int | None | Number of samples to process |
| `--seed` | int | 42 | Random seed |
| `--checkpoint_interval` | int | 50 | Save checkpoint every N samples |
| `--resume` | flag | False | Resume from checkpoint |
| `--batch` | flag | False | Enable parallel batch processing |
| `--batch_size` | int | 5 | Number of parallel workers for batch processing |

## Output Format

### Input (Extracted)
```json
{
  "task_id": "extracted-gsm8k-train-0",
  "question": "<original problem text>",
  "answer": "<gold answer>",
  "function": "<the main question being asked>",
  "arguments": [
    {"argument_id": 1, "argument": "<a fact with a numeric value V>"},
    {"argument_id": 2, "argument": "<another fact needed to solve>"}
  ]
}
```

### Output (Counterfactual)
```json
{
  "task_id": "extracted-gsm8k-train-0",
  "question": "<original problem text>",
  "answer": "<gold answer>",
  "function": "<the main question being asked>",
  "arguments": [
    {
      "argument_id": 1,
      "argument": "<a fact with a numeric value V>",
      "counterfactual_arguments": [
        {
          "counterfactual_argument": "<same fact with value V1>",
          "original_value": "V",
          "counterfactual_value": "V1",
          "reasoning": "Changed the numeric value from V to V1..."
        },
        {
          "counterfactual_argument": "<same fact with value V2>",
          "original_value": "V",
          "counterfactual_value": "V2",
          "reasoning": "Changed the numeric value from V to V2..."
        }
      ]
    },
    {"argument_id": 2, "argument": "...", "counterfactual_arguments": [...]}
  ],
  "counterfactual_info": {
    "num_counterfactuals_requested": 2,
    "total_arguments": 2,
    "successful_counterfactuals": 4,
    "dataset_type": "math"
  }
}
```

## SQL (BIRD-SQL): Programmatic Counterfactual

For BIRD-SQL we use a deterministic, programmatic generator,
[`generate_counterfactuals_sql.py`](./generate_counterfactuals_sql.py), with **zero LLM
dependency**. It swaps WHERE-clause values with real alternative values pulled
directly from the database and validates every counterfactual by executing the
counterfactual SQL.

### Pipeline (per sample)

1. **Build slots.** Take every WHERE predicate from the stage-1 `arguments`
   array.
2. **DB-ground each slot.** Pull a pool of plausible alternative values
   directly from the actual database column.
3. **Swap and validate.** Each candidate is swapped into the gold SQL,
   executed against the SQLite DB, and accepted only when the result is
   **non-empty** AND the counterfactual final answer **differs** from the original.
4. **Emit.** NL phrasing for each accepted counterfactual is produced via fixed
   templates (no model call). Output schema feeds directly into `predecessor`
   and `situated_simulation`.

### Usage

```bash
python generate_counterfactuals_sql.py \
    --input ../../intent_extraction/output/bird_sql/extracted.json \
    --output output/bird_sql/argument_counterfactual.json \
    --num_counterfactuals 3 \
    --num_workers 4
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | required | Stage-1 extracted BIRD-SQL JSON |
| `--output` | str | required | Output JSON path |
| `--num_counterfactuals` | int | 3 | Max counterfactuals per argument |
| `--num_workers` | int | 4 | Parallel sample workers |
| `--num_samples` | int | None | Limit to first N samples |
| `--seed` | int | 42 | Random seed |
| `--checkpoint_interval` | int | 50 | Save a resumable checkpoint every N samples |
| `--resume` | flag | False | Resume from a previous checkpoint |
| `--sql_timeout` | int | 30 | Per-query SQLite timeout (seconds) |

## Combining with Function Predecessor

You can run both `counterfactual` and `predecessor` on the same data. They add separate fields and don't overwrite each other:

```bash
# Step 1: Counterfactual arguments
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/argument_counterfactual.json

# Step 2: Predecessor functions (on top of argument counterfactual)
cd ../predecessor
python generate_predecessors.py \
    --input ../counterfactual/output/gsm8k/argument_counterfactual.json \
    --output output/gsm8k/both_counterfactual.json
```

## Retry Failed Samples

If some samples fail during counterfactual, use `retry_failed.py` to recover them:

```bash
# Auto-detect and retry failed samples
python retry_failed.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/argument_counterfactual.json \
    --model gpt-5.1

# Retry specific task_ids
python retry_failed.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/argument_counterfactual.json \
    --model gpt-5.1 \
    --task_ids extracted-gsm8k-test-135
```

The script will:
1. Auto-detect missing samples (comparing input vs output)
2. Retry with a stronger model (default: gpt-5.1)
3. Auto-merge results into the output file
