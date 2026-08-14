# Intent Construction (Data Pipeline)

This module transforms existing benchmarks into structured data for EvolvingIntent. The pipeline consists of three sequential stages:

```
Stage 1 (Extraction) → Stage 2 (Argument Counterfactual) → Stage 3 (Function Predecessor)
```

## 📁 Structure

```
intent_construction/
├── intent_extraction/        # Stage 1: Function & Argument Extraction
├── retrospective_expansion/
│   ├── counterfactual/       # Stage 2: Argument Counterfactual
│   └── predecessor/          # Stage 3: Function Predecessor
└── scripts/                  # Per-dataset pipeline runners
README.md                     # This file (data-construction guide)
```

---

## 🚀 Quick Start

### Automated Pipeline (Recommended)

Run the full pipeline with a single command from the project root:

Each dataset has its own self-contained runner under
`intent_construction/scripts/`. Run from the project root:

```bash
./intent_construction/scripts/gsm8k.sh 20 gpt-5.1 4 3
```

The GSM8K example above takes `[workers] [model] [num_counterfactuals]
[num_predecessors]`. **Argument order differs per runner** — pass `--help`-style
defaults by position as documented in each script's header:

| Runner | Positional arguments |
|--------|----------------------|
| `gsm8k.sh` | `[workers=20] [model=gpt-5.1] [num_counterfactuals=4] [num_predecessors=3]` |
| `browsecomp_plus.sh` | `[workers=20] [model=gpt-5.1] [num_counterfactuals=4] [num_predecessors=3]` |
| `bird_sql.sh` | `[N=200] [model=gpt-5.1] [workers=8] [counterfactuals=3] [tag=v2_n200]` |
| `swe_bench_verified.sh` | `[model=gpt-5.1] [workers=8] [counterfactuals=2] [seed=42] [target]` |

The available runners are:

- **GSM8K** (math): `./intent_construction/scripts/gsm8k.sh`
- **BrowseComp-Plus** (search): `./intent_construction/scripts/browsecomp_plus.sh`
- **BIRD-SQL** (multi-clause SQL rewrite): `./intent_construction/scripts/bird_sql.sh`
- **SWE-bench Verified** (impl-precursor function chain): `./intent_construction/scripts/swe_bench_verified.sh`
  (see `evaluation/SWE_README.md` → "Data generation")

### Manual Pipeline

```bash
# Stage 1: Extract
cd intent_construction/intent_extraction
python generate.py --dataset gsm8k --split test --batch --batch_size 20

# Stage 2: Counterfactual Arguments
cd ../retrospective_expansion/counterfactual
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/argument_counterfactual.json \
    --num_counterfactuals 4

# Stage 3: Function Predecessor (predecessor inference — default)
cd ../predecessor
python generate_predecessors.py \
    --input ../counterfactual/output/gsm8k/argument_counterfactual.json \
    --output output/gsm8k/predecessor.json \
    --dataset_type gsm8k \
    --num_predecessors 3 \
    --model gpt-5.1
```

---

## Stage 1: Extraction (`intent_construction/intent_extraction/`)

**Purpose**: Decompose benchmark problems into **Function** + **Arguments** + **Answer**

### Input/Output

- **Input**: Original dataset (e.g., GSM8K, BrowseComp-Plus, BIRD-SQL, SWE-bench Verified)
- **Output**: Structured JSON with extracted components

### Example

**Original Problem**:
> A word problem with several quantities, asking how much/many of some final quantity results from combining them.

**Extracted Output**:
```json
{
  "task_id": "extracted-gsm8k-test-0",
  "question": "<original problem text>",
  "answer": "<gold answer>",
  "function": "<the main question being asked>",
  "arguments": [
    {"argument_id": 1, "argument": "<fact 1 needed to solve>"},
    {"argument_id": 2, "argument": "<fact 2 needed to solve>"},
    {"argument_id": 3, "argument": "<fact 3 needed to solve>"},
    {"argument_id": 4, "argument": "<fact 4 needed to solve>"}
  ]
}
```

### Key Features

- **LLM-based decomposition**: Uses GPT to extract function and arguments
- **Verification**: Ensures extracted arguments are sufficient to solve the problem
- **Auto-retry**: Failed extractions automatically retry with stronger models

### Usage

```bash
python generate.py --dataset gsm8k --split test --batch --batch_size 10
```

See [intent_extraction/README.md](intent_extraction/README.md) for full documentation.

---

## Stage 2: Argument Counterfactual (`intent_construction/retrospective_expansion/counterfactual/`)

**Purpose**: Generate **value variants** of arguments for robustness training

### Input/Output

- **Input**: Extracted JSON from Stage 1
- **Output**: JSON with `counterfactual_arguments` added to each argument

### Example

**Input Argument**:
```json
{"argument_id": 1, "argument": "<a fact with a numeric value V>"}
```

**Output with Counterfactuals**:
```json
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
}
```

### Key Features

- **Context-aware**: Uses function + all arguments to generate plausible counterfactuals
- **Dataset-specific prompts**: Optimized for math, search, and general domains
- **Configurable**: Control number of counterfactuals per argument

### Usage

```bash
python generate_counterfactuals.py \
    --input ../../intent_extraction/output/gsm8k/extracted_test.json \
    --output output/gsm8k/argument_counterfactual.json \
    --num_counterfactuals 4
```

See [retrospective_expansion/counterfactual/README.md](retrospective_expansion/counterfactual/README.md) for full documentation.

---

## Stage 3: Function Predecessor (`intent_construction/retrospective_expansion/predecessor/`)

**Purpose**: Generate **related but different functions** that share arguments with the original

### Input/Output

- **Input**: JSON from Stage 1 or Stage 2
- **Output**: JSON with `predecessor_functions` added

### Predecessor Inference (`generate_predecessors.py`)

Generates **causally-linked predecessor functions** via chained predecessor generation. Each predecessor's answer naturally enables or motivates the next question. **This is the default approach** for the pipeline.

```
Final turn (t)  →  generate t-1 from t  →  generate t-2 from t-1 (not t!)
```

Key features:
- **Chained generation**: t-2 is generated from t-1 (not t), creating causal dependency chains
- **Future functions context**: Each generation step sees the full downstream chain to prevent overlap
- **8 chain archetypes**: identify_then_seek, survey_then_focus, trace_then_follow, pivot_inquiry, lookup_then_compute, total_then_component, compute_then_extend, reframe_problem
- **Cross-turn relevance verification**: LLM-based check that fabricated arguments from one turn don't help answer another turn's question
- **Production-ready**: Parallel processing, checkpointing, resume support, fallback model escalation

See [retrospective_expansion/predecessor/README.md](retrospective_expansion/predecessor/README.md) for full documentation.

---

## 📤 Output Files

After running the full pipeline, each stage writes intermediate output, and the
runner copies the final result into `final_dataset/`:

```
intent_construction/intent_extraction/output/{dataset}/extracted_test.json                      # Stage 1 (bird_sql: extracted.json)
intent_construction/retrospective_expansion/counterfactual/output/{dataset}/argument_counterfactual.json  # Stage 2
intent_construction/retrospective_expansion/predecessor/output/{dataset}/predecessor.json              # Stage 3 (swe: paired_g1_implprec.json)
final_dataset/{dataset}_final.json                                                               # Final (copied from Stage 3)
```

The final dataset (`final_dataset/{dataset}_final.json`) contains all extracted
data plus counterfactuals and predecessors, ready for use with
`situated_simulation`.

---

## 🔁 Reproducibility

The evaluation subsets are fixed samples drawn from the source benchmarks. The
exact source IDs are recorded in [`eval_indices/`](eval_indices/) so the
benchmarks stay reproducible even after the generated data is removed:

| Index file | Dataset | Split | N | Source file |
|------------|---------|-------|---|-------------|
| `gsm8k_eval_ids.json` | GSM8K | test | 200 | `gsm8k_n200.json` |
| `bird_sql_eval_ids.json` | BIRD-SQL | train + dev | 100 | `bird_sql_n100.json` |
| `browsecomp_plus_eval_ids.json` | BrowseComp+ | test | 100 | `browsecomp_plus_n100.json` |
| `swe_bench_verified_eval_ids.json` | SWE-bench Verified | test | 50 | `swe_bench_verified_n50.json` |

Each file lists the `samples` used, mapping the internal `task_id` to the
`original_id` in the source dataset. To regenerate a subset, select the listed
`original_id`s from the corresponding source benchmark and re-run the
extraction → expansion pipeline. See [`eval_indices/README.md`](eval_indices/README.md)
for details.
