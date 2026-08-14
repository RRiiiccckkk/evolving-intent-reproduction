# Evaluation

Systematic evaluation framework for EvolvingIntent.

## Directory Structure

```
evaluation/
├── runners/                           # Experiment execution
│   ├── run_experiment.py              # Main runner (math / search / SQL domains)
│   ├── run_browsecomp_experiment.py   # BrowseComp+ runner (agentic search domain)
│   └── run_swe_mini_agent.py          # SWE-bench Verified runner (mini-swe-agent v2)
│
├── common/                            # Shared utilities (SWE evaluator, harness, scaffold, SQL judge)
│
├── scripts/                           # Per-dataset run scripts + helpers
│   ├── run_gsm8k.sh                   # GSM8K     (single + evolve scenarios)
│   ├── run_bird.sh                    # BIRD-SQL  (single + evolve scenarios)
│   ├── run_browsecomp.sh              # BrowseComp+ (single + evolve scenarios)
│   ├── run_swe.sh                     # SWE-bench Verified (single + evolve scenarios)
│   ├── prepull_swe_images.sh          # Pre-pull SWE docker images
│   ├── filter_valid_samples.py        # Curate task-id subsets
│   └── llm_judge_bird_sql.py          # LLM judge for SQL answers
│
├── experiments/                       # Experiment results (generated on first run, gitignored)
└── logs/                              # Execution logs (generated, gitignored)
```

> `experiments/` and `logs/` are created on the first run and are not tracked
> in git. The published evaluation subsets are the fixed task-id lists in
> `intent_construction/eval_indices/`.

## Scenario Auto-Inference

Scenarios are **automatically inferred** from parameters:

| Scenario | Parameters |
|----------|------------|
| `fully_specified` | `num_turns=1`, `num_revisions=0`, `num_switches=0` |
| `argument_reveal` | `num_turns>1`, `num_revisions=0`, `num_switches=0` |
| `argument_revision` | `num_revisions>0`, `num_switches=0` |
| `function_switch` | `num_revisions=0`, `num_switches>0` |
| `combined` | `num_revisions>0`, `num_switches>0` |

> Note: the `argument_reveal` scenario is identified internally (and in output
> directory names) as `under_specified`.

## Quick Start (per-dataset scripts)

Each dataset has a self-contained script that runs **two scenarios per model** —
a single-turn baseline (`single`: t=1, p=0, g=0) and an evolving-intent scenario
(`evolve`: t=7, p=2, g=2):

```bash
cd evaluation

# GSM8K (pass one or more model names)
./scripts/run_gsm8k.sh gpt-5.1

# BIRD-SQL
./scripts/run_bird.sh gpt-5.1

# BrowseComp+ (requires the retriever setup below)
./scripts/run_browsecomp.sh gpt-5.1

# SWE-bench Verified (requires Docker)
./scripts/run_swe.sh gpt-5.1
```

Each script reads its eval subset from `intent_construction/eval_indices/` and
writes per-sample results under `evaluation/experiments/`. Re-runs are safe: a
`(model, scenario)` whose output already exists is skipped.

**Common environment overrides** (see each script header for the full list):

| Variable | Meaning | Default |
|----------|---------|---------|
| `NUM_WORKERS` | Parallel workers | `8` |
| `REASONING_EFFORT` | For reasoning models only (`""` to skip) | `medium` |
| `DATA_PATH` | Generated dataset pool (regenerate via the construction pipeline) | `final_dataset/<dataset>_final.json` |
| `TASK_IDS_FILE` | Published eval subset | `intent_construction/eval_indices/<dataset>_task_ids.json` |
| `DATASET_NAME` | `experiments/` output label | e.g. `gsm8k_n200`, `bird_sql_n100` |

> `DATA_PATH` points at the full generated dataset (the output of the
> `intent_construction` pipeline). It is gitignored — regenerate it with
> `intent_construction/scripts/<dataset>.sh` (see `intent_construction/README.md`).

## Calling a runner directly

The scripts wrap `runners/run_experiment.py`. You can call it directly to sweep
arbitrary turn / revision / switch counts (the scenario is auto-inferred):

```bash
cd evaluation

# Fully-specified baseline (num_turns=1)
python runners/run_experiment.py \
    --data_path ../final_dataset/gsm8k_final.json \
    --task_ids_file ../intent_construction/eval_indices/gsm8k_task_ids.json \
    --models gpt-5.1 \
    --dataset_name gsm8k_n200 \
    --num_turns 1

# Argument-revision (num_revisions>0)
python runners/run_experiment.py \
    --data_path ../final_dataset/gsm8k_final.json \
    --task_ids_file ../intent_construction/eval_indices/gsm8k_task_ids.json \
    --models gpt-5.1 \
    --dataset_name gsm8k_n200 \
    --num_turns 3 \
    --num_revisions 2

# Combined: function + argument change (both >0), 8 workers
python runners/run_experiment.py \
    --data_path ../final_dataset/gsm8k_final.json \
    --task_ids_file ../intent_construction/eval_indices/gsm8k_task_ids.json \
    --models gpt-5.1 \
    --dataset_name gsm8k_n200 \
    --num_turns 5 \
    --num_switches 2 \
    --num_revisions 2 \
    --num_workers 8
```

To analyze the effect of switches (g) and revisions (p), invoke the runner once
per `(g, p)` pair you care about (the total turn count is `1 + g + p`) over the
same task-id subset, then compare accuracy across configs.

## Output Format

Each experiment saves per-sample results:

```json
{
  "task_id": "math_001",
  "prediction": "42",
  "correct": true,
  "ground_truth": "42",
  "decoding": ["Step 1: ...", "Final answer: 42"],
  "metadata": {
    "scenario": "fully-specified",
    "num_turns": 1,
    "num_arguments": 3
  }
}
```

## Fair Comparison

When comparing across experiments:
1. Find **common task_ids** across experiments
2. Compute accuracy only on common samples
3. Report both individual accuracy and drop from baseline

```python
# Example analysis
baseline = load_results("fully_specified/gsm8k_n200/gpt-5.1.json")
evolve = load_results("combined_independent/gsm8k_n200/gpt-5.1_t7_g2_p2.json")

common_ids = set(baseline.keys()) & set(evolve.keys())
baseline_acc = mean([baseline[id]["correct"] for id in common_ids])
evolve_acc = mean([evolve[id]["correct"] for id in common_ids])
drop = baseline_acc - evolve_acc
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | Path to the generated dataset pool | required |
| `--models` | Models to evaluate | required |
| `--dataset_name` | Dataset name for output folder | required |
| `--num_turns` | Number of turns (1=fully-specified, >1=multi-turn) | 1 |
| `--num_revisions` | Total revisions (round-robin) | 0 |
| `--num_switches` | Number of switches | 0 |
| `--ordering` | Ordering strategy: sequential, interleaved, mixed, random | interleaved |
| `--task_ids_file` | Path to JSON file with list of task IDs to filter | None |
| `--temperature` | Sampling temperature (0 for greedy) | 0 |
| `--num_workers` | Parallel workers | 1 |
| `--num_samples` | Limit samples (for testing) | None |
| `--naturalizer_model` | Model for LLM naturalizer (online turn rephrasing) | None (rule-based) |
| `--prefix_style` | Prefix style for function changes: `base` (default) or `function-naturalized` (SQL-only: short correction-style prefixes without repeating the full function sentence) | base |
| `--output_suffix` | Custom suffix appended to output filename | None |

## Curating a task-id subset

`scripts/filter_valid_samples.py` produces curated `task_id` lists. It keeps only
samples that pass **both** (1) quality checks (`verification_passed` and
`independence_passed`) and (2) simulation viability — i.e. the sample survives
every evaluation config in `PLAN_CONFIGS`, so the same set runs across all
g/p/turn combinations. Multiple input files may be passed (e.g. an initial run
plus retries); later files override earlier ones by `task_id`.

```bash
python scripts/filter_valid_samples.py \
    --data_paths ../intent_construction/retrospective_expansion/predecessor/output/browsecomp_plus/predecessor.json \
                 ../intent_construction/retrospective_expansion/predecessor/output/browsecomp_plus/predecessor_retry.json \
    --domain search \
    --output my_task_ids.json
```

Key flags: `--domain {math,search}`, `--output` (filtered `task_id` JSON),
`--save_merged` (optional path to dump the merged dataset). The published
subsets used for the paper live in `intent_construction/eval_indices/`.

## 🔍 BrowseComp+ (Search Domain) Evaluation

BrowseComp+ experiments use a dedicated runner with agentic search capabilities.

### Setup (external dependency)

BrowseComp-Plus is **not vendored** in this repo — the retriever needs the
upstream code and large FAISS indexes, plus `torch` / `transformers` / `faiss`
(see the optional deps in `requirements.txt`). Set it up once:

1. **Clone the upstream repo into the project root** (the runner looks for it at
   `<repo_root>/BrowseComp-Plus`):
   ```bash
   git clone https://github.com/texttron/BrowseComp-Plus.git
   ```
2. **Download the pre-built indexes** (BM25 + Qwen3-Embedding) per the upstream
   README — they land under `BrowseComp-Plus/indexes/`:
   ```bash
   cd BrowseComp-Plus && bash scripts_build_index/download_indexes.sh && cd ..
   ```
   The runner reads `BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl`
   (override with `--index_pattern`).
3. **Corpus** (`Tevatron/browsecomp-plus-corpus`) is auto-downloaded from
   Hugging Face on first use.
4. The retriever embedder needs **torch + a GPU** (`RETRIEVER_DEVICE`, default
   `cuda:0`).

For the **data-extraction** side (decrypting the obfuscated queries that the
pipeline consumes), see
`intent_construction/intent_extraction/dataset_impl/browsecomp_plus/README.md`.

### Running

Use the per-dataset script (recommended — loads the retriever once):

```bash
./scripts/run_browsecomp.sh gpt-5.1
```

Or call the runner directly:

```bash
python runners/run_browsecomp_experiment.py \
    --data_path ../final_dataset/browsecomp_plus_final.json \
    --task_ids_file ../intent_construction/eval_indices/browsecomp_plus_task_ids.json \
    --dataset_name browsecomp_plus_n100 \
    --models gpt-5.1 \
    --num_workers 8 \
    --run_plan
```

Key parameters specific to BrowseComp+:
- `--search_k`: Documents returned per search call (default: 5)
- `--max_search_iterations`: Max LLM round-trips per turn for agentic search (default: 15)
- `--max_tool_calls`: Optional cap on total tool calls per turn (default: unlimited)

### Multi-Turn Context Handling

In multi-turn BrowseComp+ conversations, **all tool calls and search results persist across turns**. When the user revises an argument (e.g., "I meant 2021, not 2022"), the model's next agentic search turn can see all prior search results in its conversation context. This matches the standard multi-turn tool use pattern recommended by OpenAI and Anthropic.

```
Turn 1: [user] → [assistant(tool_call)] → [tool(search_results)] → ... → [assistant(answer)]
Turn 2: [user(revision)] → [assistant(tool_call)] → [tool(search_results)] → [assistant(answer)]
         ↑ model sees all of Turn 1 including tool results
```

## SWE-bench Verified

The SWE-bench Verified domain has its own runner and operational details — see
[SWE_README.md](SWE_README.md).
