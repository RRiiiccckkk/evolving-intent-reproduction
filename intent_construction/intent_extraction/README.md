# Function & Argument Extraction

Extract **Function** + **Arguments** from benchmark problems.

## Structure

```
intent_extraction/
├── core/                     # Common modules
│   ├── base_extractor.py    # Abstract base class for extractors
│   ├── base_verifier.py     # Abstract base class for verifiers
│   └── llm_utils.py         # LLM API utilities
│
├── dataset_impl/             # Dataset-specific implementations
│   ├── gsm8k/               # Grade school math
│   │   ├── extractor.py
│   │   ├── verifier.py
│   │   └── prompts/
│   ├── browsecomp_plus/     # BrowseComp-Plus (search/retrieval)
│   ├── swe_bench_verified/  # SWE-bench Verified (software engineering)
│   └── bird_sql/            # BIRD-SQL (text-to-SQL)
│
├── registry.py              # Dataset registry
├── generate.py              # Unified CLI
└── output/                  # Extracted data
```

## Usage

### Extract Function + Arguments

```bash
# Sequential processing
python generate.py --dataset gsm8k --num_samples 100 --split test

# Batch (parallel) processing - faster!
python generate.py --dataset gsm8k --num_samples 100 --batch --batch_size 10

# With options
python generate.py \
    --dataset gsm8k \
    --model gpt-5.1 \
    --num_samples 100 \
    --output output/gsm8k/extracted_test.json \
    --batch --batch_size 5 \
    --max_retries 5 \
    --disable_model_verification
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | required | Dataset to process |
| `--split` | test | Dataset split (train/test) |
| `--num_samples` | None | Number of samples (None=all) |
| `--model` | gpt-5.1 | LLM model for extraction |
| `--verif_model` | gpt-5.1 | Model for verification |
| `--batch` | False | Enable parallel processing |
| `--batch_size` | 5 | Number of parallel workers |
| `--max_retries` | 5 | Max verification retry attempts |
| `--disable_model_verification` | False | Skip solvability check |

### Output Path

Default: `output/{dataset}/extracted_{split}.json`

## Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    4-Step Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  1. Decomposition    │ Extract function + arguments           │
│  2. Transform        │ Convert to conversational format    │
│  3. Coverage Check   │ LLM verifies completeness           │
│  4. Solvability      │ Model verification (optional)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (if Step 4 fails after retries)
┌─────────────────────────────────────────────────────────────┐
│               2nd Pass: LLM-as-Judge                        │
│  Compare original vs reconstructed for info equivalence    │
└─────────────────────────────────────────────────────────────┘
```

## Output Format

### Set-Like Arguments

Arguments are extracted as an **unordered set of independent facts**. Each argument must be self-contained and understandable on its own, without reading other arguments. This is enforced by the segmentation prompt (rule 6) to ensure arguments can be presented in any order during multi-turn conversations.

```json
{
  "task_id": "extracted-gsm8k-test-0",
  "original_id": 0,
  "question": "<original problem text>",
  "answer": "<gold answer>",
  "fully_specified_question": "<original problem text>",
  "function": "<the main question being asked>",
  "arguments": [
    {"argument_id": 1, "argument": "<fact 1 needed to solve>"},
    {"argument_id": 2, "argument": "<fact 2 needed to solve>"},
    {"argument_id": 3, "argument": "<fact 3 needed to solve>"}
  ],
  "num_arguments": 3,
  "model_name": "gpt-5.1"
}
```

## Retry Failed Samples

Use `retry_failed.py` to automatically retry failed extractions:

```bash
# Auto-detect failed samples and retry
python retry_failed.py --dataset gsm8k --model gpt-5.1

# With options
python retry_failed.py \
  --dataset gsm8k \
  --model gpt-5.1 \
  --split test \
  --batch_size 10 \
  --max_retries 5
```

**Features**:
- Auto-detects missing samples from main output file
- Retries with specified model (use stronger model like gpt-5.1 for hard cases)
- Auto-merges results back into main output
- Cleans up retry checkpoint files after merge

---

## Adding New Datasets

1. Create `dataset_impl/{name}/` with:
   ```
   dataset_impl/{name}/
   ├── __init__.py
   ├── extractor.py    # Inherit from BaseExtractor
   ├── verifier.py     # Inherit from BaseVerifier
   └── prompts/
       ├── segmentation.txt   # Decomposition prompt (function + arguments)
       ├── conversational.txt
       ├── verification.txt
       └── llm_judge.txt  # Optional, for hard datasets
   ```

2. Implement required methods in `extractor.py`:
   ```python
   class MyDatasetExtractor(BaseExtractor):
       def get_dataset_name(self) -> str: ...
       def get_prompts_dir(self) -> Path: ...
       def decompose(self, sample) -> Dict: ...  # Returns {"function": ..., "arguments": [...]}
       def to_conversational(self, sample, decomposed) -> Dict: ...
       def verify_coverage(self, sample, extracted) -> bool: ...
       def verify_solvability(self, sample, extracted) -> bool: ...
       def build_output(self, sample, extracted) -> Dict: ...
       # Optional: verify_with_llm_judge(self, sample, extracted) -> bool
   ```

3. Register in `registry.py` (inside `_auto_register()`):
   ```python
   from intent_construction.intent_extraction.dataset_impl.{name} import {Name}Extractor, {Name}Verifier
   register_dataset("{name}", {Name}Extractor, {Name}Verifier)
   ```

4. Add loader in `generate.py`:
   ```python
   def load_{name}_samples(split, num_samples, shuffle, seed):
       # Load from HuggingFace or local file
       ...
   
   DATASET_LOADERS["{name}"] = load_{name}_samples
   ```
