# User Simulation (`situated_simulation/`)

A DataLoader-like environment that turns the constructed dataset
(`final_dataset/{dataset}_final.json`) into multi-turn conversations for
evaluating how well an LLM tracks **evolving user intent**.

All five scenarios are produced by a single **plan-first turn scheduler**
(`turn_scheduler.py`); the scenario is auto-inferred from the parameters you
pass. For the scheduler design, prefix logic, and domain-specific internals,
see [INTERNALS.md](INTERNALS.md).

## Quick Start

```python
from situated_simulation.user_simulation import EvolvingIntent

# Argument-revision: 3 turns, 2 argument revisions
sim = EvolvingIntent(
    data_path="final_dataset/gsm8k_final.json",
    mode="eval",
    num_turns=3,
    num_revisions=2,
)

print(f"Total samples: {len(sim)}")

for sample in sim:
    turns = sample.turns      # [{"role": "user", "content": "..."}, ...]
    label = sample.label      # ground-truth answer

# Indexing / slicing
sample = sim[0]
batch = sim[0:10]
```

Filter to a published evaluation subset (the fixed task-id lists in
`intent_construction/eval_indices/`):

```python
import json

with open("intent_construction/eval_indices/gsm8k_task_ids.json") as f:
    task_ids = json.load(f)["task_ids"]

sim = EvolvingIntent(
    data_path="final_dataset/gsm8k_final.json",
    num_turns=3,
    num_revisions=2,
    task_ids=task_ids,
)
```

## Scenario Auto-Inference

The scenario is determined by `num_turns`, `num_revisions`, and `num_switches`:

| Scenario | Condition | Description |
|----------|-----------|-------------|
| `fully-specified` | `num_turns=1` | Single-turn baseline (all arguments at once) |
| `argument-reveal` | `num_turns>1`, no changes | Arguments revealed incrementally, no changes |
| `argument-revision` | `num_revisions>0` | Argument values change mid-conversation |
| `function-switch` | `num_switches>0` | The question/function changes mid-conversation |
| `combined` | `num_revisions>0` **and** `num_switches>0` | Both functions and arguments change |

> The `argument-reveal` scenario is identified as `under-specified` (hyphen) in
> `sim.scenario`, but the evaluation output directories use `under_specified`
> (underscore).

The minimum number of turns for any configuration is `1 + num_switches +
num_revisions`; requesting fewer skips the sample (the scheduler returns no
turns for it rather than raising).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str \| Path | **required** | Pipeline output JSON (`final_dataset/{dataset}_final.json`) |
| `mode` | `"eval"` \| `"train"` | `"eval"` | Deterministic (eval) or randomized/seeded (train) sampling |
| `domain` | str | `"math"` | One of `math`, `search`, `sql`, `swe_bench_verified` |
| `num_turns` | int | `1` | Target number of conversation turns |
| `num_revisions` | int | `0` | Total argument revisions, distributed round-robin (see INTERNALS) |
| `num_switches` | int | `0` | Number of function switches |
| `ordering` | str | `"interleaved"` | Accepted for backward compatibility; the plan-first scheduler currently ignores it |
| `system_prompt` | str \| None | None | Overrides the per-domain default system prompt |
| `instruction` | str \| None | None | Wraps the first user turn (use `{content}` placeholder); overrides the per-domain default |
| `task_ids` | list[str] \| set[str] \| None | None | Restrict to specific task IDs |
| `naturalizer_model` | str \| None | None | LLM model for online turn naturalization (`search` only; other domains fall back to rule-based) |
| `recap_method` | str \| None | None | Per-turn recap mode: `"prompt"`, `"dump"`, `"ground_truth"`, or None (see INTERNALS) |
| `prefix_style` | str \| None | None → `"base"` | `"base"` or `"function-naturalized"` (SQL only) |
| `include_evidence` | bool | True | SQL only: include the BIRD `evidence` field in user turn 1 |
| `seed` | int | `42` | Random seed |

## Sample Access

Each item is an `IntentSample`:

```python
@dataclass
class IntentSample:
    task_id: str
    turns: list[dict[str, str]]   # [{"role": "user", "content": "..."}]
    label: str                     # ground-truth answer
    metadata: dict                 # see below
```

### Direct access
Iterate `sample.turns` (all user turns) and read `sample.label`.

### Step-wise delivery
For interleaved conversations, drive the sample with `reset()` / `step()` /
`is_done()`:

```python
sample = sim[idx]
messages = sample.reset()              # system prompt + first user turn

while not sample.is_done():
    response = call_llm(messages)
    messages.append({"role": "assistant", "content": response})
    messages.extend(sample.step(response))   # next user turn(s); may be empty
```

`step()` always returns a list (never `None`); use `is_done()` to terminate.

### Online naturalization
Pass `naturalizer_model=` to the constructor to rephrase rule-based turns with
an LLM while preserving critical values. Only the `search` domain has an LLM
naturalizer; other domains fall back to the rule-based renderer.

```python
sim = EvolvingIntent(
    data_path="final_dataset/browsecomp_plus_final.json",
    domain="search",
    num_turns=3,
    num_revisions=2,
    naturalizer_model="gpt-5.1",
)
```

## Metadata Fields

`sample.metadata` (built in `turn_scheduler.py`) contains:

| Field | Description |
|-------|-------------|
| `task_id` | Sample identifier |
| `scenario` | Inferred scenario (`fully-specified` / `under-specified` / `argument-revision` / `function-switch` / `combined`) |
| `mode` | `eval` or `train` |
| `num_turns` | Actual number of turns produced |
| `requested_num_turns` | Originally requested `num_turns` |
| `num_arguments` | Number of source arguments |
| `num_counterfactual_arguments` | Number of arguments that were revised |
| `num_predecessor_functions` | Number of function switches applied |
| `function` / `source_function` | The final (target) function |
| `answer` | Ground-truth answer |
| `change_plan` | Serialized `ChangePlan` — the full intent trajectory (see INTERNALS) |
| `recap_method` | Active recap mode, if any |
| `per_turn_gold` | Per-turn intermediate gold (SQL domain) |
| `instruction_id_list`, `kwargs`, `data_source` | Passthrough fields from the source sample |

SQL samples additionally carry SQL-specific metadata (see INTERNALS).

## Domains

| Domain | Notes |
|--------|-------|
| `math` | GSM8K-style; rule-based rendering |
| `search` | BrowseComp+; supports LLM naturalization |
| `sql` | BIRD-SQL; per-turn partial-SQL gold, `function-naturalized` prefixes, `include_evidence` |
| `swe_bench_verified` | SWE-bench Verified; developer-tone prefixes and a dedicated scheduling overlay |

The `sql` and `swe_bench_verified` paths have additional behavior documented in
[INTERNALS.md](INTERNALS.md).
