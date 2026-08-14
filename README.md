# LLMs Get Lost in Evolving User Intent

A research project from [Microsoft Research, AI Interaction and Learning (AIIL)](https://www.microsoft.com/en-us/research/group/ai-interaction-and-learning/).

Authors: [Jihoon Tack](https://jihoontack.github.io/), [Philippe Laban](https://tingofurro.github.io/), [Jennifer Neville](https://jenneville.github.io/)

[![Microsoft Research](https://img.shields.io/badge/Microsoft-Research-0078D4?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMyAyMyI%2BPHBhdGggZmlsbD0iI2YyNTAyMiIgZD0iTTAgMGgxMXYxMUgweiIvPjxwYXRoIGZpbGw9IiM3ZmJhMDAiIGQ9Ik0xMiAwaDExdjExSDEyeiIvPjxwYXRoIGZpbGw9IiMwMGE0ZWYiIGQ9Ik0wIDEyaDExdjExSDB6Ii8%2BPHBhdGggZmlsbD0iI2ZmYjkwMCIgZD0iTTEyIDEyaDExdjExSDEyeiIvPjwvc3ZnPg%3D%3D)](https://www.microsoft.com/en-us/research/) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2607.20734)

---

## Overview

<p align="center">
  <img src="resource/simulation_concept.png" alt="Simulating conversations with evolving user intent" width="100%">
</p>

This work studies how well LLMs track and act on
user intent as it *evolves* over a conversation. Genuine interaction is
inherently dynamic: users rarely specify their intent upfront, instead
disclosing, revising, and at times redirecting it as the dialogue unfolds. Yet
LLMs are still predominantly evaluated and trained in single-turn,
fully-specified settings, leaving open a fundamental question—*how well do LLMs
follow user intent as it changes across turns?* To study this, our framework
transforms static, single-turn tasks into dynamic multi-turn conversations in
which the user's intent evolves across turns—incrementally revealed, revised,
and sometimes redirected mid-conversation—while preserving each task's original
evaluation protocol.

## Contents

- [Overview](#overview)
- [Results](#results)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Directory Structure](#directory-structure)
- [Framework](#framework)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Trademarks](#trademarks)
- [Citation](#citation)

## Results

We evaluate LLMs across multiple benchmarks under a single-turn
(static) setting and our evolving-intent setting. Even strong models degrade
substantially as user intent evolves over the conversation—after only a few
intent transitions.

<p align="center">
  <img src="resource/results.png" alt="Evolving-intent results across model families" width="100%">
</p>

## Installation

```bash
# Create and activate conda environment
conda create -n evolvingintent python=3.10 -y
conda activate evolvingintent

# Install dependencies
pip install -r requirements.txt

# Install the project as an editable package (enables `import evaluation`,
# `import intent_construction`, `import situated_simulation` from any directory)
pip install -e .
```

### API keys

Model calls go through the OpenAI Python SDK against either Azure OpenAI or
OpenAI. Configure credentials via environment variables:

```bash
# Option A — Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
# Optional: map model ids to your Azure deployment names (JSON)
export AZURE_OPENAI_DEPLOYMENT_MAP='{"gpt-5.1": "my-gpt51-deployment"}'

# Option B — OpenAI
export OPENAI_API_KEY="sk-..."
```

When both are set, choose explicitly with `LLM_BACKEND=openai` or
`LLM_BACKEND=azure`. The Azure API versions can be overridden via
`AZURE_OPENAI_API_VERSION` and `AZURE_OPENAI_RESPONSES_API_VERSION`.

### Quick Test

```bash
# Test extraction pipeline
cd intent_construction/intent_extraction
python generate.py --dataset gsm8k --num_samples 3 --split test --output output/test.json
```

## Core Concepts

Instead of creating datasets from scratch (expensive and time-consuming), we transform existing benchmarks into dynamic environments by decomposing problems into their fundamental components:

### Terminology

| Term | Description |
|------|-------------|
| `function` | The main question to answer |
| `argument_id: 1, 2, ...` | Arguments needed to solve the problem |

## Directory Structure

```
evolving-intent/
├── intent_construction/        # Data construction (Stages 1-3)
│   ├── intent_extraction/      # Stage 1: function & argument extraction
│   ├── retrospective_expansion/
│   │   ├── counterfactual/     # Stage 2: argument counterfactual
│   │   └── predecessor/        # Stage 3: function predecessor
│   ├── eval_indices/           # Fixed evaluation subsets (source IDs)
│   ├── scripts/                # Per-dataset pipeline runners
│   └── README.md               # Data construction guide
│
├── situated_simulation/        # User-simulation builder
│   ├── user_simulation.py      # DataLoader-like user-simulation interface
│   ├── turn_scheduler.py       # Plan-first turn scheduler
│   └── README.md
│
└── evaluation/                 # Evaluation
    ├── runners/                # Experiment runners (per domain)
    ├── common/                 # Shared utilities
    ├── scripts/                # Per-dataset eval scripts
    ├── experiments/            # Results by scenario
    └── SWE_README.md           # SWE-bench evaluation details
```

## Framework

### Supported Datasets

| Category | Dataset | Status |
|----------|---------|--------|
| Simple math | GSM8K | ✅ Done |
| Text-to-SQL | BIRD-SQL | ✅ Done |
| Agentic search | BrowseComp+ | ✅ Done |
| Software engineering | SWE-Bench Verified | ✅ Done — see [evaluation/SWE_README.md](evaluation/SWE_README.md) |

### Data Construction (`intent_construction/`)

Transforms existing benchmarks into structured data through three stages
(followed by user-simulation build and evaluation as Stages 4-5 of the overall pipeline):

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `intent_extraction/` | Decompose problems into Function + Arguments + Answer |
| 2 | `retrospective_expansion/counterfactual/` | Generate value variants of arguments |
| 3 | `retrospective_expansion/predecessor/` | Generate related but different functions |
| 4 | `situated_simulation/` | Assemble multi-turn conversations from counterfactual data |
| 5 | `evaluation/` | Run models, score answers, analyze results |

**Quick Start**:
```bash
# Run a dataset's full construction pipeline (example: GSM8K)
# Args: workers model num_counterfactuals num_predecessors
./intent_construction/scripts/gsm8k.sh 20 gpt-5.1 4 3
```

See [intent_construction/README.md](intent_construction/README.md) for detailed documentation.

### User Simulation (`situated_simulation/`)

Construct multi-turn conversation samples with dynamic argument/function changes.

**Scenarios**:
| Scenario | Description |
|----------|-------------|
| `fully-specified` | All info in 1 turn (baseline) |
| `argument-reveal` | Arguments revealed incrementally across turns, no changes (the under-specified setting) |
| `argument-revision` | Arguments change mid-conversation; LLM must track corrections |
| `function-switch` | Functions change mid-conversation; LLM must adapt to new questions |
| `combined` | Argument reveals, argument revisions, and function switches can all occur in the same conversation |

Scenario is auto-inferred from `num_turns` / `num_revisions` /
`num_switches`; you don't pass it explicitly.

**Example Usage**:
```python
from situated_simulation.user_simulation import EvolvingIntent

sim = EvolvingIntent(
    data_path="final_dataset/gsm8k_final.json",
    mode="eval",
    num_turns=4,
    num_revisions=2,  # → auto-inferred as "argument-revision"
)

for sample in sim:
    turns = sample.turns      # Multi-turn conversation
    label = sample.label      # Ground truth answer
```

**Key Features**:
- DataLoader-like interface (`__len__`, `__getitem__`, `__iter__`)
- Eval mode (deterministic) and train mode (random sampling)
- Online naturalization: optionally rephrase turns via `--naturalizer_model`

See [situated_simulation/README.md](situated_simulation/README.md) for details.

### Evaluation (`evaluation/`)

Run experiments using OpenAI / Azure OpenAI API calls.

**Features**:
- API-based evaluation pipeline (`runners/`)
- Multi-model comparison
- Per-dataset scripts that run a single-turn baseline and an evolving-intent scenario (`scripts/run_{gsm8k,bird,browsecomp,swe}.sh`)
- Online naturalization support: `--naturalizer_model gpt-5.1`

See [evaluation/README.md](evaluation/README.md) for details.

## Documentation

- [Intent Construction](intent_construction/README.md) - Extraction & Counterfactual stages
- [User Simulation](situated_simulation/README.md) - User-simulation interface
- [Evaluation](evaluation/README.md) - Running evaluations

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{tack2026llms,
    title={LLMs Get Lost in Evolving User Intent},
    author={Tack, Jihoon and Laban, Philippe and Neville, Jennifer},
    journal={arXiv preprint arXiv:2607.20734},
    year={2026},
}
```
