# Plan A: GSM8K directional reproduction

This workflow runs a small, paired reproduction of the paper on GSM8K. It is
for checking the reported direction of the effect, not for reproducing the
paper's exact percentages.

## Fixed scope

- Primary size: 20 samples. Deadline fallback: 10 samples.
- Selection: the first N entries in the upstream published
  `intent_construction/eval_indices/gsm8k_eval_ids.json`, not the first N raw
  GSM8K rows.
- Seed: 42.
- Construction: two counterfactuals per argument and two predecessor
  functions per sample.
- Evaluation: one model, four paired settings.

| Setting | User turns | Composition |
|---|---:|---|
| `single_t1` | 1 | fully specified source problem |
| `evolve_t4_g1_p1` | 4 | initial + one reveal + one switch + one revision |
| `repeat_control_t7` | 7 | the four-turn setting + three exact repeats of its final user turn |
| `evolve_t7_g2_p2` | 7 | initial + two reveals + two switches + two revisions |

The repeat control follows Table 5 and Appendix F.4: it adds turns without
changing intent. It does not repeat the fully specified single-turn prompt.

The exact primary and fallback IDs are in
`reproduction/config/selected_gsm8k_n20.json`. Every live run also writes them
into its own `manifest.json`.

## Setup

Use Python 3.10 or newer. This GSM8K-only dependency set avoids the large SWE,
PyTorch, and Transformers packages in the full upstream environment.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/requirements.txt
```

The model helper supports OpenAI, Azure OpenAI, and OpenAI-compatible APIs.
Plan A defaults to `kimi-k2.6`, the paper model, with provider-default sampling
(`temperature: null`) and no reasoning-effort override. Otherwise-unbounded
construction calls use a 16384-token output limit; evaluation calls use 8192.
Empty or length-truncated final answers are recorded as failed calls rather
than treating hidden reasoning as an answer.
Set credentials only
in the environment or an untracked `.env.local`; never put a key in a command,
manifest, or tracked file.

For Kimi, start from `.env.local.example`. Replace the model ID and USD-per-one-
million-token prices with the exact values in your provider's current docs:

```bash
set -a
source .env.local
set +a
```

`LLM_MODEL_MAP` maps the workflow name `kimi-k2.6` to the provider's actual
model ID. `LLM_PRICE_MAP` must contain input and output prices; cached-input and
reasoning prices can also be supplied. A live run exits before its first call
if the price map is absent. The example limits otherwise-unbounded construction
calls to 16384 output tokens; this reduces empty JSON when Kimi's internal
reasoning consumes a smaller limit.

Run the offline check before adding credentials:

```bash
python -m reproduction.plan_a dry-run \
  --run-id plan-a-check \
  --sample-count 20 \
  --ignore-deadline
```

Run all stages after credentials are available:

```bash
python -m reproduction.plan_a all \
  --run-id plan-a-20260814 \
  --sample-count 10
```

The default operational cutoff is `2026-08-14T23:59:00+08:00`. With automatic
fallback enabled, a run initialized with less than 210 minutes remaining uses
N=10. `--sample-count 20` or `--sample-count 10` makes the choice explicit.
The workflow stops starting paid batches 15 minutes before the cutoff and
keeps its checkpoints.

## Resume and inspect

Construction and evaluation files are updated after each small batch. Re-run
the same command and `--run-id` to resume. Failed evaluation calls can be
retried without replacing successful results. Evaluation completes all four
settings for each worker-sized task batch before starting the next batch, so a
deadline-limited run retains a paired subset.

```bash
python -m reproduction.plan_a inspect --run-id plan-a-20260814
python -m reproduction.plan_a evaluate --run-id plan-a-20260814 --retry-failed
python -m reproduction.plan_a aggregate --run-id plan-a-20260814
```

Use `--allow-partial` only when the deadline matters more than full N. The
manifest records both the selected IDs and the common paired subset accepted
by every scheduler setting.

## Outputs

Each run lives under `reproduction/runs/<run-id>/` and contains:

- `manifest.json`: source commit, exact IDs, models, settings, coverage, and
  artifact hashes.
- `data/`: resumable outputs from all three construction stages.
- `results/`: one checkpointed JSON file per evaluation setting.
- `summary.json`, `summary.csv`, and `summary.html`: accuracy, Wilson 95%
  intervals, and paired differences from the single-turn baseline.
- `cost_ledger.jsonl`: stage timing, per-call provider token usage, cost,
  reservations, and releases.

The workflow sets `LLM_USAGE_LEDGER_PATH` to the run ledger and records
provider-reported token usage and estimated cost for every usable response.
Plan A has no local dollar cutoff; provider-side account limits still apply.

## Interpretation limit

N=20 has wide confidence intervals; N=10 is wider. Treat the result as a trend
check. Exact paper-scale estimates require all 200 published GSM8K samples,
the paper's model versions, and substantially more inference budget.
