# SWE-bench Verified Evaluation

Evaluation pipeline for the **SWE-bench Verified** domain in EvolvingIntent.
This path is isolated from math / IF / SQL / search domains.

## What this evaluates

For SWE-bench, each scenario is a multi-turn conversation that ends in a real
GitHub bug fix; the agent's final patch is graded by the official SWE-bench
harness (`FAIL_TO_PASS` + `PASS_TO_PASS`). Scenarios are auto-inferred from
turn / revision / switch counts — see the scenario table in
[README.md](README.md#scenario-auto-inference).

SWE-specific notes:
- `function-switch` and `combined` use the **implementation-precursor** design
  (see *Function-change design* below); the docker container is pinned to one
  instance, so an arbitrary distractor bug can't be swapped in.
- The headline evolving-intent config is `combined` at **t=7, g=2, p=2** — the
  extra repo-orientation turns make it 7 rather than `1 + g + p`. `run_swe.sh`
  ships this as `evolve` alongside a `single` (t=1) baseline; call
  `run_swe_mini_agent.py` directly to run the other scenario types.
- `argument-reveal` is identified internally (and in output directory names,
  e.g. `swe_under_specified_mini_agent/`) as `under_specified`.

## Stack

```
evaluation/
├─ common/
│  ├─ swe_harness.py            # Stateless wrapper around swebench library
│  └─ swe_minisweagent_scaffold.py
│      # Custom mini-swe-agent v2 scaffold:
│      #   - LLMTextbasedModel  (regex bash blocks, legacy)
│      #   - LLMToolModel       (chat completions native function calling)
│      #   - LLMResponseToolModel (Responses API; needed for gpt-5.5)
│      # Multi-turn user injection via Submitted-intercept; per-turn step cap.
├─ runners/
│  └─ run_swe_mini_agent.py     # Single entrypoint for all 5 scenarios
├─ scripts/
│  ├─ run_swe.sh                # single + evolve scenarios per model
│  └─ prepull_swe_images.sh     # pre-pull docker images for the eval subset
└─ swe_workspace/               # Created on first run
   ├─ cache/                    # Per-(instance_id, sha1(patch)) HarnessResult cache
   ├─ predictions/              # One-shot prediction files for swebench
   └─ logs/                     # Raw harness logs

intent_construction/eval_indices/
└─ swe_bench_verified_task_ids.json  # canonical 50-task evaluation subset

final_dataset/                                # generated (gitignored)
└─ swe_bench_verified_final.json   # ★ generated eval input pool (the 50-task subset above is selected from this)

intent_construction/retrospective_expansion/predecessor/
├─ prompts/generate_impl_precursor_swe.txt
├─ generate_impl_precursors_swe.py
└─ output/swe_bench_verified/                 # generated (gitignored)
│     ├─ paired.json                        # real-bug pairing (G2)
│     ├─ paired_g1.json                     # + G1 orientation
│     └─ paired_g1_implprec.json            # impl-precursor G2 (final Stage-5 output)
```

## Function-change design (impl-precursor)

For SWE, function-switch scenarios face a unique constraint: the docker
container is pinned to one instance's `base_commit` and is verified
against that instance's `FAIL_TO_PASS`/`PASS_TO_PASS` tests. We can't
just substitute another bug as a "distractor function" because that other
bug may not even exist in our docker, and any code edits the agent
makes for the distractor contaminate the patch we ultimately verify.

Solution — implementation-planning-precursor:

- **Turn 1**: short orientation question about the repo
  (e.g. "How are django.db.models, django.db.backends, and
  django.db.migrations related?")
- **Turn 2**: implementation-PLANNING request for a feature in an
  **adjacent / orthogonal** part of the same repo. Read-only:
  *"analyze the codebase and outline the approach, don't write code yet."*
  Specifically constructed to NOT traverse the buggy class's
  `__repr__` / `__str__` / `__deepcopy__` / `info()` / `summary()` /
  `format()` paths — see `prompts/generate_impl_precursor_swe.txt`.
- **Turn 3**: pivot — *"Hmm, before I dive deeper, I just remembered
  there's an outstanding issue in [different area] I should fix first."*
  The original bug fix follows.

Net effect:
- Agent reads a meaningful amount of unrelated code for the impl plan
  (context contamination, what we want).
- Repo state stays clean (planning only, no code edits).
- Final user intent in turn 3 = original SWE-bench bug, so the harness
  still grades against the right tests.

Two LLM evaluator agents (Opus 4.7 1M, GPT-5.5) audited the design;
mean "distraction quality" score is 3.8 / 5 (mild poison) with no
sample scoring as accidental hint.

## Data generation

The full function-switch dataset is built end-to-end by
`intent_construction/scripts/swe_bench_verified.sh`, which auto-copies its final
output to `final_dataset/swe_bench_verified_final.json` (the eval input). The
script header documents the five stages (extraction → argument counterfactual →
real-bug pairing → G1 orientation → impl-precursor); the impl-precursor function
design is described in *Function-change design* above.

```bash
# ./intent_construction/scripts/swe_bench_verified.sh [MODEL=gpt-5.1] [WORKERS=8] [COUNTERFACTUALS=2] [SEED=42] [TARGET=<extracted.json>]
./intent_construction/scripts/swe_bench_verified.sh gpt-5.1 8 2
```

`TARGET` is the curated G3 target subset fed into the argument-counterfactual
stage; it defaults to the full extraction output. Point it at a curated subset
to reproduce the paper set.

## Conventions

- **Eval input (`DATA_PATH`)**: the generated dataset pool. `run_swe.sh`
  defaults `DATA_PATH` to `final_dataset/swe_bench_verified_final.json`; override
  it to point at your generated pool (impl-precursor V3 set). The published eval
  subset is selected from the pool via `TASK_IDS_FILE`.
- **Task subset**: the canonical 50-sample evaluation subset is recorded in
  `intent_construction/eval_indices/swe_bench_verified_eval_ids.json` (and the
  runner-consumable `swe_bench_verified_task_ids.json` alongside it).
- **Step caps**: per-turn step cap, set via `--step_limit_per_turn`. The
  `run_swe.sh` script uses `single`=100 and `evolve`=200 (the larger cap covers
  the extra turns). Set `--step_limit_per_turn -1` to fall back to a single
  total-trajectory cap (`--step_limit`, runner default 250).
- **Reasoning effort**: `medium` for GPT-5.x.
- **Tool calling mode**: native function calling
  (`--use_tool_calling`); routes through Chat Completions for most
  models and Responses API for gpt-5.5 (chat completions rejects function tools
  on chat completions for gpt-5.5).

## Running

### Single-instance smoke (fastest sanity check)

```bash
echo '{"task_ids": ["extracted-swe_bench_verified-test-django__django-10999"]}' \
    > /tmp/one_task.json

python evaluation/runners/run_swe_mini_agent.py \
    --data_path final_dataset/swe_bench_verified_final.json \
    --models gpt-5.1 \
    --task_ids_file /tmp/one_task.json \
    --num_workers 1 \
    --reasoning_effort medium \
    --use_tool_calling \
    --num_turns 3 --num_switches 2
```

### Full evaluation

`run_swe.sh` runs the `single` and `evolve` scenarios for each model you pass:

```bash
# One or more models; runs single + evolve per model
bash evaluation/scripts/run_swe.sh gpt-5.1
```

The script:
- Iterates scenarios sequentially, models concurrently within a scenario.
- Uses `NUM_WORKERS=8` (overridable via env).
- Skips a `(model, scenario)` whose output already exists, so re-runs resume.
- Writes per-model logs under `evaluation/logs/swe/`.

### Outputs

```
evaluation/experiments/
├─ swe_original_mini_agent/swe_bench_verified/{model}_native.json     # fully-specified
├─ swe_under_specified_mini_agent/swe_bench_verified/{model}_native_t3_g0_p0.json
├─ swe_argument_revision_mini_agent/swe_bench_verified/{model}_native_t3_g0_p2.json
├─ swe_function_switch_mini_agent/swe_bench_verified/{model}_native_t3_g2_p0.json
└─ swe_combined_independent_mini_agent/swe_bench_verified/{model}_native_t7_g2_p2.json
```

Per-instance JSON includes:
- `prediction` (final unified diff)
- `correct` (= `swe_eval.resolved`, official SWE-bench definition)
- `swe_eval.{ftp_pass, ftp_fail, ptp_pass, ptp_fail, harness_error}`
- `metadata.{n_steps, per_turn_steps, step_limit_per_turn,
   exit_status, scaffold_mode, scenario}`
- `user_messages` — the rendered turn texts as fed to the agent

## How scoring works

1. Conversation runs (multi-turn EvolvingIntent via mini-swe-agent v2 scaffold).
2. The agent issues `bash` tool calls in a docker container; on
   `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` it submits a final patch.
3. `SWEHarness.verify_patch(instance_id, patch)` is called.
   - Cache hit (`cache/<instance>__<sha1>.json`) returns immediately.
   - Otherwise, swebench's `run_evaluation.main` runs `FAIL_TO_PASS`
     and `PASS_TO_PASS` tests in the canonical `swebench/` namespace
     image, parses `report.json`.
4. Resolved iff every `FAIL_TO_PASS` passes and every `PASS_TO_PASS`
   still passes — standard SWE-bench definition.

## Validated baselines

As a scaffold sanity check (numbers are from a local run; results files are
generated under the gitignored `evaluation/experiments/`):

- **GPT-5.1, fully-specified, native tool calling, step=250**: 32/50
  = 64.0% on the 50-sample stratified subset. Compare to mini-swe-agent's
  v1.15.0 stock leaderboard: 33/50 = 66.0% on the same instances.
  Within 1 instance — the scaffold is faithful.

## Disk / time

- ~3 GB per swebench image; first run pulls them; subsequent runs
  cached on the host.
- Single instance wall time: 5-30 min (mostly tied to verify queue and
  model latency).
- Full sweep (5 scenarios × 5 models × 100 instances): ~10-15 hours
  per server with `NUM_WORKERS=100`.

### Pre-pulling images (recommended)

mini-agent's `docker run` has a 120s timeout that the first cold pull of a
~3 GB image can exceed. Pre-pull the eval subset's images sequentially
beforehand to avoid this. The script is idempotent — it skips images already
present locally.

```bash
# Defaults to the canonical eval subset
# (intent_construction/eval_indices/swe_bench_verified_task_ids.json)
bash evaluation/scripts/prepull_swe_images.sh

# Or pass a custom task_ids file
bash evaluation/scripts/prepull_swe_images.sh /tmp/one_task.json
```

It prints a per-image PULL/SKIP log and a final summary (total / skipped /
pulled / failed).

## Where to read the design history

- Project root `README.md` — high-level overview.
- This file — operational details for the SWE pipeline.
