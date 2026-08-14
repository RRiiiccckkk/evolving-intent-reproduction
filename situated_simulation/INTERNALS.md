# User Simulation — Internals

Design notes for `situated_simulation/`. For the public API and quick start, see
[README.md](README.md).

## Files

| File | Role |
|------|------|
| `user_simulation.py` | `EvolvingIntent` (DataLoader-like interface), `IntentSample`, per-domain prefix pools, default prompts/instructions |
| `turn_scheduler.py` | Unified plan-first scheduler — `create_sample()` and the five scheduling/rendering steps |
| `turn_scheduler_swe.py` | SWE-bench Verified scheduling overlay (wired in via a `post_fill_hook`) |
| `user_intent.py` | Paper formalization data structures: `Argument`, `UserIntent`, `IntentTransition`, `ChangePlan` |
| `naturalizer.py` | Online naturalization (`create_naturalizer`, rule-based + `search` LLM naturalizer) |
| `sql_partial.py` | SQL AST manipulation for per-turn partial-SQL gold answers |

## Plan-First Scheduler

`create_sample()` builds every scenario through one pipeline. The key property is
that **scheduling never fails** when `t >= 1 + g + p` and the raw sample has
enough counterfactuals.

| Step | Name | What it does |
|------|------|--------------|
| 0 | **Select** | Pick predecessor functions (farthest-first) and argument counterfactuals (round-robin) |
| 1 | **Schedule events** | Assign function-switch and correction events to turns by deadline — no text yet |
| 2 | **Fill arguments** | Distribute argument reveals across turns; defer Turn-0 arguments into otherwise-empty later slots |
| 3 | **Fill texts** | Populate function text and correction text for each slot |
| 4 | **Render** | Add prefixes and join each slot into the final turn string |

### Turn-count math

```
min_turns = 1 + num_switches + num_revisions          # applies to ALL scenarios
```

`create_sample()` returns `None` (sample skipped) if `t < min_turns` or if the
sample lacks the requested switches/revisions. The actual turn count is capped at
`min_turns + deferable` so no empty turns are produced; trailing empty turns are
trimmed after Step 2.

### `num_revisions` round-robin

`num_revisions` is the **total** number of argument counterfactuals, distributed
round-robin across eligible arguments. With arguments `C1, C2, C3` (each with
counterfactual versions available):

```
num_revisions=1 → C1v1
num_revisions=2 → C1v1, C2v1
num_revisions=4 → C1v1, C2v1, C3v1, C1v2   (wraps around)
```

For each argument, the source (true) value is always scheduled after all of its
counterfactual versions, and same-`argument_id` items always land in different
turns.

## Rendering & Prefixes

Within a turn, slots are rendered in a fixed order:

```
[function]  →  [correction(s)]  →  [reveal(s)]
```

- **Turn 0** carries the initial function + arguments with **no prefix**.
- A **function switch** gets a function-change prefix (`_get_function_change_prefix`),
  chosen from the domain's pool by taxonomy/transition type.
- Each **correction** event gets a correction prefix (`_get_correction_prefix`).
- **Reveals** get a reveal prefix, or a "new info" prefix when they follow a
  correction or function change in the same turn.

In `eval` mode, prefixes are selected deterministically (cycled) for
reproducibility; in `train` mode the category and phrasing are sampled from the
seeded RNG. The literal prefix strings live in `user_simulation.py` (per domain:
`*_PREFIXES` / `*_CORRECTION_PREFIXES` etc.) — treat that module as the source of
truth rather than copying strings here.

## Function-Switch Design

1. **Farthest predecessor first** — predecessor functions are ordered so the
   conversation starts at the farthest predecessor and converges to the target:
   `PG[g-1] → ... → PG[0] → source_function`.
2. **Function at turn start** — a function always begins a turn, never appears
   mid-turn.
3. **Block-based distribution** — items are grouped into blocks (function + its
   arguments); turns are allocated per-block minimums first, then the remainder
   is distributed proportionally to block size.
4. **Shared arguments** (`is_shared=True`) are treated as already revealed and not
   repeated across functions.

### `function-naturalized` prefix style (SQL only)

For SQL function switches, `prefix_style="function-naturalized"` replaces the full
function sentence with a short, correction-style cue derived from the
`transition_reason`, so the model must infer the new function from the cue alone:

- Aggregate swap: `"How about {new} instead?"` where `{new}` maps via `AGG_NL_MAP`
  (`count → "the total number"`, `average → "the average"`, …).
- Column swap: `"How about {new} instead?"` using the raw column name.

The cue names only the **new** target (it does not spell out the old one), and the
style is ignored by non-SQL domains.

## Combined Scenario

Combined samples interleave corrections across the function phases: each
correction lands in a turn before the function that needs the corrected argument
as source. Invariants:

1. An argument's full correction chain completes before the function that needs
   its source value.
2. A function switch may share a turn with reveals, but corrections are rendered
   in their own (earlier) turns relative to the function they precede.
3. The final turn has only source (true) values active.

## ChangePlan / UserIntent (`user_intent.py`)

Frozen dataclasses mapping the paper's formalization to code. Every
`IntentSample` serializes a `ChangePlan` into `metadata["change_plan"]`.

| Paper | Code | Description |
|-------|------|-------------|
| `I_t = (f_t, C_t, C_rev_t, y_t)` | `UserIntent` | User-intent state at turn `t` |
| `c_i ∈ C_t` | `Argument` | One argument with source + counterfactual variants |
| `ΔI_t` | `IntentTransition` | State change between consecutive turns |
| — | `ChangePlan` | Full intent trajectory (`intent_trajectory` + `transitions`) |

`ChangePlan.scenario` is one of `fully-specified`, `under-specified`,
`argument-revision`, `function-switch`, `combined`.

## SQL Domain (`sql_partial.py`)

For SQL, each turn has an intermediate gold answer reflecting only the
information revealed so far. `sql_partial.py` uses SQLGlot to:

- strip WHERE predicates whose arguments have not yet been revealed,
- prune now-unused JOINs (`prune_unused_joins`),
- rebuild and execute the partial query to produce `{"sql", "answer"}` per turn
  (`build_per_turn_gold`).

The result is exposed as `metadata["per_turn_gold"]`. `include_evidence=True`
adds the BIRD `evidence` field to user turn 1.

## SWE-bench Verified Overlay (`turn_scheduler_swe.py`)

SWE samples run through `create_sample` with a `post_fill_hook` that adjusts the
schedule to match a realistic developer conversation:

- `symptom`-category arguments are stripped before scheduling, then redistributed
  evenly across turns with a ceil-front allocation,
- symptoms are re-injected at the **start** of each receiving turn's arguments,
  yielding the order *function → symptoms → other arguments*.

SWE also uses developer-tone prefix pools (`SWE_*_PREFIXES`) and a domain-level
system prompt that instructs the model to track the **final** state of intent.

### `recap_method`

`recap_method` (`"prompt"`, `"dump"`, `"ground_truth"`, or None) controls an
optional per-turn recap of the conversation state, computed during rendering and
surfaced through `IntentSample.recap_texts` / `metadata["recap_method"]`.

## Adding a Domain

1. Add the domain's prefix pools and (optionally) default prompt/instruction to
   `user_simulation.py`.
2. Register the domain in `SUPPORTED_DOMAINS`.
3. Optionally register an LLM naturalizer in `naturalizer.py`
   (`_DOMAIN_NATURALIZERS`); without one, the domain uses the rule-based renderer.
4. Add any domain-specific scheduling via a `post_fill_hook` (see the SWE overlay)
   rather than branching the core scheduler.
