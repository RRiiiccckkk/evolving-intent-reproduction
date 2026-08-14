# Evaluation Sample Indices

These files record **which source examples** were used to build the evaluation
subsets, so the benchmarks remain **reproducible after the generated data is
removed** from the repository.

| File | Dataset | Split | N | Source dataset ID |
|------|---------|-------|---|-------------------|
| `gsm8k_eval_ids.json` | GSM8K | test | 200 | row index (`original_id`) |
| `bird_sql_eval_ids.json` | BIRD-SQL | train + dev | 100 | row index (`original_id` / `original_index`) |
| `browsecomp_plus_eval_ids.json` | BrowseComp+ | test | 100 | query id (`original_id` / `query_id`) |
| `swe_bench_verified_eval_ids.json` | SWE-bench Verified | test | 50 | instance id (`original_id`) |

Each file:

```json
{
  "dataset": "gsm8k",
  "split": "test",
  "num_samples": 200,
  "source_file": "final_dataset/gsm8k_final.json",
  "id_field": "original_id",
  "samples": [{"task_id": "...", "original_id": 12}, ...]
}
```

To regenerate a subset, select the listed `original_id`s from the corresponding
source dataset and re-run the extraction → expansion pipeline.
