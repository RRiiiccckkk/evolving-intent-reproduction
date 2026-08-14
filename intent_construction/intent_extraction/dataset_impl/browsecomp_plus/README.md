# BrowseComp-Plus dataset

BrowseComp-Plus is an **external dependency** — its queries are obfuscated in
the public benchmark and its retrieval corpus / indexes are large, so neither
is vendored in this repo. You download them from the upstream project
([texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus),
[🤗 Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus)).

This page covers the **data extraction input** (the decrypted queries this
pipeline reads in Stage 1). For the **evaluation-time retriever** (the cloned
repo + FAISS indexes), see `evaluation/README.md` → "BrowseComp+ (Search
Domain) Evaluation → Setup".

## 1. Decrypt the queries

The extractor (`intent_construction/intent_extraction/generate.py`) reads decrypted queries
from:

```
intent_construction/intent_extraction/output/browsecomp_plus/raw_data.jsonl
```

Generate that file with the upstream decryption script (run inside a clone of
`texttron/BrowseComp-Plus`; you may need `huggingface-cli login` first):

```bash
# inside the cloned BrowseComp-Plus repo
python scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
```

Then copy/symlink the decrypted JSONL into this repo at the path above:

```bash
mkdir -p intent_construction/intent_extraction/output/browsecomp_plus
cp <BrowseComp-Plus>/data/browsecomp_plus_decrypted.jsonl \
   intent_construction/intent_extraction/output/browsecomp_plus/raw_data.jsonl
```

Each line is a JSON object with at least `query`, `answer`, and the
relevance/evidence fields used by the extractor.

## 2. Corpus (auto-downloaded)

The retrieval corpus is **not** obfuscated and is pulled automatically from
Hugging Face (`Tevatron/browsecomp-plus-corpus`) the first time it is needed —
no manual step required.

## 3. Run extraction

```bash
cd intent_construction/intent_extraction
python generate.py \
    --dataset browsecomp_plus \
    --split test \
    --output output/browsecomp_plus/extracted_test.json
```

The output feeds the rest of the pipeline (counterfactual → predecessor).
