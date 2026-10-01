# Civic RAG rebuild

The repository contains three stages:

- [`data_analysis/`](data_analysis/): privacy-aware source profiling and field semantics;
- [`retrieval_baseline/`](retrieval_baseline/): time-isolated raw-text BM25, BGE-M3,
  and case-level hybrid retrieval for historical cases and observed knowledge-entry
  recommendation;
- [`semantic_extraction/`](semantic_extraction/): evidence-grounded Qwen3 extraction from
  `case_content` for controlled retrieval experiments.

Start with [`retrieval_baseline/README.md`](retrieval_baseline/README.md) for the current
CPU-only dataset and development baseline. It does not require LLM annotations.
Prepare the separate Conda environment and ModelScope model using
[`deploy/RETRIEVAL_SETUP.md`](deploy/RETRIEVAL_SETUP.md).
Then continue with [`retrieval_baseline/DENSE.md`](retrieval_baseline/DENSE.md) for local
BGE-M3 encoding, resumable vector indexing and paired BM25/dense evaluation.
After both runs, use the [case-level hybrid procedure](retrieval_baseline/README.md#案例级混合检索)
to fuse their saved candidates and compare all three methods without new model calls.
Use [`semantic_extraction/README.md`](semantic_extraction/README.md) for the separate
Conda and H100 extraction pilot procedure. Source data, derived samples, model files,
runtime logs, caches, and private deployment settings are excluded from Git.

After the 20-record Qwen3 v4 pilot, use
[`semantic_extraction/EVALUATION.md`](semantic_extraction/EVALUATION.md) to build
the private human gold set and score extraction errors before changing the
schema or running the remaining pilot records.

Use the [offline annotation page](semantic_extraction/ANNOTATION.md) to read,
annotate, and export the 20-record worksheet in a browser.
