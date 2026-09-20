# Civic RAG rebuild

The repository currently contains two deliberately separate stages:

- [`data_analysis/`](data_analysis/): privacy-aware source profiling and field semantics;
- [`semantic_extraction/`](semantic_extraction/): evidence-grounded Qwen3 extraction from
  `case_content` for controlled retrieval experiments.

Start with [`semantic_extraction/README.md`](semantic_extraction/README.md) for
the Conda and H100 pilot procedure. Source data, derived samples, model files,
runtime logs, caches, and private deployment settings are excluded from Git.
