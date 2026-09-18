"""Run global and taxonomy-oracle popularity retrieval baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.retrieval_baseline import (
    load_qrels,
    load_query_metadata,
    popularity_rankings,
    write_trec_run,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval popularity baselines.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval/title-v1",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--top-k", type=_positive_int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "run-records/retrieval-baselines",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    train_qrels = load_qrels(args.dataset / "qrels/train.tsv")
    target_qrels = load_qrels(args.dataset / f"qrels/{args.split}.tsv")
    query_path = args.dataset / "queries.jsonl"
    train_metadata = load_query_metadata(query_path, set(train_qrels))
    target_metadata = load_query_metadata(query_path, set(target_qrels))

    for category_aware, name in ((False, "global-popularity"), (True, "category-popularity")):
        rankings = popularity_rankings(
            train_qrels,
            train_metadata,
            target_metadata,
            top_k=args.top_k,
            category_aware=category_aware,
        )
        output = args.output_dir / f"{name}.{args.split}.trec"
        write_trec_run(output, rankings, run_name=name)
        print(f"run={name} queries={len(rankings)} output={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
