"""Evaluate a TREC run against observed knowledge citations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.retrieval_baseline import (
    evaluate_run,
    knowledge_frequencies,
    load_qrels,
    load_trec_run,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ticket-to-knowledge run.")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--train-qrels",
        type=Path,
        help="Training qrels used only to define head/mid/tail frequency buckets.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qrels = load_qrels(args.qrels)
    rankings = load_trec_run(args.run)
    frequencies = (
        knowledge_frequencies(load_qrels(args.train_qrels))
        if args.train_qrels is not None
        else None
    )
    result = evaluate_run(qrels, rankings, training_frequencies=frequencies)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"output={args.output.resolve()}")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
