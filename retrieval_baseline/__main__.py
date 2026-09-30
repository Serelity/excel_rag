from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_dataset
from .lexical import build_index, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only historical ticket retrieval baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--input", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--dev-start", default="2026-01-01")
    build.add_argument("--test-start", default="2026-02-01")
    build.add_argument("--queries-per-split", type=int, default=200)
    build.add_argument("--seed", type=int, default=42)
    index = commands.add_parser("index")
    index.add_argument("--dataset", required=True, type=Path)
    index.add_argument("--output", required=True, type=Path)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--dataset", required=True, type=Path)
    evaluate_parser.add_argument("--index", required=True, type=Path)
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate_parser.add_argument("--case-k", type=int, default=50)
    evaluate_parser.add_argument("--max-terms", type=int, default=32)
    args = parser.parse_args()
    if args.command == "build":
        result = build_dataset(
            args.input, args.output, dev_start=args.dev_start, test_start=args.test_start,
            queries_per_split=args.queries_per_split, seed=args.seed,
        )
    elif args.command == "index":
        result = build_index(args.dataset, args.output)
    else:
        result = evaluate(
            args.dataset, args.index, args.output, split=args.split,
            case_k=args.case_k, max_terms=args.max_terms,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
