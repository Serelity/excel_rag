"""CLI for constructing the title-level retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from pipeline.retrieval_dataset import DEFAULT_SPLITS, SplitWindow, build_retrieval_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build BEIR-compatible ticket-to-knowledge-title retrieval data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/raw/t_order_master.sanitized.v1_9.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval/title-v1",
    )
    parser.add_argument("--train-start", type=_date, default=DEFAULT_SPLITS[0].start)
    parser.add_argument("--train-end", type=_date, default=DEFAULT_SPLITS[0].end)
    parser.add_argument("--dev-start", type=_date, default=DEFAULT_SPLITS[1].start)
    parser.add_argument("--dev-end", type=_date, default=DEFAULT_SPLITS[1].end)
    parser.add_argument("--test-start", type=_date, default=DEFAULT_SPLITS[2].start)
    parser.add_argument("--test-end", type=_date, default=DEFAULT_SPLITS[2].end)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--progress-every", type=_positive_int, default=100_000)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for the temporary SQLite database; defaults to the system temp dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory only after a successful rebuild.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    splits = (
        SplitWindow("train", args.train_start, args.train_end),
        SplitWindow("dev", args.dev_start, args.dev_end),
        SplitWindow("test", args.test_start, args.test_end),
    )
    manifest = build_retrieval_dataset(
        args.input,
        args.output,
        splits=splits,
        overwrite=args.overwrite,
        limit=args.limit,
        progress_every=args.progress_every,
        work_dir=args.work_dir,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
