from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .client import ClientConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evidence-grounded Qwen3 extraction")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument("--limit", type=int)
    boundary.add_argument("--full", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-30B-A3B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--segment-chars", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    api_key = os.getenv("VLLM_API_KEY") or "EMPTY"
    config = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        segment_chars=args.segment_chars,
    )
    stats = await run_pipeline(
        input_path=args.input,
        output_path=args.output,
        errors_path=args.errors,
        cache_path=args.cache,
        client_config=config,
        limit=args.limit,
        full=args.full,
        resume=args.resume,
        overwrite=args.overwrite,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        " ".join(
            (
                f"scanned={stats.scanned}",
                f"skipped={stats.skipped}",
                f"submitted={stats.submitted}",
                f"succeeded={stats.succeeded}",
                f"failed={stats.failed}",
                f"cache_hits={stats.cache_hits}",
                f"model_calls={stats.model_calls}",
                f"grounded_spans={stats.grounded_spans}",
                f"ambiguous_span_matches={stats.ambiguous_span_matches}",
            )
        )
    )
    return 1 if stats.failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
