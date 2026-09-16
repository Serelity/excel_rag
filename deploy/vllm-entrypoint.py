#!/usr/bin/env python3
"""Launch vLLM while keeping its optional API key out of parsed CLI arguments."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


def server_argv(
    arguments: Sequence[str],
) -> list[str]:
    # vLLM 0.8.5 reads VLLM_API_KEY natively. Do not mirror it into argparse,
    # because vLLM logs its parsed argument namespace during startup.
    return ["vllm.entrypoints.openai.api_server", *arguments]


def main() -> None:
    sys.argv = server_argv(sys.argv[1:])
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
