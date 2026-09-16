"""Fail-fast validation for the pinned single-H100 runtime environments."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

EXPECTED = {
    "extract": {
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "vllm": "0.8.5",
    },
    "index": {
        "FlagEmbedding": "1.3.4",
        "qdrant-client": "1.14.2",
        "torch": "2.6.0",
        "transformers": "4.51.3",
    },
}
MIN_H100_BYTES = 70 * 1024**3


def _base_version(value: str) -> str:
    return value.split("+", maxsplit=1)[0]


def validate(mode: str) -> None:
    mismatches = []
    resolved = {}
    for distribution, expected in EXPECTED[mode].items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed")
            continue
        resolved[distribution] = actual
        if _base_version(actual) != expected:
            mismatches.append(f"{distribution}={actual}, expected {expected}")
    if mismatches:
        raise RuntimeError("runtime version gate failed: " + "; ".join(mismatches))

    import torch

    cuda_version = torch.version.cuda or ""
    if not cuda_version.startswith("12.4"):
        found_cuda = cuda_version or "none"
        raise RuntimeError(f"expected a CUDA 12.4 PyTorch wheel, found CUDA {found_cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA; check the NVIDIA driver and allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"exactly one GPU must be visible to this stage, found {torch.cuda.device_count()}"
        )

    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name.upper():
        raise RuntimeError(f"expected an H100, found {properties.name}")
    if properties.total_memory < MIN_H100_BYTES:
        gib = properties.total_memory / 1024**3
        raise RuntimeError(f"expected an approximately 80 GB H100 allocation, found {gib:.1f} GiB")

    if mode == "extract":
        try:
            import vllm  # noqa: F401
            import vllm._C  # noqa: F401
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "vLLM native extensions cannot be imported; check the wheel, PyTorch, "
                "CUDA, and driver ABI"
            ) from error

    versions = " ".join(f"{name}={version}" for name, version in sorted(resolved.items()))
    print(f"runtime gate passed: mode={mode} {versions} cuda={cuda_version} gpu={properties.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=sorted(EXPECTED))
    args = parser.parse_args()
    try:
        validate(args.mode)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
