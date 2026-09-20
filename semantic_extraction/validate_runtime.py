from __future__ import annotations

import importlib.metadata as metadata
import sys


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(f"Python {sys.version.split()[0]} is unsupported; expected 3.11.x")

    expected = {
        "openai": "1.75.0",
        "pydantic": "2.11.4",
        "torch": "2.6.0",
        "transformers": "4.51.3",
        "vllm": "0.8.5",
    }
    for package, wanted in expected.items():
        actual = metadata.version(package).split("+", 1)[0]
        if actual != wanted:
            raise SystemExit(f"{package}={actual}; expected {wanted}")

    import torch

    if not (torch.version.cuda or "").startswith("12.4"):
        raise SystemExit(f"PyTorch CUDA runtime is {torch.version.cuda}; expected 12.4.x")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"exactly one visible GPU is required; found {torch.cuda.device_count()}")
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in properties.name.upper():
        raise SystemExit(f"visible GPU is not an H100: {properties.name}")
    if properties.major != 9 or properties.total_memory < 75 * 1024**3:
        raise SystemExit(
            f"a full non-MIG H100 80GB is required; got capability={properties.major}."
            f"{properties.minor} memory_bytes={properties.total_memory}"
        )
    print(f"runtime_gpu={properties.name}")
    print(f"runtime_gpu_memory_bytes={properties.total_memory}")
    print("runtime_check=passed")


if __name__ == "__main__":
    main()
