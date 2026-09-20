from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_WEIGHT_BYTES = 61_064_245_248
EXPECTED_SHARDS = 16


def _object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def validate_model(path: Path) -> None:
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("model path must be an existing absolute directory")
    config = _object(path / "config.json")
    if config.get("model_type") != "qwen3_moe":
        raise ValueError("model_type is not qwen3_moe")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen3MoeForCausalLM" not in architectures:
        raise ValueError("model architecture is not Qwen3MoeForCausalLM")
    if config.get("num_hidden_layers") != 48 or config.get("num_experts") != 128:
        raise ValueError("model configuration does not match Qwen3-30B-A3B")
    declared_dtypes = [
        config[key] for key in ("torch_dtype", "dtype") if config.get(key) is not None
    ]
    if not declared_dtypes or any(value != "bfloat16" for value in declared_dtypes):
        raise ValueError("model configuration does not declare BF16 weights")
    if config.get("quantization_config") is not None:
        raise ValueError("quantized weights are not accepted by this BF16 pilot")

    index = _object(path / "model.safetensors.index.json")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != EXPECTED_WEIGHT_BYTES:
        raise ValueError("indexed tensor bytes do not match Qwen3-30B-A3B BF16")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model weight index is empty")
    shard_names = set(weight_map.values())
    if len(shard_names) != EXPECTED_SHARDS:
        raise ValueError(f"expected {EXPECTED_SHARDS} weight shards")
    for name in shard_names:
        if not isinstance(name, str):
            raise ValueError("invalid shard name")
        relative = PurePosixPath(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("weight index contains a non-local shard path")
        shard = path / name
        if not shard.is_file() or shard.stat().st_size < 1024 * 1024:
            raise ValueError(f"weight shard is missing or truncated: {name}")
    actual_bytes = sum((path / name).stat().st_size for name in shard_names)
    header_bytes = actual_bytes - EXPECTED_WEIGHT_BYTES
    if not 0 < header_bytes <= 64 * 1024 * 1024:
        raise ValueError("weight shard sizes are inconsistent with the tensor index")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError("tokenizer_config.json is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    validate_model(args.model_dir)
    print("model_structure=Qwen3-30B-A3B-BF16")
    print(f"model_shards={EXPECTED_SHARDS}")
    print(f"indexed_tensor_bytes={EXPECTED_WEIGHT_BYTES}")


if __name__ == "__main__":
    main()
