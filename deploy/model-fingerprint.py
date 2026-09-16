#!/usr/bin/env python3
"""Validate a local Qwen3-MoE snapshot and compute a stable content fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

FINGERPRINT_DOMAIN = b"civic-rag-qwen-model-fingerprint-v1\0"
FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")
REQUIRED_METADATA = (
    "config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
OPTIONAL_METADATA = (
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
)
TOKENIZER_ASSET_GROUPS = (
    ("tokenizer.json",),
    ("tokenizer.model",),
    ("vocab.json", "merges.txt"),
)
EXPECTED_MODEL_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": False,
}
EXPECTED_WEIGHT_TOTAL_BYTES = 61_064_245_248
EXPECTED_WEIGHT_SHARD_COUNT = 16
MIN_SHARD_FILE_BYTES = 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


class FingerprintError(RuntimeError):
    pass


def _validate_model_identity(config: dict[str, Any]) -> None:
    if config.get("model_type") != "qwen3_moe":
        raise FingerprintError("config.json model_type is not qwen3_moe")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or "Qwen3MoeForCausalLM" not in architectures:
        raise FingerprintError("config.json architectures does not contain Qwen3MoeForCausalLM")

    for key, expected in EXPECTED_MODEL_CONFIG.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise FingerprintError(f"config.json {key} does not match Qwen3-30B-A3B ({expected})")

    declared_dtypes: list[str] = []
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise FingerprintError(f"config.json {key} must be a string")
        declared_dtypes.append(value)
    if not declared_dtypes or any(value != "bfloat16" for value in declared_dtypes):
        raise FingerprintError("config.json must declare Qwen3-30B-A3B bfloat16 weights")
    if config.get("quantization_config") is not None:
        raise FingerprintError("quantized Qwen3 snapshots are not accepted for the BF16 run")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FingerprintError(f"invalid JSON metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise FingerprintError(f"JSON metadata is not an object: {path.name}")
    return value


def _validate_snapshot(model_dir: Path) -> list[tuple[str, Path]]:
    if not model_dir.is_absolute():
        raise FingerprintError("model directory must be absolute")
    if not model_dir.is_dir():
        raise FingerprintError("model directory does not exist")

    broken_links = [
        entry for entry in model_dir.rglob("*") if entry.is_symlink() and not entry.exists()
    ]
    if broken_links:
        raise FingerprintError(f"model directory has {len(broken_links)} broken symlink(s)")

    for name in REQUIRED_METADATA:
        path = model_dir / name
        if not path.is_file():
            raise FingerprintError(f"required model metadata is missing: {name}")

    if not any(
        all((model_dir / name).is_file() for name in asset_group)
        for asset_group in TOKENIZER_ASSET_GROUPS
    ):
        raise FingerprintError(
            "tokenizer assets are missing; expected tokenizer.json, tokenizer.model, "
            "or vocab.json plus merges.txt"
        )

    config = _load_json(model_dir / "config.json")
    _validate_model_identity(config)

    independent_templates = sorted(
        {
            path
            for pattern in (
                "chat_template*.jinja",
                "chat_template*.json",
                "chat_templates/*.jinja",
                "chat_templates/*.json",
            )
            for path in model_dir.glob(pattern)
            if path.is_file()
        }
    )
    tokenizer = _load_json(model_dir / "tokenizer_config.json")
    chat_template = tokenizer.get("chat_template")
    embedded_template_text = ""
    if isinstance(chat_template, str):
        embedded_template_text = chat_template
    elif isinstance(chat_template, dict | list):
        embedded_template_text = json.dumps(chat_template, ensure_ascii=False)
    independent_template_text = ""
    try:
        independent_template_text = "\n".join(
            path.read_text(encoding="utf-8") for path in independent_templates
        )
    except (OSError, UnicodeError) as exc:
        raise FingerprintError("an independent chat template is unreadable") from exc
    if "enable_thinking" not in embedded_template_text + independent_template_text:
        raise FingerprintError("tokenizer metadata cannot select enable_thinking=false")

    weight_index = _load_json(model_dir / "model.safetensors.index.json")
    metadata = weight_index.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("total_size") != EXPECTED_WEIGHT_TOTAL_BYTES:
        raise FingerprintError(
            "safetensors index total_size does not match the official "
            f"Qwen3-30B-A3B BF16 snapshot ({EXPECTED_WEIGHT_TOTAL_BYTES})"
        )
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FingerprintError("safetensors index has no weight_map")
    if not all(isinstance(name, str) and name for name in weight_map.values()):
        raise FingerprintError("safetensors index contains an invalid shard name")

    referenced_shards: set[str] = set()
    for shard_name in weight_map.values():
        relative = PurePosixPath(shard_name)
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name != shard_name:
            raise FingerprintError("safetensors index contains a non-top-level shard path")
        referenced_shards.add(shard_name)

    if len(referenced_shards) != EXPECTED_WEIGHT_SHARD_COUNT:
        raise FingerprintError(
            "safetensors index shard count does not match the official "
            f"Qwen3-30B-A3B snapshot ({EXPECTED_WEIGHT_SHARD_COUNT})"
        )

    actual_shards = {path.name for path in model_dir.glob("*.safetensors") if path.is_file()}
    missing_shards = referenced_shards - actual_shards
    extra_shards = actual_shards - referenced_shards
    if missing_shards:
        raise FingerprintError(
            f"safetensors index references {len(missing_shards)} missing shard(s)"
        )
    if extra_shards:
        raise FingerprintError(
            f"model directory has {len(extra_shards)} unindexed safetensors shard(s)"
        )
    shard_sizes = {name: (model_dir / name).stat().st_size for name in referenced_shards}
    if any(size < MIN_SHARD_FILE_BYTES for size in shard_sizes.values()):
        raise FingerprintError("a safetensors shard is too small and may be truncated or a pointer")
    actual_weight_bytes = sum(shard_sizes.values())
    header_bytes = actual_weight_bytes - EXPECTED_WEIGHT_TOTAL_BYTES
    if not 0 < header_bytes <= MAX_SAFETENSORS_HEADER_BYTES:
        raise FingerprintError(
            "safetensors shard sizes are inconsistent with the indexed tensor total"
        )

    selected_paths = {model_dir / name for name in set(REQUIRED_METADATA) | referenced_shards}
    selected_paths.update(
        model_dir / name for name in OPTIONAL_METADATA if (model_dir / name).is_file()
    )
    selected_paths.update(independent_templates)
    return sorted(
        ((path.relative_to(model_dir).as_posix(), path) for path in selected_paths),
        key=lambda item: item[0],
    )


def _hash_file(path: Path) -> tuple[bytes, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise FingerprintError(f"model file changed while it was hashed: {path.name}")
    return digest.digest(), after.st_size


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    state = path.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def fingerprint(model_dir: Path, *, quiet: bool) -> tuple[str, int, int]:
    files = _validate_snapshot(model_dir)
    identities = {relative_name: _file_identity(path) for relative_name, path in files}
    aggregate = hashlib.sha256(FINGERPRINT_DOMAIN)
    total_bytes = 0
    for index, (relative_name, path) in enumerate(files, start=1):
        if not quiet:
            print(
                f"hashing_model_file={index}/{len(files)} name={relative_name}",
                file=sys.stderr,
                flush=True,
            )
        file_digest, size = _hash_file(path)
        relative = relative_name.encode("utf-8")
        aggregate.update(len(relative).to_bytes(4, "big"))
        aggregate.update(relative)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(file_digest)
        total_bytes += size

    final_files = _validate_snapshot(model_dir)
    if [name for name, _ in final_files] != [name for name, _ in files]:
        raise FingerprintError("model file set changed while it was hashed")
    for relative_name, path in final_files:
        if _file_identity(path) != identities[relative_name]:
            raise FingerprintError(f"model file changed while it was hashed: {relative_name}")
    return f"sha256:{aggregate.hexdigest()}", len(files), total_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and fingerprint a local Qwen3-30B-A3B ModelScope snapshot"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expect", help="Required sha256:<64hex> fingerprint")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-file progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model_dir.is_absolute():
        print("ERROR: --model-dir must be absolute", file=sys.stderr)
        return 2
    if args.expect is not None and FINGERPRINT_PATTERN.fullmatch(args.expect) is None:
        print("ERROR: --expect must have the form sha256:<64hex>", file=sys.stderr)
        return 2

    try:
        value, file_count, total_bytes = fingerprint(args.model_dir.resolve(), quiet=args.quiet)
    except (FingerprintError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("model_fingerprint_algorithm=civic-rag-qwen-model-fingerprint-v1")
    print(f"model_fingerprint_file_count={file_count}")
    print(f"model_fingerprint_total_bytes={total_bytes}")
    print(f"model_fingerprint_sha256={value}")
    if args.expect is not None:
        matches = value == args.expect.lower()
        print(f"model_fingerprint_match={str(matches).lower()}")
        return 0 if matches else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
