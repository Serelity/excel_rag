"""Command-line entry point for embedding and indexing problem chunks."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from pipeline.embedding import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    BGEM3Embedder,
)
from pipeline.indexer import (
    DEFAULT_COLLECTION,
    DEFAULT_DISTANCE,
    DEFAULT_UPSERT_BATCH_SIZE,
    DEFAULT_UPSERT_MAX_ATTEMPTS,
    DEFAULT_UPSERT_RETRY_BACKOFF_SECONDS,
    DEFAULT_VECTOR_SIZE,
    build_embedding_contract,
    create_qdrant_client,
    index,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading the YAML config requires PyYAML. Install the locked "
            "application dependencies before running this CLI."
        ) from exc

    config_path = _resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as source:
        loaded = yaml.safe_load(source)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")
    return loaded


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream problem chunks through BGE-M3 into Qdrant."
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--input", help="Problem chunk JSONL; defaults to data.output")
    parser.add_argument("--model", help="BGE-M3 model name or local path")
    parser.add_argument(
        "--model-revision",
        help="Expected loaded Hugging Face commit hash or explicit local model revision",
    )
    parser.add_argument("--device", help="FlagEmbedding device, for example cuda:0")
    parser.add_argument("--embedding-batch-size", type=_positive_int)
    parser.add_argument("--max-length", type=_positive_int)
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 embedding inference",
    )
    parser.add_argument("--host", help="Qdrant host")
    parser.add_argument("--port", type=_positive_int, help="Qdrant REST port")
    parser.add_argument("--url", help="Qdrant URL; takes precedence over host and port")
    parser.add_argument(
        "--https",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable HTTPS for Qdrant",
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing the Qdrant API key",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=_positive_float,
        help="Qdrant request timeout in seconds",
    )
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--vector-size", type=_positive_int)
    parser.add_argument("--distance", help="cosine, dot, euclid, or manhattan")
    parser.add_argument("--upsert-batch-size", type=_positive_int)
    parser.add_argument("--upsert-max-attempts", type=_positive_int)
    parser.add_argument(
        "--upsert-retry-backoff-seconds",
        type=_nonnegative_float,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate an existing collection (destructive)",
    )
    return parser


def _value(cli_value: Any, section: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return section.get(key, default)


def _configured_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    return _configured_bool(value, name=name)


def _positive_configured_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero")
    return value


def _positive_configured_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number greater than zero")
    return float(value)


def _nonnegative_configured_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or null")
    return value.strip()


def _required_text(value: Any, *, name: str) -> str:
    normalized = _optional_text(value, name=name)
    if normalized is None:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _environment_api_key(variable_name: str | None) -> str | None:
    if variable_name is None:
        return None
    value = os.environ.get(variable_name)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} config section must be a mapping")
    return section


def _resolve_model_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("embedding.model must be a non-empty string")
    model_name = value.strip()
    candidate = Path(model_name).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return str(project_candidate.resolve())
    return model_name


def _loaded_model_revision(embedder: BGEM3Embedder) -> str | None:
    snapshot_revision = getattr(embedder, "resolved_revision", None)
    backend = getattr(embedder, "_model", None)
    model = getattr(backend, "model", None)
    config = getattr(model, "config", None)
    model_revision = getattr(config, "_commit_hash", None)

    tokenizer = getattr(backend, "tokenizer", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", None)
    tokenizer_revision = (
        tokenizer_kwargs.get("_commit_hash") if isinstance(tokenizer_kwargs, dict) else None
    )

    detected = {
        revision.strip()
        for revision in (snapshot_revision, model_revision, tokenizer_revision)
        if isinstance(revision, str) and revision.strip()
    }
    if len(detected) > 1:
        raise RuntimeError("Loaded embedding model and tokenizer use different revisions")
    return next(iter(detected), None)


def _resolve_model_revision(
    embedder: BGEM3Embedder,
    configured_revision: Any,
) -> str:
    expected = _optional_text(configured_revision, name="embedding.revision")
    detected = _loaded_model_revision(embedder)
    if expected is not None and detected is not None and expected != detected:
        if getattr(embedder, "resolved_revision", None) == detected:
            return detected
        raise ValueError(
            "Configured embedding revision does not match the loaded model revision "
            f"({expected!r} != {detected!r})"
        )
    if expected is not None:
        return expected
    if detected is not None:
        return detected
    raise ValueError(
        "Unable to determine the loaded embedding model revision; set "
        "embedding.revision or use --model-revision"
    )


def _verify_normalization(embedder: BGEM3Embedder, expected: bool) -> None:
    backend = getattr(embedder, "_model", None)
    actual = getattr(backend, "normalize_embeddings", None)
    if not isinstance(actual, bool):
        raise RuntimeError("Unable to determine whether the loaded embedder normalizes vectors")
    if actual != expected:
        raise ValueError(
            "Configured embedding.normalize_embeddings does not match the loaded embedder"
        )


def _close_qdrant_client(client: Any) -> None:
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        raise RuntimeError(f"Unable to close Qdrant client ({type(exc).__name__})") from None


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    data_config = _config_section(config, "data")
    embedding_config = _config_section(config, "embedding")
    vector_config = _config_section(config, "vectorstore")

    input_value = _value(args.input, data_config, "output", None)
    if input_value is None:
        raise ValueError("No input JSONL configured; use --input or set data.output")
    input_path = _resolve_project_path(_required_text(input_value, name="data.output"))
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")

    model_name = _resolve_model_name(_value(args.model, embedding_config, "model", DEFAULT_MODEL))
    configured_revision = _optional_text(
        _value(
            args.model_revision,
            embedding_config,
            "revision",
            None,
        ),
        name="embedding.revision",
    )
    embedding_batch_size = _positive_configured_int(
        _value(
            args.embedding_batch_size,
            embedding_config,
            "batch_size",
            DEFAULT_BATCH_SIZE,
        ),
        name="embedding.batch_size",
    )
    max_length = _positive_configured_int(
        _value(args.max_length, embedding_config, "max_length", DEFAULT_MAX_LENGTH),
        name="embedding.max_length",
    )
    device = _optional_text(
        _value(args.device, embedding_config, "device", None),
        name="embedding.device",
    )
    configured_fp16 = _configured_bool(
        embedding_config.get("use_fp16", True),
        name="embedding.use_fp16",
    )
    use_fp16 = configured_fp16 and not args.no_fp16
    normalize_embeddings = _configured_bool(
        embedding_config.get("normalize_embeddings", True),
        name="embedding.normalize_embeddings",
    )

    host = _optional_text(
        _value(args.host, vector_config, "host", "localhost"),
        name="vectorstore.host",
    )
    port = _positive_configured_int(
        _value(args.port, vector_config, "port", 6333),
        name="vectorstore.port",
    )
    url = _optional_text(
        _value(args.url, vector_config, "url", None),
        name="vectorstore.url",
    )
    https = _optional_bool(
        _value(args.https, vector_config, "https", None),
        name="vectorstore.https",
    )
    api_key_env = _optional_text(
        _value(args.api_key_env, vector_config, "api_key_env", None),
        name="vectorstore.api_key_env",
    )
    api_key = _environment_api_key(api_key_env)
    timeout = _positive_configured_number(
        _value(args.qdrant_timeout, vector_config, "timeout_seconds", 60.0),
        name="vectorstore.timeout_seconds",
    )
    collection = _required_text(
        _value(
            args.collection,
            vector_config,
            "collection",
            vector_config.get("collection_name", DEFAULT_COLLECTION),
        ),
        name="vectorstore.collection",
    )
    vector_size = _positive_configured_int(
        _value(args.vector_size, vector_config, "vector_size", DEFAULT_VECTOR_SIZE),
        name="vectorstore.vector_size",
    )
    distance = _required_text(
        _value(args.distance, vector_config, "distance", DEFAULT_DISTANCE),
        name="vectorstore.distance",
    )
    configured_upsert_batch = vector_config.get(
        "upsert_batch_size",
        vector_config.get("upsert_batch", DEFAULT_UPSERT_BATCH_SIZE),
    )
    upsert_batch_size = _positive_configured_int(
        args.upsert_batch_size if args.upsert_batch_size is not None else configured_upsert_batch,
        name="vectorstore.upsert_batch_size",
    )
    max_attempts = _positive_configured_int(
        _value(
            args.upsert_max_attempts,
            vector_config,
            "upsert_max_attempts",
            DEFAULT_UPSERT_MAX_ATTEMPTS,
        ),
        name="vectorstore.upsert_max_attempts",
    )
    retry_backoff_seconds = _nonnegative_configured_number(
        _value(
            args.upsert_retry_backoff_seconds,
            vector_config,
            "upsert_retry_backoff_seconds",
            DEFAULT_UPSERT_RETRY_BACKOFF_SECONDS,
        ),
        name="vectorstore.upsert_retry_backoff_seconds",
    )

    embedder = BGEM3Embedder(
        model_name,
        revision=configured_revision,
        batch_size=embedding_batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        normalize_embeddings=normalize_embeddings,
        device=device,
    )
    _verify_normalization(embedder, normalize_embeddings)
    revision = _resolve_model_revision(embedder, configured_revision)
    embedding_contract = build_embedding_contract(
        model=model_name,
        revision=revision,
        max_length=max_length,
        normalize_embeddings=normalize_embeddings,
        use_fp16=use_fp16,
    )

    client = create_qdrant_client(
        host,
        port,
        url=url,
        api_key=api_key,
        https=https,
        timeout=timeout,
    )
    try:
        indexed = index(
            input_path,
            embedder=embedder,
            client=client,
            embedding_contract=embedding_contract,
            collection_name=collection,
            vector_size=vector_size,
            distance=distance,
            upsert_batch_size=upsert_batch_size,
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            recreate=args.recreate,
        )
    except BaseException:
        try:
            _close_qdrant_client(client)
        except BaseException:
            pass
        raise
    else:
        _close_qdrant_client(client)

    return {
        "indexed": indexed,
        "collection": collection,
        "input": str(input_path),
        "embedding_fingerprint": embedding_contract["fingerprint"],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        summary = run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"indexing failed: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
