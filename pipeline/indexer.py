"""Streaming Qdrant indexing for validated problem chunk JSONL files."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

DEFAULT_COLLECTION = "problem_chunks_bge_m3_v1"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_DISTANCE = "cosine"
DEFAULT_UPSERT_BATCH_SIZE = 256
DEFAULT_UPSERT_MAX_ATTEMPTS = 3
DEFAULT_UPSERT_RETRY_BACKOFF_SECONDS = 1.0
EMBEDDING_CONTRACT_VERSION = "1.0.0"
POINT_ID_NAMESPACE = uuid.NAMESPACE_URL
EMBEDDING_CONTRACT_FIELDS = frozenset(
    {
        "contract_version",
        "model",
        "revision",
        "max_length",
        "normalize_embeddings",
        "use_fp16",
        "fingerprint",
    }
)


def _load_qdrant_client_class():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant indexing requires qdrant-client. Install the locked "
            "indexing dependencies before running the indexing CLI."
        ) from exc
    return QdrantClient


def _load_qdrant_models():
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError(
            "Qdrant indexing requires qdrant-client. Install the locked "
            "indexing dependencies before running the indexing CLI."
        ) from exc
    return models


def create_qdrant_client(
    host: str | None = "localhost",
    port: int = 6333,
    *,
    url: str | None = None,
    api_key: str | None = None,
    https: bool | None = None,
    timeout: float = 60.0,
):
    """Create a Qdrant client using either a URL or host/port settings."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("Qdrant timeout must be greater than zero")
    client_kwargs: dict[str, Any] = {"timeout": timeout}
    if url is not None:
        normalized_url = str(url).strip()
        if not normalized_url:
            raise ValueError("Qdrant url must not be empty")
        client_kwargs["url"] = normalized_url
    else:
        if host is None or not str(host).strip():
            raise ValueError("Qdrant host must not be empty when url is not set")
        if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
            raise ValueError("Qdrant port must be greater than zero")
        client_kwargs.update(host=str(host).strip(), port=port)
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Qdrant api_key must be a non-empty string or null")
        client_kwargs["api_key"] = api_key
    if https is not None:
        if not isinstance(https, bool):
            raise ValueError("Qdrant https must be a boolean")
        client_kwargs["https"] = https

    client_class = _load_qdrant_client_class()
    try:
        return client_class(**client_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Unable to configure Qdrant client ({type(exc).__name__})") from None


def _format_validation_error(error: Any) -> str:
    details = []
    for item in error.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in item.get("loc", ())) or "record"
        details.append(f"{location}: {item.get('msg', 'invalid value')}")
    return "; ".join(details) or "record does not match problem-chunk-v1"


def _validate_problem_chunk(record: Any, *, location: str) -> dict[str, Any]:
    try:
        from pydantic import ValidationError

        from schemas.chunk import ProblemChunk
    except ImportError as exc:
        raise RuntimeError(
            "Problem chunk validation requires the locked Pydantic dependency."
        ) from exc

    if not isinstance(record, Mapping):
        raise ValueError(f"Record at {location} must be a JSON object")
    try:
        validated = ProblemChunk.model_validate(record)
    except ValidationError as exc:
        details = _format_validation_error(exc)
        raise ValueError(f"Record at {location} violates problem-chunk-v1: {details}") from None
    return validated.model_dump(mode="json")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield strict problem-chunk-v1 records without loading the whole file."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source_path} at line {line_number}: {exc.msg}"
                ) from exc
            yield _validate_problem_chunk(
                record,
                location=f"{source_path}:{line_number}",
            )


def preflight_jsonl(path: str | Path) -> int:
    """Validate every record and reject empty inputs before any database mutation."""

    record_count = sum(1 for _ in iter_jsonl(path))
    if record_count == 0:
        raise ValueError(f"Input JSONL contains no problem chunks: {Path(path)}")
    return record_count


def batched(items: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def stable_point_id(source_id: str | int) -> str:
    normalized_id = str(source_id).strip()
    if not normalized_id:
        raise ValueError("source id must not be empty")
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"rag:problem-chunk:{normalized_id}"))


def build_embedding_contract(
    *,
    model: str,
    revision: str,
    max_length: int,
    normalize_embeddings: bool,
    use_fp16: bool = True,
) -> dict[str, Any]:
    """Build a stable contract persisted with every indexed point."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("embedding model must not be empty")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("embedding model revision must not be empty")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("embedding max_length must be greater than zero")
    if not isinstance(normalize_embeddings, bool):
        raise ValueError("normalize_embeddings must be a boolean")
    if not isinstance(use_fp16, bool):
        raise ValueError("use_fp16 must be a boolean")

    settings: dict[str, Any] = {
        "contract_version": EMBEDDING_CONTRACT_VERSION,
        "model": model.strip(),
        "revision": revision.strip(),
        "max_length": max_length,
        "normalize_embeddings": normalize_embeddings,
        "use_fp16": use_fp16,
    }
    canonical = json.dumps(settings, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    settings["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return settings


def _validated_embedding_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("embedding_contract must be a mapping")
    supplied_fields = set(contract)
    if supplied_fields != EMBEDDING_CONTRACT_FIELDS:
        missing = sorted(EMBEDDING_CONTRACT_FIELDS - supplied_fields)
        unexpected = sorted(supplied_fields - EMBEDDING_CONTRACT_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError(f"Invalid embedding contract ({'; '.join(details)})")
    if contract.get("contract_version") != EMBEDDING_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported embedding contract version; create a new collection "
            "for a different contract"
        )

    expected = build_embedding_contract(
        model=contract.get("model"),
        revision=contract.get("revision"),
        max_length=contract.get("max_length"),
        normalize_embeddings=contract.get("normalize_embeddings"),
        use_fp16=contract.get("use_fp16"),
    )
    if dict(contract) != expected:
        raise ValueError("Embedding contract fingerprint or settings are inconsistent")
    return expected


def _distance_value(distance: str, models: Any) -> Any:
    if not isinstance(distance, str):
        raise ValueError("Qdrant distance must be a string")
    normalized = distance.strip().lower()
    choices = {
        "cosine": "COSINE",
        "dot": "DOT",
        "euclid": "EUCLID",
        "euclidean": "EUCLID",
        "manhattan": "MANHATTAN",
    }
    member = choices.get(normalized)
    if member is None or not hasattr(models.Distance, member):
        supported = ", ".join(sorted(choices))
        raise ValueError(f"Unsupported Qdrant distance {distance!r}; use one of {supported}")
    return getattr(models.Distance, member)


def _existing_vector_params(collection_info: Any) -> Any:
    try:
        return collection_info.config.params.vectors
    except AttributeError as exc:
        raise RuntimeError("Unable to inspect the existing Qdrant collection") from exc


def _validate_existing_embedding_contract(
    client: Any,
    collection_name: str,
    expected: Mapping[str, Any],
) -> None:
    records, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=["embedding"],
        with_vectors=False,
    )
    if not records:
        return
    payload = getattr(records[0], "payload", None)
    existing = payload.get("embedding") if isinstance(payload, Mapping) else None
    existing_fingerprint = existing.get("fingerprint") if isinstance(existing, Mapping) else None
    expected_fingerprint = expected.get("fingerprint")
    if not existing_fingerprint:
        raise ValueError(
            f"Collection {collection_name!r} has no embedding contract; "
            "use a new collection or explicitly recreate it"
        )
    if existing_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Collection {collection_name!r} uses embedding fingerprint "
            f"{existing_fingerprint}; expected {expected_fingerprint}. "
            "Use a new collection or explicitly recreate it."
        )


def ensure_collection(
    client: Any,
    collection_name: str = DEFAULT_COLLECTION,
    *,
    embedding_contract: Mapping[str, Any],
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: str = DEFAULT_DISTANCE,
    recreate: bool = False,
) -> None:
    """Create or validate a collection; deletion requires recreate=True."""

    if not collection_name:
        raise ValueError("Qdrant collection name must not be empty")
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than zero")

    validated_contract = _validated_embedding_contract(embedding_contract)
    models = _load_qdrant_models()
    distance_value = _distance_value(distance, models)
    exists = client.collection_exists(collection_name=collection_name)

    if exists and recreate:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance_value),
        )
        return

    params = _existing_vector_params(client.get_collection(collection_name=collection_name))
    if isinstance(params, Mapping):
        raise ValueError(
            f"Collection {collection_name!r} uses named vectors; a single dense vector is required"
        )
    existing_size = getattr(params, "size", None)
    existing_distance = getattr(params, "distance", None)
    if existing_size != vector_size or existing_distance != distance_value:
        raise ValueError(
            f"Collection {collection_name!r} has vector config "
            f"size={existing_size}, distance={existing_distance}; expected "
            f"size={vector_size}, distance={distance_value}. Use --recreate "
            "only if deleting the existing collection is intended."
        )
    _validate_existing_embedding_contract(client, collection_name, validated_contract)


def _validated_vector(vector: Any, vector_size: int, source_id: str) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Embedding for source id {source_id!r} is not numeric") from exc
    if len(values) != vector_size:
        raise ValueError(
            f"Embedding for source id {source_id!r} has dimension {len(values)}; "
            f"expected {vector_size}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Embedding for source id {source_id!r} contains NaN or Inf")
    return values


def _is_transient_qdrant_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError | ConnectionError | OSError):
        return True
    try:
        from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    except ImportError:
        return False
    if isinstance(error, ResponseHandlingException):
        return True
    if isinstance(error, UnexpectedResponse):
        return error.status_code in {408, 425, 429, 500, 502, 503, 504}
    return False


def _safe_source_id(source_id: Any) -> str:
    representation = repr(str(source_id))
    return representation if len(representation) <= 100 else f"{representation[:97]}..."


def _upsert_with_retry(
    client: Any,
    *,
    collection_name: str,
    points: list[Any],
    wait: bool,
    batch_number: int,
    first_source_id: str,
    last_source_id: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    sleep: Callable[[float], None],
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            client.upsert(collection_name=collection_name, points=points, wait=wait)
            return
        except Exception as exc:
            retryable = _is_transient_qdrant_error(exc)
            if retryable and attempt < max_attempts:
                sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            status = getattr(exc, "status_code", None)
            status_note = f", HTTP status {status}" if status is not None else ""
            raise RuntimeError(
                f"Qdrant upsert failed for batch {batch_number} "
                f"(source IDs {_safe_source_id(first_source_id)} through "
                f"{_safe_source_id(last_source_id)}) after {attempt} attempt(s); "
                f"error type {type(exc).__name__}{status_note}"
            ) from None


def upsert_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    embedder: Any,
    client: Any,
    embedding_contract: Mapping[str, Any],
    collection_name: str = DEFAULT_COLLECTION,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    max_attempts: int = DEFAULT_UPSERT_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_UPSERT_RETRY_BACKOFF_SECONDS,
    wait: bool = True,
    point_factory: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Embed and upsert a stream of chunks, returning the indexed count."""

    if (
        isinstance(upsert_batch_size, bool)
        or not isinstance(upsert_batch_size, int)
        or upsert_batch_size <= 0
    ):
        raise ValueError("upsert_batch_size must be greater than zero")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if (
        isinstance(retry_backoff_seconds, bool)
        or not isinstance(retry_backoff_seconds, int | float)
        or not math.isfinite(retry_backoff_seconds)
        or retry_backoff_seconds < 0
    ):
        raise ValueError("retry_backoff_seconds must not be negative")

    validated_contract = _validated_embedding_contract(embedding_contract)

    def validated_chunks() -> Iterator[dict[str, Any]]:
        for record_number, chunk in enumerate(chunks, start=1):
            yield _validate_problem_chunk(
                chunk,
                location=f"upsert input record {record_number}",
            )

    indexed = 0
    for batch_number, chunk_batch in enumerate(
        batched(validated_chunks(), upsert_batch_size), start=1
    ):
        if point_factory is None:
            point_factory = _load_qdrant_models().PointStruct
        texts = [chunk["text"] for chunk in chunk_batch]
        vectors = embedder.encode(texts)
        if len(vectors) != len(chunk_batch):
            raise RuntimeError(
                f"Embedder returned the wrong vector count for batch {batch_number} "
                f"(source IDs {_safe_source_id(chunk_batch[0]['id'])} through "
                f"{_safe_source_id(chunk_batch[-1]['id'])})"
            )

        points = []
        for chunk, vector in zip(chunk_batch, vectors, strict=True):
            source_id = str(chunk["id"])
            payload = {
                "schema_version": chunk["schema_version"],
                "source_id": source_id,
                "type": chunk["type"],
                "text": chunk["text"],
                "source_text_hash": chunk["source_text_hash"],
                "extraction_run_id": chunk["extraction_run_id"],
                "extraction": dict(chunk["extraction"]),
                "metadata": dict(chunk["metadata"]),
                "embedding": dict(validated_contract),
            }
            points.append(
                point_factory(
                    id=stable_point_id(source_id),
                    vector=_validated_vector(vector, vector_size, source_id),
                    payload=payload,
                )
            )

        _upsert_with_retry(
            client,
            collection_name=collection_name,
            points=points,
            wait=wait,
            batch_number=batch_number,
            first_source_id=str(chunk_batch[0]["id"]),
            last_source_id=str(chunk_batch[-1]["id"]),
            max_attempts=max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep=sleep,
        )
        indexed += len(points)
    return indexed


def index(
    input_path: str | Path,
    *,
    embedder: Any,
    client: Any,
    embedding_contract: Mapping[str, Any],
    collection_name: str = DEFAULT_COLLECTION,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    distance: str = DEFAULT_DISTANCE,
    upsert_batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    max_attempts: int = DEFAULT_UPSERT_MAX_ATTEMPTS,
    retry_backoff_seconds: float = DEFAULT_UPSERT_RETRY_BACKOFF_SECONDS,
    recreate: bool = False,
    wait: bool = True,
) -> int:
    """Preflight, ensure the collection, then stream the JSONL into it."""

    expected_count = preflight_jsonl(input_path)
    ensure_collection(
        client,
        collection_name,
        embedding_contract=embedding_contract,
        vector_size=vector_size,
        distance=distance,
        recreate=recreate,
    )
    indexed = upsert_chunks(
        iter_jsonl(input_path),
        embedder=embedder,
        client=client,
        embedding_contract=embedding_contract,
        collection_name=collection_name,
        vector_size=vector_size,
        upsert_batch_size=upsert_batch_size,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        wait=wait,
    )
    if indexed != expected_count:
        raise RuntimeError(
            f"Input changed during indexing: preflight found {expected_count} records, "
            f"but {indexed} were indexed"
        )
    return indexed
