from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from .cache import ExtractionCache, contract_key
from .client import ClientConfig, DocumentExtraction, ExtractionError, Qwen3ExtractionClient
from .loader import SourceRecord, load_records
from .schema import PROMPT_VERSION, SCHEMA_VERSION, SemanticExtraction
from .views import build_retrieval_views

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is Linux; this enables local Windows tests.
    fcntl = None  # type: ignore[assignment]
    import msvcrt

MANIFEST_VERSION = "1.0.0"


@dataclass(slots=True)
class RunStats:
    scanned: int = 0
    skipped: int = 0
    submitted: int = 0
    succeeded: int = 0
    failed: int = 0
    cache_hits: int = 0
    model_calls: int = 0
    alignment_repairs: int = 0


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            os.chmod(temporary, 0o600)
            json.dump(value, target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+", encoding="ascii") as handle:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                handle.seek(0)
                if not handle.read(1):
                    handle.write(" ")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError) as exc:
            raise RuntimeError(f"another extraction process holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _repair_and_read_jsonl(path: Path, *, id_key: str) -> dict[str, str]:
    if not path.exists():
        return {}
    records: dict[str, str] = {}
    truncate_at: int | None = None
    append_newline = False
    with path.open("rb") as source:
        line_number = 0
        while raw := source.readline():
            line_number += 1
            start = source.tell() - len(raw)
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if raw.endswith(b"\n"):
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                truncate_at = start
                break
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            source_id = value.get(id_key)
            source_hash = value.get("content_sha256")
            if not isinstance(source_id, str) or not source_id:
                raise ValueError(f"missing {id_key} at {path}:{line_number}")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError(f"invalid content_sha256 at {path}:{line_number}")
            if source_id in records:
                raise ValueError(f"duplicate {id_key} at {path}:{line_number}")
            if id_key == "source_id" and "result" in value:
                SemanticExtraction.model_validate(value["result"])
            records[source_id] = source_hash
            append_newline = not raw.endswith(b"\n")
    if truncate_at is not None:
        with path.open("r+b") as target:
            target.truncate(truncate_at)
    elif append_newline:
        with path.open("ab") as target:
            target.write(b"\n")
    return records


def _contract(config: ClientConfig, model_fingerprint: str | None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "model": config.model,
        "model_artifact_fingerprint": model_fingerprint,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
        "segment_chars": config.segment_chars,
        "thinking": False,
    }


def _manifest(
    *,
    input_path: Path,
    input_hash: str,
    output_path: Path,
    errors_path: Path,
    cache_path: Path,
    run_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "state": "ready",
        "run_id": run_id,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_hash,
            "size_bytes": input_path.stat().st_size,
        },
        "files": {
            "output": str(output_path.resolve()),
            "errors": str(errors_path.resolve()),
            "cache": str(cache_path.resolve()),
        },
        "contract": dict(contract),
        "code_commit": os.getenv("RAG_CODE_COMMIT") or None,
    }


def _open_output(path: Path, *, mode: str) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if mode == "exclusive":
        flags |= os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "a", encoding="utf-8", buffering=1)


def _error_code(error: BaseException) -> str:
    code = getattr(error, "code", None)
    if isinstance(code, str) and code and len(code) <= 64:
        return code
    status = getattr(error, "status_code", None)
    if status == 429:
        return "UPSTREAM_RATE_LIMITED"
    if isinstance(status, int):
        return "UPSTREAM_HTTP_ERROR"
    if "timeout" in type(error).__name__.casefold():
        return "UPSTREAM_TIMEOUT"
    if "connection" in type(error).__name__.casefold():
        return "UPSTREAM_CONNECTION_ERROR"
    return "UNEXPECTED_ERROR"


def _retryable(error: BaseException) -> bool:
    if isinstance(error, ExtractionError | ValueError | TypeError):
        return False
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status in {408, 409, 429} or status >= 500
    return True


async def _extract_with_attempts(
    client: Qwen3ExtractionClient,
    content: str,
    *,
    max_attempts: int,
) -> tuple[DocumentExtraction | None, BaseException | None, int]:
    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.extract_document(content), None, attempt
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt | SystemExit | asyncio.CancelledError):
                raise
            last = exc
            if not _retryable(exc) or attempt == max_attempts:
                break
            await asyncio.sleep((2 ** (attempt - 1)) * (0.75 + random.random() * 0.5))
    return None, last, max_attempts


async def _process_batch(
    records: list[SourceRecord],
    *,
    client: Qwen3ExtractionClient,
    cache: ExtractionCache,
    contract: Mapping[str, Any],
    concurrency: int,
    max_attempts: int,
) -> dict[str, tuple[DocumentExtraction | None, BaseException | None, int, bool]]:
    by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_hash[content_sha256(record.case_content)].append(record)

    results: dict[str, tuple[DocumentExtraction | None, BaseException | None, int, bool]] = {}
    misses: dict[str, SourceRecord] = {}
    for source_hash, group in by_hash.items():
        key = contract_key(contract, source_hash)
        cached = cache.get(key, content_sha256=source_hash, source=group[0].case_content)
        if cached is not None:
            results[source_hash] = (
                DocumentExtraction(cached, 0, 0, 0),
                None,
                0,
                True,
            )
        else:
            misses[source_hash] = group[0]

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(record: SourceRecord):
        async with semaphore:
            return await _extract_with_attempts(
                client,
                record.case_content,
                max_attempts=max_attempts,
            )

    hashes = list(misses)
    calls = await asyncio.gather(*(run_one(misses[source_hash]) for source_hash in hashes))
    for source_hash, (document, error, attempts) in zip(hashes, calls, strict=True):
        results[source_hash] = (document, error, attempts, False)
        if document is not None:
            cache.put(contract_key(contract, source_hash), source_hash, document.extraction)
    cache.commit()
    return results


async def run_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    cache_path: Path,
    client_config: ClientConfig,
    limit: int | None,
    full: bool,
    resume: bool,
    overwrite: bool,
    concurrency: int,
    max_attempts: int,
    checkpoint_every: int = 25,
    client: Qwen3ExtractionClient | None = None,
) -> RunStats:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if limit is None and not full:
        raise ValueError("an explicit --limit is required unless --full is used")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if concurrency < 1 or max_attempts < 1 or checkpoint_every < 1:
        raise ValueError("concurrency, max_attempts and checkpoint_every must be positive")
    resolved = [path.resolve() for path in (input_path, output_path, errors_path, cache_path)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input, output, errors and cache paths must be distinct")

    manifest_path = Path(f"{output_path}.manifest.json")
    lock_path = Path(f"{output_path}.lock")
    model_fingerprint = os.getenv("QWEN_MODEL_FINGERPRINT_SHA256") or None
    contract = _contract(client_config, model_fingerprint)
    input_hash = file_sha256(input_path)
    stats = RunStats()

    with _exclusive_lock(lock_path):
        if resume:
            if not manifest_path.is_file():
                raise ValueError(f"cannot resume without manifest: {manifest_path}")
            missing_results = [path for path in (output_path, errors_path) if not path.is_file()]
            if missing_results:
                raise ValueError(f"cannot resume with missing result files: {missing_results}")
            with manifest_path.open("r", encoding="utf-8") as source:
                existing_manifest = json.load(source)
            run_id = existing_manifest.get("run_id")
            expected = _manifest(
                input_path=input_path,
                input_hash=input_hash,
                output_path=output_path,
                errors_path=errors_path,
                cache_path=cache_path,
                run_id=run_id,
                contract=contract,
            )
            if existing_manifest != expected:
                raise ValueError(
                    "resume manifest differs from the current input or extraction contract"
                )
            completed = _repair_and_read_jsonl(output_path, id_key="source_id")
            failed = _repair_and_read_jsonl(errors_path, id_key="source_id")
            output_mode = "append"
        else:
            existing = [path for path in (output_path, errors_path, manifest_path) if path.exists()]
            if existing and not overwrite:
                raise FileExistsError(
                    f"result files already exist: {existing}; use --resume or --overwrite"
                )
            run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S%fZ")
            completed = {}
            failed = {}
            for path in (output_path, errors_path):
                if overwrite:
                    path.unlink(missing_ok=True)
            current_manifest = _manifest(
                input_path=input_path,
                input_hash=input_hash,
                output_path=output_path,
                errors_path=errors_path,
                cache_path=cache_path,
                run_id=run_id,
                contract=contract,
            )
            _atomic_json(manifest_path, current_manifest)
            output_mode = "exclusive"

        overlap = set(completed) & set(failed)
        if overlap:
            raise ValueError("source IDs occur in both successful and quarantine outputs")

        extraction_client = client or Qwen3ExtractionClient(client_config)
        with (
            _open_output(output_path, mode=output_mode) as output,
            _open_output(errors_path, mode=output_mode) as errors,
            ExtractionCache(cache_path) as cache,
        ):
            batch: list[SourceRecord] = []
            seen_input: dict[str, str] = {}
            writes_since_checkpoint = 0

            async def flush_batch() -> None:
                nonlocal writes_since_checkpoint
                if not batch:
                    return
                results = await _process_batch(
                    batch,
                    client=extraction_client,
                    cache=cache,
                    contract=contract,
                    concurrency=concurrency,
                    max_attempts=max_attempts,
                )
                emitted_hashes: set[str] = set()
                for record in batch:
                    source_hash = content_sha256(record.case_content)
                    document, error, attempts, cache_hit = results[source_hash]
                    if document is not None:
                        reused = cache_hit or source_hash in emitted_hashes
                        model_calls = 0 if reused else document.model_calls
                        segments = 0 if reused else document.segments
                        alignment_repairs = 0 if reused else document.alignment_repairs
                        value = {
                            "schema_version": SCHEMA_VERSION,
                            "source_id": record.source_id,
                            "source_row": record.source_row,
                            "content_sha256": source_hash,
                            "extraction_run_id": run_id,
                            "result": document.extraction.model_dump(mode="json"),
                            "retrieval_views": build_retrieval_views(document.extraction),
                            "provenance": contract,
                            "processing": {
                                "cache_hit": reused,
                                "attempts": 0 if reused else attempts,
                                "model_calls": model_calls,
                                "segments": segments,
                                "alignment_repairs": alignment_repairs,
                            },
                        }
                        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                        output.write("\n")
                        completed[record.source_id] = source_hash
                        stats.succeeded += 1
                        stats.cache_hits += int(reused)
                        stats.model_calls += model_calls
                        stats.alignment_repairs += alignment_repairs
                        emitted_hashes.add(source_hash)
                    else:
                        assert error is not None
                        value = {
                            "schema_version": SCHEMA_VERSION,
                            "source_id": record.source_id,
                            "source_row": record.source_row,
                            "content_sha256": source_hash,
                            "extraction_run_id": run_id,
                            "stage": "qwen3_semantic_extraction",
                            "error_code": _error_code(error),
                            "error_type": type(error).__name__,
                            "attempts": attempts,
                            "provenance": contract,
                        }
                        errors.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                        errors.write("\n")
                        failed[record.source_id] = source_hash
                        stats.failed += 1
                    writes_since_checkpoint += 1
                    if writes_since_checkpoint >= checkpoint_every:
                        output.flush()
                        errors.flush()
                        os.fsync(output.fileno())
                        os.fsync(errors.fileno())
                        writes_since_checkpoint = 0
                batch.clear()

            for record in load_records(input_path):
                stats.scanned += 1
                source_hash = content_sha256(record.case_content)
                previous = seen_input.get(record.source_id)
                if previous is not None:
                    raise ValueError(f"input contains duplicate source_id: {record.source_id}")
                seen_input[record.source_id] = source_hash

                prior_hash = completed.get(record.source_id) or failed.get(record.source_id)
                if prior_hash is not None:
                    if prior_hash != source_hash:
                        raise ValueError(
                            f"source content changed for completed source_id: {record.source_id}"
                        )
                    stats.skipped += 1
                    continue
                if limit is not None and stats.submitted >= limit:
                    break
                batch.append(record)
                stats.submitted += 1
                if len(batch) >= max(concurrency * 4, checkpoint_every):
                    await flush_batch()
            await flush_batch()
            output.flush()
            errors.flush()
            os.fsync(output.fileno())
            os.fsync(errors.fileno())
    return stats
