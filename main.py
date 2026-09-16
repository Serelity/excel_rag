from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml
from pydantic import ValidationError

from pipeline.chunker import (
    CHUNK_SCHEMA_VERSION,
    CHUNK_TEMPLATE_VERSION,
    SOURCE_HASH_VERSION,
    build_problem_chunk,
    source_text_hash,
)
from pipeline.cleaner import NORMALIZATION_VERSION, clean_ticket
from pipeline.extractor import ProblemExtractionError, ProblemExtractor
from pipeline.loader import load_tickets
from schemas.chunk import ProblemChunk, QuarantineRecord
from schemas.ticket import Ticket

LOGGER = logging.getLogger("rag.extract")
PROJECT_ROOT = Path(__file__).resolve().parent
RUN_MANIFEST_VERSION = "3.0.0"
RUN_STATE_INITIALIZING = "initializing"
RUN_STATE_READY = "ready"
QUARANTINE_SCHEMA_VERSION = "1.0.0"
GIT_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
EXTRACTION_RUN_ID_PATTERN = re.compile(r"run_[A-Za-z0-9][A-Za-z0-9_.-]{0,123}")
MAX_BOUNDED_EXTRACTION_LIMIT = 100


@dataclass(slots=True)
class RunStats:
    seen: int = 0
    skipped: int = 0
    submitted: int = 0
    succeeded: int = 0
    failed: int = 0

    @property
    def processed(self) -> int:
        return self.succeeded + self.failed


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    for section in ("data", "pipeline", "llm"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for key in ("input", "output"):
        if not config["data"].get(key):
            raise ValueError(f"Missing configuration value: data.{key}")
    return config


def _integer_setting(value: Any, label: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}")
    return value


def _number_setting(value: Any, label: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} must be a finite number of at least {minimum}")
    return parsed


def _required_code_commit() -> str:
    commit = os.environ.get("RAG_CODE_COMMIT", "")
    if GIT_OBJECT_ID_PATTERN.fullmatch(commit) is None:
        raise ValueError("RAG_CODE_COMMIT must be a 40- or 64-character lowercase Git object ID")
    return commit


def _extraction_run_id() -> str:
    configured = os.environ.get("RAG_EXTRACTION_RUN_ID")
    if configured is None:
        return datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S%fZ")
    if EXTRACTION_RUN_ID_PATTERN.fullmatch(configured) is None:
        raise ValueError(
            "RAG_EXTRACTION_RUN_ID must start with run_ and contain at most 128 safe characters"
        )
    return configured


def _manifest_path(output_path: Path) -> Path:
    return Path(f"{output_path}.manifest.json")


def _lock_path(output_path: Path) -> Path:
    return Path(f"{output_path}.lock")


def _validate_distinct_paths(**paths: Path) -> None:
    resolved: dict[Path, str] = {}
    file_identities: dict[tuple[int, int], str] = {}
    for label, path in paths.items():
        canonical = path.resolve()
        previous = resolved.get(canonical)
        if previous is not None:
            raise ValueError(
                f"Configured paths must be distinct: {previous} and {label} "
                f"both resolve to {canonical}"
            )
        resolved[canonical] = label
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            continue
        identity = (file_stat.st_dev, file_stat.st_ino)
        previous = file_identities.get(identity)
        if previous is not None:
            raise ValueError(
                f"Configured paths must be distinct: {previous} and {label} refer to the same file"
            )
        file_identities[identity] = label


@contextmanager
def _exclusive_output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    lock_handle = os.fdopen(descriptor, "r+", encoding="ascii")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another extraction process holds the output lock: {path}") from exc
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.flush()
        yield
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


@contextmanager
def _exclusive_run_locks(*paths: Path):
    with ExitStack() as stack:
        for path in sorted(set(paths)):
            stack.enter_context(_exclusive_output_lock(path))
        yield


RecordValidator = Callable[[dict[str, Any], Path, int], dict[str, Any]]


def _scan_and_repair_jsonl(
    path: Path,
    *,
    id_key: str | None = None,
    validator: RecordValidator | None = None,
    reject_duplicate_ids: bool = False,
) -> set[str]:
    if not path.exists():
        return set()

    record_ids: set[str] = set()
    truncate_at: int | None = None
    append_newline = False
    with path.open("rb") as handle:
        line_number = 0
        while raw_line := handle.readline():
            line_number += 1
            line_start = handle.tell() - len(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if raw_line.endswith(b"\n"):
                    raise ValueError(
                        f"Cannot resume: invalid JSONL at {path}:{line_number}"
                    ) from exc
                truncate_at = line_start
                LOGGER.warning("Removing incomplete final line from %s", path)
                break
            if not isinstance(record, dict):
                raise ValueError(f"Cannot resume: non-object JSON at {path}:{line_number}")
            if validator is not None:
                record = validator(record, path, line_number)
            if id_key is not None:
                record_id = str(record.get(id_key, "")).strip()
                if not record_id:
                    raise ValueError(f"Cannot resume: missing {id_key!r} at {path}:{line_number}")
                if reject_duplicate_ids and record_id in record_ids:
                    raise ValueError(f"Cannot resume: duplicate {id_key!r} at {path}:{line_number}")
                record_ids.add(record_id)
            append_newline = not raw_line.endswith(b"\n")

    if truncate_at is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_at)
    elif append_newline:
        with path.open("ab") as handle:
            handle.write(b"\n")
    return record_ids


def _validate_chunk_record(
    record: dict[str, Any],
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    try:
        chunk = ProblemChunk.model_validate(record)
    except ValidationError:
        raise ValueError(
            f"Cannot resume: invalid problem chunk contract at {path}:{line_number}"
        ) from None
    return chunk.model_dump(mode="json")


def _validate_quarantine_record(
    record: dict[str, Any],
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    try:
        failure = QuarantineRecord.model_validate(record)
    except ValidationError:
        raise ValueError(
            f"Cannot resume: invalid quarantine record at {path}:{line_number}"
        ) from None
    return failure.model_dump(mode="json", exclude_none=True)


def read_completed_ids(
    path: Path,
    *,
    expected_extraction: Mapping[str, Any] | None = None,
) -> set[str]:
    return set(
        read_completed_hashes(
            path,
            expected_extraction=expected_extraction,
        )
    )


def read_quarantined_ids(
    path: Path,
    *,
    expected_extraction: Mapping[str, Any] | None = None,
) -> set[str]:
    return set(
        read_quarantined_hashes(
            path,
            expected_extraction=expected_extraction,
        )
    )


def read_quarantined_hashes(
    path: Path,
    *,
    expected_extraction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source_hashes: dict[str, str] = {}

    def validate(
        record: dict[str, Any],
        record_path: Path,
        line_number: int,
    ) -> dict[str, Any]:
        validated = _validate_quarantine_record(record, record_path, line_number)
        extraction = validated["extraction"]
        if expected_extraction is not None and (
            extraction["model"] != expected_extraction["model"]
            or extraction["prompt_version"] != expected_extraction["prompt_version"]
        ):
            raise ValueError(
                "Cannot resume: quarantine extraction provenance differs from the run manifest "
                f"at {record_path}:{line_number}"
            )
        ticket_id = validated["ticket_id"]
        source_hash = validated["source_text_hash"]
        previous_hash = source_hashes.get(ticket_id)
        if previous_hash is not None and previous_hash != source_hash:
            raise ValueError(
                "Cannot resume: quarantine contains conflicting source hashes "
                f"for ticket id {ticket_id!r}"
            )
        source_hashes[ticket_id] = source_hash
        return validated

    _scan_and_repair_jsonl(
        path,
        id_key="ticket_id",
        validator=validate,
    )
    return source_hashes


def read_completed_hashes(
    path: Path,
    *,
    expected_extraction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    source_hashes: dict[str, str] = {}

    def validate(
        record: dict[str, Any],
        record_path: Path,
        line_number: int,
    ) -> dict[str, Any]:
        validated = _validate_chunk_record(record, record_path, line_number)
        extraction = validated["extraction"]
        if expected_extraction is not None and (
            extraction["model"] != expected_extraction["model"]
            or extraction["prompt_version"] != expected_extraction["prompt_version"]
        ):
            raise ValueError(
                "Cannot resume: chunk extraction provenance differs from the run manifest "
                f"at {record_path}:{line_number}"
            )
        source_hashes[validated["id"]] = validated["source_text_hash"]
        return validated

    _scan_and_repair_jsonl(
        path,
        id_key="id",
        validator=validate,
        reject_duplicate_ids=True,
    )
    return source_hashes


def _open_or_create(path: Path) -> tuple[int, bool]:
    try:
        return os.open(path, os.O_WRONLY | os.O_APPEND), False
    except FileNotFoundError:
        return os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600), True


def _open_outputs(
    output_path: Path,
    quarantine_path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> tuple[TextIO, TextIO]:
    _validate_distinct_paths(output=output_path, quarantine=quarantine_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and not overwrite:
        existing = [path for path in (output_path, quarantine_path) if path.exists()]
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {names}. Use --resume or --overwrite.")

    descriptors: list[int] = []
    created_paths: list[Path] = []
    try:
        for path in (output_path, quarantine_path):
            if not resume and not overwrite:
                descriptor = os.open(
                    path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                created = True
            else:
                descriptor, created = _open_or_create(path)
            descriptors.append(descriptor)
            if created:
                created_paths.append(path)

        for descriptor, path in zip(descriptors, (output_path, quarantine_path), strict=True):
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError(f"Output path is not a regular file: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is writing output file: {path}") from exc

        output_descriptor, quarantine_descriptor = descriptors
        output_handle = os.fdopen(output_descriptor, "a", encoding="utf-8", buffering=1)
        descriptors[0] = -1
        quarantine_handle = os.fdopen(
            quarantine_descriptor,
            "a",
            encoding="utf-8",
            buffering=1,
        )
        descriptors[1] = -1
        return output_handle, quarantine_handle
    except BaseException:
        for descriptor in descriptors:
            if descriptor >= 0:
                os.close(descriptor)
        for path in created_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _truncate_outputs(*handles: TextIO) -> None:
    for handle in handles:
        handle.flush()
        os.ftruncate(handle.fileno(), 0)
        os.fsync(handle.fileno())


def _input_fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"Input changed while its fingerprint was calculated: {path}")
    return {"sha256": digest.hexdigest(), "size": after.st_size}


def _build_run_manifest(
    input_path: Path,
    output_path: Path,
    quarantine_path: Path,
    extractor: ProblemExtractor,
) -> dict[str, Any]:
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "state": RUN_STATE_READY,
        "code": {"git_commit": _required_code_commit()},
        "input": _input_fingerprint(input_path),
        "files": {
            "quarantine": os.path.relpath(
                quarantine_path.resolve(),
                start=output_path.parent.resolve(),
            )
        },
        "extraction": {
            "model": extractor.model,
            "model_revision": getattr(extractor, "model_revision", None),
            "model_source_repo": getattr(extractor, "model_source_repo", None),
            "model_artifact_fingerprint": getattr(
                extractor,
                "model_artifact_fingerprint",
                None,
            ),
            "prompt_version": extractor.prompt_version,
            "enable_thinking": getattr(extractor, "enable_thinking", False),
            "temperature": extractor.temperature,
            "max_tokens": extractor.max_tokens,
            "max_input_chars": extractor.max_input_chars,
            "seed": extractor.seed,
        },
        "chunk": {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "template_version": CHUNK_TEMPLATE_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "source_hash_version": SOURCE_HASH_VERSION,
        },
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid run manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Run manifest must contain a JSON object: {path}")
    return manifest


def _write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _restore_manifest(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".rollback",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            handle.write(previous)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_resume_manifest(
    path: Path,
    expected: dict[str, Any],
    *,
    output_exists: bool,
    quarantine_exists: bool,
) -> bool:
    if not path.exists():
        if not output_exists and not quarantine_exists:
            return False
        raise ValueError(
            f"Cannot resume existing files without their run manifest: {path}. "
            "Use a new output path or --overwrite."
        )

    current = _read_manifest(path)
    if current.get("state") != RUN_STATE_READY:
        raise ValueError(
            f"Cannot resume: run manifest is not ready: {path}. "
            "The previous overwrite did not finish; rerun with --overwrite."
        )
    if current != expected:
        raise ValueError(
            "Cannot resume: input, code, model, prompt, decoding, or chunk contract differs "
            f"from {path}. Use a new output path or --overwrite."
        )

    missing_files = []
    if not output_exists:
        missing_files.append("completed output")
    if not quarantine_exists:
        missing_files.append("quarantine output")
    if missing_files:
        raise ValueError(
            "Cannot resume: run manifest exists but required result files are missing "
            f"({', '.join(missing_files)}). Use --overwrite to start a new run."
        )
    return True


async def _extract_one(
    extractor: ProblemExtractor,
    ticket: Ticket,
    *,
    extraction_run_id: str,
    max_attempts: int,
    retry_backoff_seconds: float,
    source_hash: str | None = None,
) -> tuple[dict[str, Any] | None, Exception | None, int]:
    if not ticket.content.strip() and not ticket.goal.strip():
        return (
            None,
            ProblemExtractionError(
                "ticket contains no text to extract",
                code="EMPTY_INPUT",
            ),
            1,
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            problem = await extractor.extract(ticket)
            return (
                build_problem_chunk(
                    ticket,
                    problem,
                    extraction_run_id=extraction_run_id,
                    model=extractor.model,
                    prompt_version=extractor.prompt_version,
                    source_hash=source_hash,
                ),
                None,
                attempt,
            )
        except Exception as error:
            last_error = error
            if not _is_retryable(error):
                return None, error, attempt
            if attempt < max_attempts:
                jitter = 0.75 + random.random() * 0.5
                await asyncio.sleep(retry_backoff_seconds * (2 ** (attempt - 1)) * jitter)

    return None, last_error, max_attempts


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, ProblemExtractionError | TypeError | ValueError):
        return False
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500
    return True


def _quarantine_error_code(error: Exception) -> str:
    explicit_code = getattr(error, "code", None)
    if (
        isinstance(explicit_code, str)
        and explicit_code
        and len(explicit_code) <= 64
        and "A" <= explicit_code[0] <= "Z"
        and all(
            "A" <= character <= "Z" or "0" <= character <= "9" or character == "_"
            for character in explicit_code
        )
    ):
        return explicit_code

    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return "UPSTREAM_RATE_LIMITED"
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return "UPSTREAM_HTTP_ERROR"
    if isinstance(error, TimeoutError) or "timeout" in error.__class__.__name__.casefold():
        return "UPSTREAM_TIMEOUT"
    if isinstance(error, ValidationError | TypeError | ValueError):
        return "LOCAL_VALIDATION_ERROR"
    if "connection" in error.__class__.__name__.casefold():
        return "UPSTREAM_CONNECTION_ERROR"
    return "UNEXPECTED_ERROR"


async def run_pipeline(
    config: dict[str, Any],
    *,
    limit: int | None,
    resume: bool,
    overwrite: bool,
    concurrency_override: int | None,
    retry_failures: bool = False,
) -> RunStats:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if retry_failures and not resume:
        raise ValueError("retry_failures requires resume")

    data_config = config["data"]
    input_path = _resolve_project_path(str(data_config["input"]))
    output_path = _resolve_project_path(str(data_config["output"]))
    quarantine_path = _resolve_project_path(
        str(data_config.get("quarantine", "data/processed/problem_chunks.errors.jsonl"))
    )
    manifest_path = _manifest_path(output_path)
    output_lock_path = _lock_path(output_path.resolve())
    quarantine_lock_path = _lock_path(quarantine_path.resolve())

    if not input_path.is_file():
        raise FileNotFoundError(f"Input TSV does not exist: {input_path}")
    _validate_distinct_paths(
        input=input_path,
        output=output_path,
        quarantine=quarantine_path,
        manifest=manifest_path,
        output_lock=output_lock_path,
        quarantine_lock=quarantine_lock_path,
    )

    with _exclusive_run_locks(output_lock_path, quarantine_lock_path):
        return await _run_pipeline_locked(
            config,
            input_path=input_path,
            output_path=output_path,
            quarantine_path=quarantine_path,
            manifest_path=manifest_path,
            limit=limit,
            resume=resume,
            overwrite=overwrite,
            concurrency_override=concurrency_override,
            retry_failures=retry_failures,
        )


async def _run_pipeline_locked(
    config: dict[str, Any],
    *,
    input_path: Path,
    output_path: Path,
    quarantine_path: Path,
    manifest_path: Path,
    limit: int | None,
    resume: bool,
    overwrite: bool,
    concurrency_override: int | None,
    retry_failures: bool,
) -> RunStats:
    pipeline_config = config["pipeline"]

    concurrency_value = (
        concurrency_override
        if concurrency_override is not None
        else pipeline_config.get("concurrency", 4)
    )
    concurrency = _integer_setting(concurrency_value, "pipeline.concurrency", minimum=1)
    max_attempts = _integer_setting(
        pipeline_config.get("max_attempts", 3),
        "pipeline.max_attempts",
        minimum=1,
    )
    retry_backoff_seconds = _number_setting(
        pipeline_config.get("retry_backoff_seconds", 1.0),
        "pipeline.retry_backoff_seconds",
        minimum=0.0,
    )
    checkpoint_every = _integer_setting(
        pipeline_config.get("checkpoint_every", 25),
        "pipeline.checkpoint_every",
        minimum=1,
    )
    log_every = _integer_setting(
        pipeline_config.get("log_every", 100),
        "pipeline.log_every",
        minimum=1,
    )

    extraction_run_id = _extraction_run_id()
    LOGGER.info("extraction_run_id=%s", extraction_run_id)
    extractor = ProblemExtractor(config["llm"])
    stats = RunStats()
    started_at = time.perf_counter()
    pending: set[asyncio.Task] = set()
    output_handle: TextIO | None = None
    quarantine_handle: TextIO | None = None

    try:
        output_existed = output_path.exists()
        quarantine_existed = quarantine_path.exists()

        if not resume and not overwrite:
            existing = [
                path for path in (output_path, quarantine_path, manifest_path) if path.exists()
            ]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise FileExistsError(
                    f"Output already exists: {names}. Use --resume or --overwrite."
                )

        expected_manifest = _build_run_manifest(
            input_path,
            output_path,
            quarantine_path,
            extractor,
        )

        manifest_previous: bytes | None = None
        manifest_attempted = False
        outputs_invalidated = False
        resume_manifest_exists = False
        if resume:
            resume_manifest_exists = _validate_resume_manifest(
                manifest_path,
                expected_manifest,
                output_exists=output_existed,
                quarantine_exists=quarantine_existed,
            )
        elif overwrite:
            if manifest_path.exists():
                manifest_previous = manifest_path.read_bytes()

        try:
            if overwrite:
                initializing_manifest = dict(expected_manifest)
                initializing_manifest["state"] = RUN_STATE_INITIALIZING
                manifest_attempted = True
                _write_manifest_atomic(manifest_path, initializing_manifest)

            output_handle, quarantine_handle = _open_outputs(
                output_path,
                quarantine_path,
                resume=resume,
                overwrite=overwrite,
            )
            if overwrite:
                outputs_invalidated = True
                _truncate_outputs(output_handle, quarantine_handle)
                _write_manifest_atomic(manifest_path, expected_manifest)
            elif not resume or not resume_manifest_exists:
                manifest_attempted = True
                _write_manifest_atomic(manifest_path, expected_manifest)

            if resume:
                completed_hashes = read_completed_hashes(
                    output_path,
                    expected_extraction=expected_manifest["extraction"],
                )
                quarantined_hashes = read_quarantined_hashes(
                    quarantine_path,
                    expected_extraction=expected_manifest["extraction"],
                )
                conflicting_ids = {
                    ticket_id
                    for ticket_id in completed_hashes.keys() & quarantined_hashes.keys()
                    if completed_hashes[ticket_id] != quarantined_hashes[ticket_id]
                }
                if conflicting_ids:
                    raise ValueError(
                        "Cannot resume: completed and quarantine files contain conflicting "
                        f"source hashes for {len(conflicting_ids)} ticket ID(s)"
                    )
            else:
                completed_hashes = {}
                quarantined_hashes = {}
        except BaseException:
            if output_handle is not None:
                output_handle.close()
            if quarantine_handle is not None:
                quarantine_handle.close()
            if manifest_attempted and (not overwrite or not outputs_invalidated):
                _restore_manifest(manifest_path, manifest_previous)
            if not overwrite and not output_existed and not quarantine_existed:
                output_path.unlink(missing_ok=True)
                quarantine_path.unlink(missing_ok=True)
            raise

        def sync_outputs() -> None:
            output_handle.flush()
            quarantine_handle.flush()
            os.fsync(output_handle.fileno())
            os.fsync(quarantine_handle.fileno())

        def record_result(
            ticket: Ticket,
            source_hash: str,
            chunk: dict[str, Any] | None,
            error: Exception | None,
            attempts: int,
        ) -> None:
            if chunk is not None:
                output_handle.write(
                    json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                stats.succeeded += 1
            else:
                error = error or RuntimeError("Unknown extraction failure")
                error_record: dict[str, Any] = {
                    "schema_version": QUARANTINE_SCHEMA_VERSION,
                    "ticket_id": ticket.ticket_id,
                    "source_text_hash": source_hash,
                    "stage": "llm_extraction",
                    "error_code": _quarantine_error_code(error),
                    "error_type": error.__class__.__name__,
                    "attempts": attempts,
                    "extraction_run_id": extraction_run_id,
                    "extraction": {
                        "model": extractor.model,
                        "prompt_version": extractor.prompt_version,
                    },
                }
                status_code = getattr(error, "status_code", None)
                if isinstance(status_code, int) and not isinstance(status_code, bool):
                    error_record["status_code"] = status_code
                validated_error = QuarantineRecord.model_validate(error_record).model_dump(
                    mode="json",
                    exclude_none=True,
                )
                quarantine_handle.write(
                    json.dumps(validated_error, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                stats.failed += 1
                LOGGER.warning(
                    "ticket_id=%s failed after %d attempt(s): %s",
                    ticket.ticket_id,
                    attempts,
                    error.__class__.__name__,
                )

            if stats.processed % checkpoint_every == 0:
                sync_outputs()
            if stats.processed % log_every == 0:
                elapsed = max(time.perf_counter() - started_at, 0.001)
                LOGGER.info(
                    "processed=%d succeeded=%d failed=%d skipped=%d rate=%.2f records/s",
                    stats.processed,
                    stats.succeeded,
                    stats.failed,
                    stats.skipped,
                    stats.processed / elapsed,
                )

        async def drain_completed() -> None:
            nonlocal pending
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                ticket, source_hash, chunk, error, attempts = await task
                record_result(ticket, source_hash, chunk, error, attempts)

        async def process(ticket: Ticket, source_hash: str):
            chunk, error, attempts = await _extract_one(
                extractor,
                ticket,
                extraction_run_id=extraction_run_id,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                source_hash=source_hash,
            )
            return ticket, source_hash, chunk, error, attempts

        with output_handle, quarantine_handle:
            seen_input_ids: set[str] = set()
            input_fully_scanned = True
            for ticket in load_tickets(input_path):
                stats.seen += 1
                if ticket.ticket_id in seen_input_ids:
                    raise ValueError(f"Input contains duplicate ticket id: {ticket.ticket_id!r}")
                seen_input_ids.add(ticket.ticket_id)
                raw_source_hash = source_text_hash(ticket)
                if ticket.ticket_id in completed_hashes:
                    if completed_hashes[ticket.ticket_id] != raw_source_hash:
                        raise ValueError(
                            "Cannot resume: completed chunk source hash differs for ticket id "
                            f"{ticket.ticket_id!r}"
                        )
                    stats.skipped += 1
                    continue
                if ticket.ticket_id in quarantined_hashes:
                    if quarantined_hashes[ticket.ticket_id] != raw_source_hash:
                        raise ValueError(
                            "Cannot resume: quarantine source hash differs for ticket id "
                            f"{ticket.ticket_id!r}"
                        )
                    if not retry_failures:
                        stats.skipped += 1
                        continue
                if limit is not None and stats.submitted >= limit:
                    input_fully_scanned = False
                    break

                ticket = clean_ticket(ticket)
                pending.add(asyncio.create_task(process(ticket, raw_source_hash)))
                stats.submitted += 1
                if len(pending) >= concurrency:
                    await drain_completed()

            while pending:
                await drain_completed()
            if resume and input_fully_scanned:
                foreign_completed = completed_hashes.keys() - seen_input_ids
                foreign_quarantined = quarantined_hashes.keys() - seen_input_ids
                if foreign_completed or foreign_quarantined:
                    details = []
                    if foreign_completed:
                        details.append(f"completed={len(foreign_completed)}")
                    if foreign_quarantined:
                        details.append(f"quarantined={len(foreign_quarantined)}")
                    raise ValueError(
                        "Cannot resume: resume files contain ticket IDs absent from the input "
                        f"({', '.join(details)})"
                    )
            sync_outputs()
    finally:
        primary_error = sys.exception()
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        try:
            close = getattr(extractor, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        except BaseException as close_error:
            if primary_error is None:
                raise
            LOGGER.error(
                "Extractor close failed with %s while preserving primary %s",
                close_error.__class__.__name__,
                primary_error.__class__.__name__,
            )

    elapsed = max(time.perf_counter() - started_at, 0.001)
    LOGGER.info(
        "complete seen=%d submitted=%d succeeded=%d failed=%d skipped=%d elapsed=%.1fs",
        stats.seen,
        stats.submitted,
        stats.succeeded,
        stats.failed,
        stats.skipped,
        elapsed,
    )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract candidate problem chunks from ticket TSV")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/config.yaml",
        help="YAML configuration path",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many new records")
    parser.add_argument("--concurrency", type=int, help="Override pipeline concurrency")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true", help="Append and skip completed IDs")
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and quarantine files",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="With --resume, retry IDs already present in quarantine",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.concurrency is not None and args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.retry_failures and not args.resume:
        parser.error("--retry-failures requires --resume")
    full_extraction_allowed = os.getenv("RAG_ALLOW_FULL_EXTRACTION") == "1"
    if args.limit is None and not full_extraction_allowed:
        parser.error(
            "unbounded extraction is disabled; use deploy/run-extraction-job.sh --full --resume"
        )
    if (
        args.limit is not None
        and args.limit > MAX_BOUNDED_EXTRACTION_LIMIT
        and not full_extraction_allowed
    ):
        parser.error(
            f"bounded extraction accepts at most --limit {MAX_BOUNDED_EXTRACTION_LIMIT}; "
            "use deploy/run-extraction-job.sh --full --resume"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = load_config(args.config.resolve())
        stats = asyncio.run(
            run_pipeline(
                config,
                limit=args.limit,
                resume=args.resume,
                overwrite=args.overwrite,
                concurrency_override=args.concurrency,
                retry_failures=args.retry_failures,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; rerun with --resume to continue")
        return 130

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(cli())
