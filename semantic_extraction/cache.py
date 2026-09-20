from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .alignment import validate_aligned_extraction
from .schema import SemanticExtraction


def contract_key(contract: Mapping[str, Any], content_sha256: str) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded + b"\0" + content_sha256.encode("ascii"))
    return digest.hexdigest()


class ExtractionCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_cache (
                cache_key TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                extraction_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()

    def get(
        self,
        key: str,
        *,
        content_sha256: str,
        source: str,
    ) -> SemanticExtraction | None:
        row = self.connection.execute(
            "SELECT content_sha256, extraction_json FROM extraction_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        stored_hash, raw = row
        if stored_hash != content_sha256:
            raise ValueError("extraction cache key has a conflicting content hash")
        try:
            extraction = SemanticExtraction.model_validate_json(raw)
        except Exception as exc:
            raise ValueError("extraction cache contains an invalid result") from exc
        validate_aligned_extraction(extraction, source)
        return extraction

    def put(self, key: str, content_sha256: str, extraction: SemanticExtraction) -> None:
        serialized = extraction.model_dump_json()
        self.connection.execute(
            """
            INSERT INTO extraction_cache(cache_key, content_sha256, extraction_json)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO NOTHING
            """,
            (key, content_sha256, serialized),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ExtractionCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
