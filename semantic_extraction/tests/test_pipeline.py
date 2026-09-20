from __future__ import annotations

import json

import pytest

from semantic_extraction.client import ClientConfig, DocumentExtraction
from semantic_extraction.pipeline import run_pipeline
from semantic_extraction.schema import SemanticExtraction


class FakeExtractionClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def extract_document(self, content: str) -> DocumentExtraction:
        self.calls.append(content)
        return DocumentExtraction(
            extraction=SemanticExtraction(events=[]),
            model_calls=1,
            segments=1,
            grounded_spans=0,
            ambiguous_span_matches=0,
        )


def write_input(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_pipeline_deduplicates_content_and_resumes(tmp_path) -> None:
    source = tmp_path / "pilot.jsonl"
    output = tmp_path / "output.jsonl"
    errors = tmp_path / "errors.jsonl"
    cache = tmp_path / "cache.sqlite3"
    write_input(
        source,
        [
            {"source_id": "1", "source_row": 1, "case_content": "相同正文"},
            {"source_id": "2", "source_row": 2, "case_content": "相同正文"},
            {"source_id": "3", "source_row": 3, "case_content": "另一正文"},
        ],
    )
    fake = FakeExtractionClient()
    config = ClientConfig(model="fake")

    first = await run_pipeline(
        input_path=source,
        output_path=output,
        errors_path=errors,
        cache_path=cache,
        client_config=config,
        limit=2,
        full=False,
        resume=False,
        overwrite=False,
        concurrency=2,
        max_attempts=1,
        client=fake,
    )
    second = await run_pipeline(
        input_path=source,
        output_path=output,
        errors_path=errors,
        cache_path=cache,
        client_config=config,
        limit=1,
        full=False,
        resume=True,
        overwrite=False,
        concurrency=2,
        max_attempts=1,
        client=fake,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert first.succeeded == 2
    assert first.model_calls == 1
    assert first.cache_hits == 1
    assert second.skipped == 2
    assert second.succeeded == 1
    assert fake.calls == ["相同正文", "另一正文"]
    assert rows[0]["processing"]["cache_hit"] is False
    assert rows[1]["processing"]["cache_hit"] is True
    assert errors.read_text(encoding="utf-8") == ""
