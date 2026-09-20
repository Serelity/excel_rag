from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .alignment import (
    EvidenceAlignmentError,
    align_extraction,
    merge_extractions,
    shift_extraction,
)
from .prompt import SYSTEM_PROMPT, user_message
from .schema import SemanticExtraction


class ExtractionError(RuntimeError):
    def __init__(self, message: str, *, code: str = "EXTRACTION_ERROR") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ClientConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3-30B-A3B"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 3072
    timeout_seconds: float = 180.0
    max_retries: int = 1
    seed: int = 42
    segment_chars: int = 8000


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    extraction: SemanticExtraction
    model_calls: int
    segments: int
    alignment_repairs: int


def split_document(text: str, max_chars: int) -> list[tuple[int, str]]:
    if max_chars < 1000:
        raise ValueError("segment_chars must be at least 1000")
    if len(text) <= max_chars:
        return [(0, text)]

    segments: list[tuple[int, str]] = []
    start = 0
    boundaries = "\n。！？；;"
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            search_start = max(start + max_chars // 2, hard_end - 800)
            best = max(text.rfind(mark, search_start, hard_end) for mark in boundaries)
            if best >= search_start:
                end = best + 1
        if end <= start:
            end = hard_end
        segments.append((start, text[start:end]))
        start = end
    return segments


class Qwen3ExtractionClient:
    def __init__(self, config: ClientConfig, *, client: Any | None = None) -> None:
        self.config = config
        if client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        else:
            self.client = client

        schema = SemanticExtraction.model_json_schema()
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_extraction_v1",
                "strict": True,
                "schema": schema,
            },
        }

    async def extract_segment(self, content: str) -> tuple[SemanticExtraction, int]:
        if not content.strip():
            return SemanticExtraction(events=[]), 0
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(content)},
                ],
                response_format=self.response_format,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
                timeout=self.config.timeout_seconds,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            raise

        choices = getattr(response, "choices", None)
        if not choices:
            raise ExtractionError("model returned no choices", code="EMPTY_RESPONSE")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in (None, "stop"):
            code = "OUTPUT_TRUNCATED" if finish_reason == "length" else "INCOMPLETE_RESPONSE"
            raise ExtractionError("model response did not finish normally", code=code)
        message = getattr(choice, "message", None)
        raw = getattr(message, "content", None)
        if not isinstance(raw, str) or not raw.strip():
            raise ExtractionError("model returned empty content", code="EMPTY_RESPONSE")

        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExtractionError("model content is not valid JSON", code="INVALID_JSON") from exc
        try:
            extraction = SemanticExtraction.model_validate(value)
            aligned = align_extraction(extraction, content)
        except ValidationError as exc:
            raise ExtractionError(
                "model JSON does not match semantic-extraction-v1",
                code="SCHEMA_VALIDATION_FAILED",
            ) from exc
        except EvidenceAlignmentError as exc:
            raise ExtractionError(str(exc), code="EVIDENCE_ALIGNMENT_FAILED") from exc
        return aligned.extraction, aligned.repairs

    async def extract_document(self, content: str) -> DocumentExtraction:
        if not content.strip():
            raise ExtractionError("case_content is empty", code="EMPTY_CONTENT")

        segments = split_document(content, self.config.segment_chars)
        parts: list[SemanticExtraction] = []
        repairs = 0
        for offset, segment in segments:
            extraction, segment_repairs = await self.extract_segment(segment)
            parts.append(shift_extraction(extraction, offset))
            repairs += segment_repairs
        try:
            merged = merge_extractions(parts)
        except EvidenceAlignmentError as exc:
            raise ExtractionError(str(exc), code="TOO_MANY_EVENTS") from exc
        return DocumentExtraction(
            extraction=merged,
            model_calls=len(segments),
            segments=len(segments),
            alignment_repairs=repairs,
        )
