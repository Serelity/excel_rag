from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .alignment import EvidenceAlignmentError, merge_extractions, shift_extraction
from .grounding import EvidenceGroundingError, GroundingResult, ground_extraction
from .prompt import SYSTEM_PROMPT, TRUNCATION_RECOVERY_PROMPT, user_message
from .schema import ModelSemanticExtraction, SemanticExtraction


class ExtractionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "EXTRACTION_ERROR",
        processing: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.processing = processing or {}


@dataclass(frozen=True, slots=True)
class ClientConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3-30B-A3B"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 6144
    timeout_seconds: float = 180.0
    max_retries: int = 1
    seed: int = 42
    segment_chars: int = 8000


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    extraction: SemanticExtraction
    model_calls: int
    segments: int
    grounded_spans: int
    ambiguous_span_matches: int
    proposed_evidence_quotes: int
    rejected_evidence_quotes: int
    trigger_fallbacks: int
    dropped_events: int
    polarity_repairs: int
    merged_duplicate_events: int
    truncation_retries: int
    truncation_recoveries: int


@dataclass(frozen=True, slots=True)
class _SegmentExtraction:
    grounding: GroundingResult
    model_calls: int
    truncation_retries: int
    truncation_recoveries: int


def _response_format(schema: dict[str, Any], *, name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _compact_recovery_schema(schema: dict[str, Any]) -> dict[str, Any]:
    compact = deepcopy(schema)
    definitions = compact["$defs"]
    definitions["EvidenceQuote"]["properties"]["text"]["maxLength"] = 48
    event_properties = definitions["ModelExtractedEvent"]["properties"]
    for field in (
        "actors",
        "objects",
        "behaviors",
        "impacts",
        "requests",
        "locations",
        "time_expressions",
    ):
        event_properties[field]["maxItems"] = 2
    event_properties["search_terms"]["maxItems"] = 4
    return compact


def _request_processing(
    *, model_calls: int, truncation_retries: int, truncation_recoveries: int = 0
) -> dict[str, int]:
    return {
        "model_calls": model_calls,
        "truncation_retries": truncation_retries,
        "truncation_recoveries": truncation_recoveries,
    }


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

        schema = ModelSemanticExtraction.model_json_schema()
        self.response_format = _response_format(
            schema,
            name="case_content_retrieval_issue_v4",
        )
        self.recovery_response_format = _response_format(
            _compact_recovery_schema(schema),
            name="case_content_retrieval_issue_v4_recovery",
        )

    async def extract_segment(self, content: str) -> _SegmentExtraction:
        if not content.strip():
            return _SegmentExtraction(
                grounding=GroundingResult(
                    extraction=SemanticExtraction(events=[]),
                    grounded_spans=0,
                    ambiguous_matches=0,
                    proposed_evidence_quotes=0,
                    rejected_evidence_quotes=0,
                    trigger_fallbacks=0,
                    dropped_events=0,
                    polarity_repairs=0,
                ),
                model_calls=0,
                truncation_retries=0,
                truncation_recoveries=0,
            )

        for recovery in (False, True):
            model_calls = 1 + int(recovery)
            truncation_retries = int(recovery)
            processing = _request_processing(
                model_calls=model_calls,
                truncation_retries=truncation_retries,
            )
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": TRUNCATION_RECOVERY_PROMPT if recovery else SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": user_message(content)},
                ],
                response_format=(
                    self.recovery_response_format if recovery else self.response_format
                ),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
                timeout=self.config.timeout_seconds,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            choices = getattr(response, "choices", None)
            if not choices:
                raise ExtractionError(
                    "model returned no choices",
                    code="EMPTY_RESPONSE",
                    processing=processing,
                )
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length" and not recovery:
                continue
            if finish_reason not in (None, "stop"):
                code = "OUTPUT_TRUNCATED" if finish_reason == "length" else "INCOMPLETE_RESPONSE"
                raise ExtractionError(
                    "model response did not finish normally",
                    code=code,
                    processing=processing,
                )
            message = getattr(choice, "message", None)
            raw = getattr(message, "content", None)
            if not isinstance(raw, str) or not raw.strip():
                raise ExtractionError(
                    "model returned empty content",
                    code="EMPTY_RESPONSE",
                    processing=processing,
                )

            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ExtractionError(
                    "model content is not valid JSON",
                    code="INVALID_JSON",
                    processing=processing,
                ) from exc
            try:
                model_result = ModelSemanticExtraction.model_validate(value)
                grounded = ground_extraction(model_result, content)
            except ValidationError as exc:
                raise ExtractionError(
                    "model JSON does not match case-content-retrieval-issue-v4",
                    code="SCHEMA_VALIDATION_FAILED",
                    processing=processing,
                ) from exc
            except EvidenceGroundingError as exc:
                raise ExtractionError(
                    str(exc),
                    code=exc.code,
                    processing={**processing, **exc.processing},
                ) from exc
            return _SegmentExtraction(
                grounding=grounded,
                model_calls=model_calls,
                truncation_retries=truncation_retries,
                truncation_recoveries=int(recovery),
            )

        raise AssertionError("truncation recovery loop exited unexpectedly")

    async def extract_document(self, content: str) -> DocumentExtraction:
        if not content.strip():
            raise ExtractionError("case_content is empty", code="EMPTY_CONTENT")

        segments = split_document(content, self.config.segment_chars)
        parts: list[SemanticExtraction] = []
        grounded_spans = 0
        ambiguous_matches = 0
        proposed_evidence_quotes = 0
        rejected_evidence_quotes = 0
        trigger_fallbacks = 0
        dropped_events = 0
        polarity_repairs = 0
        model_calls = 0
        truncation_retries = 0
        truncation_recoveries = 0
        for offset, segment in segments:
            try:
                segment_result = await self.extract_segment(segment)
            except ExtractionError as exc:
                processing = dict(exc.processing)
                processing["model_calls"] = model_calls + processing.get("model_calls", 0)
                processing["truncation_retries"] = truncation_retries + processing.get(
                    "truncation_retries", 0
                )
                processing["truncation_recoveries"] = truncation_recoveries + processing.get(
                    "truncation_recoveries", 0
                )
                raise ExtractionError(
                    str(exc),
                    code=exc.code,
                    processing=processing,
                ) from exc
            grounded = segment_result.grounding
            parts.append(shift_extraction(grounded.extraction, offset))
            model_calls += segment_result.model_calls
            truncation_retries += segment_result.truncation_retries
            truncation_recoveries += segment_result.truncation_recoveries
            grounded_spans += grounded.grounded_spans
            ambiguous_matches += grounded.ambiguous_matches
            proposed_evidence_quotes += grounded.proposed_evidence_quotes
            rejected_evidence_quotes += grounded.rejected_evidence_quotes
            trigger_fallbacks += grounded.trigger_fallbacks
            dropped_events += grounded.dropped_events
            polarity_repairs += grounded.polarity_repairs
        event_count_before_merge = sum(len(part.events) for part in parts)
        try:
            merged = merge_extractions(parts)
        except EvidenceAlignmentError as exc:
            raise ExtractionError(
                str(exc),
                code="TOO_MANY_EVENTS",
                processing=_request_processing(
                    model_calls=model_calls,
                    truncation_retries=truncation_retries,
                    truncation_recoveries=truncation_recoveries,
                ),
            ) from exc
        return DocumentExtraction(
            extraction=merged,
            model_calls=model_calls,
            segments=len(segments),
            grounded_spans=grounded_spans,
            ambiguous_span_matches=ambiguous_matches,
            proposed_evidence_quotes=proposed_evidence_quotes,
            rejected_evidence_quotes=rejected_evidence_quotes,
            trigger_fallbacks=trigger_fallbacks,
            dropped_events=dropped_events,
            polarity_repairs=polarity_repairs,
            merged_duplicate_events=event_count_before_merge - len(merged.events),
            truncation_retries=truncation_retries,
            truncation_recoveries=truncation_recoveries,
        )
