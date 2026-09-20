from __future__ import annotations

from collections.abc import Iterable

from .schema import EvidenceSpan, SemanticExtraction


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _span_text(spans: Iterable[EvidenceSpan]) -> list[str]:
    return _unique(span.text for span in spans)


def _line(label: str, values: Iterable[str]) -> str | None:
    items = _unique(values)
    return f"{label}：{'；'.join(items)}" if items else None


def build_retrieval_views(extraction: SemanticExtraction) -> dict[str, str]:
    evidence_blocks: list[str] = []
    term_blocks: list[str] = []
    factual_blocks: list[str] = []
    request_blocks: list[str] = []
    location_blocks: list[str] = []

    for event in extraction.events:
        evidence_lines: list[str] = []
        for label, values in (
            ("原文触发", [event.trigger.text]),
            ("主体", _span_text(event.actors)),
            ("对象", _span_text(event.objects)),
            ("行为现象", _span_text(event.behaviors)),
            ("影响", _span_text(event.impacts)),
        ):
            value = _line(label, values)
            if value:
                evidence_lines.append(value)
        evidence_blocks.append("\n".join(evidence_lines))

        term_lines = [f"事件：{event.normalized_event_type}"]
        terms_line = _line("检索词", event.search_terms)
        if terms_line:
            term_lines.append(terms_line)
        term_blocks.append("\n".join(term_lines))

        factual = [f"事件：{event.normalized_event_type}"]
        for label, values in (
            ("原文触发", [event.trigger.text]),
            ("主体", _span_text(event.actors)),
            ("对象", _span_text(event.objects)),
            ("行为现象", _span_text(event.behaviors)),
            ("影响", _span_text(event.impacts)),
            ("检索词", event.search_terms),
        ):
            value = _line(label, values)
            if value:
                factual.append(value)
        factual_block = "\n".join(factual)
        factual_blocks.append(factual_block)

        request_line = _line("诉求", _span_text(event.requests))
        request_blocks.append(
            factual_block if request_line is None else f"{factual_block}\n{request_line}"
        )

        place_values = [
            location.normalized_name or location.evidence.text for location in event.locations
        ]
        place_line = _line("地点", place_values)
        time_line = _line("时间", _span_text(event.time_expressions))
        additions = [value for value in (request_line, place_line, time_line) if value]
        location_blocks.append("\n".join([factual_block, *additions]))

    return {
        "evidence_core": "\n\n".join(evidence_blocks),
        "semantic_terms": "\n\n".join(term_blocks),
        "semantic_core": "\n\n".join(factual_blocks),
        "semantic_with_request": "\n\n".join(request_blocks),
        "semantic_with_location": "\n\n".join(location_blocks),
    }
