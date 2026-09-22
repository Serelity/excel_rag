from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    MAX_RETRIEVAL_ISSUES,
    EvidenceSpan,
    ExtractedEvent,
    LocationMention,
    SemanticExtraction,
)


class EvidenceAlignmentError(ValueError):
    """Raised when model evidence cannot be mapped unambiguously to the source."""


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    extraction: SemanticExtraction
    repairs: int


def _occurrences(source: str, needle: str) -> list[int]:
    positions: list[int] = []
    offset = 0
    while True:
        position = source.find(needle, offset)
        if position < 0:
            return positions
        positions.append(position)
        offset = position + 1


def align_span(span: EvidenceSpan, source: str) -> tuple[EvidenceSpan, bool]:
    if span.end <= len(source) and source[span.start : span.end] == span.text:
        return span, False

    positions = _occurrences(source, span.text)
    if not positions:
        raise EvidenceAlignmentError("evidence text is not a contiguous source substring")

    # Wrong offsets are common in LLM output. Exact text makes a deterministic repair safe;
    # for repeated text, the reported offset disambiguates by nearest occurrence.
    best_distance = min(abs(position - span.start) for position in positions)
    nearest = [position for position in positions if abs(position - span.start) == best_distance]
    if len(nearest) != 1:
        raise EvidenceAlignmentError("repeated evidence has an ambiguous source position")
    start = nearest[0]
    return EvidenceSpan(text=span.text, start=start, end=start + len(span.text)), True


def _align_many(spans: list[EvidenceSpan], source: str) -> tuple[list[EvidenceSpan], int]:
    aligned: list[EvidenceSpan] = []
    repairs = 0
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        fixed, repaired = align_span(span, source)
        key = (fixed.start, fixed.end, fixed.text)
        if key not in seen:
            aligned.append(fixed)
            seen.add(key)
        repairs += int(repaired)
    return aligned, repairs


def align_extraction(extraction: SemanticExtraction, source: str) -> AlignmentResult:
    events: list[ExtractedEvent] = []
    repairs = 0
    for event in extraction.events:
        trigger, repaired = align_span(event.trigger, source)
        repairs += int(repaired)
        actors, count = _align_many(event.actors, source)
        repairs += count
        objects, count = _align_many(event.objects, source)
        repairs += count
        behaviors, count = _align_many(event.behaviors, source)
        repairs += count
        impacts, count = _align_many(event.impacts, source)
        repairs += count
        requests, count = _align_many(event.requests, source)
        repairs += count
        times, count = _align_many(event.time_expressions, source)
        repairs += count

        locations: list[LocationMention] = []
        seen_locations: set[tuple[int, int, str]] = set()
        for location in event.locations:
            evidence, repaired = align_span(location.evidence, source)
            repairs += int(repaired)
            key = (evidence.start, evidence.end, location.kind)
            if key not in seen_locations:
                locations.append(location.model_copy(update={"evidence": evidence}))
                seen_locations.add(key)

        events.append(
            event.model_copy(
                update={
                    "trigger": trigger,
                    "actors": actors,
                    "objects": objects,
                    "behaviors": behaviors,
                    "impacts": impacts,
                    "requests": requests,
                    "locations": locations,
                    "time_expressions": times,
                    "search_terms": list(dict.fromkeys(event.search_terms)),
                }
            )
        )

    return AlignmentResult(extraction=SemanticExtraction(events=events), repairs=repairs)


def shift_extraction(extraction: SemanticExtraction, offset: int) -> SemanticExtraction:
    if offset == 0:
        return extraction

    def shift(span: EvidenceSpan) -> EvidenceSpan:
        return span.model_copy(update={"start": span.start + offset, "end": span.end + offset})

    events: list[ExtractedEvent] = []
    for event in extraction.events:
        events.append(
            event.model_copy(
                update={
                    "trigger": shift(event.trigger),
                    "actors": [shift(item) for item in event.actors],
                    "objects": [shift(item) for item in event.objects],
                    "behaviors": [shift(item) for item in event.behaviors],
                    "impacts": [shift(item) for item in event.impacts],
                    "requests": [shift(item) for item in event.requests],
                    "locations": [
                        item.model_copy(update={"evidence": shift(item.evidence)})
                        for item in event.locations
                    ],
                    "time_expressions": [shift(item) for item in event.time_expressions],
                }
            )
        )
    return SemanticExtraction(events=events)


def merge_extractions(parts: list[SemanticExtraction]) -> SemanticExtraction:
    grouped: dict[tuple[str, str], list[ExtractedEvent]] = {}
    for part in parts:
        for event in part.events:
            grouped.setdefault((event.normalized_event_type, event.polarity), []).append(event)

    events = [_merge_event_group(group) for group in grouped.values()]
    if len(events) > MAX_RETRIEVAL_ISSUES:
        raise EvidenceAlignmentError(
            f"merged document contains more than {MAX_RETRIEVAL_ISSUES} distinct issues"
        )
    return SemanticExtraction(events=events)


def _unique_spans(events: list[ExtractedEvent], field: str, limit: int) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for event in events:
        for span in getattr(event, field):
            key = (span.start, span.end, span.text)
            if key not in seen:
                spans.append(span)
                seen.add(key)
    return spans[:limit]


def _unique_locations(events: list[ExtractedEvent]) -> list[LocationMention]:
    locations: list[LocationMention] = []
    seen: set[tuple[int, int, str, str | None]] = set()
    for event in events:
        for location in event.locations:
            key = (
                location.evidence.start,
                location.evidence.end,
                location.kind,
                location.normalized_name,
            )
            if key not in seen:
                locations.append(location)
                seen.add(key)
    return locations[:5]


def _merge_event_group(events: list[ExtractedEvent]) -> ExtractedEvent:
    first = events[0]
    trigger = min((event.trigger for event in events), key=lambda span: (span.start, span.end))
    search_terms = list(dict.fromkeys(term for event in events for term in event.search_terms))[:8]
    return ExtractedEvent(
        normalized_event_type=first.normalized_event_type,
        trigger=trigger,
        actors=_unique_spans(events, "actors", 4),
        objects=_unique_spans(events, "objects", 5),
        behaviors=_unique_spans(events, "behaviors", 5),
        impacts=_unique_spans(events, "impacts", 4),
        requests=_unique_spans(events, "requests", 4),
        locations=_unique_locations(events),
        time_expressions=_unique_spans(events, "time_expressions", 4),
        search_terms=search_terms,
        polarity=first.polarity,
    )


def validate_aligned_extraction(extraction: SemanticExtraction, source: str) -> None:
    aligned = align_extraction(extraction, source)
    if aligned.repairs:
        raise EvidenceAlignmentError("stored extraction contains unaligned evidence offsets")
