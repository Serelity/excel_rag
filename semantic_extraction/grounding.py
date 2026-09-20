from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    EvidenceQuote,
    EvidenceSpan,
    ExtractedEvent,
    LocationMention,
    ModelSemanticExtraction,
    SemanticExtraction,
)


class EvidenceGroundingError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class GroundingResult:
    extraction: SemanticExtraction
    grounded_spans: int
    ambiguous_matches: int


def _occurrences(source: str, needle: str) -> list[int]:
    positions: list[int] = []
    offset = 0
    while True:
        position = source.find(needle, offset)
        if position < 0:
            return positions
        positions.append(position)
        offset = position + 1


def _ground_quote(
    quote: EvidenceQuote,
    source: str,
    *,
    anchor: int | None,
    excluded: set[int] | None = None,
) -> tuple[EvidenceSpan, bool]:
    positions = _occurrences(source, quote.text)
    if not positions:
        raise EvidenceGroundingError(
            "evidence quote is not a contiguous source substring",
            code="EVIDENCE_TEXT_NOT_FOUND",
        )
    candidates = positions
    if excluded:
        unused = [position for position in positions if position not in excluded]
        if unused:
            candidates = unused
    start = (
        candidates[0]
        if anchor is None
        else min(candidates, key=lambda pos: (abs(pos - anchor), pos))
    )
    return (
        EvidenceSpan(text=quote.text, start=start, end=start + len(quote.text)),
        len(positions) > 1,
    )


def _ground_many(
    quotes: list[EvidenceQuote],
    source: str,
    *,
    anchor: int,
) -> tuple[list[EvidenceSpan], int]:
    spans: list[EvidenceSpan] = []
    ambiguous = 0
    seen: set[tuple[int, int, str]] = set()
    for quote in quotes:
        span, repeated = _ground_quote(quote, source, anchor=anchor)
        key = (span.start, span.end, span.text)
        if key not in seen:
            spans.append(span)
            seen.add(key)
        ambiguous += int(repeated)
    return spans, ambiguous


def ground_extraction(
    model_result: ModelSemanticExtraction,
    source: str,
) -> GroundingResult:
    events: list[ExtractedEvent] = []
    ambiguous = 0
    used_trigger_positions: dict[str, set[int]] = {}

    for event in model_result.events:
        excluded = used_trigger_positions.setdefault(event.trigger.text, set())
        trigger, repeated = _ground_quote(
            event.trigger,
            source,
            anchor=None,
            excluded=excluded,
        )
        excluded.add(trigger.start)
        ambiguous += int(repeated)

        actors, count = _ground_many(event.actors, source, anchor=trigger.start)
        ambiguous += count
        objects, count = _ground_many(event.objects, source, anchor=trigger.start)
        ambiguous += count
        behaviors, count = _ground_many(event.behaviors, source, anchor=trigger.start)
        ambiguous += count
        impacts, count = _ground_many(event.impacts, source, anchor=trigger.start)
        ambiguous += count
        requests, count = _ground_many(event.requests, source, anchor=trigger.start)
        ambiguous += count
        times, count = _ground_many(event.time_expressions, source, anchor=trigger.start)
        ambiguous += count

        locations: list[LocationMention] = []
        seen_locations: set[tuple[int, int, str]] = set()
        for location in event.locations:
            evidence, repeated = _ground_quote(
                location.evidence,
                source,
                anchor=trigger.start,
            )
            ambiguous += int(repeated)
            key = (evidence.start, evidence.end, location.kind)
            if key not in seen_locations:
                locations.append(
                    LocationMention(
                        evidence=evidence,
                        kind=location.kind,
                        normalized_name=location.normalized_name,
                    )
                )
                seen_locations.add(key)

        events.append(
            ExtractedEvent(
                normalized_event_type=event.normalized_event_type,
                trigger=trigger,
                actors=actors,
                objects=objects,
                behaviors=behaviors,
                impacts=impacts,
                requests=requests,
                locations=locations,
                time_expressions=times,
                search_terms=list(dict.fromkeys(event.search_terms)),
                polarity=event.polarity,
            )
        )

    extraction = SemanticExtraction(events=events)
    grounded_spans = sum(
        1
        + len(event.actors)
        + len(event.objects)
        + len(event.behaviors)
        + len(event.impacts)
        + len(event.requests)
        + len(event.locations)
        + len(event.time_expressions)
        for event in events
    )
    return GroundingResult(
        extraction=extraction,
        grounded_spans=grounded_spans,
        ambiguous_matches=ambiguous,
    )
