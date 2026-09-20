from __future__ import annotations

from dataclasses import dataclass

from .schema import (
    EvidenceQuote,
    EvidenceSpan,
    ExtractedEvent,
    LocationMention,
    ModelExtractedEvent,
    ModelSemanticExtraction,
    SemanticExtraction,
)


class EvidenceGroundingError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        proposed_evidence_quotes: int = 0,
        rejected_evidence_quotes: int = 0,
        trigger_fallbacks: int = 0,
        dropped_events: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.processing = {
            "proposed_evidence_quotes": proposed_evidence_quotes,
            "rejected_evidence_quotes": rejected_evidence_quotes,
            "trigger_fallbacks": trigger_fallbacks,
            "dropped_events": dropped_events,
        }


@dataclass(frozen=True, slots=True)
class GroundingResult:
    extraction: SemanticExtraction
    grounded_spans: int
    ambiguous_matches: int
    proposed_evidence_quotes: int
    rejected_evidence_quotes: int
    trigger_fallbacks: int
    dropped_events: int


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
) -> tuple[list[EvidenceSpan], int, int]:
    spans: list[EvidenceSpan] = []
    ambiguous = 0
    rejected = 0
    seen: set[tuple[int, int, str]] = set()
    for quote in quotes:
        try:
            span, repeated = _ground_quote(quote, source, anchor=anchor)
        except EvidenceGroundingError:
            rejected += 1
            continue
        key = (span.start, span.end, span.text)
        if key not in seen:
            spans.append(span)
            seen.add(key)
        ambiguous += int(repeated)
    return spans, ambiguous, rejected


def _event_quotes(event: ModelExtractedEvent) -> list[EvidenceQuote]:
    return [
        *event.behaviors,
        *event.objects,
        *event.impacts,
        *event.requests,
        *(location.evidence for location in event.locations),
        *event.time_expressions,
        *event.actors,
    ]


def ground_extraction(
    model_result: ModelSemanticExtraction,
    source: str,
) -> GroundingResult:
    events: list[ExtractedEvent] = []
    ambiguous = 0
    proposed_quotes = 0
    rejected_quotes = 0
    trigger_fallbacks = 0
    dropped_events = 0
    used_trigger_positions: dict[str, set[int]] = {}

    for event in model_result.events:
        fallback_quotes = _event_quotes(event)
        proposed_quotes += 1 + len(fallback_quotes)
        excluded = used_trigger_positions.setdefault(event.trigger.text, set())
        try:
            trigger, repeated = _ground_quote(
                event.trigger,
                source,
                anchor=None,
                excluded=excluded,
            )
            ambiguous += int(repeated)
        except EvidenceGroundingError:
            rejected_quotes += 1
            trigger = None
            for quote in fallback_quotes:
                quote_excluded = used_trigger_positions.setdefault(quote.text, set())
                try:
                    trigger, repeated = _ground_quote(
                        quote,
                        source,
                        anchor=None,
                        excluded=quote_excluded,
                    )
                except EvidenceGroundingError:
                    continue
                ambiguous += int(repeated)
                trigger_fallbacks += 1
                break
            if trigger is None:
                rejected_quotes += len(fallback_quotes)
                dropped_events += 1
                continue
        used_trigger_positions.setdefault(trigger.text, set()).add(trigger.start)

        actors, count, rejected = _ground_many(event.actors, source, anchor=trigger.start)
        ambiguous += count
        rejected_quotes += rejected
        objects, count, rejected = _ground_many(event.objects, source, anchor=trigger.start)
        ambiguous += count
        rejected_quotes += rejected
        behaviors, count, rejected = _ground_many(event.behaviors, source, anchor=trigger.start)
        ambiguous += count
        rejected_quotes += rejected
        impacts, count, rejected = _ground_many(event.impacts, source, anchor=trigger.start)
        ambiguous += count
        rejected_quotes += rejected
        requests, count, rejected = _ground_many(event.requests, source, anchor=trigger.start)
        ambiguous += count
        rejected_quotes += rejected
        times, count, rejected = _ground_many(
            event.time_expressions,
            source,
            anchor=trigger.start,
        )
        ambiguous += count
        rejected_quotes += rejected

        locations: list[LocationMention] = []
        seen_locations: set[tuple[int, int, str]] = set()
        for location in event.locations:
            try:
                evidence, repeated = _ground_quote(
                    location.evidence,
                    source,
                    anchor=trigger.start,
                )
            except EvidenceGroundingError:
                rejected_quotes += 1
                continue
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

    if model_result.events and not events:
        raise EvidenceGroundingError(
            "no model event contains a verbatim source quote",
            code="NO_GROUNDED_EVENTS",
            proposed_evidence_quotes=proposed_quotes,
            rejected_evidence_quotes=rejected_quotes,
            trigger_fallbacks=trigger_fallbacks,
            dropped_events=dropped_events,
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
        proposed_evidence_quotes=proposed_quotes,
        rejected_evidence_quotes=rejected_quotes,
        trigger_fallbacks=trigger_fallbacks,
        dropped_events=dropped_events,
    )
