from __future__ import annotations

import pytest
from pydantic import ValidationError

from semantic_extraction.alignment import (
    EvidenceAlignmentError,
    align_extraction,
    merge_extractions,
)
from semantic_extraction.schema import ModelSemanticExtraction, SemanticExtraction
from semantic_extraction.views import build_retrieval_views


def extraction_value(*, start: int = 0, end: int = 4) -> dict:
    return {
        "events": [
            {
                "normalized_event_type": "道路积水",
                "trigger": {"text": "道路积水", "start": start, "end": end},
                "actors": [],
                "objects": [{"text": "道路", "start": start, "end": start + 2}],
                "behaviors": [{"text": "积水", "start": start + 2, "end": end}],
                "impacts": [],
                "requests": [{"text": "希望排水", "start": 5, "end": 9}],
                "locations": [
                    {
                        "evidence": {"text": "汉江路", "start": 10, "end": 13},
                        "kind": "road",
                        "normalized_name": "汉江路",
                    }
                ],
                "time_expressions": [],
                "search_terms": ["道路积水", "排水"],
                "polarity": "occurred",
            }
        ]
    }


def test_alignment_repairs_offsets_when_evidence_text_is_present() -> None:
    source = "道路积水，希望排水，汉江路"
    value = extraction_value(start=99, end=103)
    value["events"][0]["objects"][0] = {"text": "道路", "start": 99, "end": 101}
    value["events"][0]["behaviors"][0] = {"text": "积水", "start": 101, "end": 103}

    result = align_extraction(SemanticExtraction.model_validate(value), source)

    assert result.extraction.events[0].trigger.start == 0
    assert result.repairs == 3


def test_alignment_rejects_text_absent_from_source() -> None:
    value = extraction_value()
    value["events"][0]["impacts"] = [{"text": "影响安全", "start": 0, "end": 4}]

    with pytest.raises(EvidenceAlignmentError, match="not a contiguous"):
        align_extraction(SemanticExtraction.model_validate(value), "道路积水，希望排水，汉江路")


def test_retrieval_core_keeps_location_as_separate_ablation() -> None:
    extraction = align_extraction(
        SemanticExtraction.model_validate(extraction_value()),
        "道路积水，希望排水，汉江路",
    ).extraction

    views = build_retrieval_views(extraction)

    assert "道路积水" in views["evidence_core"]
    assert "问题：道路积水" not in views["evidence_core"]
    assert "问题：道路积水" in views["semantic_terms"]
    assert "汉江路" not in views["semantic_core"]
    assert "希望排水" not in views["semantic_core"]
    assert "希望排水" in views["semantic_with_request"]
    assert "汉江路" in views["semantic_with_location"]


def test_merge_extractions_combines_same_retrieval_issue() -> None:
    first = SemanticExtraction.model_validate(extraction_value())
    second_value = extraction_value(start=14, end=18)
    second_event = second_value["events"][0]
    second_event["trigger"] = {"text": "道路积水", "start": 14, "end": 18}
    second_event["objects"] = [{"text": "道路", "start": 14, "end": 16}]
    second_event["behaviors"] = [{"text": "积水", "start": 16, "end": 18}]
    second_event["requests"] = [{"text": "要求处理", "start": 19, "end": 23}]
    second_event["locations"] = []
    second = SemanticExtraction.model_validate(second_value)

    result = merge_extractions([first, second])

    assert len(result.events) == 1
    assert [span.text for span in result.events[0].requests] == ["希望排水", "要求处理"]
    assert len(result.events[0].behaviors) == 2


def test_merge_extractions_keeps_different_retrieval_issues() -> None:
    first = SemanticExtraction.model_validate(extraction_value())
    second_value = extraction_value()
    second_value["events"][0]["normalized_event_type"] = "路灯故障"
    second = SemanticExtraction.model_validate(second_value)

    result = merge_extractions([first, second])

    assert [event.normalized_event_type for event in result.events] == [
        "道路积水",
        "路灯故障",
    ]


def test_model_schema_rejects_more_than_three_retrieval_issues() -> None:
    event = {
        "normalized_event_type": "道路积水",
        "trigger": {"text": "道路积水"},
        "actors": [],
        "objects": [],
        "behaviors": [],
        "impacts": [],
        "requests": [],
        "locations": [],
        "time_expressions": [],
        "search_terms": [],
        "polarity": "occurred",
    }

    with pytest.raises(ValidationError):
        ModelSemanticExtraction.model_validate({"events": [event.copy() for _ in range(4)]})
