from __future__ import annotations

import pytest

from semantic_extraction.grounding import EvidenceGroundingError, ground_extraction
from semantic_extraction.schema import ModelSemanticExtraction


def model_value(trigger: str = "路灯不亮") -> dict:
    return {
        "events": [
            {
                "normalized_event_type": "路灯故障",
                "trigger": {"text": trigger},
                "actors": [],
                "objects": [{"text": "路灯"}],
                "behaviors": [{"text": "路灯不亮"}],
                "impacts": [],
                "requests": [{"text": "希望维修"}],
                "locations": [
                    {
                        "evidence": {"text": "汉江路"},
                        "kind": "road",
                        "normalized_name": "汉江路",
                    }
                ],
                "time_expressions": [],
                "search_terms": ["路灯故障", "照明设施"],
                "polarity": "occurred",
            }
        ]
    }


def test_grounding_assigns_offsets_without_model_character_arithmetic() -> None:
    source = "汉江路路灯不亮，希望维修"

    result = ground_extraction(ModelSemanticExtraction.model_validate(model_value()), source)

    event = result.extraction.events[0]
    assert source[event.trigger.start : event.trigger.end] == "路灯不亮"
    assert event.locations[0].evidence.start == 0
    assert result.grounded_spans == 5
    assert result.ambiguous_matches == 0
    assert result.proposed_evidence_quotes == 5
    assert result.rejected_evidence_quotes == 0
    assert result.trigger_fallbacks == 0
    assert result.dropped_events == 0
    assert result.polarity_repairs == 0


def test_grounding_uses_unused_trigger_occurrences_in_event_order() -> None:
    value = model_value()
    value["events"] = [value["events"][0], value["events"][0].copy()]
    source = "汉江路路灯不亮，希望维修；汉江路另一处路灯不亮，希望维修"

    result = ground_extraction(ModelSemanticExtraction.model_validate(value), source)

    first, second = result.extraction.events
    assert first.trigger.start == source.find("路灯不亮")
    assert second.trigger.start == source.rfind("路灯不亮")
    assert result.ambiguous_matches > 0


def test_grounding_falls_back_when_trigger_is_paraphrased() -> None:
    value = model_value(trigger="照明设施损坏")

    result = ground_extraction(
        ModelSemanticExtraction.model_validate(value),
        "汉江路路灯不亮，希望维修",
    )

    assert result.extraction.events[0].trigger.text == "路灯不亮"
    assert result.trigger_fallbacks == 1
    assert result.rejected_evidence_quotes == 1


def test_grounding_drops_only_paraphrased_optional_evidence() -> None:
    value = model_value()
    value["events"][0]["impacts"] = [{"text": "影响道路通行"}]

    result = ground_extraction(
        ModelSemanticExtraction.model_validate(value),
        "汉江路路灯不亮，希望维修",
    )

    assert result.extraction.events[0].impacts == []
    assert result.proposed_evidence_quotes == 6
    assert result.rejected_evidence_quotes == 1
    assert result.dropped_events == 0


def test_grounding_quarantines_when_no_event_has_verbatim_evidence() -> None:
    value = model_value(trigger="照明设施损坏")
    event = value["events"][0]
    event["objects"] = [{"text": "照明设备"}]
    event["behaviors"] = [{"text": "灯具发生故障"}]
    event["requests"] = [{"text": "请求有关部门修复"}]
    event["locations"] = []

    with pytest.raises(EvidenceGroundingError) as caught:
        ground_extraction(
            ModelSemanticExtraction.model_validate(value),
            "汉江路路灯不亮，希望维修",
        )

    assert caught.value.code == "NO_GROUNDED_EVENTS"
    assert caught.value.processing == {
        "proposed_evidence_quotes": 4,
        "rejected_evidence_quotes": 4,
        "trigger_fallbacks": 0,
        "dropped_events": 1,
    }


def test_grounding_repairs_possible_event_type_polarity() -> None:
    value = model_value()
    value["events"][0]["normalized_event_type"] = "疑似路灯损坏"

    result = ground_extraction(
        ModelSemanticExtraction.model_validate(value),
        "汉江路路灯不亮，希望维修",
    )

    assert result.extraction.events[0].polarity == "possible"
    assert result.polarity_repairs == 1
