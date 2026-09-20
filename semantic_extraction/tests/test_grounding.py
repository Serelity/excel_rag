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


def test_grounding_uses_unused_trigger_occurrences_in_event_order() -> None:
    value = model_value()
    value["events"] = [value["events"][0], value["events"][0].copy()]
    source = "汉江路路灯不亮，希望维修；汉江路另一处路灯不亮，希望维修"

    result = ground_extraction(ModelSemanticExtraction.model_validate(value), source)

    first, second = result.extraction.events
    assert first.trigger.start == source.find("路灯不亮")
    assert second.trigger.start == source.rfind("路灯不亮")
    assert result.ambiguous_matches > 0


def test_grounding_rejects_paraphrased_evidence() -> None:
    value = model_value(trigger="照明设施损坏")

    with pytest.raises(EvidenceGroundingError) as caught:
        ground_extraction(
            ModelSemanticExtraction.model_validate(value),
            "汉江路路灯不亮，希望维修",
        )

    assert caught.value.code == "EVIDENCE_TEXT_NOT_FOUND"
