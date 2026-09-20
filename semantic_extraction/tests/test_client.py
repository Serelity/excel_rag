from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from semantic_extraction.client import ClientConfig, Qwen3ExtractionClient, split_document


class FakeCompletions:
    def __init__(self, value: dict) -> None:
        self.value = value
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.value, ensure_ascii=False))
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class FakeClient:
    def __init__(self, value: dict) -> None:
        self.completions = FakeCompletions(value)
        self.chat = SimpleNamespace(completions=self.completions)


def valid_value() -> dict:
    return {
        "events": [
            {
                "normalized_event_type": "施工噪声",
                "trigger": {"text": "施工噪声"},
                "actors": [],
                "objects": [],
                "behaviors": [{"text": "施工噪声"}],
                "impacts": [],
                "requests": [],
                "locations": [],
                "time_expressions": [{"text": "夜间"}],
                "search_terms": ["施工噪声"],
                "polarity": "occurred",
            }
        ]
    }


@pytest.mark.asyncio
async def test_client_sends_only_case_content_and_disables_thinking() -> None:
    fake = FakeClient(valid_value())
    client = Qwen3ExtractionClient(ClientConfig(model="test-model"), client=fake)

    result = await client.extract_document("夜间施工噪声")

    assert result.extraction.events[0].normalized_event_type == "施工噪声"
    assert result.extraction.events[0].trigger.start == 2
    assert result.grounded_spans == 3
    assert result.proposed_evidence_quotes == 3
    assert result.rejected_evidence_quotes == 0
    call = fake.completions.calls[0]
    assert json.loads(call["messages"][1]["content"]) == {"case_content": "夜间施工噪声"}
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["name"] == "case_content_semantic_v3"
    required = call["response_format"]["json_schema"]["schema"]["$defs"]["ModelExtractedEvent"][
        "required"
    ]
    assert "requests" in required
    assert "locations" in required
    quote_schema = call["response_format"]["json_schema"]["schema"]["$defs"]["EvidenceQuote"]
    assert set(quote_schema["properties"]) == {"text"}


def test_long_document_splits_on_sentence_boundary_without_loss() -> None:
    text = "甲" * 700 + "。" + "乙" * 700 + "。"
    parts = split_document(text, 1000)

    assert len(parts) == 2
    assert "".join(part for _, part in parts) == text
    assert parts[1][0] == len(parts[0][1])
