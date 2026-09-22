from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from semantic_extraction.client import (
    ClientConfig,
    ExtractionError,
    Qwen3ExtractionClient,
    split_document,
)


class FakeCompletions:
    def __init__(self, value: dict, finish_reasons: list[str] | None = None) -> None:
        self.value = value
        self.finish_reasons = finish_reasons or ["stop"]
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.finish_reasons) - 1)
        message = SimpleNamespace(content=json.dumps(self.value, ensure_ascii=False))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reasons[index])]
        )


class FakeClient:
    def __init__(self, value: dict, finish_reasons: list[str] | None = None) -> None:
        self.completions = FakeCompletions(value, finish_reasons)
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
    assert result.model_calls == 1
    assert result.truncation_retries == 0
    assert result.truncation_recoveries == 0
    call = fake.completions.calls[0]
    assert json.loads(call["messages"][1]["content"]) == {"case_content": "夜间施工噪声"}
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call["max_tokens"] == 6144
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["name"] == "case_content_retrieval_issue_v4"
    required = call["response_format"]["json_schema"]["schema"]["$defs"]["ModelExtractedEvent"][
        "required"
    ]
    assert "requests" in required
    assert "locations" in required
    quote_schema = call["response_format"]["json_schema"]["schema"]["$defs"]["EvidenceQuote"]
    assert set(quote_schema["properties"]) == {"text"}
    assert call["response_format"]["json_schema"]["schema"]["properties"]["events"]["maxItems"] == 3


@pytest.mark.asyncio
async def test_client_recovers_once_from_truncated_structured_output() -> None:
    fake = FakeClient(valid_value(), finish_reasons=["length", "stop"])
    client = Qwen3ExtractionClient(ClientConfig(model="test-model"), client=fake)

    result = await client.extract_document("夜间施工噪声")

    assert result.model_calls == 2
    assert result.truncation_retries == 1
    assert result.truncation_recoveries == 1
    assert len(fake.completions.calls) == 2
    recovery = fake.completions.calls[1]
    assert "精简恢复模式" in recovery["messages"][0]["content"]
    response_schema = recovery["response_format"]["json_schema"]
    assert response_schema["name"] == "case_content_retrieval_issue_v4_recovery"
    definitions = response_schema["schema"]["$defs"]
    assert definitions["EvidenceQuote"]["properties"]["text"]["maxLength"] == 48
    event_properties = definitions["ModelExtractedEvent"]["properties"]
    assert event_properties["behaviors"]["maxItems"] == 2
    assert event_properties["search_terms"]["maxItems"] == 4


@pytest.mark.asyncio
async def test_client_quarantines_when_compact_recovery_is_also_truncated() -> None:
    fake = FakeClient(valid_value(), finish_reasons=["length", "length"])
    client = Qwen3ExtractionClient(ClientConfig(model="test-model"), client=fake)

    with pytest.raises(ExtractionError) as caught:
        await client.extract_document("夜间施工噪声")

    assert caught.value.code == "OUTPUT_TRUNCATED"
    assert caught.value.processing == {
        "model_calls": 2,
        "truncation_retries": 1,
        "truncation_recoveries": 0,
    }


def test_long_document_splits_on_sentence_boundary_without_loss() -> None:
    text = "甲" * 700 + "。" + "乙" * 700 + "。"
    parts = split_document(text, 1000)

    assert len(parts) == 2
    assert "".join(part for _, part in parts) == text
    assert parts[1][0] == len(parts[0][1])
