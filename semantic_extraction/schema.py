from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "semantic-extraction-v2"
PROMPT_VERSION = "case-content-semantic-v2"

ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
EventTypeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=48),
]
SearchTermText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=32),
]


class EvidenceSpan(BaseModel):
    """A half-open character span copied exactly from case_content."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: ShortText = Field(description="case_content 中连续出现的原文，不得改写或补字。")
    start: int = Field(
        ge=0,
        description="text 首字符在 case_content 中从 0 开始的字符下标。",
    )
    end: int = Field(
        ge=1,
        description="text 末字符之后的字符下标；case_content[start:end] 必须等于 text。",
    )

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


LocationKind = Literal[
    "administrative_area",
    "street_or_town",
    "road",
    "community",
    "building",
    "poi",
    "institution",
    "other",
]


class LocationMention(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence: EvidenceSpan
    kind: LocationKind
    normalized_name: str | None = Field(
        max_length=120,
        description=(
            "仅规范原文已有地名的空格、简称或行政后缀；无法可靠规范时必须为 null，"
            "不得补全原文没有的地址层级。"
        ),
    )


Polarity = Literal["occurred", "possible", "negated", "consultation"]


class ExtractedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    normalized_event_type: EventTypeText = Field(
        description="简短、通用、适合检索的事件类型；保留咨询、疑似等语义。"
    )
    trigger: EvidenceSpan = Field(description="最能指示该事件的原文短语。")
    actors: list[EvidenceSpan] = Field(
        max_length=4,
        description="原文明示的涉事主体角色或机构；不要推断省略的主体。",
    )
    objects: list[EvidenceSpan] = Field(
        max_length=5,
        description="事件涉及的设施、商品、服务或事项。",
    )
    behaviors: list[EvidenceSpan] = Field(
        max_length=5,
        description="已经陈述的行为或可观察现象，不含办理诉求。",
    )
    impacts: list[EvidenceSpan] = Field(
        max_length=4,
        description="原文明示的实际影响或风险，不得补造常识性后果。",
    )
    requests: list[EvidenceSpan] = Field(
        max_length=4,
        description="来电人的咨询、希望、申请、建议或要求。",
    )
    locations: list[LocationMention] = Field(
        max_length=5,
        description="与事件直接相关的地点、地址或场所。",
    )
    time_expressions: list[EvidenceSpan] = Field(
        max_length=4,
        description="与事件直接相关的时间或频率表达。",
    )
    search_terms: list[SearchTermText] = Field(
        max_length=8,
        description=(
            "由该事件归一化得到的通用检索概念；不放具体人名、地址、机构名、电话、"
            "工号或原文没有依据的新事实。"
        ),
    )
    polarity: Polarity = Field(
        description=(
            "occurred 表示原文按已发生陈述，不代表外部核实；possible 表示怀疑、疑似或"
            "尚未确认；negated 表示原文明示未发生；consultation 表示仅咨询规则或办理方式。"
        )
    )


class SemanticExtraction(BaseModel):
    """The grounded, persisted extraction for one case_content value."""

    model_config = ConfigDict(extra="forbid", strict=True)

    events: list[ExtractedEvent] = Field(
        max_length=6,
        description=(
            "case_content 中可独立检索的事件。没有足够信息时返回空数组；不得创建"
            "“未知问题”占位事件。"
        ),
    )


class EvidenceQuote(BaseModel):
    """Exact source text requested from Qwen; offsets are assigned in code."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: ShortText = Field(description="case_content 中连续出现的原文，不得改写或补字。")


class ModelLocationMention(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evidence: EvidenceQuote
    kind: LocationKind
    normalized_name: str | None = Field(
        max_length=120,
        description=(
            "仅规范原文已有地名的空格、简称或行政后缀；无法可靠规范时必须为 null，"
            "不得补全原文没有的地址层级。"
        ),
    )


class ModelExtractedEvent(BaseModel):
    """Compact model response that avoids unreliable character arithmetic."""

    model_config = ConfigDict(extra="forbid", strict=True)

    normalized_event_type: EventTypeText = Field(
        description="简短、通用、适合检索的事件类型；保留咨询、疑似等语义。"
    )
    trigger: EvidenceQuote = Field(description="最能指示该事件的最短原文短语。")
    actors: list[EvidenceQuote] = Field(
        max_length=4,
        description="原文明示的涉事主体角色或机构；不要推断省略的主体。",
    )
    objects: list[EvidenceQuote] = Field(
        max_length=5,
        description="事件涉及的设施、商品、服务或事项。",
    )
    behaviors: list[EvidenceQuote] = Field(
        max_length=5,
        description="已经陈述的行为或可观察现象，不含办理诉求。",
    )
    impacts: list[EvidenceQuote] = Field(
        max_length=4,
        description="原文明示的实际影响或风险，不得补造常识性后果。",
    )
    requests: list[EvidenceQuote] = Field(
        max_length=4,
        description="来电人的咨询、希望、申请、建议或要求。",
    )
    locations: list[ModelLocationMention] = Field(
        max_length=5,
        description="与事件直接相关的地点、地址或场所。",
    )
    time_expressions: list[EvidenceQuote] = Field(
        max_length=4,
        description="与事件直接相关的时间或频率表达。",
    )
    search_terms: list[SearchTermText] = Field(
        max_length=8,
        description="基于该事件的通用检索概念，不含专名和新事实。",
    )
    polarity: Polarity = Field(description="事件的原文确定性或咨询属性。")


class ModelSemanticExtraction(BaseModel):
    """The complete Qwen-owned response before deterministic grounding."""

    model_config = ConfigDict(extra="forbid", strict=True)

    events: list[ModelExtractedEvent] = Field(
        max_length=6,
        description="可独立检索的事件；没有足够信息时必须返回空数组。",
    )
