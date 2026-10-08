from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SPEC_VERSION = "case-content-extraction-v1"
PROMPT_VERSION = "case-content-extraction-v1-p1"

Text = Annotated[str, StringConstraints(min_length=1)]
IssueId = Annotated[str, StringConstraints(pattern=r"^I[1-9][0-9]*$")]
FactId = Annotated[str, StringConstraints(pattern=r"^F[1-9][0-9]*$")]
Kind = Literal[
    "topic", "intent", "actor", "object", "problem", "request", "impact",
    "place", "condition", "time", "status",
]
Role = Literal[
    "subject", "consultation", "complaint", "report", "service_request", "suggestion",
    "followup", "withdrawal", "supplement", "employer", "merchant", "affected_person",
    "respondent", "service_provider", "target_institution", "handler", "facility",
    "product", "service", "account", "document", "benefit", "transaction", "phenomenon",
    "behavior", "obstacle", "dispute", "desired_action", "desired_state", "information",
    "experienced", "risk", "avoided", "incident", "residence", "registration", "insurance",
    "treatment", "origin", "destination", "policy_area", "landmark", "organization_location",
    "identity", "eligibility", "amount", "quantity", "prerequisite", "event_time",
    "action_time", "period", "frequency", "relative", "issue_state", "ticket_state",
    "action_state", "other", "unspecified",
]
ROLES_BY_KIND: dict[str, set[str]] = {
    "topic": {"subject"},
    "intent": {"consultation", "complaint", "report", "service_request", "suggestion",
               "followup", "withdrawal", "supplement"},
    "actor": {"employer", "merchant", "affected_person", "respondent", "service_provider",
              "target_institution", "handler", "other", "unspecified"},
    "object": {"facility", "product", "service", "account", "document", "benefit",
               "transaction", "other", "unspecified"},
    "problem": {"phenomenon", "behavior", "obstacle", "dispute", "unspecified"},
    "request": {"desired_action", "desired_state", "information", "unspecified"},
    "impact": {"experienced", "risk", "avoided", "unspecified"},
    "place": {"incident", "residence", "registration", "insurance", "treatment", "transaction",
              "origin", "destination", "policy_area", "landmark", "organization_location",
              "unspecified"},
    "condition": {"identity", "eligibility", "amount", "quantity", "prerequisite",
                  "other", "unspecified"},
    "time": {"event_time", "action_time", "period", "frequency", "relative", "unspecified"},
    "status": {"issue_state", "ticket_state", "action_state"},
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CaseInput(StrictModel):
    case_content: str = Field(description="唯一语义输入；保留原始字符，允许空文本。")


class Quote(StrictModel):
    text: Text = Field(description="原文连续片段，不改字、不去空白、不补地址或年份。")
    occurrence: int | None = Field(
        ge=0,
        description="唯一出现填 null；重复时填从0开始的出现序号，或扩大引文至唯一。",
    )


class Fact(StrictModel):
    fact_id: FactId
    kind: Kind
    role: Role = Field(description="角色须符合字段类型；不明用 unspecified，不按距离猜。")
    evidence: list[Quote] = Field(min_length=1, description="共同支持该字段的原文片段。")
    modality: Literal[
        "asserted", "possible", "negated", "hypothetical", "unresolved", "not_applicable",
    ] = Field(description="原文陈述方式，asserted仅指有人这样陈述，不表示外部核实。")
    phase: Literal["current", "background", "followup", "unspecified"]
    attribution: Quote | None = Field(description="原文明示的信息来源/说话者；没说明填null。")
    time_context: Quote | None = Field(description="原文明示且适用于本字段的时间；否则null。")

    @model_validator(mode="after")
    def role_matches_kind(self) -> Self:
        if self.role not in ROLES_BY_KIND[self.kind]:
            raise ValueError(f"role {self.role!r} is not allowed for kind {self.kind!r}")
        return self


class Issue(StrictModel):
    issue_id: IssueId
    facts: list[Fact] = Field(min_length=1)

    @model_validator(mode="after")
    def has_one_topic(self) -> Self:
        if sum(f.kind == "topic" for f in self.facts) != 1:
            raise ValueError("each issue must contain exactly one grounded topic")
        if any(f.role == "ticket_state" for f in self.facts):
            raise ValueError("ticket_state belongs in procedure, not in an underlying issue")
        return self


class ReviewFlag(StrictModel):
    code: Literal[
        "missing_context", "ambiguous_reference", "uncertain_grouping", "conflicting_claims",
        "ambiguous_role", "insufficient_content",
    ]
    issue_ids: list[IssueId]
    fact_ids: list[FactId]
    evidence: list[Quote]
    note: Text = Field(description="说明待核对之处，不补造缺失的业务信息。")


class CaseExtraction(StrictModel):
    spec_version: Literal["case-content-extraction-v1"]
    route: Literal[
        "substantive", "substantive_with_followup", "procedure_only", "insufficient_content",
    ]
    issues: list[Issue] = Field(description="按可区分的问题目标组织，不设三项截断。")
    procedure: list[Fact] = Field(description="热线受理、催办、撤单、转接等流程；不是原问题状态。")
    review_flags: list[ReviewFlag]

    @model_validator(mode="after")
    def coherent_structure(self) -> Self:
        issue_ids = [i.issue_id for i in self.issues]
        facts = [f for i in self.issues for f in i.facts] + self.procedure
        fact_ids = [f.fact_id for f in facts]
        if len(set(issue_ids)) != len(issue_ids) or len(set(fact_ids)) != len(fact_ids):
            raise ValueError("issue IDs and document-wide fact IDs must be unique")
        substantive = self.route in {"substantive", "substantive_with_followup"}
        if substantive != bool(self.issues):
            raise ValueError("route and presence of underlying issues disagree")
        has_followup = bool(self.procedure) or any(f.phase == "followup" for f in facts)
        if self.route == "substantive" and has_followup:
            raise ValueError("followup content requires substantive_with_followup route")
        if self.route == "substantive_with_followup" and not has_followup:
            raise ValueError("substantive_with_followup requires followup evidence")
        if self.route == "procedure_only" and not self.procedure:
            raise ValueError("procedure_only requires procedure evidence")
        if self.route == "insufficient_content" and (facts or not any(
            flag.code == "insufficient_content" for flag in self.review_flags
        )):
            raise ValueError("insufficient_content requires no facts and an explicit review flag")
        for fact in self.procedure:
            if fact.kind not in {"intent", "actor", "request", "condition", "time", "status"}:
                raise ValueError("underlying topic/object/problem/place/impact cannot be procedure")
            if fact.role == "issue_state":
                raise ValueError("issue_state must be attached to its underlying issue")
        owner = {f.fact_id: i.issue_id for i in self.issues for f in i.facts}
        for flag in self.review_flags:
            if not set(flag.issue_ids) <= set(issue_ids):
                raise ValueError("review flag refers to an unknown issue")
            if not set(flag.fact_ids) <= set(fact_ids):
                raise ValueError("review flag refers to an unknown fact")
            if any(owner.get(fid) not in flag.issue_ids for fid in flag.fact_ids if fid in owner):
                raise ValueError("review flag must name the issues owning its facts")
            if flag.code == "conflicting_claims" and len(set(flag.fact_ids)) < 2:
                raise ValueError("conflicting_claims must preserve at least two distinct claims")
            if flag.code not in {"missing_context", "insufficient_content"} and not flag.evidence:
                raise ValueError("this review flag requires explicit source evidence")
        return self
