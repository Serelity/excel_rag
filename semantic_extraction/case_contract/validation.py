from __future__ import annotations

import hashlib

from .schema import PROMPT_VERSION, SPEC_VERSION, CaseExtraction, Quote

KIND_LABELS = {
    "topic": "主题", "intent": "意图", "actor": "主体", "object": "对象", "problem": "现象",
    "request": "诉求", "impact": "影响", "place": "地点", "condition": "条件", "time": "时间",
    "status": "状态",
}
ROLE_LABELS = {
    "subject": "主题", "consultation": "咨询", "complaint": "投诉", "report": "举报",
    "service_request": "服务请求", "suggestion": "建议", "followup": "催办/跟进",
    "withdrawal": "撤单", "supplement": "补充", "employer": "雇主", "merchant": "商家",
    "affected_person": "受影响者", "respondent": "被反映主体", "service_provider": "服务方",
    "target_institution": "被查询机构", "handler": "处理方", "facility": "设施",
    "product": "产品", "service": "服务", "account": "账户", "document": "证明/材料",
    "benefit": "权益/待遇", "transaction": "交易", "phenomenon": "异常现象",
    "behavior": "行为", "obstacle": "障碍", "dispute": "争议", "desired_action": "希望动作",
    "desired_state": "希望状态", "information": "所问信息", "experienced": "已述影响",
    "risk": "风险", "avoided": "未实际发生的后果", "incident": "事发地", "residence": "居住地",
    "registration": "登记/户籍地", "insurance": "参保地", "treatment": "就医地",
    "origin": "起点/迁出地", "destination": "终点/目的地", "policy_area": "适用地域",
    "landmark": "定位参照", "organization_location": "机构所在地", "identity": "身份",
    "eligibility": "资格条件", "amount": "金额", "quantity": "数量", "prerequisite": "前提",
    "event_time": "事项时间", "action_time": "行动时间", "period": "期间",
    "frequency": "频率", "relative": "相对时间", "issue_state": "原问题状态",
    "ticket_state": "工单状态", "action_state": "措施状态", "other": "其他", "unspecified": "未明",
}
MODALITY_LABELS = {
    "asserted": "原文陈述", "possible": "疑似", "negated": "否定", "hypothetical": "假设/计划",
    "unresolved": "待核实", "not_applicable": "不适用",
}
PHASE_LABELS = {
    "current": "当前", "background": "背景", "followup": "后续", "unspecified": "时序未明",
}


class GroundingError(ValueError):
    """The contract is structurally valid but a quote cannot be located unambiguously."""


def ground_quote(case_content: str, quote: Quote, path: str) -> dict:
    starts = []
    position = case_content.find(quote.text)
    while position >= 0:
        starts.append(position)
        position = case_content.find(quote.text, position + 1)
    if not starts:
        raise GroundingError(f"{path}: quote is absent from the unchanged case_content")
    occurrence = quote.occurrence
    if occurrence is None:
        if len(starts) != 1:
            raise GroundingError(f"{path}: repeated quote requires an explicit occurrence")
        occurrence = 0
    if occurrence >= len(starts):
        raise GroundingError(f"{path}: occurrence is outside the available matches")
    start = starts[occurrence]
    return {
        "path": path, "text": quote.text, "occurrence": occurrence,
        "start": start, "end": start + len(quote.text), "all_occurrences": starts,
    }


def _render_fact(fact) -> str:
    qualifiers = [PHASE_LABELS[fact.phase]]
    if fact.modality != "not_applicable":
        qualifiers.append(MODALITY_LABELS[fact.modality])
    if fact.attribution:
        qualifiers.append(f"来源：{fact.attribution.text}")
    if fact.time_context:
        qualifiers.append(f"时间：{fact.time_context.text}")
    label = f"{KIND_LABELS[fact.kind]}（{ROLE_LABELS[fact.role]}；{'；'.join(qualifiers)}）"
    return f"{label}：{' / '.join(q.text for q in fact.evidence)}"


def build_representations(case_content: str, extraction: CaseExtraction) -> dict:
    lines = [f"事项 {i.issue_id}：" + "；".join(_render_fact(f) for f in i.facts)
             for i in extraction.issues]
    if extraction.procedure:
        lines.append("流程：" + "；".join(_render_fact(f) for f in extraction.procedure))
    if extraction.review_flags:
        # Include uncertainty type and scope, never free-form analytical notes as query facts.
        lines.append("待复核：" + "；".join(
            f"{f.code}（{','.join(f.issue_ids + f.fact_ids) or '全文'}）"
            for f in extraction.review_flags
        ))
    brief = "\n".join(lines) if extraction.route != "insufficient_content" else ""
    scope = "substantive" if extraction.issues else (
        "procedure" if extraction.procedure else "insufficient"
    )
    return {
        "A_raw": case_content, "B_evidence": brief,
        "C_raw_plus_evidence": case_content + ("\n字段证据表示：\n" + brief if brief else ""),
        "scope": scope, "usage": "development_only_not_automatically_dispatched",
    }


def validate_response(case_content: str, response: dict) -> dict:
    if not isinstance(case_content, str):
        raise TypeError("case_content must be an unchanged string")
    parsed = CaseExtraction.model_validate(response)
    grounded = []

    def walk(value, path):
        if isinstance(value, dict):
            if set(value) == {"text", "occurrence"}:
                grounded.append(ground_quote(case_content, Quote.model_validate(value), path))
            else:
                for key, child in value.items():
                    walk(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")

    payload = parsed.model_dump()
    walk(payload, "")
    return {
        "spec_version": SPEC_VERSION, "contract_prompt_version": PROMPT_VERSION,
        "case_content_sha256": hashlib.sha256(case_content.encode("utf-8")).hexdigest(),
        "extraction": payload, "evidence_grounding": grounded,
        "offset_unit": "python_unicode_codepoint_half_open",
        "representations": build_representations(case_content, parsed),
        "validation": {
            "structure": "passed", "grounding": "passed", "semantics": "not_evaluated",
            "completeness": "not_evaluated", "retrieval_utility": "not_evaluated",
        },
    }
