"""Invented regression examples. These are not the user's 80 reviewed cases."""

from __future__ import annotations

from .schema import SPEC_VERSION


def quote(text: str, occurrence: int | None = None) -> dict:
    return {"text": text, "occurrence": occurrence}


def fact(number, kind, role, text, *, modality="not_applicable", phase="current",
         attribution=None, time_context=None, occurrence=None) -> dict:
    return {
        "fact_id": f"F{number}", "kind": kind, "role": role,
        "evidence": [quote(text, occurrence)], "modality": modality, "phase": phase,
        "attribution": quote(attribution) if attribution else None,
        "time_context": quote(time_context) if time_context else None,
    }


def response(issues=(), procedure=(), flags=(), route="substantive") -> dict:
    return {
        "spec_version": SPEC_VERSION, "route": route,
        "issues": [{"issue_id": f"I{n}", "facts": facts} for n, facts in enumerate(issues, 1)],
        "procedure": list(procedure), "review_flags": list(flags),
    }


def flag(code, note, *, issue_ids=(), fact_ids=(), evidence=()) -> dict:
    return {"code": code, "note": note, "issue_ids": list(issue_ids),
            "fact_ids": list(fact_ids), "evidence": [quote(t) for t in evidence]}


def synthetic_examples() -> list[dict]:
    examples = []

    def add(name, raw, answer, rules):
        examples.append({"example_id": name, "source": "invented_not_private_case",
                         "input": {"case_content": raw}, "response": answer, "rules": rules})

    add("S01_object_phenomenon",
        "居民反映青桥镇柳叶路停车场旁的公共厕所淹水，有粪便漂浮，无法进入，要求恢复公共厕所使用。",
        response([[
            fact(1, "topic", "subject", "公共厕所淹水"),
            fact(2, "actor", "affected_person", "居民"),
            fact(3, "object", "facility", "公共厕所", occurrence=0),
            fact(4, "problem", "phenomenon", "淹水", modality="asserted", attribution="居民"),
            fact(5, "problem", "phenomenon", "有粪便漂浮", modality="asserted", attribution="居民"),
            fact(6, "impact", "experienced", "无法进入", modality="asserted", attribution="居民"),
            fact(7, "request", "desired_state", "恢复公共厕所使用"),
            fact(8, "intent", "service_request", "要求"),
            fact(9, "place", "incident", "青桥镇柳叶路停车场旁"),
        ]]), ["R04", "R05", "R07"])
    add("S02_company_and_place",
        "来电人称在青禾公司工作，工作地点为白石镇，公司拖欠12月工资3200元，希望支付。",
        response([[
            fact(1, "topic", "subject", "公司拖欠12月工资3200元"),
            fact(2, "actor", "employer", "青禾公司"),
            fact(3, "place", "incident", "白石镇"),
            fact(4, "problem", "behavior", "拖欠12月工资3200元", modality="asserted",
                 attribution="来电人", time_context="12月"),
            fact(5, "condition", "amount", "3200元"),
            fact(6, "time", "period", "12月"),
            fact(7, "request", "desired_action", "希望支付"),
            fact(8, "intent", "service_request", "希望支付"),
        ]]), ["R05", "R06", "R09"])
    add("S03_consultation_unresolved", "咨询孩子医保是否参保成功。",
        response([[
            fact(1, "topic", "subject", "咨询孩子医保是否参保成功"),
            fact(2, "intent", "consultation", "咨询"),
            fact(3, "condition", "identity", "孩子"),
            fact(4, "object", "benefit", "医保"),
            fact(5, "status", "issue_state", "是否参保成功", modality="unresolved"),
            fact(6, "request", "information", "是否参保成功"),
        ]]), ["R07"])
    add("S04_procedure_only", "来电催办原工单，原事项仍在处理中，本次催办登记已办结。",
        response(procedure=[
            fact(1, "intent", "followup", "催办原工单"),
            fact(2, "status", "action_state", "原事项仍在处理中", modality="asserted"),
            fact(3, "status", "ticket_state", "本次催办登记已办结", modality="asserted"),
        ], flags=[flag("missing_context", "没有原问题主题，无法生成同类原投诉查询。")],
            route="procedure_only"), ["R02", "R09"])
    add("S05_conflicting_claims", "社区称噪声已消失，居民随后反映噪声仍持续。",
        response([[
            fact(1, "topic", "subject", "噪声", occurrence=0),
            fact(2, "status", "issue_state", "噪声已消失", modality="asserted",
                 attribution="社区", phase="background"),
            fact(3, "status", "issue_state", "噪声仍持续", modality="asserted",
                 attribution="居民", time_context="随后", phase="followup"),
        ]], flags=[flag("conflicting_claims", "双方对噪声状态描述不同，可能含时序变化，待核对。",
                        issue_ids=["I1"], fact_ids=["F2", "F3"],
                        evidence=["社区称噪声已消失", "居民随后反映噪声仍持续"])],
            route="substantive_with_followup"), ["R08", "R09", "R10"])
    add("S06_four_independent_issues", "分别反映水管漏水、路面破损、电梯停运、网络断线。",
        response([[fact(n, "topic", "subject", text)] for n, text in enumerate(
            ["水管漏水", "路面破损", "电梯停运", "网络断线"], 1)]), ["R03"])
    add("S07_insufficient", "（没有有效对话）",
        response(flags=[flag("insufficient_content", "未给出可识别的业务主题或流程。",
                             evidence=["（没有有效对话）"])],
                 route="insufficient_content"), ["R02"])
    add("S08_location_roles_and_plan", "在柳州市参保，计划去青湾市就医，咨询异地备案。",
        response([[
            fact(1, "topic", "subject", "咨询异地备案"),
            fact(2, "intent", "consultation", "咨询"),
            fact(3, "place", "insurance", "柳州市"),
            fact(4, "place", "treatment", "青湾市", modality="hypothetical"),
            fact(5, "condition", "prerequisite", "计划去青湾市就医", modality="hypothetical"),
            fact(6, "request", "information", "异地备案"),
        ]]), ["R05", "R07", "R09"])
    add("S09_uncertain_grouping", "咨询账户管理费和转账手续费。",
        response([[
            fact(1, "topic", "subject", "账户管理费和转账手续费"),
            fact(2, "intent", "consultation", "咨询"),
        ]], flags=[flag("uncertain_grouping", "两个费用是独立咨询还是同一扣费组成，原文未说明。",
                        issue_ids=["I1"], fact_ids=["F1"],
                        evidence=["账户管理费和转账手续费"])]), ["R03", "R10"])
    return examples
