#!/usr/bin/env python3
"""Build a privacy-aware, reproducible profile of the civic ticket TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "civic-data-profile-v2"
NULL_TOKENS = {"null", "none", "nan", "n/a", "na"}
TEXT_FIELDS = {
    "case_content",
    "case_goal",
    "address_detail",
    "return_visit_reason",
    "custom_form_data_str",
}
SENSITIVE_FIELDS = {
    "id",
    "order_no",
    "order_id",
    "case_content",
    "case_goal",
    "address_detail",
    "knowledge_quote",
    "deptName",
    "return_visit_reason",
    "custom_form_data_str",
}
SAFE_TOP_VALUE_FIELDS = {
    "service_object_type",
    "case_public",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "order_type",
    "case_is_visit",
    "case_is_urgent",
    "info_protect",
    "hotspot",
    "case_labels",
    "order_source",
    "special_type",
    "case_accord_type_one_name",
    "case_accord_type_two_name",
    "case_accord_type_three_name",
    "case_accord_type_four_name",
    "case_accord_type_five_name",
    "order_status",
    "order_invalid_type",
    "delete_flag",
    "order_source_detail",
    "area_code",
    "is_accuracy",
    "belong_platform",
    "isOverTime",
    "isSignOverTime",
    "resultSatisfied",
    "visitCount",
    "visitResult",
    "firstVisitSatisfied",
    "appeal_status",
    "form_type",
}
DATETIME_FIELDS = {"call_time", "case_complete_time"}
TAXONOMY_FIELDS = tuple(
    f"case_accord_type_{name}_name" for name in ("one", "two", "three", "four", "five")
)
GEOGRAPHY_FIELDS = ("area_code_city", "area_code_area", "area_code_street")
WORKFLOW_FIELDS = (
    "order_status",
    "order_invalid_type",
    "case_complete_time",
    "is_accuracy",
    "deptName",
    "isOverTime",
    "isSignOverTime",
    "resultSatisfied",
    "visitCount",
    "visitResult",
    "firstVisitSatisfied",
    "appeal_status",
)

SEMANTIC_STATUS_LABELS = {
    "data_confirmed": "数据证据确认",
    "inferred": "字段名和值域推断",
    "unknown": "业务口径待确认",
}

# These definitions describe what the sanitized export supports. They are not a
# substitute for the source system's official data dictionary. Ambiguous fields
# remain explicitly marked instead of being promoted from a name-based guess to
# a business fact.
FIELD_SEMANTICS: dict[str, dict[str, str | bool]] = {
    "id": {
        "business_name": "记录标识",
        "definition": "当前导出中每条记录的唯一标识；只证明行级唯一，不证明它等同于唯一工单。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "system",
        "serving_availability": "system_key",
        "rag_role": "record_key",
        "requires_confirmation": False,
        "caution": "不得据此推断用户身份。",
    },
    "order_no": {
        "business_name": "业务编号",
        "definition": "疑似面向业务展示或外部交换的工单编号，确切生成规则和唯一性需确认。",
        "semantic_status": "inferred",
        "lifecycle_stage": "system",
        "serving_availability": "unknown",
        "rag_role": "audit_only",
        "requires_confirmation": True,
        "caution": "缺失率高且不是稳定主键，不进入模型文本。",
    },
    "order_id": {
        "business_name": "业务关联标识",
        "definition": "可关联多条记录的标识；组内既可能是流程快照，也可能是内容修订或连续提交。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "system",
        "serving_availability": "system_key",
        "rag_role": "association_key",
        "requires_confirmation": True,
        "caution": "不能直接解释为父工单、用户ID，也不能据此无条件合并记录。",
    },
    "service_object_type": {
        "business_name": "诉求类型",
        "definition": "咨询、求助、投诉举报、意见建议等诉求性质分类。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_metadata",
        "requires_confirmation": True,
        "caution": "需确认是用户选择、坐席标注还是后处理结果。",
    },
    "case_content": {
        "business_name": "工单正文",
        "definition": "对事件、问题和上下文的主要文本描述；是否为用户原话或坐席转写尚未确认。",
        "semantic_status": "inferred",
        "lifecycle_stage": "intake_or_editing",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "primary_query_candidate",
        "requires_confirmation": True,
        "caution": "含隐私信息且可能在流程中修订；建模前需脱敏并确认版本时点。",
    },
    "case_goal": {
        "business_name": "诉求目标",
        "definition": "通常比正文更短的办理目标或诉求摘要，可能由坐席归纳。",
        "semantic_status": "inferred",
        "lifecycle_stage": "intake_or_classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "query_view_candidate",
        "requires_confirmation": True,
        "caution": "若在检索之后生成，作为输入会造成未来信息泄漏。",
    },
    "case_public": {
        "business_name": "是否公开",
        "definition": "工单或案例是否允许公开的控制标志。",
        "semantic_status": "inferred",
        "lifecycle_stage": "governance",
        "serving_availability": "unknown",
        "rag_role": "policy_filter",
        "requires_confirmation": True,
        "caution": "用于合规控制，不作为语义相关性特征。",
    },
    "area_code_city": {
        "business_name": "市级地域名称",
        "definition": "工单关联的市级地域文本。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_routing",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_filter_or_metadata",
        "requires_confirmation": True,
        "caution": "需确认表示事发地、诉求人所在地还是承办归属地。",
    },
    "area_code_area": {
        "business_name": "区县级地域名称",
        "definition": "工单关联的区县或市本级地域文本。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_routing",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_filter_or_metadata",
        "requires_confirmation": True,
        "caution": "存在市本级、不涉及和历史名称，不能直接当标准行政区。",
    },
    "area_code_street": {
        "business_name": "街道乡镇名称",
        "definition": "工单关联的街道或乡镇文本。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_routing",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_filter_or_metadata",
        "requires_confirmation": True,
        "caution": "缺失率高，且地域含义仍需与源系统确认。",
    },
    "address_detail": {
        "business_name": "详细地址",
        "definition": "源字段用于详细地点；当前脱敏表中的有效值已统一替换，已不保留地址语义。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake",
        "serving_availability": "unusable_after_sanitization",
        "rag_role": "excluded",
        "requires_confirmation": False,
        "caution": "当前数据中仅有一个脱敏占位值，禁止把它输入模型。",
    },
    "order_type": {
        "business_name": "诉求主体类型",
        "definition": "从个人、企业等值域看，表示诉求主体或服务对象类别，而非流程类型。",
        "semantic_status": "inferred",
        "lifecycle_stage": "intake",
        "serving_availability": "likely_at_intake",
        "rag_role": "candidate_metadata",
        "requires_confirmation": True,
        "caution": "字段名与实际值域不完全一致，需确认官方名称。",
    },
    "case_is_visit": {
        "business_name": "回访标志",
        "definition": "疑似表示是否需要或是否进入回访，具体是计划状态还是执行结果尚不明确。",
        "semantic_status": "unknown",
        "lifecycle_stage": "workflow_or_follow_up",
        "serving_availability": "unknown",
        "rag_role": "excluded_until_confirmed",
        "requires_confirmation": True,
        "caution": "存在少量越域数字值；不能仅凭是/否确定业务含义。",
    },
    "case_is_urgent": {
        "business_name": "紧急程度",
        "definition": "一般、紧急、非常紧急等工单优先级。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_triage",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_metadata",
        "requires_confirmation": True,
        "caution": "通常影响处理优先级，不必然影响知识相关性。",
    },
    "info_protect": {
        "business_name": "信息保护标志",
        "definition": "表示记录是否需要执行额外的信息保护。",
        "semantic_status": "inferred",
        "lifecycle_stage": "governance",
        "serving_availability": "likely_at_intake",
        "rag_role": "policy_filter",
        "requires_confirmation": True,
        "caution": "只能用于访问控制和脱敏策略，不能作为检索语义。",
    },
    "hotspot": {
        "business_name": "热点标志",
        "definition": "疑似标记热点事件或热点工单。",
        "semantic_status": "inferred",
        "lifecycle_stage": "triage_or_monitoring",
        "serving_availability": "unknown",
        "rag_role": "audit_or_slice",
        "requires_confirmation": True,
        "caution": "正例极少，适合作为分析切片而非主要输入特征。",
    },
    "case_labels": {
        "business_name": "补充业务标签",
        "definition": "稀疏的专题、渠道或问题标签，可包含多个层级化标签。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification_or_routing",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "assisted_mode_metadata",
        "requires_confirmation": True,
        "caution": "可能是人工或规则后标注，直接输入可能产生标签泄漏。",
    },
    "order_source": {
        "business_name": "受理渠道大类",
        "definition": "电话、互联网、承办转办等工单来源大类。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake",
        "serving_availability": "at_intake",
        "rag_role": "candidate_metadata",
        "requires_confirmation": False,
        "caution": "渠道分布高度偏向电话，评测应按渠道分层。",
    },
    "special_type": {
        "business_name": "特殊工单类型",
        "definition": "用于少量特殊关怀或特殊流程工单的附加类型。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake_or_triage",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "audit_or_slice",
        "requires_confirmation": True,
        "caution": "覆盖率极低，不适合作为通用模型特征。",
    },
    "case_accord_type_one_name": {
        "business_name": "事项分类一级",
        "definition": "工单所属事项分类体系的一级名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "assisted_mode_metadata",
        "requires_confirmation": True,
        "caution": "需确认分类发生在知识检索之前还是之后。",
    },
    "case_accord_type_two_name": {
        "business_name": "事项分类二级",
        "definition": "工单所属事项分类体系的二级名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "assisted_mode_metadata",
        "requires_confirmation": True,
        "caution": "可能与被引用知识高度相关，存在标签泄漏风险。",
    },
    "case_accord_type_three_name": {
        "business_name": "事项分类三级",
        "definition": "工单所属事项分类体系的三级名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "assisted_mode_metadata",
        "requires_confirmation": True,
        "caution": "粒度较细，只能在坐席辅助场景单独评测。",
    },
    "case_accord_type_four_name": {
        "business_name": "事项分类四级",
        "definition": "少量记录使用的事项分类四级名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "assisted_mode_metadata",
        "requires_confirmation": True,
        "caution": "覆盖率极低且可能直接表达问题答案。",
    },
    "case_accord_type_five_name": {
        "business_name": "事项分类五级",
        "definition": "极少量记录使用的事项分类五级名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "classification",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "excluded_from_baseline",
        "requires_confirmation": True,
        "caution": "仅3条有效记录，不能形成稳定特征。",
    },
    "order_status": {
        "business_name": "流程状态码",
        "definition": "数字化工单流程状态；当前数据没有代码到状态名称的映射。",
        "semantic_status": "unknown",
        "lifecycle_stage": "workflow",
        "serving_availability": "changes_over_time",
        "rag_role": "audit_only",
        "requires_confirmation": True,
        "caution": "必须取得状态码表，不能按数值大小解释流程先后。",
    },
    "order_invalid_type": {
        "business_name": "无效工单原因",
        "definition": "骚扰电话、无声电话、拨错号码、无效多诉求等无效原因。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "triage_or_workflow",
        "serving_availability": "potentially_post_intake",
        "rag_role": "eligibility_filter",
        "requires_confirmation": True,
        "caution": "应作为样本纳入/排除规则，不作为知识检索文本。",
    },
    "call_time": {
        "business_name": "受理时间",
        "definition": "电话渠道下疑似来电或受理时间；非电话渠道的统一含义需确认。",
        "semantic_status": "inferred",
        "lifecycle_stage": "intake",
        "serving_availability": "at_intake",
        "rag_role": "temporal_split_key",
        "requires_confirmation": True,
        "caution": "用于时间切分前需确认跨渠道口径和历史回灌情况。",
    },
    "case_complete_time": {
        "business_name": "工单完成时间",
        "definition": "疑似流程完成或办结时间。",
        "semantic_status": "inferred",
        "lifecycle_stage": "closure",
        "serving_availability": "post_resolution",
        "rag_role": "audit_only",
        "requires_confirmation": True,
        "caution": "属于检索后的未来信息，不能进入在线检索输入。",
    },
    "knowledge_quote": {
        "business_name": "知识引用记录",
        "definition": "JSON数组，每项含type、value和label，记录系统中观察到的知识条目引用。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "agent_action_or_workflow",
        "serving_availability": "target_only",
        "rag_role": "weak_observed_label",
        "requires_confirmation": True,
        "caution": "历史引用不是完整相关性金标；空值也不是负例。",
    },
    "delete_flag": {
        "business_name": "逻辑删除标志",
        "definition": "0/1逻辑删除状态。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "system",
        "serving_availability": "system_key",
        "rag_role": "eligibility_filter",
        "requires_confirmation": False,
        "caution": "建模集应默认排除已删除记录并保留数量审计。",
    },
    "order_source_detail": {
        "business_name": "受理渠道明细",
        "definition": "12345、12393、小程序、APP、网站、微信等细粒度来源。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake",
        "serving_availability": "at_intake",
        "rag_role": "candidate_metadata",
        "requires_confirmation": False,
        "caution": "渠道可能与样本分布和标注方式相关，应做分层评测。",
    },
    "area_code": {
        "business_name": "行政区划代码",
        "definition": "6位地域代码，值域符合常州市及下辖区域编码形态。",
        "semantic_status": "inferred",
        "lifecycle_stage": "routing_or_system",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_filter_or_metadata",
        "requires_confirmation": True,
        "caution": "需确认是事发地代码、受理归属代码还是承办区域代码。",
    },
    "is_accuracy": {
        "business_name": "准确性标志",
        "definition": "主要为0/1，但“准确”的对象和产生规则无法从当前数据确定。",
        "semantic_status": "unknown",
        "lifecycle_stage": "quality_control",
        "serving_availability": "unknown",
        "rag_role": "excluded_until_confirmed",
        "requires_confirmation": True,
        "caution": "存在两个时间字符串越域值；取得官方定义前不得用于过滤或训练。",
    },
    "belong_platform": {
        "business_name": "归属平台代码",
        "definition": "疑似表示受理或管辖平台的6位区域代码。",
        "semantic_status": "inferred",
        "lifecycle_stage": "routing_or_system",
        "serving_availability": "needs_timing_confirmation",
        "rag_role": "candidate_metadata",
        "requires_confirmation": True,
        "caution": "不能与事发地域或承办部门自动视为同义。",
    },
    "return_visit_reason": {
        "business_name": "回访原因",
        "definition": "回访流程中的原因字段，具体表示触发原因、未回访原因还是结果原因待确认。",
        "semantic_status": "unknown",
        "lifecycle_stage": "follow_up",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "属于后验流程信息，不进入知识检索基线。",
    },
    "deptName": {
        "business_name": "部门名称",
        "definition": "与工单关联的部门名称；是受理、当前处理还是最终承办部门待确认。",
        "semantic_status": "unknown",
        "lifecycle_stage": "routing_or_workflow",
        "serving_availability": "potentially_post_intake",
        "rag_role": "excluded_from_baseline",
        "requires_confirmation": True,
        "caution": "可能直接泄漏路由或处理结果，且报告中不输出原始值。",
    },
    "isOverTime": {
        "business_name": "办理超时标志",
        "definition": "疑似表示整体办理是否超时；当前仅观察到值1。",
        "semantic_status": "inferred",
        "lifecycle_stage": "workflow_or_closure",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "空值不能未经确认解释为未超时。",
    },
    "isSignOverTime": {
        "business_name": "签收超时标志",
        "definition": "疑似表示承办方签收是否超时；当前仅观察到值1。",
        "semantic_status": "inferred",
        "lifecycle_stage": "workflow",
        "serving_availability": "post_intake",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "空值不能未经确认解释为未超时。",
    },
    "resultSatisfied": {
        "business_name": "结果满意度",
        "definition": "满意、基本满意、不满意或未表态等结果评价。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "follow_up",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "需确认与visitResult、firstVisitSatisfied的口径差异。",
    },
    "visitCount": {
        "business_name": "回访次数",
        "definition": "记录回访次数的整数计数。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "follow_up",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": False,
        "caution": "不进入检索输入。",
    },
    "visitResult": {
        "business_name": "回访结果",
        "definition": "当前值域表现为满意度结果，而非自由文本回访记录。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "follow_up",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "需确认与结果满意度字段的适用流程差异。",
    },
    "firstVisitSatisfied": {
        "business_name": "首次回访满意度",
        "definition": "首次回访时记录的满意、基本满意、不满意或未表态。",
        "semantic_status": "inferred",
        "lifecycle_stage": "follow_up",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": True,
        "caution": "与原始表中的firstVistSatisfied拼写近似字段需核对来源。",
    },
    "appeal_status": {
        "business_name": "申诉审核状态",
        "definition": "审核中、审核通过、审核不通过等申诉流程状态。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "appeal",
        "serving_availability": "post_resolution",
        "rag_role": "outcome_analysis_only",
        "requires_confirmation": False,
        "caution": "覆盖率极低，不进入检索输入。",
    },
    "form_type": {
        "business_name": "表单模板类型",
        "definition": "通用表单或占道经营、拖欠工资等专题表单名称。",
        "semantic_status": "data_confirmed",
        "lifecycle_stage": "intake",
        "serving_availability": "likely_at_intake",
        "rag_role": "candidate_metadata",
        "requires_confirmation": True,
        "caution": "专题表单本身带有问题类别信息，需单独报告是否造成捷径学习。",
    },
    "custom_form_data_str": {
        "business_name": "自定义表单数据",
        "definition": "自定义表单提交的JSON数组；当前尚未解析内部字段语义。",
        "semantic_status": "unknown",
        "lifecycle_stage": "intake",
        "serving_availability": "likely_at_intake",
        "rag_role": "quarantine_until_schema_known",
        "requires_confirmation": True,
        "caution": "可能包含隐私信息，必须取得子字段schema后才能清洗和使用。",
    },
}

BUSINESS_CONFIRMATION_QUESTIONS = {
    "order_no": "该字段的官方名称、生成规则、唯一性范围以及为何仅部分记录存在是什么？",
    "order_id": "该标识关联的是流程版本、同一来电、同一用户连续提交，还是其他业务对象？",
    "service_object_type": "该分类由谁在什么时点产生，知识检索发生时是否已经可用？",
    "case_content": "该文本是用户原话、坐席转写还是可被后续修改的最终正文？",
    "case_goal": "该字段由谁归纳、何时写入，是否早于知识检索行为？",
    "case_public": "“公开”的对象、适用渠道和赋值时点分别是什么？",
    "area_code_city": "该地域表示事发地、诉求人所在地、受理地还是承办归属地？",
    "area_code_area": "该地域表示事发地、诉求人所在地、受理地还是承办归属地？",
    "area_code_street": "该地域表示事发地、诉求人所在地、受理地还是承办归属地？",
    "order_type": "字段官方名称是否为诉求主体类型，个人/企业之外值的口径是什么？",
    "case_is_visit": "该字段表示需要回访、已经回访还是可回访，赋值时点是什么？",
    "case_is_urgent": "紧急程度由谁判断，知识检索发生时是否已经确定？",
    "info_protect": "保护标志具体控制哪些信息和流程，0与空值是否有定义？",
    "hotspot": "热点标志由人工、规则还是事后统计产生，赋值时点是什么？",
    "case_labels": "标签由谁生成、何时写入，多个标签的分隔和层级规则是什么？",
    "special_type": "特殊类型的完整码表、触发规则和写入时点是什么？",
    "case_accord_type_one_name": "事项分类由谁完成，是否发生在知识检索之前？",
    "case_accord_type_two_name": "事项分类由谁完成，是否发生在知识检索之前？",
    "case_accord_type_three_name": "事项分类由谁完成，是否发生在知识检索之前？",
    "case_accord_type_four_name": "四级分类只在少量专题流程出现，还是历史口径缺失？",
    "case_accord_type_five_name": "五级分类仅3条有效记录的业务原因是什么？",
    "order_status": "21个状态码对应的名称、转换关系和终态分别是什么？",
    "order_invalid_type": "无效原因在哪个流程节点产生，标记后是否还可能引用知识？",
    "call_time": "电话、互联网和转办渠道下该时间分别代表来电、创建还是受理时刻？",
    "case_complete_time": "该时间表示坐席录入完成、工单办结还是流程实例结束？",
    "knowledge_quote": "type=0/2分别代表什么，引用由坐席点击还是系统自动记录，写入时点是什么？",
    "area_code": "该代码表示事发区域、受理区域、派单区域还是承办区域？",
    "is_accuracy": "准确性评价针对分类、派单、地址还是其他对象，由谁在何时评价？",
    "belong_platform": "平台代码表示受理平台、数据归属平台还是最终承办平台？",
    "return_visit_reason": "该字段表示触发回访、未回访还是回访结果原因？",
    "deptName": "该部门是受理部门、当前处理部门还是最终承办部门？",
    "isOverTime": "空值是否等于未超时，超时判断针对哪个办理时限？",
    "isSignOverTime": "空值是否等于未超时，“签收”的业务节点和时限是什么？",
    "resultSatisfied": "该满意度评价的对象和采集轮次是什么，与回访结果有何区别？",
    "visitResult": "该字段与resultSatisfied的适用流程、采集轮次和覆盖范围有何区别？",
    "firstVisitSatisfied": "该字段与原始表firstVistSatisfied是否为同一指标的不同版本？",
    "form_type": "专题表单是在用户提交前选择，还是受理后由坐席切换？",
    "custom_form_data_str": "JSON数组中各子字段的schema、表单版本和隐私等级是什么？",
}

PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DOMAIN_RULES: dict[str, set[str] | re.Pattern[str]] = {
    "case_public": {"是", "否"},
    "case_is_visit": {"是", "否"},
    "case_is_urgent": {"一般", "紧急", "非常紧急"},
    "info_protect": {"是", "否"},
    "hotspot": {"是", "否"},
    "delete_flag": re.compile(r"[01]"),
    "area_code": re.compile(r"\d{6}"),
    "order_status": re.compile(r"\d+"),
    "is_accuracy": re.compile(r"[01]"),
    "isOverTime": re.compile(r"[01]"),
    "isSignOverTime": re.compile(r"[01]"),
    "visitCount": re.compile(r"\d+"),
}


def semantic_value(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.casefold() in NULL_TOKENS:
        return None
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def short_hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12).hexdigest()


def quantile_from_counts(counts: Counter[int], probability: float) -> int | None:
    total = sum(counts.values())
    if not total:
        return None
    target = max(1, math.ceil(total * probability))
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen >= target:
            return value
    return max(counts)


def safe_display_value(value: str) -> str:
    """Prevent malformed categorical values from leaking identifiers or PII."""
    if PHONE_PATTERN.search(value) or ID_CARD_PATTERN.search(value) or EMAIL_PATTERN.search(value):
        return "<REDACTED_PII_LIKE>"
    if re.search(r"\d{7,}", value):
        return "<REDACTED_LONG_NUMBER>"
    if len(value) > 120 or CONTROL_PATTERN.search(value):
        return "<REDACTED_UNSAFE_VALUE>"
    return value


def safe_counter_items(counter: Counter[str], top_k: int, key: str) -> list[dict[str, Any]]:
    displayed: Counter[str] = Counter()
    for value, count in counter.items():
        displayed[safe_display_value(value)] += count
    return [{key: value, "count": count} for value, count in displayed.most_common(top_k)]


class HyperLogLog:
    """Small deterministic approximate-distinct counter."""

    def __init__(self, precision: int = 12) -> None:
        self.precision = precision
        self.size = 1 << precision
        self.registers = bytearray(self.size)

    def add(self, value: str) -> None:
        hashed = int.from_bytes(
            hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
        )
        index = hashed & (self.size - 1)
        remainder = hashed >> self.precision
        bits = 64 - self.precision
        rank = bits - remainder.bit_length() + 1 if remainder else bits + 1
        if rank > self.registers[index]:
            self.registers[index] = rank

    def estimate(self) -> int:
        size = self.size
        alpha = 0.7213 / (1 + 1.079 / size)
        raw = alpha * size * size / sum(2.0 ** -register for register in self.registers)
        zeroes = self.registers.count(0)
        if raw <= 2.5 * size and zeroes:
            raw = size * math.log(size / zeroes)
        return max(0, round(raw))


@dataclass
class ColumnProfile:
    name: str
    total: int = 0
    blank: int = 0
    null_literal: int = 0
    filled: int = 0
    trimmed: int = 0
    multiline: int = 0
    control_character: int = 0
    integer_like: int = 0
    length_counts: Counter[int] = field(default_factory=Counter)
    values: Counter[str] | None = None
    approximate_distinct: HyperLogLog | None = None

    def __post_init__(self) -> None:
        if self.name in SAFE_TOP_VALUE_FIELDS:
            self.values = Counter()
        else:
            self.approximate_distinct = HyperLogLog()

    def update(self, raw: str | None) -> None:
        self.total += 1
        if raw is None or raw == "":
            self.blank += 1
            return
        stripped = raw.strip()
        if stripped != raw:
            self.trimmed += 1
        if not stripped:
            self.blank += 1
            return
        if stripped.casefold() in NULL_TOKENS:
            self.null_literal += 1
            return
        self.filled += 1
        self.length_counts[len(stripped)] += 1
        if "\n" in stripped or "\r" in stripped:
            self.multiline += 1
        if CONTROL_PATTERN.search(stripped):
            self.control_character += 1
        if INTEGER_PATTERN.fullmatch(stripped):
            self.integer_like += 1
        if self.values is not None:
            self.values[stripped] += 1
        elif self.approximate_distinct is not None:
            self.approximate_distinct.add(stripped)

    def result(self, top_k: int) -> dict[str, Any]:
        missing = self.blank + self.null_literal
        if self.values is not None:
            distinct = len(self.values)
            distinct_method = "exact"
            top_values = safe_counter_items(self.values, top_k, "value")
        else:
            distinct = self.approximate_distinct.estimate() if self.approximate_distinct else 0
            distinct_method = "hyperloglog_p12"
            top_values = None
        return {
            "name": self.name,
            "total": self.total,
            "filled": self.filled,
            "missing": missing,
            "missing_rate": missing / self.total if self.total else 0.0,
            "blank": self.blank,
            "null_literal": self.null_literal,
            "trimmed_rows": self.trimmed,
            "multiline_rows": self.multiline,
            "control_character_rows": self.control_character,
            "integer_like_rows": self.integer_like,
            "distinct": distinct,
            "distinct_method": distinct_method,
            "length": {
                "min": min(self.length_counts) if self.length_counts else None,
                "p50": quantile_from_counts(self.length_counts, 0.50),
                "p95": quantile_from_counts(self.length_counts, 0.95),
                "p99": quantile_from_counts(self.length_counts, 0.99),
                "max": max(self.length_counts) if self.length_counts else None,
            },
            "top_values": top_values,
            "values_suppressed": self.name in SENSITIVE_FIELDS,
        }


@dataclass
class DateProfile:
    filled: int = 0
    valid: int = 0
    invalid: int = 0
    minimum: datetime | None = None
    maximum: datetime | None = None
    months: Counter[str] = field(default_factory=Counter)

    def update(self, value: str | None) -> datetime | None:
        normalized = semantic_value(value)
        if normalized is None:
            return None
        self.filled += 1
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            self.invalid += 1
            return None
        self.valid += 1
        self.minimum = parsed if self.minimum is None else min(self.minimum, parsed)
        self.maximum = parsed if self.maximum is None else max(self.maximum, parsed)
        self.months[parsed.strftime("%Y-%m")] += 1
        return parsed

    def result(self) -> dict[str, Any]:
        return {
            "filled": self.filled,
            "valid": self.valid,
            "invalid": self.invalid,
            "min": self.minimum.isoformat(sep=" ") if self.minimum else None,
            "max": self.maximum.isoformat(sep=" ") if self.maximum else None,
            "monthly_counts": dict(sorted(self.months.items())),
        }


class AssociationStore:
    def __init__(self, database_path: Path) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE order_groups (
                order_id TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                goal_hash TEXT NOT NULL,
                taxonomy_hash TEXT NOT NULL,
                geography_hash TEXT NOT NULL,
                knowledge_hash TEXT NOT NULL,
                call_time_hash TEXT NOT NULL,
                completion_time_hash TEXT NOT NULL,
                workflow_hash TEXT NOT NULL,
                content_variation INTEGER NOT NULL DEFAULT 0,
                goal_variation INTEGER NOT NULL DEFAULT 0,
                taxonomy_variation INTEGER NOT NULL DEFAULT 0,
                geography_variation INTEGER NOT NULL DEFAULT 0,
                knowledge_variation INTEGER NOT NULL DEFAULT 0,
                call_time_variation INTEGER NOT NULL DEFAULT 0,
                completion_time_variation INTEGER NOT NULL DEFAULT 0,
                workflow_variation INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE record_ids (
                record_id TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL
            );
            """
        )
        self.missing_order_id = 0
        self.missing_record_id = 0

    def update(self, row: dict[str, str]) -> None:
        record_id = semantic_value(row.get("id"))
        if record_id is None:
            self.missing_record_id += 1
        else:
            self.connection.execute(
                """
                INSERT INTO record_ids(record_id, row_count) VALUES (?, 1)
                ON CONFLICT(record_id) DO UPDATE SET row_count = row_count + 1
                """,
                (record_id,),
            )
        order_id = semantic_value(row.get("order_id"))
        if order_id is None:
            self.missing_order_id += 1
            return
        values = (
            short_hash(semantic_value(row.get("case_content")) or "<MISSING>"),
            short_hash(semantic_value(row.get("case_goal")) or "<MISSING>"),
            short_hash(
                "\x1f".join(
                    semantic_value(row.get(name)) or "<MISSING>"
                    for name in TAXONOMY_FIELDS
                )
            ),
            short_hash(
                "\x1f".join(
                    semantic_value(row.get(name)) or "<MISSING>"
                    for name in GEOGRAPHY_FIELDS
                )
            ),
            short_hash(semantic_value(row.get("knowledge_quote")) or "<MISSING>"),
            short_hash(semantic_value(row.get("call_time")) or "<MISSING>"),
            short_hash(semantic_value(row.get("case_complete_time")) or "<MISSING>"),
            short_hash(
                "\x1f".join(
                    semantic_value(row.get(name)) or "<MISSING>"
                    for name in WORKFLOW_FIELDS
                )
            ),
        )
        self.connection.execute(
            """
            INSERT INTO order_groups(
                order_id, row_count, content_hash, goal_hash, taxonomy_hash,
                geography_hash, knowledge_hash, call_time_hash,
                completion_time_hash, workflow_hash
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET
                row_count = row_count + 1,
                content_variation = content_variation OR content_hash <> excluded.content_hash,
                goal_variation = goal_variation OR goal_hash <> excluded.goal_hash,
                taxonomy_variation = taxonomy_variation OR taxonomy_hash <> excluded.taxonomy_hash,
                geography_variation = (
                    geography_variation OR geography_hash <> excluded.geography_hash
                ),
                knowledge_variation = (
                    knowledge_variation OR knowledge_hash <> excluded.knowledge_hash
                ),
                call_time_variation = (
                    call_time_variation OR call_time_hash <> excluded.call_time_hash
                ),
                completion_time_variation = (
                    completion_time_variation
                    OR completion_time_hash <> excluded.completion_time_hash
                ),
                workflow_variation = (
                    workflow_variation OR workflow_hash <> excluded.workflow_hash
                )
            """,
            (order_id, *values),
        )

    def result(self) -> dict[str, Any]:
        self.connection.commit()
        group_count, row_count, multirow_groups, rows_in_multi = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(row_count), 0),
                   COALESCE(SUM(row_count > 1), 0),
                   COALESCE(SUM(CASE WHEN row_count > 1 THEN row_count ELSE 0 END), 0)
            FROM order_groups
            """
        ).fetchone()
        group_sizes = {
            str(size): count
            for size, count in self.connection.execute(
                "SELECT row_count, COUNT(*) FROM order_groups "
                "GROUP BY row_count ORDER BY row_count"
            )
        }
        variation_names = (
            "content_variation",
            "goal_variation",
            "taxonomy_variation",
            "geography_variation",
            "knowledge_variation",
            "call_time_variation",
            "completion_time_variation",
            "workflow_variation",
        )
        variation_columns = ", ".join(
            f"COALESCE(SUM({name}), 0)" for name in variation_names
        )
        variation_values = self.connection.execute(
            f"SELECT {variation_columns} FROM order_groups"
        ).fetchone()
        record_id_count, duplicate_record_ids, duplicate_record_rows = self.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(row_count > 1), 0),
                   COALESCE(SUM(CASE WHEN row_count > 1 THEN row_count ELSE 0 END), 0)
            FROM record_ids
            """
        ).fetchone()
        return {
            "order_id_groups": group_count,
            "rows_with_order_id": row_count,
            "missing_order_id_rows": self.missing_order_id,
            "multirow_order_id_groups": multirow_groups,
            "rows_in_multirow_order_id_groups": rows_in_multi,
            "order_id_group_size_histogram": group_sizes,
            "order_id_group_variations": dict(
                zip(variation_names, variation_values)
            ),
            "distinct_record_ids": record_id_count,
            "missing_record_id_rows": self.missing_record_id,
            "duplicate_record_ids": duplicate_record_ids,
            "rows_using_duplicate_record_ids": duplicate_record_rows,
        }

    def close(self) -> None:
        self.connection.close()


class DatasetProfiler:
    def __init__(
        self, header: list[str], top_k: int, association_store: AssociationStore
    ) -> None:
        self.header = header
        self.top_k = top_k
        self.columns = {name: ColumnProfile(name) for name in header}
        self.association_store = association_store
        self.records = 0
        self.valid_width_records = 0
        self.invalid_width_records = 0
        self.widths: Counter[int] = Counter()
        self.full_row_distinct = HyperLogLog()
        self.pii: dict[str, Counter[str]] = {
            field: Counter() for field in TEXT_FIELDS if field in self.columns
        }
        self.domain_violations: dict[str, Counter[str]] = {
            field: Counter() for field in DOMAIN_RULES if field in self.columns
        }
        self.rows_with_domain_violation = 0
        self.rows_with_possible_pii = 0
        self.dates = {field: DateProfile() for field in DATETIME_FIELDS if field in self.columns}
        self.duration_seconds = Counter()
        self.duration_negative = 0
        self.text_relations = Counter()
        self.taxonomy_depth = Counter()
        self.taxonomy_gap_rows = 0
        self.taxonomy_paths = Counter()
        self.geography_depth = Counter()
        self.geography_gap_rows = 0
        self.geography_paths = Counter()
        self.knowledge = Counter()
        self.knowledge_types = Counter()
        self.knowledge_ids: set[str] = set()
        self.knowledge_items_per_row = Counter()
        self.custom_json = Counter()

    def mark_width(self, width: int) -> None:
        self.records += 1
        self.widths[width] += 1
        if width == len(self.header):
            self.valid_width_records += 1
        else:
            self.invalid_width_records += 1

    def update(self, values: list[str]) -> None:
        row = dict(zip(self.header, values))
        for name, value in row.items():
            self.columns[name].update(value)
        self.full_row_distinct.add("\x1e".join(values))
        self.association_store.update(row)
        self._update_domains(row)
        self._update_text(row)
        self._update_dates(row)
        self._update_hierarchies(row)
        self._update_structured(row)

    def _update_domains(self, row: dict[str, str]) -> None:
        row_has_violation = False
        for name, rule in DOMAIN_RULES.items():
            if name not in row:
                continue
            value = semantic_value(row[name])
            if value is None:
                continue
            valid = value in rule if isinstance(rule, set) else rule.fullmatch(value) is not None
            if not valid:
                self.domain_violations[name][value] += 1
                row_has_violation = True
        if row_has_violation:
            self.rows_with_domain_violation += 1

    def _update_text(self, row: dict[str, str]) -> None:
        row_has_possible_pii = False
        for name, counters in self.pii.items():
            value = semantic_value(row.get(name))
            if value is None:
                continue
            if PHONE_PATTERN.search(value):
                counters["mainland_mobile"] += 1
                row_has_possible_pii = True
            if ID_CARD_PATTERN.search(value):
                counters["citizen_id"] += 1
                row_has_possible_pii = True
            if EMAIL_PATTERN.search(value):
                counters["email"] += 1
                row_has_possible_pii = True
        if row_has_possible_pii:
            self.rows_with_possible_pii += 1
        content = semantic_value(row.get("case_content"))
        goal = semantic_value(row.get("case_goal"))
        if content is None and goal is None:
            self.text_relations["both_missing"] += 1
        elif content is None:
            self.text_relations["content_missing_only"] += 1
        elif goal is None:
            self.text_relations["goal_missing_only"] += 1
        elif content == goal:
            self.text_relations["exactly_equal"] += 1
        elif goal in content:
            self.text_relations["goal_contained_in_content"] += 1
        elif content in goal:
            self.text_relations["content_contained_in_goal"] += 1
        else:
            self.text_relations["both_present_distinct"] += 1

    def _update_dates(self, row: dict[str, str]) -> None:
        parsed = {name: profile.update(row.get(name)) for name, profile in self.dates.items()}
        start = parsed.get("call_time")
        end = parsed.get("case_complete_time")
        if start is not None and end is not None:
            seconds = round((end - start).total_seconds())
            if seconds < 0:
                self.duration_negative += 1
            else:
                bucket = 0 if seconds == 0 else 2 ** int(math.log2(seconds))
                self.duration_seconds[bucket] += 1

    @staticmethod
    def _path(row: dict[str, str], fields: Iterable[str]) -> list[str | None]:
        return [semantic_value(row.get(name)) for name in fields]

    def _update_hierarchies(self, row: dict[str, str]) -> None:
        taxonomy = self._path(row, TAXONOMY_FIELDS)
        taxonomy_depth = max(
            (index + 1 for index, value in enumerate(taxonomy) if value), default=0
        )
        self.taxonomy_depth[str(taxonomy_depth)] += 1
        if any(
            value and any(previous is None for previous in taxonomy[:index])
            for index, value in enumerate(taxonomy)
        ):
            self.taxonomy_gap_rows += 1
        if taxonomy_depth:
            path = " > ".join(
                value or "<MISSING>" for value in taxonomy[:taxonomy_depth]
            )
            self.taxonomy_paths[path] += 1

        geography = self._path(row, GEOGRAPHY_FIELDS)
        geography_depth = max(
            (index + 1 for index, value in enumerate(geography) if value), default=0
        )
        self.geography_depth[str(geography_depth)] += 1
        if any(
            value and any(previous is None for previous in geography[:index])
            for index, value in enumerate(geography)
        ):
            self.geography_gap_rows += 1
        if geography_depth:
            path = " > ".join(
                value or "<MISSING>" for value in geography[:geography_depth]
            )
            self.geography_paths[path] += 1

    def _update_structured(self, row: dict[str, str]) -> None:
        knowledge_raw = semantic_value(row.get("knowledge_quote"))
        if knowledge_raw is None:
            self.knowledge["missing"] += 1
        else:
            try:
                parsed = json.loads(knowledge_raw)
            except (TypeError, ValueError):
                self.knowledge["invalid_json"] += 1
            else:
                self.knowledge["valid_json"] += 1
                if parsed is None:
                    self.knowledge["json_null"] += 1
                elif not isinstance(parsed, list):
                    self.knowledge["non_list"] += 1
                else:
                    self.knowledge["list_rows"] += 1
                    self.knowledge_items_per_row[len(parsed)] += 1
                    seen: set[str] = set()
                    for item in parsed:
                        if not isinstance(item, dict):
                            self.knowledge["malformed_items"] += 1
                            continue
                        value = semantic_value(str(item.get("value", "")))
                        item_type = semantic_value(str(item.get("type", "")))
                        label = semantic_value(str(item.get("label", "")))
                        if value is None or item_type is None:
                            self.knowledge["items_missing_identity"] += 1
                            continue
                        identity = f"{item_type}:{value}"
                        if identity in seen:
                            self.knowledge["duplicate_items_within_row"] += 1
                        seen.add(identity)
                        self.knowledge_ids.add(identity)
                        self.knowledge_types[item_type] += 1
                        self.knowledge["items"] += 1
                        if label is None:
                            self.knowledge["items_missing_label"] += 1
                    if seen:
                        self.knowledge["rows_with_items"] += 1

        custom_raw = semantic_value(row.get("custom_form_data_str"))
        if custom_raw is None:
            self.custom_json["missing"] += 1
        else:
            try:
                parsed_custom = json.loads(custom_raw)
            except (TypeError, ValueError):
                self.custom_json["invalid_json"] += 1
            else:
                self.custom_json["valid_json"] += 1
                self.custom_json[f"root_{type(parsed_custom).__name__}"] += 1

    def result(self) -> dict[str, Any]:
        association_result = self.association_store.result()
        column_results = [self.columns[name].result(self.top_k) for name in self.header]
        exact_distinct = {
            "id": association_result["distinct_record_ids"],
            "order_id": association_result["order_id_groups"],
        }
        for column in column_results:
            if column["name"] in exact_distinct:
                column["distinct"] = exact_distinct[column["name"]]
                column["distinct_method"] = "exact_sqlite"
        duration_count = sum(self.duration_seconds.values())
        return {
            "records": {
                "logical_records": self.records,
                "valid_width_records": self.valid_width_records,
                "invalid_width_records": self.invalid_width_records,
                "record_width_histogram": dict(sorted(self.widths.items())),
                "approximate_distinct_full_rows": self.full_row_distinct.estimate(),
            },
            "columns": column_results,
            "entities": association_result,
            "domain_violations": {
                name: {
                    "count": sum(values.values()),
                    "values": safe_counter_items(values, self.top_k, "value"),
                }
                for name, values in self.domain_violations.items()
            },
            "quality_flags": {
                "rows_with_domain_violation": self.rows_with_domain_violation,
                "rows_with_possible_pii": self.rows_with_possible_pii,
            },
            "text": {
                "content_goal_relationship": dict(self.text_relations),
                "possible_pii_rows_by_field": {
                    name: dict(counters) for name, counters in self.pii.items()
                },
            },
            "dates": {
                "fields": {name: profile.result() for name, profile in self.dates.items()},
                "nonnegative_completion_duration_rows": duration_count,
                "negative_completion_duration_rows": self.duration_negative,
                "completion_duration_power_of_two_seconds_histogram": {
                    str(bucket): count for bucket, count in sorted(self.duration_seconds.items())
                },
            },
            "hierarchies": {
                "taxonomy": {
                    "depth_histogram": dict(self.taxonomy_depth),
                    "gap_rows": self.taxonomy_gap_rows,
                    "distinct_paths": len(self.taxonomy_paths),
                    "top_paths": safe_counter_items(
                        self.taxonomy_paths, self.top_k, "path"
                    ),
                },
                "geography": {
                    "depth_histogram": dict(self.geography_depth),
                    "gap_rows": self.geography_gap_rows,
                    "distinct_paths": len(self.geography_paths),
                    "top_paths": safe_counter_items(
                        self.geography_paths, self.top_k, "path"
                    ),
                },
            },
            "structured_fields": {
                "knowledge_quote": {
                    **dict(self.knowledge),
                    "distinct_knowledge_ids": len(self.knowledge_ids),
                    "knowledge_type_item_counts": {
                        safe_display_value(name): count
                        for name, count in self.knowledge_types.items()
                    },
                    "items_per_row_histogram": {
                        str(count): rows
                        for count, rows in sorted(self.knowledge_items_per_row.items())
                    },
                },
                "custom_form_data_str": dict(self.custom_json),
            },
        }


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        try:
            return next(csv.reader(source, delimiter="\t", strict=True))
        except StopIteration as exc:
            raise ValueError(f"empty TSV: {path}") from exc


def source_metadata(path: Path, *, include_hash: bool) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path) if include_hash else None,
        "sha256_skipped": not include_hash,
        "columns": len(read_header(path)),
        "header": read_header(path),
    }


def compare_sample_prefix(sample_path: Path, source_path: Path) -> dict[str, Any]:
    with (
        sample_path.open("r", encoding="utf-8-sig", newline="") as sample_file,
        source_path.open("r", encoding="utf-8-sig", newline="") as source_file,
    ):
        sample_reader = csv.reader(sample_file, delimiter="\t", strict=True)
        source_reader = csv.reader(source_file, delimiter="\t", strict=True)
        sample_header = next(sample_reader)
        source_header = next(source_reader)
        rows = 0
        mismatches = 0
        for sample_row in sample_reader:
            rows += 1
            source_row = next(source_reader, None)
            if source_row != sample_row:
                mismatches += 1
        return {
            "path": str(sample_path.resolve()),
            "size_bytes": sample_path.stat().st_size,
            "logical_records": rows,
            "header_matches": sample_header == source_header,
            "prefix_row_mismatches": mismatches,
            "is_exact_prefix": sample_header == source_header and mismatches == 0,
        }


def profile_sources(
    input_path: Path,
    raw_path: Path | None,
    *,
    top_k: int,
    database_path: Path,
    progress_every: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    association_store = AssociationStore(database_path)
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as sanitized_file:
            sanitized_reader = csv.reader(sanitized_file, delimiter="\t", strict=True)
            sanitized_header = next(sanitized_reader)
            profiler = DatasetProfiler(sanitized_header, top_k, association_store)
            raw_comparison: dict[str, Any] | None = None
            if raw_path is None:
                for values in sanitized_reader:
                    profiler.mark_width(len(values))
                    if len(values) == len(sanitized_header):
                        profiler.update(values)
                    if profiler.records % progress_every == 0:
                        print(f"profiled_records={profiler.records:,}", flush=True)
                physical_lines = sanitized_reader.line_num
            else:
                with raw_path.open("r", encoding="utf-8-sig", newline="") as raw_file:
                    # The historical export contains valid keys in rows with fewer
                    # trailing columns and quote sequences rejected by strict CSV.
                    # Keep the sanitized input strict; audit raw widths separately.
                    raw_reader = csv.reader(raw_file, delimiter="\t", strict=False)
                    raw_header = next(raw_reader)
                    shared = [name for name in sanitized_header if name in set(raw_header)]
                    raw_index = {name: raw_header.index(name) for name in shared}
                    sanitized_index = {name: sanitized_header.index(name) for name in shared}
                    mismatch_counts = Counter()
                    unavailable_counts = Counter()
                    key_mismatches = Counter()
                    raw_records = 0
                    paired_records = 0
                    missing_raw_records = 0
                    missing_sanitized_records = 0
                    invalid_raw_width_records = 0
                    for sanitized_values, raw_values in zip_longest(sanitized_reader, raw_reader):
                        if sanitized_values is None:
                            missing_sanitized_records += 1
                            raw_records += 1
                            continue
                        profiler.mark_width(len(sanitized_values))
                        if raw_values is None:
                            missing_raw_records += 1
                            if len(sanitized_values) == len(sanitized_header):
                                profiler.update(sanitized_values)
                            continue
                        raw_records += 1
                        paired_records += 1
                        if len(raw_values) != len(raw_header):
                            invalid_raw_width_records += 1
                        if len(sanitized_values) != len(sanitized_header):
                            continue
                        profiler.update(sanitized_values)
                        if profiler.records % progress_every == 0:
                            print(f"profiled_records={profiler.records:,}", flush=True)
                        for name in shared:
                            if raw_index[name] >= len(raw_values):
                                unavailable_counts[name] += 1
                            elif (
                                sanitized_values[sanitized_index[name]]
                                != raw_values[raw_index[name]]
                            ):
                                mismatch_counts[name] += 1
                        for name in ("id", "order_id"):
                            if name in raw_index and raw_index[name] < len(raw_values) and (
                                sanitized_values[sanitized_index[name]]
                                != raw_values[raw_index[name]]
                            ):
                                key_mismatches[name] += 1
                    physical_lines = sanitized_reader.line_num
                    raw_comparison = {
                        "raw_columns": len(raw_header),
                        "sanitized_columns": len(sanitized_header),
                        "shared_columns": len(shared),
                        "removed_columns": [
                            name for name in raw_header if name not in set(sanitized_header)
                        ],
                        "added_columns": [
                            name for name in sanitized_header if name not in set(raw_header)
                        ],
                        "raw_logical_records": raw_records,
                        "paired_records": paired_records,
                        "missing_raw_records": missing_raw_records,
                        "missing_sanitized_records": missing_sanitized_records,
                        "invalid_raw_width_records": invalid_raw_width_records,
                        "raw_physical_lines": raw_reader.line_num,
                        "raw_parser": "python_csv_tsv_permissive",
                        "positional_key_mismatches": dict(key_mismatches),
                        "shared_field_mismatch_counts": dict(sorted(mismatch_counts.items())),
                        "raw_field_unavailable_counts": dict(sorted(unavailable_counts.items())),
                        "raw_values_emitted": False,
                    }
            return profiler.result(), raw_comparison, physical_lines
    finally:
        association_store.close()


def build_field_dictionary(columns: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    unmapped_fields: list[str] = []
    for column in columns:
        name = column["name"]
        semantic = FIELD_SEMANTICS.get(name)
        if semantic is None:
            unmapped_fields.append(name)
            semantic = {
                "business_name": "未定义字段",
                "definition": "当前画像版本没有该字段的业务定义。",
                "semantic_status": "unknown",
                "lifecycle_stage": "unknown",
                "serving_availability": "unknown",
                "rag_role": "excluded_until_confirmed",
                "requires_confirmation": True,
                "caution": "必须补充源系统业务口径后才能使用。",
            }
        entries.append(
            {
                "name": name,
                **semantic,
                "requires_confirmation": name in BUSINESS_CONFIRMATION_QUESTIONS,
                "confirmation_question": BUSINESS_CONFIRMATION_QUESTIONS.get(name),
                "observed": {
                    "filled": column["filled"],
                    "missing_rate": column["missing_rate"],
                    "distinct": column["distinct"],
                    "distinct_method": column["distinct_method"],
                },
            }
        )
    return {
        "scope": "sanitized_export",
        "status_labels": SEMANTIC_STATUS_LABELS,
        "official_source_dictionary_present_in_repository": False,
        "entries": entries,
        "unmapped_fields": unmapped_fields,
        "fields_requiring_business_confirmation": [
            entry["name"] for entry in entries if entry["requires_confirmation"]
        ],
    }


def write_columns_csv(columns: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "name",
        "total",
        "filled",
        "missing",
        "missing_rate",
        "blank",
        "null_literal",
        "distinct",
        "distinct_method",
        "length_min",
        "length_p50",
        "length_p95",
        "length_p99",
        "length_max",
        "trimmed_rows",
        "multiline_rows",
        "control_character_rows",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for column in columns:
            writer.writerow(
                {
                    **{name: column.get(name) for name in fields},
                    **{f"length_{name}": value for name, value in column["length"].items()},
                }
            )


def write_field_dictionary_csv(dictionary: dict[str, Any], path: Path) -> None:
    fields = [
        "name",
        "business_name",
        "definition",
        "semantic_status",
        "lifecycle_stage",
        "serving_availability",
        "rag_role",
        "requires_confirmation",
        "confirmation_question",
        "caution",
        "filled",
        "missing_rate",
        "distinct",
        "distinct_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for entry in dictionary["entries"]:
            writer.writerow(
                {
                    **{name: entry.get(name) for name in fields},
                    **entry["observed"],
                }
            )


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_dictionary_markdown(profile: dict[str, Any]) -> str:
    dictionary = profile["semantic_layer"]
    lines = []
    for entry in dictionary["entries"]:
        lines.append(
            "| `{name}` | {business_name} | {definition} | {status} | {stage} | "
            "{availability} | `{role}` | {confirmation} | {caution} |".format(
                name=markdown_cell(entry["name"]),
                business_name=markdown_cell(entry["business_name"]),
                definition=markdown_cell(entry["definition"]),
                status=markdown_cell(
                    SEMANTIC_STATUS_LABELS[entry["semantic_status"]]
                ),
                stage=markdown_cell(entry["lifecycle_stage"]),
                availability=markdown_cell(entry["serving_availability"]),
                role=markdown_cell(entry["rag_role"]),
                confirmation="是" if entry["requires_confirmation"] else "否",
                caution=markdown_cell(entry["caution"]),
            )
        )
    confirmations = "\n".join(
        f"- `{entry['name']}`：{entry['confirmation_question']}"
        for entry in dictionary["entries"]
        if entry["requires_confirmation"]
    )
    return f"""# 字段业务语义字典

画像协议：`{profile['schema_version']}`

本字典描述当前脱敏导出能够支持的解释，不冒充源系统官方数据字典。语义状态分为“数据证据确认”、
“字段名和值域推断”和“业务口径待确认”。后两类字段在获得业务确认前不能升级为生产特征。

| 字段 | 业务名称 | 当前定义 | 证据状态 | 生命周期 | 在线可用性 | RAG角色 | 待确认 | 风险与限制 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(lines)}

## 待业务确认

{confirmations}
"""


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown(profile: dict[str, Any]) -> str:
    dataset = profile["dataset"]
    analysis = profile["analysis"]
    records = analysis["records"]
    entities = analysis["entities"]
    knowledge = analysis["structured_fields"]["knowledge_quote"]
    pii = analysis["text"]["possible_pii_rows_by_field"]
    domain_total = sum(item["count"] for item in analysis["domain_violations"].values())
    pii_total = sum(sum(types.values()) for types in pii.values())
    quality_flags = analysis.get("quality_flags", {})
    columns_by_name = {column["name"]: column for column in analysis["columns"]}
    content_column = columns_by_name.get("case_content", {})
    goal_column = columns_by_name.get("case_goal", {})
    content_counts = (
        f"{content_column.get('filled', 0):,} / {content_column.get('missing', 0):,}"
    )
    goal_counts = f"{goal_column.get('filled', 0):,} / {goal_column.get('missing', 0):,}"
    call_time = analysis["dates"]["fields"].get("call_time", {})
    call_time_missing = records["logical_records"] - call_time.get("valid", 0)
    call_time_counts = f"{call_time.get('valid', 0):,} / {call_time_missing:,}"
    taxonomy = analysis["hierarchies"]["taxonomy"]
    geography = analysis["hierarchies"]["geography"]
    dictionary = profile["semantic_layer"]
    semantic_by_name = {
        entry["name"]: entry for entry in dictionary["entries"]
    }
    semantic_counts = Counter(
        entry["semantic_status"] for entry in dictionary["entries"]
    )
    column_lines = []
    for column in analysis["columns"]:
        semantic = semantic_by_name[column["name"]]
        column_lines.append(
            f"| `{column['name']}` | {markdown_cell(semantic['business_name'])} | "
            f"{markdown_cell(semantic['definition'])} | "
            f"{SEMANTIC_STATUS_LABELS[semantic['semantic_status']]} | "
            f"`{semantic['rag_role']}` | {percent(column['missing_rate'])} | "
            f"{column['distinct']:,} ({column['distinct_method']}) |"
        )
    raw_comparison = profile.get("raw_to_sanitized_comparison")
    comparison_text = "未执行原始表对照。"
    if raw_comparison:
        comparison_text = (
            f"原始表 {raw_comparison['raw_columns']} 列，脱敏表 {raw_comparison['sanitized_columns']} 列，"
            f"保留 {raw_comparison['shared_columns']} 个同名字段，删除 "
            f"{len(raw_comparison['removed_columns'])} 个字段；成对记录 "
            f"{raw_comparison['paired_records']:,} 条，记录ID/关联键位置不一致计数为 "
            f"{sum(raw_comparison['positional_key_mismatches'].values()):,}，原始表非标准列宽记录 "
            f"{raw_comparison['invalid_raw_width_records']:,} 条。"
        )
    variations = entities["order_id_group_variations"]
    confirmation_lines = "\n".join(
        f"- `{entry['name']}`：{entry['confirmation_question']}"
        for entry in dictionary["entries"]
        if entry["requires_confirmation"]
    )
    return f"""# 数据画像报告

生成时间：`{profile['generated_at']}`
画像协议：`{profile['schema_version']}`

## 1. 数据集身份

- 脱敏数据：`{dataset['sanitized']['path']}`
- 文件大小：{dataset['sanitized']['size_bytes']:,} bytes
- SHA256：`{dataset['sanitized']['sha256']}`
- 逻辑记录：{records['logical_records']:,}
- 物理行：{dataset['sanitized']['physical_lines']:,}
- 字段数：{dataset['sanitized']['columns']}
- 列宽异常记录：{records['invalid_width_records']:,}

物理行数不能作为记录数：正文中的合法换行会增加物理行，而 TSV 解析器识别的是逻辑记录。

## 2. 原始表与脱敏表关系

{comparison_text}

本报告不输出原始表字段值、工单正文、地址、ID、部门名、知识标题或 PII 命中内容。

## 3. 记录与关联组

- 唯一记录 ID：{entities['distinct_record_ids']:,}
- 重复记录 ID：{entities['duplicate_record_ids']:,}
- 不同 `order_id`：{entities['order_id_groups']:,}
- 多行 `order_id` 组：{entities['multirow_order_id_groups']:,}
- 位于多行组中的记录：{entities['rows_in_multirow_order_id_groups']:,}
- 正文/诉求变化组：{variations['content_variation']:,} / {variations['goal_variation']:,}
- 分类/地域变化组：{variations['taxonomy_variation']:,} / {variations['geography_variation']:,}
- 知识引用变化组：{variations['knowledge_variation']:,}
- 受理时间/完成时间变化组：{variations['call_time_variation']:,} / {variations['completion_time_variation']:,}
- 任一流程字段变化组：{variations['workflow_variation']:,}

组内变化详见 `profile.json` 的 `analysis.entities.order_id_group_variations`。`order_id` 只被定义为
业务关联键：它既不证明组内记录是不同工单，也不证明它们必须合并。每个 `id` 原样保留；后续
建模样本需要根据可用输入版本另行构造，同时让同一关联组留在同一数据切分中。

## 4. 业务语义覆盖

- 数据证据可确认字段：{semantic_counts['data_confirmed']}
- 依据字段名和值域推断字段：{semantic_counts['inferred']}
- 当前业务含义不明确字段：{semantic_counts['unknown']}
- 仍需至少确认一个业务口径的字段：{len(dictionary['fields_requiring_business_confirmation'])}
- 未映射字段：{len(dictionary['unmapped_fields'])}

当前仓库没有源系统官方字段说明，因此报告严格区分已观察事实与推断。完整定义、生命周期、
在线可用性、RAG角色和风险见 `data_dictionary.md` 与 `field_dictionary.csv`。

## 5. 文本与隐私风险

- 启发式 PII 命中记录：{quality_flags.get('rows_with_possible_pii', 0):,}（字段命中 {pii_total:,}）
- 受约束字段域异常记录：{quality_flags.get('rows_with_domain_violation', 0):,}（字段命中 {domain_total:,}）
- `case_content` 有效/缺失：{content_counts}
- `case_goal` 有效/缺失：{goal_counts}
- 正文/诉求关系：`{json.dumps(analysis['text']['content_goal_relationship'], ensure_ascii=False)}`

PII 检测是启发式正则筛查，既可能误报，也远非完整覆盖。命中记录在进入训练、
embedding 或外部服务前必须隔离复核。

## 6. 知识引用

- 有有效知识项的记录：{knowledge.get('rows_with_items', 0):,}
- 引用项总数：{knowledge.get('items', 0):,}
- 唯一知识 ID：{knowledge.get('distinct_knowledge_ids', 0):,}
- 非法 JSON：{knowledge.get('invalid_json', 0):,}
- JSON null：{knowledge.get('json_null', 0):,}
- 行内重复引用：{knowledge.get('duplicate_items_within_row', 0):,}

空引用表示“未观察到引用”，不是知识负例。后续召回评估只能把已引用知识作为 observed positive。

## 7. 时间与层级覆盖

- `call_time` 有效/缺失：{call_time_counts}
- `call_time` 范围：`{call_time.get('min')}` 至 `{call_time.get('max')}`
- 负处理时长：{analysis['dates']['negative_completion_duration_rows']:,}
- 分类路径：{taxonomy['distinct_paths']:,}，层级缺口记录：{taxonomy['gap_rows']:,}
- 地域路径：{geography['distinct_paths']:,}，层级缺口记录：{geography['gap_rows']:,}

月度量存在明显不连续区间，时间切分前必须区分真实业务波动、历史回灌和批次缺失，不能只按
随机比例划分。

## 8. 字段语义与质量

| 字段 | 业务名称 | 当前定义 | 证据状态 | RAG角色 | 缺失率 | 基数 |
|---|---|---|---|---|---:|---:|
{chr(10).join(column_lines)}

完整安全枚举、长度、时间月分布、分类/地域路径、异常域值计数见 `profile.json`；纯统计表见
`columns.csv`，业务数据字典见 `data_dictionary.md` 和 `field_dictionary.csv`。

## 9. 待业务确认

以下问题不能仅靠字段名或统计分布得出结论：

{confirmation_lines}

## 10. 数据处理阶段建议

1. **冻结数据契约**：以文件 SHA256、45 列顺序和逻辑记录数作为输入门禁，拒绝静默换表。
2. **统一缺失值**：把空串及大小写不同的 `NULL/null/None/NaN/N/A` 统一为真正缺失值，保留原始缺失类型审计列。
3. **先隔离再清洗**：列宽异常、字段域异常、无效时间、负处理时长和 PII 命中进入 quarantine，不自动猜测修复。
4. **保留事件与关联**：每个 `id` 保留为原始事件；`order_id` 只建关联组和变化标记，不做无条件聚合或删除。
5. **保留双文本视图**：`case_content` 与 `case_goal` 分开清洗，同时构造 joint 视图；不要用生成摘要覆盖原文。
6. **分类分层处理**：保留原始层级和规范化层级；层级缺口与罕见路径不直接回填。
7. **知识引用结构化**：解析为 `type:value` 稳定 ID，标签只作展示文本；空引用保持 unknown，不造负样本。
8. **时间切分防泄漏**：确认 `call_time` 的跨渠道口径后再做 train/dev/test；相同关联组或相同文本指纹不能跨集合。
9. **处理版本化**：清洗结果写入新目录，附输入哈希、规则版本、记录计数和拒绝原因；绝不覆盖 `data/raw/`。
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/t_order_master.sanitized.v1_9.tsv"),
        help="Sanitized TSV used for all value-level profiling.",
    )
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=Path("data/raw/t_order_master.tsv"),
        help="Original TSV used only for schema/key/projection comparison.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("data/raw/t_order_master_100.sanitized.v1_9.tsv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data_analysis/output"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--skip-raw-comparison", action="store_true")
    parser.add_argument("--skip-raw-hash", action="store_true")
    return parser


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    raw_path = None if args.skip_raw_comparison else args.raw_source.resolve()
    sample_path = args.sample.resolve() if args.sample else None
    for path in (input_path, raw_path, sample_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than zero")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="civic-profile-") as temporary:
        analysis, comparison, physical_lines = profile_sources(
            input_path,
            raw_path,
            top_k=args.top_k,
            database_path=Path(temporary) / "entities.sqlite3",
            progress_every=args.progress_every,
        )
    sanitized_metadata = source_metadata(input_path, include_hash=True)
    sanitized_metadata["physical_lines"] = physical_lines
    semantic_layer = build_field_dictionary(analysis["columns"])
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "privacy": {
            "raw_values_emitted": False,
            "sensitive_top_values_suppressed": sorted(SENSITIVE_FIELDS),
            "pii_like_or_long_numeric_categorical_values_redacted": True,
            "pii_detection_is_heuristic": True,
        },
        "dataset": {
            "sanitized": sanitized_metadata,
            "raw_source": source_metadata(
                args.raw_source.resolve(), include_hash=not args.skip_raw_hash
            )
            if raw_path is not None
            else None,
            "sample": compare_sample_prefix(sample_path, input_path) if sample_path else None,
        },
        "raw_to_sanitized_comparison": comparison,
        "semantic_layer": semantic_layer,
        "analysis": analysis,
    }
    (output / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_columns_csv(analysis["columns"], output / "columns.csv")
    write_field_dictionary_csv(semantic_layer, output / "field_dictionary.csv")
    (output / "data_dictionary.md").write_text(
        build_dictionary_markdown(profile), encoding="utf-8"
    )
    (output / "profile.md").write_text(build_markdown(profile), encoding="utf-8")
    return profile


def main() -> int:
    args = build_parser().parse_args()
    profile = run_analysis(args)
    print(
        json.dumps(
            {
                "records": profile["analysis"]["records"]["logical_records"],
                "order_id_groups": profile["analysis"]["entities"]["order_id_groups"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
