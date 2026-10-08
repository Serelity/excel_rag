"""Build private, evidence-linked development notes from assistant-authored drafts.

This prepares a qualitative research worksheet, not human gold or a formal run.
No model, retrieval, or network calls are made. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


TAGS = {
    "MC": "上下文或条件不足",
    "PO": "仅有工单流程意图",
    "CR": "办结与实质解决需区分",
    "MI": "多个主题或请求的拆分待裁决",
    "LP": "同一事项的过程与关联环节",
    "LR": "地点角色或空间关系",
    "RC": "转述、指控或未核实判断",
    "CF": "表述冲突或来源不一致",
    "NS": "否定、未发生及作用域",
    "TA": "时间表达与锚点",
    "NU": "数值、单位或近似限定",
    "BP": "转接、号码、工单及模板信息",
    "DQ": "原文格式或文本质量",
    "RB": "主体、对象与业务角色",
    "RF": "诉求、计划、风险与既成事实",
    "NC": "无足够实体问题内容",
}

ROUTES = {
    "substantive": "有实质主题",
    "substantive_with_followup": "实质主题与办理反馈并存",
    "procedure_only": "仅流程意图，原问题缺失",
    "insufficient_content": "内容不足以识别实质主题",
}

STRUCTURES = {
    "single": "暂作一个事项",
    "linked": "关联环节，先保留同一目标",
    "multiple_candidate": "多个问题候选，待裁决拆分",
    "no_underlying_issue": "没有可恢复的原事项",
}

# These are research dimensions, not a frozen model output schema.
FIELDS = {
    "problem": {"label": "问题或咨询主题", "use": "召回与排序", "transform": "保留逐字依据；概括另列，不能扩大事实", "boundary": "咨询主题不自动变成故障、侵权或事实认定"},
    "request": {"label": "请求或目标动作", "use": "召回、排序与展示", "transform": "可简短概括，但区分原文请求与分析性推断", "boundary": "希望、要求、咨询与已采取行动分开"},
    "actor": {"label": "主体及角色", "use": "角色核对与候选比较", "transform": "优先保留角色，不为每个人名建检索条件", "boundary": "被投诉方、办理方、联系方、记录方不互换"},
    "object": {"label": "问题对象", "use": "召回、排序与展示", "transform": "保留对象层级；不把部件问题扩大为整体问题", "boundary": "设备、产品、权益、证明与主体分开"},
    "place": {"label": "地点与关系", "use": "展示、地址查询或经审核的限制条件", "transform": "定位原文并绑定角色；不自动补行政层级", "boundary": "事发、居住、参保、就医、交易和线路端点不能混用"},
    "condition": {"label": "适用条件或数量", "use": "排序和条件核对", "transform": "身份、金额、单位、方向、近似词与对象绑定", "boundary": "缺失不补全；只有条件可靠且历史侧可比时才硬过滤"},
    "time": {"label": "时间表达", "use": "时点核对与展示", "transform": "先保留表达及所属动作；绝对日期解析另设任务", "boundary": "没有文本锚点时不按今天推算年份或日期"},
    "status": {"label": "状态或处置陈述", "use": "状态核对与展示", "transform": "保留说话者、对象与先后；不默认最新一句覆盖所有事项", "boundary": "工单办结、措施实施、当事人认可和问题解决分别对待"},
    "impact": {"label": "影响或风险", "use": "排序与展示", "transform": "保留原文因果表达与确定性，允许同句证据复用", "boundary": "风险、目的、险些发生与实际后果区分"},
    "uncertainty": {"label": "陈述来源或不确定性", "use": "事实核对与展示", "transform": "保留否认、怀疑、转述与说话者", "boundary": "声称和举报不升级为独立核实的事实"},
}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_key(value: str) -> str:
    return digest_bytes(re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).encode())


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def lines(value: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in value)


def md(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def build(source: Path, spec: Path, base_exclusions: Path, output: Path, template: Path) -> dict:
    source_bytes, spec_bytes = source.read_bytes(), spec.read_bytes()
    candidates, drafts = jsonl(source), jsonl(spec)
    if len(candidates) != 80 or len(drafts) != 80:
        raise ValueError("This study requires exactly 80 prepared candidates and 80 draft notes")
    if [r["candidate_position"] for r in candidates] != list(range(1, 81)):
        raise ValueError("Candidate positions must be 1 through 80 in source order")
    if [r["i"] for r in drafts] != list(range(1, 81)):
        raise ValueError("Draft notes must preserve source positions 1 through 80")
    if len({r["source_id"] for r in candidates}) != 80 or len({str(r["group_id"]) for r in candidates}) != 80:
        raise ValueError("Duplicate source or group")
    if any(r["source_type"] != "dataset_dev" or content_key(r["content"]) != r["text_hash"] for r in candidates):
        raise ValueError("Unexpected source type or source content fingerprint")
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    records, provenance, snapshot = [], [], []
    source_sha, spec_sha = digest_bytes(source_bytes), digest_bytes(spec_bytes)
    for source_row, note in zip(candidates, drafts, strict=True):
        sample_id, raw = f"B{note['i']:03d}", source_row["content"]
        if note["route"] not in ROUTES or note["structure"] not in STRUCTURES:
            raise ValueError(f"Unknown research dimension for {sample_id}")
        if not set(note["tags"]) <= TAGS.keys() or len(note["tags"]) != len(set(note["tags"])):
            raise ValueError(f"Invalid tags for {sample_id}")
        if not note["query"].strip() or not note["note"].strip() or not note["evidence"]:
            raise ValueError(f"Incomplete draft for {sample_id}")
        evidence = []
        for number, (field, quote, role) in enumerate(note["evidence"], 1):
            if field not in FIELDS or not quote:
                raise ValueError(f"Invalid field for {sample_id}")
            offsets = [m.start() for m in re.finditer(re.escape(quote), raw)]
            if not offsets:
                raise ValueError(f"Quote {number} does not match the source for {sample_id}")
            start = offsets[0]
            evidence.append({
                "evidence_id": f"{sample_id}-E{number:02d}", "field": field,
                "quote": quote, "start": start, "end": start + len(quote),
                "all_occurrences": offsets,
                "occurrence_selection": "unique" if len(offsets) == 1 else "first_occurrence_requires_review",
                "role_draft": role, "purpose_draft": FIELDS[field]["use"],
                "allowed_transform_draft": FIELDS[field]["transform"],
                "generic_boundary": FIELDS[field]["boundary"],
                "semantic_review": "pending",
            })
        structure = note["structure"]
        scope = "procedure" if note["route"] == "procedure_only" else "insufficient" if note["route"] == "insufficient_content" else "substantive"
        records.append({
            "sample_id": sample_id, "candidate_position": note["i"],
            "analysis_kind": "assistant_qualitative_development_draft",
            "case_content": raw, "case_content_sha256": digest_bytes(raw.encode()),
            "route_draft": note["route"], "issue_structure_draft": structure,
            "issue_linking_status": "hypothesis_only_not_formal_issue_graph",
            "boundary_tags_draft": note["tags"], "boundary_note_draft": note["note"],
            "query_draft": note["query"], "query_scope_draft": scope,
            "evidence": evidence,
            "representations": {
                "A_raw": raw, "B_brief_draft": note["query"],
                "C_raw_plus_brief_draft": raw + "\n问题表达：" + note["query"],
                "status": "not_retrieved_and_not_human_validated",
                "substantive_search_readiness": "needs_human_review" if scope == "substantive" else "not_ready_for_underlying_issue_search",
            },
            "human_review": {"status": "pending", "reviewer": "", "edited_query": "", "comment": ""},
        })
        provenance.append({
            "sample_id": sample_id,
            **{key: source_row[key] for key in ("source_id", "source_row", "group_id", "text_hash", "source_type", "candidate_position")},
            "input_candidates_sha256": source_sha,
            "exposure": "read_for_field_boundary_development",
        })
        snapshot.append({"sample_id": sample_id, "case_content": raw, "case_content_sha256": digest_bytes(raw.encode())})

    routes = dict(Counter(r["route_draft"] for r in records))
    structures = dict(Counter(r["issue_structure_draft"] for r in records))
    tags = dict(Counter(t for r in records for t in r["boundary_tags_draft"]))
    fields = dict(Counter(e["field"] for r in records for e in r["evidence"]))
    lengths = sorted(len(r["case_content"]) for r in records)
    summary = {
        "analysis_kind": "assistant_qualitative_development_draft", "sample_count": 80,
        "total_characters": sum(lengths), "median_characters": (lengths[39] + lengths[40]) / 2,
        "max_characters": max(lengths), "route_counts_draft": routes,
        "issue_structure_counts_draft": structures, "tag_counts_draft": tags,
        "field_evidence_counts": fields, "exact_quote_count": sum(fields.values()),
        "ambiguous_occurrence_count": sum(e["occurrence_selection"] != "unique" for r in records for e in r["evidence"]),
        "human_complete_count": 0,
        "important_limitations": [
            "Semantic labels, roles, summaries, and boundary flags are assistant drafts, not human gold.",
            "Exact string offsets validate quotation existence, not role or semantic correctness.",
            "All 80 records have been used for development; exclude their sources/groups from later formal sampling.",
            "No retrieval or model comparison was run; field utility and effect sizes remain unmeasured.",
            "These 80 dev representatives are not a population sample or a property-only subset.",
            "Tag counts are qualitative draft coverage, not model error rates; tags can overlap.",
        ],
    }
    exposure = {
        "schema_version": "field-boundary-development-exposure-v1",
        "study_id": output.name, "created_at": now, "source_sha256": source_sha,
        "reason": "All 80 candidate texts were read and used for assistant-authored field-boundary development.",
        "excluded_source_ids": [r["source_id"] for r in candidates],
        "excluded_group_ids": [r["group_id"] for r in candidates],
        "not_a_human_review": True,
    }
    base = json.loads(base_exclusions.read_text(encoding="utf-8-sig"))
    if base["schema_version"] != "phase1-source-exclusions-v1":
        raise ValueError("Unexpected base source exclusion schema")
    merged = dict(base)
    merged.update({"version": base["version"] + "+" + output.name, "history_complete": False,
                   "reviewed_by": "", "reviewed_at": "", "status": "awaiting_human_history_review",
                   "updated_at": now})
    merged["excluded_source_ids"] = sorted(set(base["excluded_source_ids"]) | set(exposure["excluded_source_ids"]))
    groups = {str(v): v for v in base["excluded_group_ids"] + exposure["excluded_group_ids"]}
    merged["excluded_group_ids"] = [groups[k] for k in sorted(groups)]
    merged["limitations"] = list(base["limitations"]) + [
        "本研究的80个来源及关联组已用于开发，新增排除是已发生的暴露记录；其余历史仍需人工核定。",
        "原sampling-001仅作为本次开发材料；正式抽样需使用本合并排除表重新prepare，不沿用旧清单。",
    ]

    out = io.StringIO(newline="")
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(["sample_id", "evidence_id", "field", "quote", "start", "end", "role_draft", "purpose_draft", "allowed_transform", "boundary_note", "human_status"])
    for r in records:
        for e in r["evidence"]:
            writer.writerow([r["sample_id"], e["evidence_id"], FIELDS[e["field"]]["label"], e["quote"], e["start"], e["end"], e["role_draft"], e["purpose_draft"], e["allowed_transform_draft"], r["boundary_note_draft"], "pending"])

    report = ["# 80 条字段用途与边界样例草案", "", "仅限本地研究。全部判断由助手初读形成，尚无人审；不作为 gold、正式查询资格或相关性标签。原文仅来自 case_content。", "",
              "证据位置使用原始 Python Unicode 字符索引 [start,end)，未整理空白；浏览器 UTF-16 索引不是本文件的坐标单位。", ""]
    for r in records:
        report.extend([f"## {r['sample_id']}", "", f"**原文**：{md(r['case_content'])}", "",
                       f"**问题表达草案**：{r['query_draft']}", "",
                       f"**初步判断**：{ROUTES[r['route_draft']]}；{STRUCTURES[r['issue_structure_draft']]}。", "",
                       f"**边界说明**：{r['boundary_note_draft']}", "",
                       "| 候选维度 | 原文证据 | 角色或归属草案 | 用途 |", "| --- | --- | --- | --- |"])
        for e in r["evidence"]:
            report.append(f"| {FIELDS[e['field']]['label']} | {md(e['quote'])} | {md(e['role_draft'])} | {e['purpose_draft']} |")
        report.extend(["", "人工复核：待填写。", ""])

    payload = {"study_id": output.name, "source_sha256": source_sha, "spec_sha256": spec_sha,
               "summary": summary, "tags": TAGS, "routes": ROUTES, "structures": STRUCTURES,
               "fields": FIELDS, "records": records}
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = template.read_text(encoding="utf-8")
    if html.count("__STUDY_JSON__") != 1:
        raise ValueError("Template must have exactly one data placeholder")

    readme = f"""# 字段用途与边界开发研究 001

本次覆盖全部80条候选，原文共{sum(lengths)}字符。所有语义判断为助手草案，人工复核完成0条。

- `样例审阅.html`：本地审阅页，可按边界筛选、查看逐字依据、改写问题表达并导出复核记录。没有外部资源或网络请求。
- `样例分析.md`：80条完整阅读版；`字段用途与边界样例表.tsv`：每个原文片段一行。
- `analysis-draft.jsonl`：结构化分析草案与A/B/C表示；B/C均未经人审，也未跑检索。
- `source-snapshot.jsonl`、`source-provenance.jsonl`：原文快照与私有来源映射；坐标基于未经改写的原文。
- `draft-spec.jsonl`：助手初读笔记；`analysis-summary.json`：覆盖统计，不能解读为模型错误率。
- `development-exposure.json`：80个来源/关联组的开发暴露记录。
- `source-exclusions.after-study.json`：与原正式排除表合并后的新草稿。原排除表未改；之后正式prepare应显式使用本文件，补齐历史审核后再冻结。不能沿用原80条候选。

浏览器保存只保留在当前浏览器存储中；完成后点击“导出复核记录”，保留下载文件。导出不是正式gold或case_eval标签格式，不会自动改写研究笔记或原候选review。

来源哈希：`{source_sha}`。草案哈希：`{spec_sha}`。

没有联网、运行模型、生成案例相关性评分或修改既有评测金标。本目录位于Git忽略的data下；包含真实业务文本，不放入公开研究报告。
"""
    artifacts = {
        "source-snapshot.jsonl": lines(snapshot), "source-provenance.jsonl": lines(provenance),
        "analysis-draft.jsonl": lines(records), "analysis-summary.json": dumps(summary),
        "field-dictionary-draft.json": dumps(FIELDS), "development-exposure.json": dumps(exposure),
        "source-exclusions.after-study.json": dumps(merged),
        "字段用途与边界样例表.tsv": out.getvalue(), "样例分析.md": "\n".join(report),
        "样例审阅.html": html.replace("__STUDY_JSON__", embedded), "README.md": readme,
    }
    validation = {
        "scope": "artifact_structure_and_evidence_alignment_only", "sample_count": 80,
        "all_source_content_hashes_match": True, "all_quotes_exact": True,
        "all_offsets_roundtrip": all(r["case_content"][e["start"]:e["end"]] == e["quote"] for r in records for e in r["evidence"]),
        "source_order_preserved": True, "all_raw_representations_unchanged": all(r["representations"]["A_raw"] == c["content"] for r, c in zip(records, candidates, strict=True)),
        "human_reviews_pending": 80,
        "exposure_sources_and_groups_covered": all(r["source_id"] in merged["excluded_source_ids"] and r["group_id"] in merged["excluded_group_ids"] for r in candidates),
        "source_candidates_unchanged": source.read_bytes() == source_bytes,
        "not_validated": ["semantic correctness", "issue membership", "retrieval utility", "human agreement"],
    }
    if not all(validation[key] for key in ("all_offsets_roundtrip", "all_raw_representations_unchanged", "exposure_sources_and_groups_covered", "source_candidates_unchanged")):
        raise ValueError("Artifact validation failed")
    artifacts["validation.json"] = dumps(validation)
    manifest = {
        "schema_version": "field-boundary-study-v1", "study_id": output.name,
        "created_at": now, "status": "assistant_draft_awaiting_human_review",
        "source_sha256": source_sha, "draft_spec_sha256": spec_sha,
        "base_exclusions_sha256": digest_bytes(base_exclusions.read_bytes()),
        "generator_sha256": digest_bytes(Path(__file__).read_bytes()),
        "template_sha256": digest_bytes(template.read_bytes()),
        "input_field": "case_content", "sample_count": 80, "formal_experiment": False,
        "artifacts": {name: digest_bytes(text.encode()) for name, text in artifacts.items()},
    }
    artifacts["manifest.json"] = dumps(manifest)
    output.mkdir(parents=True, exist_ok=True)
    conflicts = [name for name in artifacts if (output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite study outputs: " + ", ".join(conflicts))
    for name, text in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(text)
    return {"sample_count": 80, "evidence_count": summary["exact_quote_count"],
            "route_counts_draft": routes, "issue_structure_counts_draft": structures,
            "tag_counts_draft": tags, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--base-exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).parents[1] / "templates" / "field-boundary-review.html")
    args = parser.parse_args()
    print(dumps(build(args.source, args.spec, args.base_exclusions, args.output, args.template)))


if __name__ == "__main__":
    main()
