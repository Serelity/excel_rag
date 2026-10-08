"""Import a version-matched human review and apply explicit, evidence-linked revisions.

The review export is preserved verbatim. Natural-language feedback never becomes
a retrieval query automatically: a separate, hash-bound application spec is required.
This creates development material, not extraction gold or case-relevance labels.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def dump_lines(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)


def md(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def anchor(raw: str, change: dict, evidence_id: str, fields: dict) -> dict:
    field, quote = change["field"], change["quote"]
    if field not in fields or not quote:
        raise ValueError("Unknown field or empty quotation")
    occurrences = [m.start() for m in re.finditer(re.escape(quote), raw)]
    if not occurrences:
        raise ValueError(f"Revised quote is absent from source: {evidence_id}")
    start = occurrences[0]
    return {
        "evidence_id": evidence_id, "field": field, "quote": quote,
        "start": start, "end": start + len(quote), "all_occurrences": occurrences,
        "occurrence_selection": "unique" if len(occurrences) == 1 else "first_occurrence_requires_review",
        "role_draft": change["role_draft"], "purpose_draft": fields[field]["use"],
        "allowed_transform_draft": fields[field]["transform"], "generic_boundary": fields[field]["boundary"],
        "semantic_review": "assistant_revision_based_on_sample_level_human_feedback",
    }


def apply_review(study: Path, review_path: Path, spec_path: Path, output: Path) -> dict:
    manifest = load(study / "manifest.json")
    review_bytes = review_path.read_bytes()
    review, spec = load(review_path), load(spec_path)
    if review.get("schema_version") != "field-boundary-human-review-v1":
        raise ValueError("Unexpected review schema")
    if spec.get("schema_version") != "field-boundary-review-application-v1":
        raise ValueError("Unexpected revision spec schema")
    if spec["review_export_sha256"] != sha(review_bytes):
        raise ValueError("Revision spec refers to another review export")
    for key in ("study_id", "source_sha256", "draft_spec_sha256"):
        if review[key] != manifest[key]:
            raise ValueError(f"Review provenance mismatch: {key}")
    if review.get("formal_gold") is not False or not review.get("reviewer", "").strip():
        raise ValueError("Reviewer is required and review must remain nonformal")
    for name, expected in manifest["artifacts"].items():
        if sha((study / name).read_bytes()) != expected:
            raise ValueError(f"Original study artifact changed: {name}")
    source_manifest_sha = sha((study / "manifest.json").read_bytes())
    originals = load_lines(study / "analysis-draft.jsonl")
    ids = {r["sample_id"] for r in originals}
    reviews = review["reviews"]
    if len(reviews) != len(ids) or len({r["sample_id"] for r in reviews}) != len(ids):
        raise ValueError("Review sample IDs are missing or duplicated")
    if {r["sample_id"] for r in reviews} != ids:
        raise ValueError("Review sample set differs from the original study")
    review_by_id = {r["sample_id"]: r for r in reviews}
    for r in reviews:
        if r["status"] not in {"accept", "revise", "defer"}:
            raise ValueError("The import requires all samples to have a completed review action")
        if not isinstance(r.get("edited_query"), str) or not isinstance(r.get("comment"), str):
            raise ValueError("Review text must be a string")
        reviewed_at = datetime.fromisoformat(r["reviewed_at"].replace("Z", "+00:00"))
        if reviewed_at.tzinfo is None:
            raise ValueError("Review timestamps must include a timezone")
    revisions = {r["sample_id"]: r for r in spec["revisions"]}
    if len(revisions) != len(spec["revisions"]):
        raise ValueError("Duplicated revision IDs")
    needs_action = {r["sample_id"] for r in reviews if r["status"] != "accept" or r["edited_query"].strip() or r["comment"].strip()}
    if set(revisions) != needs_action:
        raise ValueError("Every review with feedback or a non-accept action needs an application entry")
    if any(r["status"] == "defer" for r in reviews):
        raise ValueError("Deferred rows require a separate unresolved-review workflow")
    fields = load(study / "field-dictionary-draft.json")
    records, log, representations = [], [], []
    for original in originals:
        sid = original["sample_id"]
        result = copy.deepcopy(original)
        human = review_by_id[sid]
        change = revisions.get(sid)
        before_query, changes = original["query_draft"], []
        if change:
            if "query" in change:
                if not change["query"].strip():
                    raise ValueError(f"Empty revised query: {sid}")
                result["query_draft"] = change["query"]
            positions = {e["evidence_id"]: i for i, e in enumerate(result["evidence"])}
            for replacement in change.get("replace_evidence", []):
                eid = replacement["evidence_id"]
                if eid not in positions:
                    raise ValueError(f"Unknown original evidence ID: {eid}")
                i = positions[eid]
                updated = anchor(result["case_content"], replacement, eid, fields)
                changes.append({"operation": "replace", "before": result["evidence"][i], "after": updated})
                result["evidence"][i] = updated
            for addition in change.get("add_evidence", []):
                number = max(int(e["evidence_id"].rsplit("E", 1)[1]) for e in result["evidence"]) + 1
                updated = anchor(result["case_content"], addition, f"{sid}-E{number:02d}", fields)
                changes.append({"operation": "add", "before": None, "after": updated})
                result["evidence"].append(updated)
            if "boundary_note" in change:
                result["boundary_note_draft"] = change["boundary_note"]
            log.append({
                "sample_id": sid, "review_status_original": human["status"],
                "review_feedback_verbatim": human["edited_query"], "review_comment_verbatim": human["comment"],
                "revision_kind": change["kind"], "application_explanation": change["interpretation"],
                "query_before": before_query, "query_after": result["query_draft"],
                "evidence_changes": changes,
                "requires_semantic_reconfirmation": change.get("requires_semantic_reconfirmation", False),
            })
        result["original_query_draft"] = before_query
        result["original_boundary_note_draft"] = original["boundary_note_draft"]
        result["analysis_kind"] = "human_sample_review_with_traced_assistant_revisions"
        result["human_review"] = {**human, "reviewer": review["reviewer"]}
        result["review_application"] = {
            "status": "assistant_interpretation_to_reconfirm" if change and change.get("requires_semantic_reconfirmation") else "human_feedback_applied" if change else "accepted_without_changes",
            "scope": "sample_level_review_not_independent_span_labels",
            "review_export_sha256": sha(review_bytes),
            "explanation": change["interpretation"] if change else "保留人工同意且未提出修改的原草案。",
        }
        result["representations"]["B_brief_draft"] = result["query_draft"]
        result["representations"]["C_raw_plus_brief_draft"] = result["case_content"] + "\n问题表达：" + result["query_draft"]
        result["representations"]["status"] = "review_imported_development_only_not_retrieved"
        records.append(result)
        representations.append({
            "sample_id": sid, "query_scope": result["query_scope_draft"],
            "A_raw": result["representations"]["A_raw"],
            "B_review_applied": result["query_draft"],
            "C_raw_plus_review_applied": result["representations"]["C_raw_plus_brief_draft"],
            "application_status": result["review_application"]["status"],
            "usage": "development_only_not_formal_queries_or_relevance_gold",
        })
    evidence_count = sum(len(r["evidence"]) for r in records)
    validation = {
        "review_provenance_matched": True, "original_artifact_hashes_verified": len(manifest["artifacts"]),
        "sample_count": len(records), "review_ids_complete_and_unique": True,
        "source_order_preserved": [r["sample_id"] for r in records] == [r["sample_id"] for r in originals],
        "all_original_text_and_A_unchanged": all(r["case_content"] == o["case_content"] and r["representations"]["A_raw"] == o["representations"]["A_raw"] for r, o in zip(records, originals, strict=True)),
        "all_evidence_exact_and_offsets_valid": all(r["case_content"][e["start"]:e["end"]] == e["quote"] for r in records for e in r["evidence"]),
        "uncommented_accepted_samples_unchanged": all(r["query_draft"] == o["query_draft"] and r["evidence"] == o["evidence"] and r["boundary_note_draft"] == o["boundary_note_draft"] for r, o in zip(records, originals, strict=True) if r["sample_id"] not in revisions),
        "review_export_bytes_unchanged": review_path.read_bytes() == review_bytes,
        "evidence_count": evidence_count,
        "not_validated": ["independent span-level human labels", "relevance judgments", "retrieval gains", "B015 semantic reconfirmation"],
    }
    if any(validation[key] is not True for key in ("source_order_preserved", "all_original_text_and_A_unchanged", "all_evidence_exact_and_offsets_valid", "uncommented_accepted_samples_unchanged", "review_export_bytes_unchanged")):
        raise ValueError("Post-application validation failed")
    summary = {
        "study_id": manifest["study_id"], "review_count": len(reviews),
        "original_status_counts": dict(Counter(r["status"] for r in reviews)),
        "feedback_sample_count": len(revisions),
        "accepted_with_feedback_count": sum(r["status"] == "accept" and r["sample_id"] in revisions for r in reviews),
        "uncommented_accepted_count": len(records) - len(revisions),
        "revision_kind_counts": dict(Counter(r["kind"] for r in spec["revisions"])),
        "query_changed_count": sum(r["query_draft"] != r["original_query_draft"] for r in records),
        "evidence_count_before": sum(len(r["evidence"]) for r in originals),
        "evidence_count_after": evidence_count,
        "pending_semantic_reconfirmation": [r["sample_id"] for r in records if r["review_application"]["status"] == "assistant_interpretation_to_reconfirm"],
        "human_review_granularity": "sample_level",
        "formal_gold": False, "retrieval_run": False,
        "exported_at": review["exported_at"],
    }
    changes_tsv = io.StringIO(newline="")
    writer = csv.writer(changes_tsv, delimiter="\t", lineterminator="\n")
    writer.writerow(["sample_id", "review_status", "feedback_verbatim", "change_kind", "query_before", "query_after", "application", "requires_reconfirmation"])
    for row in log:
        writer.writerow([row["sample_id"], row["review_status_original"], row["review_feedback_verbatim"], row["revision_kind"], row["query_before"], row["query_after"], row["application_explanation"], row["requires_semantic_reconfirmation"]])
    field_tsv = io.StringIO(newline="")
    writer = csv.writer(field_tsv, delimiter="\t", lineterminator="\n")
    writer.writerow(["sample_id", "evidence_id", "field", "quote", "start", "end", "role", "use", "sample_review_status", "application_status"])
    for r in records:
        for e in r["evidence"]:
            writer.writerow([r["sample_id"], e["evidence_id"], fields[e["field"]]["label"], e["quote"], e["start"], e["end"], e["role_draft"], e["purpose_draft"], r["human_review"]["status"], r["review_application"]["status"]])
    report = ["# 人工复核导入与修改落实记录", "", "80条复核记录与原始草案匹配：70条标为同意、10条标为修改。3条同意项同时有修改文字，因此按13条意见处理；其余67条保持原草案。", "", "修改框内填写的是意见，未将意见本身直接当检索查询。根据原文落实10条地点、1条主体、1条咨询意图补充；B015的对象/现象重分类为助手解释，待再确认。", "", "保留原始复核状态与导出文件，不将修改项伪装成用户已同意修订后的新文字。样例级复核也不转换为逐片段独立人工标签。", "", "## 修改对照", ""]
    for r in log:
        report.extend([f"### {r['sample_id']}", "", f"复核状态：`{r['review_status_original']}`；修改意见：{md(r['review_feedback_verbatim'])}", "", f"原表达：{r['query_before']}", "", f"现表达：{r['query_after']}", "", f"落实说明：{r['application_explanation']}", ""])
    report.extend(["## 后续使用", "", "先核对B015对象/现象解释，再安排输入表示的开发比较；本轮没有运行检索或模型。80条仍是已暴露的开发材料，不能替代正式案例相关性留出集。", "", "原始导出见review-export.json；结构化记录见reviewed-analysis.jsonl；输入表示见reviewed-representations.jsonl；全部证据见复核后字段表.tsv。"])
    artifacts = {
        "review-export.json": review_bytes,
        "reviewed-analysis.jsonl": dump_lines(records).encode(),
        "reviewed-representations.jsonl": dump_lines(representations).encode(),
        "revision-log.jsonl": dump_lines(log).encode(),
        "summary.json": dump(summary).encode(), "validation.json": dump(validation).encode(),
        "修改对照.tsv": changes_tsv.getvalue().encode(), "复核后字段表.tsv": field_tsv.getvalue().encode(),
        "复核汇总.md": ("\n".join(report) + "\n").encode(),
    }
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    result_manifest = {
        "schema_version": "field-boundary-review-import-v1", "created_at": now,
        "study_id": manifest["study_id"], "status": "review_imported_with_one_interpretation_to_reconfirm",
        "study_manifest_sha256": source_manifest_sha, "review_export_sha256": sha(review_bytes),
        "revision_spec_sha256": sha(spec_path.read_bytes()), "importer_sha256": sha(Path(__file__).read_bytes()),
        "source_sha256": manifest["source_sha256"], "draft_spec_sha256": manifest["draft_spec_sha256"],
        "human_review_scope": "sample_level", "formal_gold": False,
        "artifacts": {name: sha(value) for name, value in artifacts.items()},
    }
    artifacts["manifest.json"] = dump(result_manifest).encode()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError("Refusing to overwrite review import artifacts")
    for name, value in artifacts.items():
        with (output / name).open("xb") as target:
            target.write(value)
    return {"summary": summary, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--revision-spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(dump(apply_review(args.study, args.review, args.revision_spec, args.output)))


if __name__ == "__main__":
    main()
