"""Bind the confirmed 80-case review to v1 rule checks; do not manufacture model gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from semantic_extraction.case_contract.cli import json_text, write_new  # noqa: E402
from semantic_extraction.case_contract.schema import SPEC_VERSION  # noqa: E402

TAG_RULES = {
    "MC": ["R10"], "PO": ["R02"], "CR": ["R09"], "MI": ["R03", "R10"],
    "LP": ["R03", "R09"], "LR": ["R05"], "RC": ["R08"], "CF": ["R10"],
    "NS": ["R07", "R08"], "TA": ["R09"], "NU": ["R09"], "BP": ["R02", "R06"],
    "DQ": ["R01"], "RB": ["R04", "R06"], "RF": ["R07"], "NC": ["R02"],
}
FEEDBACK_RULES = {
    **dict.fromkeys(["B001", "B006", "B014", "B023", "B034", "B035", "B036", "B041",
                     "B049", "B080"], "R05"),
    "B032": "R06", "B040": "R07", "B015": "R04",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(folder, filename):
    manifest = json.loads((folder / filename).read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError("Manifest artifact must stay inside its source folder")
        if digest(target) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return manifest


def prepare(review_dir: Path, contract_dir: Path, output: Path) -> dict:
    manifest = verify_manifest(review_dir, "manifest.confirmed.json")
    verify_manifest(contract_dir, "manifest.json")
    if digest(review_dir / "manifest.json") != manifest["parent_import_manifest_sha256"]:
        raise ValueError("Parent review import manifest changed")
    if digest(review_dir / "review-export.json") != manifest["review_export_sha256"]:
        raise ValueError("Original human review export changed")
    rows = [json.loads(line) for line in (review_dir / "reviewed-analysis.confirmed.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    if len(rows) != 80 or {r["sample_id"] for r in rows} != {f"B{n:03}" for n in range(1, 81)}:
        raise ValueError("Expected all 80 distinct confirmed development samples")
    matrix, inputs = [], []
    evidence_count = 0
    for row in rows:
        raw, sample_id = row["case_content"], row["sample_id"]
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["case_content_sha256"]:
            raise ValueError(f"Raw text hash mismatch: {sample_id}")
        if row["human_review"]["status"] not in {"accept", "revise"}:
            raise ValueError("Unreviewed sample encountered")
        if not set(row["boundary_tags_draft"]) <= TAG_RULES.keys():
            raise ValueError("Unmapped boundary tag")
        for evidence in row["evidence"]:
            if raw[evidence["start"]:evidence["end"]] != evidence["quote"]:
                raise ValueError(f"Invalid existing evidence: {sample_id}")
            evidence_count += 1
        rules = {rule for tag in row["boundary_tags_draft"] for rule in TAG_RULES[tag]}
        if sample_id in FEEDBACK_RULES:
            rules.add(FEEDBACK_RULES[sample_id])
        matrix.append({
            "sample_id": sample_id, "case_content_sha256": row["case_content_sha256"],
            "focus_rules": sorted(rules), "all_rules_apply": True,
            "human_feedback_rule": FEEDBACK_RULES.get(sample_id),
            "reference_query": row["query_draft"], "reference_route": row["route_draft"],
            "reference_boundary_note": row["boundary_note_draft"],
            "reference_evidence": row["evidence"],
            "reference_scope": "sample_review_not_independent_field_gold",
            "model_output": None, "semantic_review_status": "not_run",
            "checks": {k: None for k in ["routing", "issue_grouping", "field_boundary",
                       "location_and_actor_retention", "intent_and_modality", "time_and_status",
                       "missing_and_conflicting_context"]},
        })
        # sample_id is transport metadata, never included in the model's user message.
        inputs.append({"sample_id": sample_id, "input": {"case_content": raw}})
    summary = {
        "spec_version": SPEC_VERSION, "sample_count": len(rows),
        "reference_evidence_count": evidence_count, "feedback_cases": len(FEEDBACK_RULES),
        "focus_rule_counts": dict(sorted(
            Counter(r for m in matrix for r in m["focus_rules"]).items()
        )),
        "status": "prepared_not_model_run", "formal_gold": False,
        "model_inference": "not_run", "semantic_evaluation": "not_run",
        "retrieval_evaluation": "not_run", "usage": "development_only",
    }
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "inputs.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs),
        "acceptance-matrix.jsonl": "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in matrix
        ),
        "summary.json": json_text(summary),
    }
    for name, content in artifacts.items():
        write_new(output / name, content)
    write_new(output / "manifest.json", json_text({
        "spec_version": SPEC_VERSION,
        "review_manifest_sha256": digest(review_dir / "manifest.confirmed.json"),
        "contract_manifest_sha256": digest(contract_dir / "manifest.json"),
        "preparation_script_sha256": digest(Path(__file__)),
        "guideline_sha256": digest(
            ROOT / "research/topics/case-content-extraction/12-extraction-spec-v1.md"
        ),
        "artifacts": {name: digest(output / name) for name in artifacts},
        "formal_gold": False, "model_run": False,
    }))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--contract-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json_text(prepare(args.review_dir, args.contract_dir, args.output)))


if __name__ == "__main__":
    main()
