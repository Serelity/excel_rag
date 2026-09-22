from __future__ import annotations

import json

import pytest

from semantic_extraction.evaluation import (
    finalize_gold,
    prepare_adjudication,
    prepare_gold_worksheet,
    score,
)
from semantic_extraction.pipeline import content_sha256


def write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_rows() -> list[dict]:
    return [
        {
            "source_id": "case-1",
            "source_row": 7,
            "case_content": "路灯不亮，希望维修。此前垃圾已清运。",
        },
        {
            "source_id": "case-2",
            "source_row": 8,
            "case_content": "来电后自主挂机。",
        },
    ]


def prediction_rows() -> list[dict]:
    first_content = source_rows()[0]["case_content"]
    second_content = source_rows()[1]["case_content"]

    def event(event_type, trigger, polarity="occurred"):
        return {
            "normalized_event_type": event_type,
            "trigger": {"text": trigger, "start": first_content.index(trigger), "end": 0},
            "actors": [],
            "objects": [],
            "behaviors": [{"text": trigger, "start": 0, "end": 0}],
            "impacts": [],
            "requests": [],
            "locations": [],
            "time_expressions": [],
            "search_terms": [event_type],
            "polarity": polarity,
        }

    return [
        {
            "schema_version": "semantic-extraction-v4",
            "source_id": "case-1",
            "source_row": 7,
            "content_sha256": content_sha256(first_content),
            "result": {
                "events": [
                    event("路灯故障", "路灯不亮"),
                    event("垃圾清运", "垃圾已清运"),
                    event("未知事项", "此前"),
                ]
            },
            "provenance": {"prompt_version": "test-v1"},
        },
        {
            "schema_version": "semantic-extraction-v4",
            "source_id": "case-2",
            "source_row": 8,
            "content_sha256": content_sha256(second_content),
            "result": {"events": []},
            "provenance": {"prompt_version": "test-v1"},
        },
    ]


def complete_worksheet(path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rows[0]["review_status"] = "complete"
    rows[0]["annotator_id"] = "reviewer-a"
    rows[0]["gold"] = {
        "case_status": "active",
        "active_retrieval_issues": [
            {
                "issue_id": "street-light-repair",
                "label": "路灯故障维修",
                "expected_polarity": "occurred",
                "is_current_request": True,
                "required_knowledge_need": "市政照明设施维修处置",
                "evidence_quotes": ["路灯不亮", "希望维修"],
            },
            {
                "issue_id": "second-active-issue",
                "label": "第二个待处理问题",
                "expected_polarity": "possible",
                "is_current_request": False,
                "required_knowledge_need": "另一类处置知识",
                "evidence_quotes": ["希望维修"],
            },
        ],
        "background_issues": [
            {
                "issue_id": "old-rubbish",
                "label": "已完成垃圾清运",
                "expected_polarity": "occurred",
                "is_current_request": False,
                "required_knowledge_need": "垃圾清运规范",
                "evidence_quotes": ["垃圾已清运"],
            }
        ],
        "annotation_notes": "用于测试",
    }
    if len(rows) > 1:
        rows[1]["review_status"] = "complete"
        rows[1]["annotator_id"] = "reviewer-a"
        rows[1]["gold"] = {
            "case_status": "resolved",
            "active_retrieval_issues": [],
            "background_issues": [],
            "annotation_notes": "没有可检索问题",
        }
    write_jsonl(path, rows)


def complete_adjudication(path) -> None:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rows[0]["adjudication"].update(
        {
            "review_status": "complete",
            "matches": [
                {
                    "event_index": 0,
                    "gold_scope": "active",
                    "gold_issue_id": "street-light-repair",
                    "match_notes": "",
                },
                {
                    "event_index": 1,
                    "gold_scope": "background",
                    "gold_issue_id": "old-rubbish",
                    "match_notes": "",
                },
            ],
            "spurious_event_indices": [2],
        }
    )
    rows[1]["adjudication"].update(
        {
            "review_status": "complete",
            "matches": [],
            "spurious_event_indices": [],
        }
    )
    write_jsonl(path, rows)


def test_gold_workflow_strips_source_text_and_scores_manual_matches(tmp_path) -> None:
    source = tmp_path / "pilot.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    worksheet = tmp_path / "worksheet.jsonl"
    gold = tmp_path / "gold.jsonl"
    adjudication = tmp_path / "adjudication.jsonl"
    write_jsonl(source, source_rows())
    write_jsonl(predictions, prediction_rows())

    assert (
        prepare_gold_worksheet(
            input_path=source,
            output_path=worksheet,
            predictions_path=predictions,
            limit=20,
            overwrite=False,
        )
        == 2
    )
    complete_worksheet(worksheet)
    assert (
        finalize_gold(
            input_path=source,
            worksheet_path=worksheet,
            output_path=gold,
            overwrite=False,
        )
        == 2
    )
    assert "case_content" not in gold.read_text(encoding="utf-8")

    assert (
        prepare_adjudication(
            gold_path=gold,
            predictions_path=predictions,
            output_path=adjudication,
            overwrite=False,
        )
        == 2
    )
    complete_adjudication(adjudication)
    report = score(
        gold_path=gold,
        predictions_path=predictions,
        adjudication_path=adjudication,
    )

    assert report["rates"]["issue_precision"] == 1 / 3
    assert report["rates"]["issue_recall"] == 1 / 2
    assert report["rates"]["current_request_recall"] == 1.0
    assert report["rates"]["background_leakage_rate"] == 1 / 3
    assert report["rates"]["spurious_issue_rate"] == 1 / 3
    assert report["rates"]["active_polarity_accuracy"] == 1.0
    assert report["rates"]["active_evidence_quote_recall"] == 1 / 3
    assert report["rates"]["matched_issue_evidence_quote_recall"] == 1 / 2
    assert report["rates"]["exact_record_rate"] == 1 / 2
    assert report["rates"]["case_status_prediction_coverage"] == 0.0
    assert report["rates"]["case_status_accuracy"] is None


def test_finalize_rejects_non_verbatim_gold_evidence(tmp_path) -> None:
    source = tmp_path / "pilot.jsonl"
    worksheet = tmp_path / "worksheet.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(source, source_rows())
    prepare_gold_worksheet(
        input_path=source,
        output_path=worksheet,
        predictions_path=None,
        limit=1,
        overwrite=False,
    )
    complete_worksheet(worksheet)
    rows = [json.loads(line) for line in worksheet.open(encoding="utf-8") if line.strip()]
    rows[0]["gold"]["active_retrieval_issues"][0]["evidence_quotes"] = ["改写证据"]
    write_jsonl(worksheet, rows[:1])

    with pytest.raises(ValueError, match="not an exact source quote"):
        finalize_gold(
            input_path=source,
            worksheet_path=worksheet,
            output_path=gold,
            overwrite=False,
        )


def test_score_rejects_unaccounted_or_stale_prediction_events(tmp_path) -> None:
    source = tmp_path / "pilot.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    worksheet = tmp_path / "worksheet.jsonl"
    gold = tmp_path / "gold.jsonl"
    adjudication = tmp_path / "adjudication.jsonl"
    write_jsonl(source, source_rows())
    write_jsonl(predictions, prediction_rows())
    prepare_gold_worksheet(
        input_path=source,
        output_path=worksheet,
        predictions_path=predictions,
        limit=2,
        overwrite=False,
    )
    complete_worksheet(worksheet)
    finalize_gold(
        input_path=source,
        worksheet_path=worksheet,
        output_path=gold,
        overwrite=False,
    )
    prepare_adjudication(
        gold_path=gold,
        predictions_path=predictions,
        output_path=adjudication,
        overwrite=False,
    )
    complete_adjudication(adjudication)
    rows = [json.loads(line) for line in adjudication.open(encoding="utf-8") if line.strip()]
    rows[0]["adjudication"]["spurious_event_indices"] = []
    write_jsonl(adjudication, rows)

    with pytest.raises(ValueError, match="every predicted event"):
        score(
            gold_path=gold,
            predictions_path=predictions,
            adjudication_path=adjudication,
        )

    complete_adjudication(adjudication)
    changed = prediction_rows()
    changed[0]["result"]["events"][0]["normalized_event_type"] = "修改后的问题"
    write_jsonl(predictions, changed)
    with pytest.raises(ValueError, match="stale"):
        score(
            gold_path=gold,
            predictions_path=predictions,
            adjudication_path=adjudication,
        )
