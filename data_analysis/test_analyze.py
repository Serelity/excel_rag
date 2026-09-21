from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from data_analysis.analyze import (
    BUSINESS_CONFIRMATION_QUESTIONS,
    FIELD_SEMANTICS,
    SEMANTIC_STATUS_LABELS,
    run_analysis,
    safe_display_value,
    semantic_value,
)


HEADER = [
    "id",
    "order_id",
    "case_content",
    "case_goal",
    "case_is_visit",
    "call_time",
    "case_complete_time",
    "knowledge_quote",
    "case_accord_type_one_name",
    "case_accord_type_two_name",
    "case_accord_type_three_name",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "custom_form_data_str",
]


class AnalyzeTests(unittest.TestCase):
    def test_semantic_catalog_is_complete_and_questions_are_explicit(self) -> None:
        self.assertEqual(len(FIELD_SEMANTICS), 45)
        marked = {
            name
            for name, definition in FIELD_SEMANTICS.items()
            if definition["requires_confirmation"]
        }
        self.assertEqual(marked, set(BUSINESS_CONFIRMATION_QUESTIONS))
        self.assertTrue(
            all(
                definition["semantic_status"] in SEMANTIC_STATUS_LABELS
                for definition in FIELD_SEMANTICS.values()
            )
        )

    def test_semantic_nulls(self) -> None:
        for value in (None, "", "  ", "NULL", "null", "NaN", "n/a"):
            self.assertIsNone(semantic_value(value))
        self.assertEqual(semantic_value("  正常值  "), "正常值")

    def test_unsafe_categorical_values_are_redacted(self) -> None:
        self.assertEqual(safe_display_value("13800138000"), "<REDACTED_PII_LIKE>")
        self.assertEqual(safe_display_value("202501081136160073091015"), "<REDACTED_LONG_NUMBER>")
        self.assertEqual(safe_display_value("正常枚举"), "正常枚举")

    def test_profile_counts_association_variations_and_suppresses_text(self) -> None:
        rows = [
            [
                "1",
                "parent-a",
                "正文 13800138000",
                "诉求",
                "是",
                "2025-01-01 10:00:00",
                "2025-01-01 11:00:00",
                '[{"label":"标题","value":"k1","type":0}]',
                "一级",
                "二级",
                "三级",
                "城市",
                "城区",
                "街道",
                "NULL",
            ],
            [
                "2",
                "parent-a",
                "另一正文\t嵌入字段",
                "诉求",
                "6015",
                "2025-01-02 10:00:00",
                "2025-01-02 09:00:00",
                "null",
                "一级",
                "二级",
                "三级",
                "城市",
                "城区",
                "街道",
                "{}",
            ],
            [
                "3",
                "parent-b",
                "NULL",
                "NULL",
                "否",
                "bad-date",
                "NULL",
                "[]",
                "一级",
                "NULL",
                "三级",
                "城市",
                "NULL",
                "街道",
                "{bad",
            ],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sanitized.tsv"
            sample = root / "sample.tsv"
            for path, selected_rows in ((source, rows), (sample, rows[:2])):
                with path.open("w", encoding="utf-8", newline="") as output:
                    writer = csv.writer(output, delimiter="\t")
                    writer.writerow(HEADER)
                    writer.writerows(selected_rows)
            args = argparse.Namespace(
                input=source,
                raw_source=source,
                sample=sample,
                output=root / "output",
                top_k=5,
                progress_every=100,
                skip_raw_comparison=True,
                skip_raw_hash=True,
            )
            profile = run_analysis(args)
            analysis = profile["analysis"]
            self.assertEqual(analysis["records"]["logical_records"], 3)
            self.assertEqual(analysis["entities"]["order_id_groups"], 2)
            self.assertEqual(
                analysis["entities"]["order_id_group_variations"][
                    "content_variation"
                ],
                1,
            )
            self.assertEqual(analysis["domain_violations"]["case_is_visit"]["count"], 1)
            self.assertEqual(analysis["quality_flags"]["rows_with_domain_violation"], 1)
            self.assertEqual(analysis["quality_flags"]["rows_with_possible_pii"], 1)
            self.assertEqual(
                analysis["quality_flags"]["case_content_tabular_contamination_rows"],
                1,
            )
            self.assertEqual(analysis["dates"]["fields"]["call_time"]["invalid"], 1)
            self.assertEqual(analysis["dates"]["negative_completion_duration_rows"], 1)
            self.assertEqual(
                analysis["text"]["possible_pii_rows_by_field"]["case_content"][
                    "mainland_mobile"
                ],
                1,
            )
            content = next(
                column for column in analysis["columns"] if column["name"] == "case_content"
            )
            self.assertIsNone(content["top_values"])
            serialized = json.dumps(profile, ensure_ascii=False)
            self.assertNotIn("13800138000", serialized)
            self.assertEqual(profile["schema_version"], "civic-data-profile-v3")
            self.assertEqual(profile["semantic_layer"]["unmapped_fields"], [])
            order_id = next(
                entry
                for entry in profile["semantic_layer"]["entries"]
                if entry["name"] == "order_id"
            )
            self.assertEqual(order_id["rag_role"], "association_key")
            self.assertIn("不能直接解释为父工单", order_id["caution"])
            self.assertTrue((root / "output/profile.md").is_file())
            self.assertTrue((root / "output/columns.csv").is_file())
            self.assertTrue((root / "output/data_dictionary.md").is_file())
            self.assertTrue((root / "output/field_dictionary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
