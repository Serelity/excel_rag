from __future__ import annotations

import csv
from collections import Counter

import pytest

from semantic_extraction.loader import load_records
from semantic_extraction.quality import TABULAR_CONTAMINATION
from semantic_extraction.selector import select_pilot


def write_tsv(path) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["id", "case_content", "case_accord_type_one_name"],
            delimiter="\t",
        )
        writer.writeheader()
        for index in range(30):
            writer.writerow(
                {
                    "id": str(index),
                    "case_content": f"正文-{index % 27}-" + "长" * (index * 10),
                    "case_accord_type_one_name": "类别甲" if index % 2 else "类别乙",
                }
            )


def test_selector_is_deterministic_and_uses_unique_content(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    write_tsv(source)

    first = select_pilot(source, size=12, seed=7)
    second = select_pilot(source, size=12, seed=7)

    assert [item.source_id for item in first] == [item.source_id for item in second]
    assert len({item.content_sha256 for item in first}) == 12
    assert len({(item.category1, item.length_bucket) for item in first}) > 2


def test_selector_excludes_tabular_contamination(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    write_tsv(source)
    with source.open("a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["id", "case_content", "case_accord_type_one_name"],
            delimiter="\t",
        )
        writer.writerow(
            {
                "id": "contaminated",
                "case_content": "真实正文\t后续字段\t后续记录",
                "case_accord_type_one_name": "类别甲",
            }
        )

    excluded: Counter[str] = Counter()
    selected = select_pilot(source, size=30, seed=7, excluded_quality=excluded)

    assert len(selected) == 30
    assert all(item.source_id != "contaminated" for item in selected)
    assert excluded == Counter({TABULAR_CONTAMINATION: 1})


def test_loader_rejects_contaminated_jsonl(tmp_path) -> None:
    source = tmp_path / "pilot.jsonl"
    source.write_text(
        '{"source_id":"1","source_row":1,"case_content":"正文\\t后续字段"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=TABULAR_CONTAMINATION):
        list(load_records(source))
