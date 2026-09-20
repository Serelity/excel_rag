from __future__ import annotations

import csv

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
