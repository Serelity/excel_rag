import csv

import pytest
from pydantic import ValidationError

from pipeline.cleaner import clean_ticket
from pipeline.loader import load_tickets
from schemas.ticket import Ticket

COLUMNS = [
    "id",
    "case_content",
    "case_goal",
    "case_accord_type_one_name",
    "case_accord_type_two_name",
    "case_accord_type_three_name",
    "area_code_city",
    "area_code_area",
    "call_time",
]


def write_tsv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_tickets_streams_quoted_multiline_tsv_and_normalizes_nulls(tmp_path):
    source = tmp_path / "tickets.tsv"
    write_tsv(
        source,
        COLUMNS,
        [
            {
                "id": " 123 ",
                "case_content": "第一行\n第二行\t详情",
                "case_goal": "NULL",
                "case_accord_type_one_name": " 城乡建设 ",
                "case_accord_type_two_name": "null",
                "case_accord_type_three_name": "NaN",
                "area_code_city": "常州市",
                "area_code_area": " 武进区 ",
                "call_time": "NaN",
            },
            {
                "id": "124",
                "case_content": '包含"引号"的内容',
                "case_goal": "处理",
                "case_accord_type_one_name": "",
                "case_accord_type_two_name": "",
                "case_accord_type_three_name": "",
                "area_code_city": "",
                "area_code_area": "",
                "call_time": "",
            },
        ],
    )

    tickets = load_tickets(source)
    first = next(tickets)

    assert first.ticket_id == "123"
    assert first.content == "第一行\n第二行\t详情"
    assert first.goal == ""
    assert first.category_path == ["城乡建设"]
    assert first.district == "武进区"
    assert first.create_time == ""

    assert clean_ticket(first).content == "第一行 第二行 详情"
    assert next(tickets).ticket_id == "124"
    with pytest.raises(StopIteration):
        next(tickets)


def test_load_tickets_rejects_missing_id_column(tmp_path):
    source = tmp_path / "missing-id.tsv"
    write_tsv(source, COLUMNS[1:], [{column: "" for column in COLUMNS[1:]}])

    with pytest.raises(ValueError, match=r"missing required column.*'id'"):
        list(load_tickets(source))


@pytest.mark.parametrize("empty_id", ["", "NULL", "null", "NaN", "  "])
def test_load_tickets_rejects_empty_id(tmp_path, empty_id):
    source = tmp_path / "empty-id.tsv"
    row = {column: "" for column in COLUMNS}
    row["id"] = empty_id
    write_tsv(source, COLUMNS, [row])

    with pytest.raises(ValueError, match=r"empty required 'id'"):
        list(load_tickets(source))


def test_ticket_validates_id_and_exposes_category_path():
    values = {
        "ticket_id": "ticket-1",
        "content": None,
        "goal": "NaN",
        "category1": "一级",
        "category2": "NULL",
        "category3": "三级",
        "city": "常州市",
        "district": "",
    }

    ticket = Ticket(**values)

    assert ticket.content == ""
    assert ticket.goal == ""
    assert ticket.category_path == ["一级", "三级"]
    assert ticket.model_dump()["category_path"] == ["一级", "三级"]

    with pytest.raises(ValidationError):
        Ticket(**{**values, "ticket_id": "NULL"})

    with pytest.raises(ValidationError):
        Ticket(**{**values, "content": ["not", "text"]})
