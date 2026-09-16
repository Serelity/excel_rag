import json

from jsonschema import Draft202012Validator

from pipeline.chunker import build_problem_chunk, source_text_hash
from schemas.problem import Problem
from schemas.ticket import Ticket


def test_problem_chunk_matches_schema() -> None:
    ticket = Ticket(
        ticket_id="SYNTHETIC-TICKET-000001",
        content="合成测试内容：公共设施出现告警",
        goal="合成测试目标：确认处置流程",
        category1="测试一级分类",
        category2="测试二级分类",
        category3="测试三级分类",
        city="测试市",
        district="测试区",
        create_time="2099-01-02 03:04:05",
    )
    problem = Problem(
        problem_type="合成设施告警",
        category=ticket.category_path,
        symptom=["测试信号异常"],
        impact=["测试流程受阻"],
        location_type="测试区域",
        keywords=["合成", "告警"],
    )

    chunk = build_problem_chunk(
        ticket,
        problem,
        extraction_run_id="run_test",
        model="test-model",
        prompt_version="test-v1",
    )

    with open("schemas/problem-chunk-v1.schema.json", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator(schema).validate(chunk)
    assert "合成测试内容" not in json.dumps(chunk, ensure_ascii=False)
    assert chunk["source_text_hash"]


def test_source_hash_has_unambiguous_field_boundaries() -> None:
    shared = {
        "ticket_id": "ticket-1",
        "category1": "",
        "category2": "",
        "category3": "",
        "city": "",
        "district": "",
        "create_time": "",
    }
    left = Ticket(content="a\x00b", goal="c", **shared)
    right = Ticket(content="a", goal="b\x00c", **shared)

    assert source_text_hash(left) != source_text_hash(right)
