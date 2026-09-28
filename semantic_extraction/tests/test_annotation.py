from __future__ import annotations

import hashlib
import json
import re

import pytest

from semantic_extraction.annotation import build_page


def worksheet(path, content):
    row = {
        "annotation_version": "semantic-gold-v1",
        "source_id": "test-1",
        "source_row": 1,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "case_content": content,
        "prediction": None,
        "review_status": "pending",
        "annotator_id": "",
        "gold": {
            "case_status": "unclear",
            "active_retrieval_issues": [],
            "background_issues": [],
            "annotation_notes": "",
        },
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return row


def test_page_preserves_unicode_and_does_not_execute_source_markup(tmp_path):
    source, target = tmp_path / "worksheet.jsonl", tmp_path / "page.html"
    content = '原文\u2028下一句\u2029</script><script>alert("source")</script>'
    row = worksheet(source, content)
    assert build_page(source, target) == 1
    page = target.read_text(encoding="utf-8")
    payload = re.search(r'id="worksheet-data">(.*?)</script>', page, re.S).group(1)
    assert json.loads(payload) == [row]
    assert '<script>alert("source")</script>' not in page
    assert "/* ANNOTATION_APP */" not in page
    assert "connect-src 'none'" in page
    with pytest.raises(FileExistsError):
        build_page(source, target)


def test_page_rejects_source_hash_changes_and_duplicate_records(tmp_path):
    source, target = tmp_path / "worksheet.jsonl", tmp_path / "page.html"
    row = worksheet(source, "原文")
    row["case_content"] = "变动"
    source.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_page(source, target)
    row = worksheet(source, "原文")
    with source.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        build_page(source, target)
    assert not target.exists()
