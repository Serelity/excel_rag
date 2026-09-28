"""Build a self-contained offline annotation page; no model or server needed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_page(worksheet: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing page: {output}")
    rows = []
    seen = set()
    with worksheet.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("annotation_version") != "semantic-gold-v1":
                raise ValueError(f"Invalid worksheet record at line {number}")
            sid = row.get("source_id")
            content = row.get("case_content")
            if not isinstance(sid, str) or not sid or sid in seen:
                raise ValueError(f"Missing or duplicate source_id at line {number}")
            if not isinstance(content, str):
                raise ValueError(f"Missing case_content at line {number}")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if row.get("content_sha256") != digest:
                raise ValueError(f"Source hash mismatch at line {number}")
            seen.add(sid)
            rows.append(row)
    if not rows:
        raise ValueError("Worksheet is empty")
    assets = Path(__file__).with_name("annotation_assets")
    template = (assets / "page.html").read_text(encoding="utf-8")
    script = (assets / "app.js").read_text(encoding="utf-8")
    # A narrative may contain literal HTML or closing script tags.
    payload = json.dumps(rows, ensure_ascii=True).replace("<", "\\u003c")
    page = template.replace("/* ANNOTATION_APP */", script)
    page = page.replace("<!-- WORKSHEET_DATA -->", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as target:
        target.write(page)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = build_page(args.worksheet, args.output)
    print(f"annotation_records={count} output={args.output}")


if __name__ == "__main__":
    main()
