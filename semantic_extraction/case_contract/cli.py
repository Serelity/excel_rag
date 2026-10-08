from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from .examples import synthetic_examples
from .prompt import SYSTEM_PROMPT, build_request
from .schema import PROMPT_VERSION, SPEC_VERSION, CaseExtraction, CaseInput
from .validation import validate_response


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def export_bundle(output: Path) -> dict:
    # Refuse any existing directory: a new prompt/spec is a new reproducible bundle.
    output.mkdir(parents=True, exist_ok=False)
    examples = synthetic_examples()
    for example in examples:
        validate_response(example["input"]["case_content"], example["response"])
    artifacts = {
        "response.schema.json": json_text(CaseExtraction.model_json_schema()),
        "system-prompt.txt": SYSTEM_PROMPT + "\n",
        "examples.synthetic.jsonl": "".join(
            json.dumps(e, ensure_ascii=False) + "\n" for e in examples
        ),
    }
    for name, content in artifacts.items():
        write_new(output / name, content)
    manifest = {
        "spec_version": SPEC_VERSION, "prompt_version": PROMPT_VERSION,
        "status": "development_contract_not_model_evaluation",
        "artifacts": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                      for name, text in artifacts.items()},
        "synthetic_example_count": len(examples),
    }
    write_new(output / "manifest.json", json_text(manifest))
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="案例抽取规范v1：导出、准备请求、离线校验。")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="导出可追溯规范包，不调用模型")
    export.add_argument("--output", required=True, type=Path)
    for name in ("request", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--input", required=True, type=Path, help="仅含case_content的JSON")
        command.add_argument("--output", required=True, type=Path)
        if name == "validate":
            command.add_argument("--response", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            export_bundle(args.output)
        else:
            raw_input = json.loads(args.input.read_text(encoding="utf-8-sig"))
            parsed = CaseInput.model_validate(raw_input)
            if args.command == "request":
                result = build_request(parsed.model_dump())
            else:
                response = json.loads(args.response.read_text(encoding="utf-8-sig"))
                result = validate_response(parsed.case_content, response)
            write_new(args.output, json_text(result))
    except (ValueError, OSError, TypeError) as exc:
        # Avoid dumping private field values from Pydantic errors in shared terminal logs.
        print(f"Failed ({type(exc).__name__}); output not accepted. "
              "Inspect the local input/response with the contract validator.", file=sys.stderr)
        if isinstance(exc, ValidationError):
            print(json.dumps([{"path": list(e["loc"]), "type": e["type"]}
                              for e in exc.errors()], ensure_ascii=False), file=sys.stderr)
        elif isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0
