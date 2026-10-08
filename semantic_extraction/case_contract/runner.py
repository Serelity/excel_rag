"""Prepare and run an auditable v1 extraction pilot against server-local vLLM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from .cli import json_text, write_new
from .prompt import build_request
from .schema import PROMPT_VERSION, SPEC_VERSION, CaseInput
from .transport import TransportError, VllmTransport, strict_json, validate_base_url
from .validation import GroundingError, validate_response

RUN_VERSION = "case-contract-run-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOKE_IDS = "B001,B015,B032,B040,B013,B016,B019,B026,B053,B069"


def now():
    return datetime.now(UTC).isoformat()


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return strict_json(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value):
    write_new(path, json_text(value))


def update_status(folder: Path, value):
    temporary = folder / "status.tmp"
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(folder / "status.json")


def verify_artifacts(folder: Path, manifest: dict):
    for name, expected in manifest["artifacts"].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or sha(target) != expected:
            raise ValueError("artifact_integrity_failure")


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "deploy/run-case-contract.sh", ROOT / "deploy/inspect-case-contract-env.py",
        ROOT / "deploy/run-qwen3-vllm.sh", ROOT / "semantic_extraction/validate_runtime.py",
        ROOT / "semantic_extraction/validate_model.py",
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def prepare(prepared_dir: Path, output: Path, *, sample_ids: list[str] | None,
            model: str, model_fingerprint: str, max_tokens=8192, timeout=300,
            retries=1, base_url="http://127.0.0.1:8000/v1") -> dict:
    validate_base_url(base_url)
    if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", model_fingerprint):
        raise ValueError("declared_model_fingerprint_required")
    if model != "Qwen3-30B-A3B" or max_tokens < 1 or timeout <= 0 or not 0 <= retries <= 2:
        raise ValueError("invalid_run_parameters")
    manifest = load(prepared_dir / "manifest.json")
    if manifest.get("spec_version") != SPEC_VERSION or manifest.get("model_run") is not False:
        raise ValueError("expected_prepared_development_input")
    verify_artifacts(prepared_dir, manifest)
    contract_dir = ROOT / "research/specs/case-content-extraction-v1"
    if manifest["contract_manifest_sha256"] != sha(contract_dir / "manifest.json"):
        raise ValueError("prepared_contract_version_mismatch")
    verify_artifacts(contract_dir, load(contract_dir / "manifest.json"))
    rows = [strict_json(line) for line in
            (prepared_dir / "inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = set()
    for row in rows:
        if set(row) != {"sample_id", "input"} or not isinstance(row["sample_id"], str):
            raise ValueError("invalid_input_envelope")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["sample_id"]) or row["sample_id"] in seen:
            raise ValueError("invalid_or_duplicate_sample_id")
        seen.add(row["sample_id"])
        CaseInput.model_validate(row["input"])
    if not rows:
        raise ValueError("empty_input")
    if sample_ids is not None:
        if not sample_ids or len(set(sample_ids)) != len(sample_ids) or not set(sample_ids) <= seen:
            raise ValueError("invalid_sample_selection")
        selected = set(sample_ids)
        rows = [row for row in rows if row["sample_id"] in selected]
    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    save(output / "inputs.json", rows)
    save(output / "prepared-manifest.json", manifest)
    config = {
        "run_version": RUN_VERSION, "spec_version": SPEC_VERSION, "prompt_version": PROMPT_VERSION,
        "model": model, "declared_model_fingerprint": model_fingerprint,
        "base_url": base_url, "max_tokens": max_tokens, "timeout_seconds": timeout,
        "transport_retries": retries, "temperature": 0, "seed": 42, "enable_thinking": False,
        "concurrency": 1, "sample_ids": [row["sample_id"] for row in rows],
        "semantic_repair": False, "truncation_repair": False,
    }
    save(output / "config.json", config)
    save(output / "plan.json", {
        "run_version": RUN_VERSION, "created_at": now(),
        "prepared_manifest_sha256": sha(prepared_dir / "manifest.json"),
        "source_sha256": source_hashes(),
        "artifacts": {name: sha(output / name) for name in
                      ["inputs.json", "config.json", "prepared-manifest.json"]},
    })
    update_status(output, {"status": "prepared", "selected": len(rows), "model_run": False})
    return {"status": "prepared", "selected": len(rows)}


def decode_reply(reply, expected_model):
    if reply.status != 200:
        return None, "http_error"
    try:
        value = strict_json(reply.body)
        if not isinstance(value, dict) or value.get("model") != expected_model:
            return None, "response_model_mismatch"
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return None, "invalid_choices"
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            return None, ("output_truncated" if choice.get("finish_reason") == "length"
                          else "incomplete_response")
        message = choice.get("message", {})
        if message.get("refusal"):
            return None, "model_refusal"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return None, "empty_response"
        return strict_json(content), None
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return None, "invalid_json_response"


def run(folder: Path, *, transport=None, sleep=time.sleep) -> dict:
    plan = load(folder / "plan.json")
    verify_artifacts(folder, plan)
    if plan["source_sha256"] != source_hashes():
        raise ValueError("source_changed_since_preparation")
    if load(folder / "status.json")["status"] != "prepared":
        raise ValueError("run_already_attempted_use_new_directory")
    config, rows = load(folder / "config.json"), load(folder / "inputs.json")
    # Persist a claim before any service call; no resume into partially written attempts.
    write_new(folder / "execution.lock", str(os.getpid()))
    started_at = now()
    update_status(folder, {"status": "running", "started_at": started_at,
                           "selected": len(rows), "model_run": False})
    totals = Counter()
    try:
        client = transport or VllmTransport(
            config["base_url"], os.environ.get("VLLM_API_KEY", ""), config["timeout_seconds"],
        )
        save(folder / "client-runtime.json", {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "pydantic": importlib.metadata.version("pydantic"),
        })
        models = client.request("models")
        (folder / "models.response.json").write_bytes(models.body)
        listing = strict_json(models.body)
        if models.status != 200 or config["model"] not in {
            entry.get("id") for entry in listing.get("data", []) if isinstance(entry, dict)
        }:
            raise ValueError("served_model_not_available")
        attempts_dir = folder / "attempts"
        attempts_dir.mkdir()
        results = []
        for row in rows:
            sample_id, case_content = row["sample_id"], row["input"]["case_content"]
            request = build_request(row["input"])
            request.update({
                "model": config["model"], "temperature": config["temperature"],
                "seed": config["seed"],
                "max_tokens": config["max_tokens"], "stream": False,
                "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]},
            })
            error, validated, attempts = None, None, []
            for attempt in range(config["transport_retries"] + 1):
                stem = f"{sample_id}-{attempt + 1:02d}"
                save(attempts_dir / f"{stem}.request.json", request)
                metadata = {"sample_id": sample_id, "attempt": attempt + 1, "started_at": now()}
                tick = time.monotonic()
                retryable = False
                totals["model_calls"] += 1
                try:
                    reply = client.request("chat/completions", request)
                    (attempts_dir / f"{stem}.response.json").write_bytes(reply.body)
                    metadata["http_status"] = reply.status
                    candidate, error = decode_reply(reply, config["model"])
                    retryable = reply.status in {408, 429, 500, 502, 503, 504}
                    if error is None:
                        try:
                            validated = validate_response(case_content, candidate)
                        except ValidationError as exc:
                            error = "schema_validation_failed"
                            metadata["validation_errors"] = [
                                {"path": list(e["loc"]), "type": e["type"]} for e in exc.errors()
                            ]
                        except GroundingError as exc:
                            error = "evidence_grounding_failed"
                            metadata["grounding_error"] = str(exc)
                except TransportError as exc:
                    error, retryable = "transport_error", True
                    metadata["transport_error"] = str(exc)
                metadata.update({"elapsed_seconds": round(time.monotonic() - tick, 3),
                                 "finished_at": now(), "error": error})
                save(attempts_dir / f"{stem}.metadata.json", metadata)
                attempts.append(stem)
                if not error or not retryable or attempt == config["transport_retries"]:
                    break
                sleep(1)
            result = {"sample_id": sample_id, "status": "validated" if not error else "rejected",
                      "error": error, "attempts": attempts, "result": validated,
                      "semantic_review_status": "not_run"}
            # Individual records survive an interruption; the final manifest marks completion.
            save(folder / "records" / f"{sample_id}.json", result)
            results.append(result)
            totals["processed"] += 1
            totals["validated" if not error else "rejected"] += 1
            if error:
                totals[error] += 1
        write_new(folder / "results.jsonl", "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in results
        ))
        status = {
            "status": "completed" if not totals["rejected"] else "completed_with_errors",
            "started_at": started_at, "finished_at": now(), "selected": len(rows),
            "counts": dict(totals), "model_run": bool(totals["model_calls"]),
            "semantic_evaluation": "not_run", "retrieval_evaluation": "not_run",
        }
        save(folder / "summary.json", status)
        save(folder / "manifest.json", {
            "run_version": RUN_VERSION, "plan_sha256": sha(folder / "plan.json"),
            "artifacts": {p.relative_to(folder).as_posix(): sha(p)
                          for p in sorted(folder.rglob("*"))
                          if p.is_file() and p.name not in {"status.json", "status.tmp",
                                                         "execution.lock", "manifest.json"}},
        })
        update_status(folder, status)
        return status
    except BaseException as exc:
        update_status(folder, {"status": "interrupted" if isinstance(exc, KeyboardInterrupt |
                              SystemExit) else "failed", "error_type": type(exc).__name__,
                              "counts": dict(totals), "finished_at": now(),
                              "model_run": bool(totals["model_calls"])})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--prepared-dir", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    selection = prep.add_mutually_exclusive_group()
    selection.add_argument("--sample-ids", default=DEFAULT_SMOKE_IDS)
    selection.add_argument("--all", action="store_true")
    prep.add_argument("--model", default="Qwen3-30B-A3B")
    prep.add_argument("--model-fingerprint", required=True)
    prep.add_argument("--max-tokens", type=int, default=8192)
    prep.add_argument("--timeout", type=float, default=300)
    prep.add_argument("--retries", type=int, default=1)
    prep.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    execute = commands.add_parser("run")
    execute.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.prepared_dir, args.output,
                             sample_ids=None if args.all else args.sample_ids.split(","),
                             model=args.model, model_fingerprint=args.model_fingerprint,
                             max_tokens=args.max_tokens, timeout=args.timeout,
                             retries=args.retries, base_url=args.base_url)
        else:
            result = run(args.run_dir)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json_text(result))
    return 0 if result["status"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
