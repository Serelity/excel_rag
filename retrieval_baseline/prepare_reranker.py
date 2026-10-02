"""Download and check the separate ModelScope reranker snapshot; no GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "BAAI/bge-reranker-v2-m3"


def verify_files(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "xlm-roberta"
        or config.get("architectures") != ["XLMRobertaForSequenceClassification"]
        or len(config.get("id2label", {"0": "score"})) != 1
    ):
        raise ValueError("Expected BGE-reranker-v2-m3 single-logit XLM-R sequence classifier")
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        json.loads((path / name).read_text(encoding="utf-8"))
    if not any(
        (path / name).is_file() and (path / name).stat().st_size
        for name in (
            "tokenizer.json",
            "sentencepiece.bpe.model",
        )
    ):
        raise ValueError("Missing reranker tokenizer")
    weights = [
        path / name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (path / name).is_file() and (path / name).stat().st_size
    ]
    if not weights:
        raise ValueError("Missing reranker weights")
    return {
        "model_files_check": "passed",
        "model_path": str(path.resolve()),
        "validation_scope": "files and classifier configuration; GPU not tested",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        if (args.output / "config.json").exists():
            config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
            if config.get("architectures") != ["XLMRobertaForSequenceClassification"]:
                raise ValueError("Output is another model; use a separate reranker directory")
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"modelscope_model={args.model_id} revision={args.revision}", flush=True)
        snapshot_download(
            args.model_id, revision=args.revision, local_dir=str(args.output.resolve())
        )
    print(json.dumps(verify_files(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
