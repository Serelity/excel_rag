"""Download a BGE-M3 snapshot from ModelScope and check files without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "Xorbits/bge-m3"
REQUIRED = (
    "config.json", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
)


def verify_files(path: Path) -> dict:
    for name in REQUIRED:
        target = path / name
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Required model file missing or empty: {name}")
        json.loads(target.read_text(encoding="utf-8"))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    pooling = json.loads((path / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if (config.get("model_type") != "xlm-roberta" or config.get("hidden_size") != 1024
            or not pooling.get("pooling_mode_cls_token")
            or pooling.get("pooling_mode_mean_tokens")):
        raise ValueError("Configuration does not match the expected BGE-M3 dense profile")
    weights = [path / name for name in ("model.safetensors", "pytorch_model.bin")
               if (path / name).is_file() and (path / name).stat().st_size > 0]
    if not weights:
        raise ValueError("No nonempty model.safetensors or pytorch_model.bin found")
    return {"model_files_check": "passed", "model_path": str(path.resolve()),
            "weight_files": [p.name for p in weights],
            "weight_bytes": sum(p.stat().st_size for p in weights),
            "validation_scope": "required files and configuration; GPU loading not tested"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        # Refuse a clearly different model directory before downloading into it.
        config_path = args.output / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_type") != "xlm-roberta" or config.get("hidden_size") != 1024:
                raise ValueError(
                    "Output contains another model; choose a separate BGE-M3 directory"
                )
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"modelscope_model={MODEL_ID} revision={args.revision}", flush=True)
        snapshot_download(MODEL_ID, revision=args.revision, local_dir=str(args.output.resolve()))
    print(json.dumps(verify_files(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
