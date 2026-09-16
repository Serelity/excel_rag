import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "deploy" / "model-fingerprint.py"
SPEC = importlib.util.spec_from_file_location("model_fingerprint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
model_fingerprint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_fingerprint)

TEST_WEIGHTS = b"test-weights"


@pytest.fixture(autouse=True)
def compact_model_contract(monkeypatch) -> None:
    monkeypatch.setattr(model_fingerprint, "EXPECTED_WEIGHT_TOTAL_BYTES", len(TEST_WEIGHTS) - 1)
    monkeypatch.setattr(model_fingerprint, "EXPECTED_WEIGHT_SHARD_COUNT", 1)
    monkeypatch.setattr(model_fingerprint, "MIN_SHARD_FILE_BYTES", 1)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_snapshot(
    root: Path,
    *,
    embedded_template: bool = True,
    config_overrides: dict[str, object] | None = None,
) -> Path:
    model_dir = root / "Qwen3-30B-A3B"
    model_dir.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    config.update(config_overrides or {})
    write_json(
        model_dir / "config.json",
        config,
    )
    tokenizer_config = {
        "chat_template": "{% if enable_thinking %}<think>{% endif %}" if embedded_template else None
    }
    write_json(model_dir / "tokenizer_config.json", tokenizer_config)
    write_json(
        model_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": len(TEST_WEIGHTS) - 1},
            "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
        },
    )
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(TEST_WEIGHTS)
    write_json(model_dir / "tokenizer.json", {"version": "1.0"})
    return model_dir.resolve()


def test_fingerprint_is_stable_and_changes_with_selected_content(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path)

    first = model_fingerprint.fingerprint(model_dir, quiet=True)
    second = model_fingerprint.fingerprint(model_dir, quiet=True)
    assert first == second
    assert first[1] == 5
    assert first[2] > 0

    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"fake-weights")
    changed = model_fingerprint.fingerprint(model_dir, quiet=True)
    assert changed[0] != first[0]


def test_independent_chat_template_is_validated_and_hashed(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path, embedded_template=False)
    template = model_dir / "chat_template.jinja"
    template.write_text("{% if enable_thinking %}<think>{% endif %}", encoding="utf-8")

    first = model_fingerprint.fingerprint(model_dir, quiet=True)
    template.write_text("{% if enable_thinking %}thinking{% endif %}", encoding="utf-8")
    second = model_fingerprint.fingerprint(model_dir, quiet=True)

    assert second[0] != first[0]
    assert second[1] == first[1]


def test_snapshot_rejects_a_missing_indexed_shard(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path)
    (model_dir / "model-00001-of-00001.safetensors").unlink()

    with pytest.raises(model_fingerprint.FingerprintError, match="missing shard"):
        model_fingerprint.fingerprint(model_dir, quiet=True)


def test_snapshot_rejects_missing_tokenizer_assets(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path)
    (model_dir / "tokenizer.json").unlink()

    with pytest.raises(model_fingerprint.FingerprintError, match="tokenizer assets are missing"):
        model_fingerprint.fingerprint(model_dir, quiet=True)


def test_snapshot_rejects_wrong_weight_total(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_size"] += 1
    write_json(index_path, index)

    with pytest.raises(model_fingerprint.FingerprintError, match="index total_size"):
        model_fingerprint.fingerprint(model_dir, quiet=True)


def test_snapshot_rejects_a_tiny_or_pointer_shard(tmp_path) -> None:
    model_dir = make_snapshot(tmp_path)
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"")

    with pytest.raises(model_fingerprint.FingerprintError, match="too small"):
        model_fingerprint.fingerprint(model_dir, quiet=True)


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        ({"hidden_size": 4096}, "hidden_size does not match Qwen3-30B-A3B"),
        ({"torch_dtype": "float16"}, "must declare Qwen3-30B-A3B bfloat16"),
        ({"torch_dtype": {"name": "bfloat16"}}, "torch_dtype must be a string"),
        (
            {"quantization_config": {"quant_method": "gptq"}},
            "quantized Qwen3 snapshots are not accepted",
        ),
    ],
)
def test_snapshot_rejects_wrong_model_identity(tmp_path, config_overrides, message) -> None:
    model_dir = make_snapshot(tmp_path, config_overrides=config_overrides)

    with pytest.raises(model_fingerprint.FingerprintError, match=message):
        model_fingerprint.fingerprint(model_dir, quiet=True)


def test_fingerprint_rejects_a_file_changed_after_it_was_hashed(tmp_path, monkeypatch) -> None:
    model_dir = make_snapshot(tmp_path)
    original_hash_file = model_fingerprint._hash_file

    def hash_and_mutate(path):
        result = original_hash_file(path)
        if path.suffix == ".safetensors":
            config_path = model_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["changed"] = True
            write_json(config_path, config)
        return result

    monkeypatch.setattr(model_fingerprint, "_hash_file", hash_and_mutate)
    with pytest.raises(model_fingerprint.FingerprintError, match="changed while it was hashed"):
        model_fingerprint.fingerprint(model_dir, quiet=True)


def test_cli_expect_returns_distinct_match_and_mismatch_codes(tmp_path, capsys) -> None:
    model_dir = make_snapshot(tmp_path)
    expected, _, _ = model_fingerprint.fingerprint(model_dir, quiet=True)

    assert (
        model_fingerprint.main(["--model-dir", str(model_dir), "--expect", expected, "--quiet"])
        == 0
    )
    assert "model_fingerprint_match=true" in capsys.readouterr().out

    mismatch = f"sha256:{'0' * 64}"
    assert (
        model_fingerprint.main(["--model-dir", str(model_dir), "--expect", mismatch, "--quiet"])
        == 3
    )
    assert "model_fingerprint_match=false" in capsys.readouterr().out
