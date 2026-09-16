import importlib.util
import os
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "deploy" / "vllm-entrypoint.py"
SPEC = importlib.util.spec_from_file_location("vllm_entrypoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
vllm_entrypoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vllm_entrypoint)


def test_server_argv_never_copies_environment_key_into_arguments(monkeypatch) -> None:
    secret = "private-test-key"
    monkeypatch.setenv("VLLM_API_KEY", secret)

    argv = vllm_entrypoint.server_argv(["--host", "127.0.0.1"])

    assert argv == [
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
    ]
    assert secret not in argv
    assert os.environ["VLLM_API_KEY"] == secret


def test_server_argv_preserves_server_arguments() -> None:
    argv = vllm_entrypoint.server_argv(["--port", "8000"])

    assert argv == ["vllm.entrypoints.openai.api_server", "--port", "8000"]


def test_main_keeps_api_key_in_environment_only(monkeypatch) -> None:
    secret = "private-test-key"
    captured: dict[str, object] = {}

    def fake_run_module(module_name: str, *, run_name: str) -> None:
        captured["module_name"] = module_name
        captured["run_name"] = run_name
        captured["argv"] = list(sys.argv)
        captured["api_key"] = os.environ.get("VLLM_API_KEY")

    monkeypatch.setenv("VLLM_API_KEY", secret)
    monkeypatch.setattr(sys, "argv", ["vllm-entrypoint.py", "--port", "8000"])
    monkeypatch.setattr(vllm_entrypoint.runpy, "run_module", fake_run_module)

    vllm_entrypoint.main()

    assert captured == {
        "module_name": "vllm.entrypoints.openai.api_server",
        "run_name": "__main__",
        "argv": ["vllm.entrypoints.openai.api_server", "--port", "8000"],
        "api_key": secret,
    }
