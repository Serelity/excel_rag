import importlib.util
import json
from types import SimpleNamespace

import pytest

from semantic_extraction.case_contract import runner
from semantic_extraction.case_contract.examples import fact, response
from semantic_extraction.case_contract.transport import (
    NoRedirect,
    Reply,
    TransportError,
    strict_json,
    validate_base_url,
)


def success(payload=None, reason="stop", model="Qwen3-30B-A3B"):
    if payload is None:
        payload = response([[fact(1, "topic", "subject", "路灯不亮")]])
    return Reply(200, json.dumps({"model": model, "choices": [{"finish_reason": reason,
        "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}).encode())


class FakeTransport:
    def __init__(self, replies, model="Qwen3-30B-A3B"):
        self.replies, self.model, self.calls = iter(replies), model, []

    def request(self, endpoint, payload=None):
        self.calls.append((endpoint, payload))
        if endpoint == "models":
            return Reply(200, json.dumps({"data": [{"id": self.model}]}).encode())
        item = next(self.replies)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def prepared(tmp_path):
    folder = tmp_path / "prepared"
    folder.mkdir()
    rows = [{"sample_id": "S01", "input": {"case_content": "路灯不亮"}}]
    (folder / "inputs.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    runner.save(folder / "manifest.json", {
        "spec_version": runner.SPEC_VERSION, "model_run": False,
        "contract_manifest_sha256": runner.sha(
            runner.ROOT / "research/specs/case-content-extraction-v1/manifest.json"
        ),
        "artifacts": {"inputs.jsonl": runner.sha(folder / "inputs.jsonl")},
    })
    return folder


@pytest.fixture
def plan(prepared, tmp_path):
    output = tmp_path / "run"
    runner.prepare(prepared, output, sample_ids=None, model="Qwen3-30B-A3B",
                   model_fingerprint="sha256:" + "a" * 64)
    return output


def test_full_run_persists_request_reply_and_grounded_output(plan):
    transport = FakeTransport([success()])
    result = runner.run(plan, transport=transport)
    assert result["status"] == "completed" and result["counts"]["validated"] == 1
    request = transport.calls[1][1]
    assert json.loads(request["messages"][1]["content"]) == {"case_content": "路灯不亮"}
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["response_format"]["json_schema"]["name"] == "case_content_extraction_v1"
    assert "Authorization" not in json.dumps(request)
    assert runner.load(plan / "records/S01.json")["semantic_review_status"] == "not_run"
    runner.verify_artifacts(plan, runner.load(plan / "manifest.json"))
    with pytest.raises(ValueError, match="already_attempted"):
        runner.run(plan, transport=transport)
    assert len(transport.calls) == 2


@pytest.mark.parametrize("first", [Reply(503, b'{"error":"busy"}'), TransportError("timeout")])
def test_only_transient_errors_retry_and_all_attempts_survive(plan, first):
    client = FakeTransport([first, success()])
    sleeps = []
    result = runner.run(plan, transport=client, sleep=sleeps.append)
    assert result["counts"]["model_calls"] == 2 and sleeps == [1]
    assert len(list((plan / "attempts").glob("*.request.json"))) == 2
    assert runner.load(plan / "attempts/S01-01.metadata.json")["error"] is not None


@pytest.mark.parametrize("reply,error", [
    (success(reason="length"), "output_truncated"),
    (success(reason="content_filter"), "incomplete_response"),
    (success(model="wrong-model"), "response_model_mismatch"),
    (Reply(400, b'{"error":"unsupported schema"}'), "http_error"),
    (Reply(200, b'invalid JSON'), "invalid_json_response"),
    (success({"events": []}), "schema_validation_failed"),
    (success(response([[fact(1, "topic", "subject", "漏水")]])), "evidence_grounding_failed"),
])
def test_semantic_schema_and_truncation_failures_are_not_repaired_or_dropped(plan, reply, error):
    client = FakeTransport([reply])
    result = runner.run(plan, transport=client)
    assert result["status"] == "completed_with_errors"
    assert result["counts"]["model_calls"] == 1
    record = runner.load(plan / "records/S01.json")
    assert record["error"] == error and record["result"] is None
    runner.verify_artifacts(plan, runner.load(plan / "manifest.json"))


def test_transport_retry_budget_is_bounded(plan):
    result = runner.run(plan, transport=FakeTransport([TransportError("timeout")] * 2),
                        sleep=lambda _: None)
    assert result["counts"]["model_calls"] == 2 and result["counts"]["rejected"] == 1


def test_wrong_served_alias_stops_before_chat_call(plan):
    client = FakeTransport([], model="other")
    with pytest.raises(ValueError, match="served_model"):
        runner.run(plan, transport=client)
    assert len(client.calls) == 1
    assert runner.load(plan / "status.json")["status"] == "failed"
    assert not (plan / "manifest.json").exists()


def test_interruption_cannot_look_like_completed_run(plan):
    with pytest.raises(KeyboardInterrupt):
        runner.run(plan, transport=FakeTransport([KeyboardInterrupt()]))
    assert runner.load(plan / "status.json")["status"] == "interrupted"
    assert not (plan / "manifest.json").exists()
    assert (plan / "attempts/S01-01.request.json").exists()


def test_changed_sources_stop_before_any_service_call(plan, monkeypatch):
    monkeypatch.setattr(runner, "source_hashes", lambda: {"changed": "hash"})
    client = FakeTransport([])
    with pytest.raises(ValueError, match="source_changed"):
        runner.run(plan, transport=client)
    assert not client.calls


def test_changed_plan_input_stops_before_service_call(plan):
    (plan / "inputs.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        runner.run(plan, transport=FakeTransport([]))


def test_preparation_checks_source_hashes_and_selection(prepared, tmp_path):
    kwargs = {"model": "Qwen3-30B-A3B", "model_fingerprint": "sha256:" + "a" * 64}
    with pytest.raises(ValueError, match="selection"):
        runner.prepare(prepared, tmp_path / "bad", sample_ids=["MISSING"], **kwargs)
    assert not (tmp_path / "bad").exists()
    (prepared / "inputs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        runner.prepare(prepared, tmp_path / "bad", sample_ids=None, **kwargs)


@pytest.mark.parametrize("url", ["https://api.example.com/v1", "http://localhost:8000/v1",
    "http://127.0.0.1:8000/v1?key=x", "http://user:secret@127.0.0.1:8000/v1",
    "http://127.0.0.1:8000/other", "http://127.0.0.1:80/v1"])
def test_transport_refuses_nonlocal_or_ambiguous_urls(url):
    with pytest.raises(ValueError):
        validate_base_url(url)


def test_redirects_and_duplicate_json_keys_are_rejected():
    with pytest.raises(TransportError, match="redirect"):
        NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com")
    for text in ['{"route":"a","route":"b"}', '{"value":NaN}']:
        with pytest.raises(ValueError):
            strict_json(text)


def load_env_inspector():
    spec = importlib.util.spec_from_file_location(
        "contract_env_inspector", runner.ROOT / "deploy/inspect-case-contract-env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(name, *, serving=False):
    module = load_env_inspector()
    return {"name": name, "prefix": f"/envs/{name}", "python": [3, 11, 0],
            "pydantic_import": True,
            "packages": dict(module.SERVE_VERSIONS) if serving else {"pydantic": "2.11.4"}}


def test_new_environment_is_used_for_both_client_and_server():
    module = load_env_inspector()
    selected = environment("civic-rag-extract-v1")
    old = environment("civic-rag-retrieval", serving=True)
    assert module.choose([selected, old], "client") == selected
    with pytest.raises(ValueError, match="incompatible_for_serve"):
        module.choose([selected, old], "serve")
    with pytest.raises(ValueError, match="missing_or_ambiguous"):
        module.choose([old], "client")


def test_old_environment_can_be_selected_explicitly():
    module = load_env_inspector()
    retrieval = environment("civic-rag-retrieval", serving=True)
    server = environment("civic-rag-extract", serving=True)
    assert module.choose([server, retrieval], "serve",
                         environment="civic-rag-retrieval") == retrieval


def test_missing_vllm_and_duplicate_env_names_do_not_trigger_installation():
    module = load_env_inspector()
    retrieval = environment("civic-rag-extract-v1")
    with pytest.raises(ValueError, match="incompatible"):
        module.choose([retrieval], "serve")
    second = dict(retrieval, prefix="/elsewhere/civic-rag-extract-v1")
    with pytest.raises(ValueError, match="ambiguous"):
        module.choose([retrieval, second], "client")


@pytest.mark.parametrize("selected", ["civic-rag-extract-v1", "civic-rag-retrieval"])
def test_inspector_probes_only_selected_environment_without_mutation(selected):
    module = load_env_inspector()
    calls = []
    probe = environment(selected, serving=True)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["env", "list"]:
            return SimpleNamespace(stdout=json.dumps({"envs": [
                "/envs/civic-rag-extract", "/envs/civic-rag-retrieval",
                "/envs/civic-rag-extract-v1",
            ]}))
        return SimpleNamespace(stdout="CASE_CONTRACT_PROBE=" + json.dumps(probe))

    report = module.inspect("conda", environment=selected, run=fake_run)
    assert report["status"] == "ready_for_gpu_preflight"
    assert report["client"]["prefix"] == report["server"]["prefix"]
    assert report["environment_changed"] is False and report["gpu_inference"] == "not_run"
    assert report["required_environment"] == selected
    assert report["probes"][0]["serve_mismatches"] == []
    assert len(calls) == 2 and calls[1][4] == f"/envs/{selected}"
    assert all("install" not in c and "create" not in c for c in calls)


def test_inspector_reports_missing_environment_without_probing_alternatives():
    module = load_env_inspector()
    report = module.inspect("conda", run=lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"envs": ["/envs/civi-rag-retrieval", "/envs/civic-rag-extract"]})
    ))
    assert report["status"] == "failed" and report["probes"] == []
    assert report["reason"] == "required_environment_missing_or_ambiguous"


def test_missing_vllm_is_reported_with_actual_and_expected_versions():
    module = load_env_inspector()
    probe = environment("civic-rag-retrieval", serving=True)
    probe["python"] = [3, 11, 16]
    probe["packages"].update({"torch": "2.6.0+cu124", "vllm": None, "pytest": None})
    assert module.compatibility_errors(probe, "client") == []
    assert module.compatibility_errors(probe, "serve") == [
        {"component": "vllm", "actual": None, "expected": "0.8.5"},
    ]


def test_inspector_default_is_independent_extraction_environment():
    module = load_env_inspector()
    assert module.ENVIRONMENT == "civic-rag-extract-v1"
    selected = environment(module.ENVIRONMENT, serving=True)
    assert module.choose([environment("civic-rag-retrieval", serving=True), selected],
                         "serve") == selected


@pytest.mark.parametrize("name", ["../env", "/envs/name", "bad name", "--help"])
def test_invalid_environment_name_fails_before_conda(name):
    module = load_env_inspector()

    def no_calls(*args, **kwargs):
        pytest.fail("invalid environment name must not invoke Conda")

    with pytest.raises(ValueError, match="invalid_environment_name"):
        module.inspect("conda", environment=name, run=no_calls)
