from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REQUIRED_COMMANDS = ("bash", "flock", "git", "ps", "setsid", "sha256sum")


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _group_members(pgid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
    )
    members = []
    for line in result.stdout.splitlines():
        pid_text, pgid_text, state = line.split(maxsplit=2)
        if int(pgid_text) == pgid and not state.startswith("Z"):
            members.append(int(pid_text))
    return members


def _wait_for_group_exit(pgid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_members(pgid):
            return
        time.sleep(0.05)
    pytest.fail(f"process group {pgid} still has live members: {_group_members(pgid)}")


def _terminate_group_from_record(path: Path) -> None:
    if not path.exists():
        return
    try:
        os.killpg(int(path.read_text(encoding="ascii")), signal.SIGKILL)
    except (ProcessLookupError, ValueError):
        pass


@pytest.fixture
def job_fixture(tmp_path):
    missing = [command for command in REQUIRED_COMMANDS if shutil.which(command) is None]
    if missing:
        pytest.skip(f"missing required commands: {', '.join(missing)}")

    source_root = Path(__file__).parents[1]
    project = tmp_path / "project"
    deploy = project / "deploy"
    fake_bin = tmp_path / "bin"
    external = tmp_path / "external"
    deploy.mkdir(parents=True)
    fake_bin.mkdir()
    external.mkdir()
    real_setsid = shutil.which("setsid")
    assert real_setsid is not None
    shutil.copy2(source_root / "deploy" / "run-extraction-job.sh", deploy)
    _write_executable(
        deploy / "verify-listener-owner.py",
        """
        #!/usr/bin/env python3
        import os
        import sys
        from pathlib import Path

        if "--expect-free" in sys.argv:
            if os.environ.get("FAKE_CURL_MODE") == "existing":
                raise SystemExit(2)
            print("listener_free=true")
            raise SystemExit(0)
        owner_mode = os.environ.get("FAKE_LISTENER_OWNER_MODE", "accept")
        if owner_mode == "reject" or (
            owner_mode == "reject_during"
            and Path(os.environ["FAKE_EXTRACTION_STARTED"]).exists()
        ):
            raise SystemExit(2)
        live_file = Path(os.environ["FAKE_VLLM_LIVE_FILE"])
        if not live_file.is_file():
            raise SystemExit(3)
        try:
            os.kill(int(live_file.read_text(encoding="ascii")), 0)
        except (OSError, ValueError):
            raise SystemExit(3)
        print("listener_owner_verified=true")
        print("listener_owner_process_group=test")
        print("listener_owner_session=test")
        print("listener_owner_process_count=1")
        """,
    )

    input_path = external / "sanitized.tsv"
    input_bytes = b"sanitized-test-input\n"
    input_path.write_bytes(input_bytes)
    output_path = external / "chunks.jsonl"
    quarantine_path = external / "errors.jsonl"
    log_dir = external / "logs"
    model_dir = external / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="ascii")

    vllm_pid_record = external / "vllm.pid"
    vllm_live_file = external / "vllm.live"
    extraction_pid_record = external / "extraction.pid"
    extraction_child_record = external / "extraction-child.pid"
    extraction_started = external / "extraction.started"
    setsid_started = external / "setsid.started"
    setsid_group_record = external / "setsid-group.pid"

    _write_executable(
        fake_bin / "setsid",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ ${1:-} == --fork && ${2:-} == --wait && \
          -n ${FAKE_SETSID_MARKER_DELAY:-} ]]; then
          shift 2
          exec "$REAL_SETSID_PATH" --fork --wait bash -c '
            printf "%s" "$$" > "$FAKE_SETSID_GROUP_RECORD"
            : > "$FAKE_SETSID_STARTED"
            sleep "$FAKE_SETSID_MARKER_DELAY"
            exec "$@"
          ' _ "$@"
        fi
        exec "$REAL_SETSID_PATH" "$@"
        """,
    )

    _write_executable(
        deploy / "run-vllm.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        # Production launcher contract markers:
        # --disable-log-requests --disable-uvicorn-access-log
        [[ -z ${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-} ]]
        [[ -z ${http_proxy:-}${https_proxy:-}${all_proxy:-} ]]
        [[ ${NO_PROXY:-} == 127.0.0.1,localhost ]]
        [[ ${no_proxy:-} == 127.0.0.1,localhost ]]
        printf '%s' "$$" > "$FAKE_VLLM_PID_RECORD"
        printf '%s' "$$" > "$FAKE_VLLM_LIVE_FILE"
        cleanup() { rm -f -- "$FAKE_VLLM_LIVE_FILE"; exit 0; }
        trap cleanup TERM INT HUP
        if [[ ${FAKE_VLLM_MODE:-stay} == exit_during ]]; then
          while [[ ! -e $FAKE_EXTRACTION_STARTED ]]; do sleep 0.05; done
          rm -f -- "$FAKE_VLLM_LIVE_FILE"
          exit 23
        fi
        while true; do sleep 0.1; done
        """,
    )
    _write_executable(
        deploy / "run-extraction.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        [[ -z ${HTTP_PROXY:-}${HTTPS_PROXY:-}${ALL_PROXY:-} ]]
        [[ -z ${http_proxy:-}${https_proxy:-}${all_proxy:-} ]]
        [[ ${NO_PROXY:-} == 127.0.0.1,localhost ]]
        [[ ${no_proxy:-} == 127.0.0.1,localhost ]]
        printf '%s' "$$" > "$FAKE_EXTRACTION_PID_RECORD"
        : > "$FAKE_EXTRACTION_STARTED"
        if [[ ${FAKE_EXTRACTION_MODE:-return} == return ]]; then
          exit "${FAKE_EXTRACTION_RC:-0}"
        fi
        (
          trap '' TERM
          while true; do sleep 0.1; done
        ) &
        child=$!
        printf '%s' "$child" > "$FAKE_EXTRACTION_CHILD_RECORD"
        trap 'exit 0' TERM INT HUP
        wait "$child"
        """,
    )
    _write_executable(
        fake_bin / "curl",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ ${FAKE_CURL_MODE:-normal} == existing ]]; then
          exit 0
        fi
        [[ -s $FAKE_VLLM_LIVE_FILE ]] || exit 22
        pid=$(cat "$FAKE_VLLM_LIVE_FILE")
        kill -0 "$pid" 2>/dev/null || exit 22
        if [[ ${FAKE_CURL_MODE:-normal} == unhealthy && -e $FAKE_EXTRACTION_STARTED ]]; then
          exit 22
        fi
        exit 0
        """,
    )
    _write_executable(
        fake_bin / "conda",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        for argument in "$@"; do
          if [[ $argument == 'import sys; print(sys.executable)' ]]; then
            command -v python3
            exit 0
          fi
        done
        if [[ -n ${VLLM_MODEL_LIST_URL:-} ]]; then
          exit 0
        fi
        for argument in "$@"; do
          if [[ $argument == */verify-listener-owner.py ]]; then
            if [[ ${FAKE_LISTENER_OWNER_MODE:-accept} == reject ]]; then
              exit 2
            fi
            printf '%s\n' \
              'listener_owner_verified=true' \
              'listener_owner_process_group=test' \
              'listener_owner_session=test' \
              'listener_owner_process_count=1'
            exit 0
          fi
          if [[ $argument == */model-fingerprint.py ]]; then
            printf '%s\n' \
              'model_fingerprint_algorithm=test' \
              'model_fingerprint_file_count=1' \
              'model_fingerprint_total_bytes=1' \
              "model_fingerprint_sha256=$QWEN_MODEL_FINGERPRINT_SHA256" \
              'model_fingerprint_match=true'
            exit 0
          fi
          if [[ $argument == */configs/config.yaml ]]; then
            printf '%s\n' \
              "$FAKE_CONFIG_INPUT" \
              "$FAKE_CONFIG_OUTPUT" \
              "$FAKE_CONFIG_QUARANTINE" \
              "$QWEN_SERVED_MODEL_NAME"
            exit 0
          fi
        done
        exit 0
        """,
    )

    env_file = deploy / ".env"
    values = {
        "CONDA_EXTRACT_ENV": "fake-extract",
        "QWEN_MODEL_PATH": str(model_dir),
        "QWEN_SERVED_MODEL_NAME": "Qwen3-30B-A3B",
        "QWEN_MODELSCOPE_REPO_ID": "Qwen/Qwen3-30B-A3B",
        "QWEN_MODEL_REVISION": "test-revision",
        "QWEN_MODEL_FINGERPRINT_SHA256": f"sha256:{'a' * 64}",
        "VLLM_HOST": "127.0.0.1",
        "VLLM_PORT": "8000",
        "VLLM_STARTUP_TIMEOUT_SECONDS": "5",
        "VLLM_HEALTH_POLL_SECONDS": "1",
        "VLLM_HEALTH_FAILURE_LIMIT": "2",
        "VLLM_SHUTDOWN_TIMEOUT_SECONDS": "1",
        "RAG_RUN_RECORDS_PATH": str(external / "records"),
        "RAG_JOB_LOG_DIR": str(log_dir),
        "RAG_JOB_LOCK_PATH": str(external / "job.lock"),
        "RAG_INPUT_PATH": str(input_path),
        "RAG_INPUT_SIZE_BYTES": str(len(input_bytes)),
        "RAG_INPUT_SHA256": hashlib.sha256(input_bytes).hexdigest(),
    }
    env_file.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )

    _git(project, "init", "-q")
    _git(project, "config", "user.email", "tests@example.invalid")
    _git(project, "config", "user.name", "Test Runner")
    _git(project, "add", ".")
    _git(project, "commit", "-qm", "fixture")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "RAG_ENV_FILE": str(env_file),
            "FAKE_CONFIG_INPUT": str(input_path),
            "FAKE_CONFIG_OUTPUT": str(output_path),
            "FAKE_CONFIG_QUARANTINE": str(quarantine_path),
            "FAKE_VLLM_PID_RECORD": str(vllm_pid_record),
            "FAKE_VLLM_LIVE_FILE": str(vllm_live_file),
            "FAKE_EXTRACTION_PID_RECORD": str(extraction_pid_record),
            "FAKE_EXTRACTION_CHILD_RECORD": str(extraction_child_record),
            "FAKE_EXTRACTION_STARTED": str(extraction_started),
            "FAKE_SETSID_STARTED": str(setsid_started),
            "FAKE_SETSID_GROUP_RECORD": str(setsid_group_record),
            "REAL_SETSID_PATH": real_setsid,
        }
    )
    fixture = SimpleNamespace(
        project=project,
        script=deploy / "run-extraction-job.sh",
        log_dir=log_dir,
        environment=environment,
        vllm_pid_record=vllm_pid_record,
        extraction_pid_record=extraction_pid_record,
        extraction_child_record=extraction_child_record,
        setsid_started=setsid_started,
        setsid_group_record=setsid_group_record,
    )
    try:
        yield fixture
    finally:
        _terminate_group_from_record(extraction_pid_record)
        _terminate_group_from_record(vllm_pid_record)


def _run_job(
    fixture,
    *arguments: str,
    **environment_overrides: str,
) -> subprocess.CompletedProcess[str]:
    environment = fixture.environment | environment_overrides
    return subprocess.run(
        ["bash", str(fixture.script), *arguments],
        cwd=fixture.project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _status_text(fixture) -> str:
    status_files = list(fixture.log_dir.glob("job-*.status"))
    assert len(status_files) == 1
    return status_files[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("extraction_rc", [0, 1, 2])
def test_job_preserves_extraction_exit_code_and_cleans_its_vllm_group(
    job_fixture,
    extraction_rc,
) -> None:
    sentinel = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        result = _run_job(
            job_fixture,
            FAKE_EXTRACTION_MODE="return",
            FAKE_EXTRACTION_RC=str(extraction_rc),
        )

        assert result.returncode == extraction_rc, result.stderr
        status = _status_text(job_fixture)
        assert f"extraction_finished exit_code={extraction_rc}" in status
        assert f"job_finished exit_code={extraction_rc}" in status
        _wait_for_group_exit(int(job_fixture.vllm_pid_record.read_text(encoding="ascii")))
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


@pytest.mark.parametrize(
    ("vllm_mode", "curl_mode", "status_marker"),
    [
        ("exit_during", "normal", "vllm_exited_during_extraction"),
        ("stay", "unhealthy", "vllm_unhealthy_during_extraction"),
    ],
)
def test_vllm_failure_stops_the_entire_extraction_group(
    job_fixture,
    vllm_mode,
    curl_mode,
    status_marker,
) -> None:
    result = _run_job(
        job_fixture,
        FAKE_VLLM_MODE=vllm_mode,
        FAKE_CURL_MODE=curl_mode,
        FAKE_EXTRACTION_MODE="long",
    )

    assert result.returncode == 2, result.stderr
    status = _status_text(job_fixture)
    assert status_marker in status
    assert "extraction_finished exit_code=2" in status
    extraction_pgid = int(job_fixture.extraction_pid_record.read_text(encoding="ascii"))
    _wait_for_group_exit(extraction_pgid)


def test_existing_healthy_service_is_not_stopped(job_fixture) -> None:
    sentinel = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        result = _run_job(job_fixture, FAKE_CURL_MODE="existing")

        assert result.returncode == 2
        assert "listener address 127.0.0.1:8000 is occupied" in result.stderr
        assert not job_fixture.vllm_pid_record.exists()
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


@pytest.mark.parametrize("arguments", [("--limit", "101"), ("--limit=101",)])
def test_bounded_pilot_rejects_more_than_one_hundred_rows(
    job_fixture,
    arguments,
) -> None:
    result = _run_job(job_fixture, *arguments)

    assert result.returncode == 2
    assert "bounded pilots accept at most --limit 100" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_bounded_pilot_accepts_one_hundred_rows(job_fixture) -> None:
    result = _run_job(job_fixture, "--limit", "100")

    assert result.returncode == 0, result.stderr
    assert job_fixture.vllm_pid_record.exists()
    assert job_fixture.extraction_pid_record.exists()
    assert "default_smoke=no" in _status_text(job_fixture)


def test_full_resume_is_not_blocked_by_the_pilot_limit(job_fixture) -> None:
    result = _run_job(job_fixture, "--full", "--resume")

    assert result.returncode == 0, result.stderr
    assert job_fixture.vllm_pid_record.exists()
    assert job_fixture.extraction_pid_record.exists()


def test_unowned_healthy_listener_stops_only_this_jobs_vllm(job_fixture) -> None:
    sentinel = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        result = _run_job(job_fixture, FAKE_LISTENER_OWNER_MODE="reject")

        assert result.returncode == 2
        assert "listener at 127.0.0.1:8000 is not owned" in result.stderr
        assert not job_fixture.extraction_pid_record.exists()
        _wait_for_group_exit(int(job_fixture.vllm_pid_record.read_text(encoding="ascii")))
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_listener_owner_loss_during_extraction_stops_the_entire_job(job_fixture) -> None:
    sentinel = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        result = _run_job(
            job_fixture,
            FAKE_LISTENER_OWNER_MODE="reject_during",
            FAKE_EXTRACTION_MODE="long",
        )

        assert result.returncode == 2
        status = _status_text(job_fixture)
        assert "listener_owner_check_failed exit_code=2" in status
        assert "job_finished exit_code=2" in status
        _wait_for_group_exit(int(job_fixture.extraction_pid_record.read_text(encoding="ascii")))
        _wait_for_group_exit(int(job_fixture.vllm_pid_record.read_text(encoding="ascii")))
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_second_job_is_rejected_while_the_h100_job_lock_is_held(job_fixture) -> None:
    environment = job_fixture.environment | {
        "FAKE_VLLM_MODE": "stay",
        "FAKE_CURL_MODE": "normal",
        "FAKE_EXTRACTION_MODE": "long",
    }
    first = subprocess.Popen(
        ["bash", str(job_fixture.script)],
        cwd=job_fixture.project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (Path(environment["FAKE_EXTRACTION_STARTED"])).exists():
            if first.poll() is not None:
                stdout, stderr = first.communicate()
                pytest.fail(f"first job exited early rc={first.returncode}: {stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("first job did not reach extraction before the timeout")
            time.sleep(0.05)

        second = _run_job(job_fixture, FAKE_EXTRACTION_MODE="return")

        assert second.returncode == 2
        assert "another extraction job holds the H100 job lock" in second.stderr
        assert first.poll() is None
    finally:
        if first.poll() is None:
            first.send_signal(signal.SIGTERM)
        first.communicate(timeout=10)


def test_second_termination_signal_does_not_interrupt_group_cleanup(job_fixture) -> None:
    environment = job_fixture.environment | {
        "FAKE_VLLM_MODE": "stay",
        "FAKE_CURL_MODE": "normal",
        "FAKE_EXTRACTION_MODE": "long",
    }
    process = subprocess.Popen(
        ["bash", str(job_fixture.script)],
        cwd=job_fixture.project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not Path(environment["FAKE_EXTRACTION_STARTED"]).exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"job exited before extraction: {stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("job did not reach extraction before the timeout")
            time.sleep(0.05)

        process.send_signal(signal.SIGTERM)
        status_deadline = time.monotonic() + 5
        while True:
            status_files = list(job_fixture.log_dir.glob("job-*.status"))
            status = status_files[0].read_text(encoding="utf-8") if status_files else ""
            if "stopping extraction" in status:
                break
            if process.poll() is not None:
                pytest.fail("job exited before its cleanup phase could be observed")
            if time.monotonic() >= status_deadline:
                pytest.fail("job did not enter extraction cleanup before the timeout")
            time.sleep(0.02)

        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)

        assert process.returncode == 143, f"{stdout}\n{stderr}"
        status = _status_text(job_fixture)
        assert status.count("job_finished exit_code=143") == 1
        _wait_for_group_exit(int(job_fixture.extraction_pid_record.read_text(encoding="ascii")))
        _wait_for_group_exit(int(job_fixture.vllm_pid_record.read_text(encoding="ascii")))
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize("signal_process_group", [False, True])
def test_termination_during_setsid_handshake_cleans_the_new_session(
    job_fixture,
    signal_process_group,
) -> None:
    environment = job_fixture.environment | {"FAKE_SETSID_MARKER_DELAY": "0.5"}
    process = subprocess.Popen(
        ["bash", str(job_fixture.script)],
        cwd=job_fixture.project,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=signal_process_group,
    )
    try:
        deadline = time.monotonic() + 10
        while not job_fixture.setsid_started.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"job exited before the setsid handshake: {stdout}\n{stderr}")
            if time.monotonic() >= deadline:
                pytest.fail("setsid handshake did not start before the timeout")
            time.sleep(0.02)

        if signal_process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)

        assert process.returncode == 143, f"{stdout}\n{stderr}"
        process_group = int(job_fixture.setsid_group_record.read_text(encoding="ascii"))
        _wait_for_group_exit(process_group)
        assert not job_fixture.extraction_pid_record.exists()
        assert "job_finished exit_code=143" in _status_text(job_fixture)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        _terminate_group_from_record(job_fixture.setsid_group_record)
