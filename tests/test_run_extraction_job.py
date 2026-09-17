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
TEST_CODE_BRANCH = "test/h100-stage-a"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _git(project: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
    real_git = shutil.which("git")
    assert real_setsid is not None
    assert real_git is not None
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
        fake_bin / "git",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ ${FAKE_GIT_MODE:-real} == unavailable ]]; then
          exit 127
        fi
        if [[ ${FAKE_GIT_MODE:-real} == status_failure ]]; then
          for argument in "$@"; do
            [[ $argument == status ]] && exit 74
          done
        fi
        exec "$REAL_GIT_PATH" "$@"
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
    _git(project, "switch", "-q", "-c", TEST_CODE_BRANCH)
    _git(project, "add", ".")
    _git(project, "commit", "-qm", "fixture")
    git_commit = _git(project, "rev-parse", "HEAD")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "RAG_CODE_COMMIT": git_commit,
            "RAG_CODE_BRANCH": TEST_CODE_BRANCH,
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
            "REAL_GIT_PATH": real_git,
        }
    )
    fixture = SimpleNamespace(
        project=project,
        script=deploy / "run-extraction-job.sh",
        env_file=env_file,
        log_dir=log_dir,
        environment=environment,
        git_commit=git_commit,
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
    **environment_overrides: str | None,
) -> subprocess.CompletedProcess[str]:
    environment = fixture.environment.copy()
    for name, value in environment_overrides.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
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


def _assert_status_value(status: str, key: str, value: str) -> None:
    assert f"{key}={value}" in status.splitlines()


def test_gitless_runtime_uses_declared_code_metadata(job_fixture) -> None:
    result = _run_job(job_fixture, FAKE_GIT_MODE="unavailable")

    assert result.returncode == 0, result.stderr
    assert "git is unavailable on this runtime" in result.stderr
    status = _status_text(job_fixture)
    _assert_status_value(status, "git_commit", job_fixture.git_commit)
    _assert_status_value(status, "git_branch", TEST_CODE_BRANCH)
    _assert_status_value(status, "code_provenance_source", "launch_environment")
    _assert_status_value(status, "git_runtime_available", "false")
    _assert_status_value(status, "git_commit_verified", "false")
    _assert_status_value(status, "git_branch_verified", "false")
    _assert_status_value(status, "git_worktree_verified", "false")
    _assert_status_value(status, "git_worktree_changes", "unknown")
    assert job_fixture.vllm_pid_record.exists()
    assert job_fixture.extraction_pid_record.exists()


def test_git_checkout_verifies_declared_code_metadata(job_fixture) -> None:
    result = _run_job(job_fixture)

    assert result.returncode == 0, result.stderr
    status = _status_text(job_fixture)
    _assert_status_value(status, "git_runtime_available", "true")
    _assert_status_value(status, "git_commit_verified", "true")
    _assert_status_value(status, "git_branch_verified", "true")
    _assert_status_value(status, "git_worktree_verified", "true")
    _assert_status_value(status, "git_worktree_changes", "0")


@pytest.mark.parametrize(
    "commit",
    [None, "", "a" * 39, "A" * 40, "g" * 40, "a" * 65],
)
def test_job_rejects_invalid_declared_code_commit(job_fixture, commit) -> None:
    result = _run_job(job_fixture, RAG_CODE_COMMIT=commit)

    assert result.returncode == 2
    assert "RAG_CODE_COMMIT" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_gitless_runtime_accepts_full_sha256_git_object_id(job_fixture) -> None:
    result = _run_job(
        job_fixture,
        FAKE_GIT_MODE="unavailable",
        RAG_CODE_COMMIT="b" * 64,
    )

    assert result.returncode == 0, result.stderr
    _assert_status_value(_status_text(job_fixture), "git_commit", "b" * 64)


@pytest.mark.parametrize("branch", ["_release", "release./child"])
def test_gitless_runtime_accepts_safe_git_branch_names(job_fixture, branch) -> None:
    result = _run_job(
        job_fixture,
        FAKE_GIT_MODE="unavailable",
        RAG_CODE_BRANCH=branch,
    )

    assert result.returncode == 0, result.stderr
    _assert_status_value(_status_text(job_fixture), "git_branch", branch)


@pytest.mark.parametrize(
    "branch",
    [
        None,
        "",
        "bad\nbranch",
        "/leading",
        "trailing/",
        "HEAD",
        "bad..branch",
        "bad@{ref",
        "release/.hidden",
        "release.lock/child",
    ],
)
def test_job_rejects_invalid_declared_code_branch(job_fixture, branch) -> None:
    result = _run_job(job_fixture, RAG_CODE_BRANCH=branch)

    assert result.returncode == 2
    assert "RAG_CODE_BRANCH" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_job_rejects_declared_commit_mismatch(job_fixture) -> None:
    result = _run_job(job_fixture, RAG_CODE_COMMIT="f" * 40)

    assert result.returncode == 2
    assert "RAG_CODE_COMMIT does not match" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_job_rejects_declared_branch_mismatch(job_fixture) -> None:
    result = _run_job(job_fixture, RAG_CODE_BRANCH="other/h100-stage-a")

    assert result.returncode == 2
    assert "RAG_CODE_BRANCH does not match" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_job_rejects_detached_head_as_a_verified_branch(job_fixture) -> None:
    _git(job_fixture.project, "checkout", "--detach", "-q")

    result = _run_job(job_fixture, RAG_CODE_BRANCH="detached")

    assert result.returncode == 2
    assert "repository HEAD must be attached" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_job_rejects_symbolic_head_outside_local_branches(job_fixture) -> None:
    _git(job_fixture.project, "tag", "symbolic-target")
    _git(job_fixture.project, "symbolic-ref", "HEAD", "refs/tags/symbolic-target")

    result = _run_job(job_fixture, RAG_CODE_BRANCH="symbolic-target")

    assert result.returncode == 2
    assert "HEAD must reference a local branch" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_ancestor_git_repository_is_not_used_for_verification(job_fixture) -> None:
    shutil.rmtree(job_fixture.project / ".git")
    parent = job_fixture.project.parent
    _git(parent, "init", "-q")
    _git(parent, "config", "user.email", "tests@example.invalid")
    _git(parent, "config", "user.name", "Test Runner")
    _git(parent, "switch", "-q", "-c", "unrelated/parent")
    sentinel = parent / "parent-sentinel.txt"
    sentinel.write_text("unrelated parent repository\n", encoding="utf-8")
    _git(parent, "add", sentinel.name)
    _git(parent, "commit", "-qm", "parent fixture")

    result = _run_job(job_fixture)

    assert result.returncode == 0, result.stderr
    assert "git found only ancestor repository metadata" in result.stderr
    status = _status_text(job_fixture)
    _assert_status_value(status, "git_runtime_available", "true")
    _assert_status_value(status, "git_commit_verified", "false")
    _assert_status_value(status, "git_branch_verified", "false")
    _assert_status_value(status, "git_worktree_verified", "false")
    _assert_status_value(status, "git_worktree_changes", "unknown")


def test_job_rejects_dirty_git_worktree(job_fixture) -> None:
    tracked_file = job_fixture.project / "deploy" / "run-vllm.sh"
    tracked_file.write_text(
        tracked_file.read_text(encoding="utf-8") + "\n# dirty test\n",
        encoding="utf-8",
    )

    result = _run_job(job_fixture)

    assert result.returncode == 2
    assert "repository worktree must be clean" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_job_fails_closed_when_git_status_fails(job_fixture) -> None:
    result = _run_job(job_fixture, FAKE_GIT_MODE="status_failure")

    assert result.returncode == 2
    assert "worktree status could not be read" in result.stderr
    assert not job_fixture.vllm_pid_record.exists()
    assert not job_fixture.extraction_pid_record.exists()


def test_launch_metadata_overrides_stale_environment_file(job_fixture) -> None:
    with job_fixture.env_file.open("a", encoding="utf-8") as environment_file:
        environment_file.write(f"RAG_CODE_COMMIT={'e' * 40}\n")
        environment_file.write("RAG_CODE_BRANCH=stale/branch\n")

    result = _run_job(job_fixture, FAKE_GIT_MODE="unavailable")

    assert result.returncode == 0, result.stderr
    status = _status_text(job_fixture)
    _assert_status_value(status, "git_commit", job_fixture.git_commit)
    _assert_status_value(status, "git_branch", TEST_CODE_BRANCH)
    assert f"git_commit={'e' * 40}" not in status.splitlines()
    assert "git_branch=stale/branch" not in status.splitlines()


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


def test_production_extraction_launcher_runs_without_git(tmp_path) -> None:
    source_root = Path(__file__).parents[1]
    project = tmp_path / "project"
    deploy = project / "deploy"
    runtime_bin = tmp_path / "runtime-bin"
    deploy.mkdir(parents=True)
    runtime_bin.mkdir()
    shutil.copy2(source_root / "deploy" / "run-extraction.sh", deploy)

    for command in ("bash", "dirname", "env"):
        executable = shutil.which(command)
        assert executable is not None
        (runtime_bin / command).symlink_to(executable)

    marker = tmp_path / "conda-invocation.txt"
    _write_executable(
        runtime_bin / "conda",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        printf 'commit=%s\nbranch=%s\nargs=%s\n' \
          "$RAG_CODE_COMMIT" "$RAG_CODE_BRANCH" "$*" > "$FAKE_CONDA_MARKER"
        """,
    )
    env_file = deploy / ".env"
    env_file.write_text(
        "".join(
            (
                "CONDA_EXTRACT_ENV=fake-extract\n",
                f"QWEN_MODEL_FINGERPRINT_SHA256=sha256:{'a' * 64}\n",
                "QWEN_MODELSCOPE_REPO_ID=Qwen/Qwen3-30B-A3B\n",
                "QWEN_MODEL_REVISION=test-revision\n",
                f"RAG_CODE_COMMIT={'e' * 40}\n",
                "RAG_CODE_BRANCH=stale/branch\n",
            )
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": str(runtime_bin),
            "RAG_ENV_FILE": str(env_file),
            "RAG_JOB_WRAPPER_ACTIVE": "1",
            "RAG_CODE_COMMIT": "d" * 40,
            "RAG_CODE_BRANCH": "release/h100-stage-a",
            "FAKE_CONDA_MARKER": str(marker),
        }
    )

    result = subprocess.run(
        [str(runtime_bin / "bash"), str(deploy / "run-extraction.sh"), "--limit", "1"],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    invocation = marker.read_text(encoding="utf-8")
    assert f"commit={'d' * 40}" in invocation.splitlines()
    assert "branch=release/h100-stage-a" in invocation.splitlines()
    assert "python main.py --limit 1" in invocation


def test_preflight_treats_gitless_code_verification_as_a_warning(tmp_path) -> None:
    source_root = Path(__file__).parents[1]
    project = tmp_path / "project"
    deploy = project / "deploy"
    fake_bin = tmp_path / "bin"
    deploy.mkdir(parents=True)
    fake_bin.mkdir()
    shutil.copy2(source_root / "deploy" / "server-preflight.sh", deploy)
    _write_executable(
        fake_bin / "git",
        """
        #!/usr/bin/env bash
        exit 127
        """,
    )
    _write_executable(
        fake_bin / "conda",
        """
        #!/usr/bin/env bash
        exit 1
        """,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "RAG_CODE_COMMIT": "c" * 40,
            "RAG_CODE_BRANCH": "release/h100-stage-a",
        }
    )

    result = subprocess.run(
        ["bash", str(deploy / "server-preflight.sh")],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 1  # Other intentionally absent preflight assets fail.
    assert "WARN: git is unavailable on this runtime" in result.stdout
    assert "FAIL: git is not available" not in result.stdout
    assert "git_runtime_available=false" in result.stdout.splitlines()
    assert "git_commit_verified=false" in result.stdout.splitlines()
    assert "git_branch_verified=false" in result.stdout.splitlines()
    assert "git_worktree_verified=false" in result.stdout.splitlines()
    assert "git_worktree_changes=unknown" in result.stdout.splitlines()
