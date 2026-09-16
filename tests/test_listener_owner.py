from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "deploy" / "verify-listener-owner.py"
SPEC = importlib.util.spec_from_file_location("verify_listener_owner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verify_listener_owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_listener_owner)


def _write_process(
    proc_root: Path,
    pid: int,
    process_group: int,
    inode: str,
    *,
    session: int | None = None,
) -> None:
    process = proc_root / str(pid)
    (process / "fd").mkdir(parents=True)
    (process / "stat").write_text(
        f"{pid} (fake worker) S 1 {process_group} "
        f"{process_group if session is None else session} 0 0\n",
        encoding="ascii",
    )
    (process / "fd" / "3").symlink_to(f"socket:[{inode}]")


def test_listener_must_belong_to_expected_process_group_and_session(tmp_path) -> None:
    (tmp_path / "net").mkdir()
    (tmp_path / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st tx_queue tm->when retrnsmt uid timeout inode\n"
        "   0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 1000 0 12345\n",
        encoding="ascii",
    )
    _write_process(tmp_path, pid=4243, process_group=4242, inode="12345")

    owners = verify_listener_owner.verify_listener_owner(
        host="127.0.0.1",
        port=8000,
        expected_process_group=4242,
        proc_root=tmp_path,
    )

    assert owners == {4243: (4242, 4242)}
    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="owner outside process group/session",
    ):
        verify_listener_owner.verify_listener_owner(
            host="127.0.0.1",
            port=8000,
            expected_process_group=9999,
            proc_root=tmp_path,
        )
    (tmp_path / "4243" / "stat").write_text(
        "4243 (fake worker) S 1 4242 7777 0 0\n",
        encoding="ascii",
    )
    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="owner outside process group/session",
    ):
        verify_listener_owner.verify_listener_owner(
            host="127.0.0.1",
            port=8000,
            expected_process_group=4242,
            proc_root=tmp_path,
        )
    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="already occupies",
    ):
        verify_listener_owner.verify_listener_free(
            host="127.0.0.1",
            port=8000,
            proc_root=tmp_path,
        )
    verify_listener_owner.verify_listener_free(
        host="127.0.0.1",
        port=8001,
        proc_root=tmp_path,
    )


def test_listener_rejects_foreign_co_owner_of_same_inode(tmp_path) -> None:
    (tmp_path / "net").mkdir()
    (tmp_path / "net" / "tcp").write_text(
        "  sl  local_address rem_address   st tx_queue tm->when retrnsmt uid timeout inode\n"
        "   0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000 1000 0 12345\n",
        encoding="ascii",
    )
    _write_process(tmp_path, pid=4243, process_group=4242, inode="12345")
    _write_process(tmp_path, pid=9001, process_group=9001, inode="12345")

    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="owner outside process group/session",
    ):
        verify_listener_owner.verify_listener_owner(
            host="127.0.0.1",
            port=8000,
            expected_process_group=4242,
            proc_root=tmp_path,
        )


def test_listener_verifier_rejects_non_loopback_host() -> None:
    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="must be IPv4 loopback",
    ):
        verify_listener_owner.verify_listener_owner(
            host="192.0.2.1",
            port=8000,
            expected_process_group=1,
        )


@pytest.mark.parametrize(
    ("table_name", "local_address"),
    [
        ("tcp", "00000000"),
        ("tcp6", "0" * 32),
        ("tcp6", "0000000000000000FFFF000000000000"),
        ("tcp6", "0000000000000000FFFF00000100007F"),
    ],
)
def test_listener_free_rejects_wildcard_listener_on_same_port(
    tmp_path,
    table_name,
    local_address,
) -> None:
    (tmp_path / "net").mkdir()
    header = "  sl  local_address rem_address   st tx_queue tm->when retrnsmt uid timeout inode\n"
    (tmp_path / "net" / "tcp").write_text(header, encoding="ascii")
    if table_name == "tcp":
        table = tmp_path / "net" / "tcp"
    else:
        table = tmp_path / "net" / "tcp6"
    table.write_text(
        header + f"   0: {local_address}:1F40 {'0' * len(local_address)}:0000 0A "
        "00000000:00000000 00:00000000 00000000 1000 0 54321\n",
        encoding="ascii",
    )

    with pytest.raises(
        verify_listener_owner.ListenerOwnershipError,
        match="already occupies",
    ):
        verify_listener_owner.verify_listener_free(
            host="127.0.0.1",
            port=8000,
            proc_root=tmp_path,
        )
