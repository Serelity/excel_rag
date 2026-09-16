#!/usr/bin/env python3
"""Verify that a loopback TCP listener belongs to an expected process group."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import sys
from pathlib import Path

LISTEN_STATE = "0A"
SOCKET_TARGET = re.compile(r"socket:\[(\d+)\]\Z")


class ListenerOwnershipError(RuntimeError):
    """Raised when a listener cannot be tied to the expected job."""


class ListenerNotReadyError(ListenerOwnershipError):
    """Raised when the expected address does not have a listener yet."""


def _listener_inodes(proc_root: Path, host: str, port: int) -> set[str]:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ListenerOwnershipError(f"invalid listener host: {host}") from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ListenerOwnershipError("only an IPv4 loopback listener is supported")
    if not address.is_loopback:
        raise ListenerOwnershipError("listener host must be IPv4 loopback")
    if not 1 <= port <= 65535:
        raise ListenerOwnershipError("listener port must be between 1 and 65535")

    expected_ipv4_address = address.packed[::-1].hex().upper()
    expected_ipv4_addresses = {expected_ipv4_address, "00000000"}
    expected_ipv6_addresses = {
        "0" * 32,
        "0000000000000000FFFF000000000000",
        "0000000000000000FFFF0000" + expected_ipv4_address,
    }
    expected_port = f"{port:04X}"
    inodes: set[str] = set()
    for table_name in ("tcp", "tcp6"):
        table = proc_root / "net" / table_name
        try:
            lines = table.read_text(encoding="ascii").splitlines()
        except FileNotFoundError:
            if table_name == "tcp6":
                continue
            raise ListenerOwnershipError(f"cannot read TCP listener table: {table}") from None
        except OSError as exc:
            raise ListenerOwnershipError(f"cannot read TCP listener table: {table}") from exc

        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != LISTEN_STATE:
                continue
            try:
                local_address, local_port = fields[1].split(":", maxsplit=1)
            except ValueError:
                continue
            if table_name == "tcp6":
                # IPv6 wildcard and IPv4-mapped loopback sockets can accept an
                # IPv4 request to 127.0.0.1 when IPV6_V6ONLY is disabled.
                address_matches = local_address in expected_ipv6_addresses
            else:
                address_matches = local_address in expected_ipv4_addresses
            if address_matches and local_port == expected_port:
                inodes.add(fields[9])
    return inodes


def _process_identity(proc_root: Path, pid: int) -> tuple[int, int] | None:
    try:
        stat_text = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        return None
    fields = stat_text[closing_parenthesis + 1 :].split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[2]), int(fields[3])
    except ValueError:
        return None


def _listener_owners(
    proc_root: Path,
    socket_inodes: set[str],
) -> dict[str, dict[int, tuple[int, int]]]:
    owners: dict[str, dict[int, tuple[int, int]]] = {inode: {} for inode in socket_inodes}
    try:
        process_entries = list(proc_root.iterdir())
    except OSError as exc:
        raise ListenerOwnershipError(f"cannot enumerate processes under {proc_root}") from exc

    for process_entry in process_entries:
        if not process_entry.name.isdecimal():
            continue
        pid = int(process_entry.name)
        try:
            with os.scandir(process_entry / "fd") as descriptors:
                for descriptor in descriptors:
                    try:
                        target = os.readlink(descriptor.path)
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        continue
                    match = SOCKET_TARGET.fullmatch(target)
                    if match is None or match.group(1) not in socket_inodes:
                        continue
                    identity = _process_identity(proc_root, pid)
                    if identity is None:
                        raise ListenerOwnershipError(
                            f"listener owner identity became unreadable for pid {pid}"
                        )
                    owners[match.group(1)][pid] = identity
        except (FileNotFoundError, NotADirectoryError, PermissionError, ProcessLookupError):
            continue
    return owners


def verify_listener_owner(
    *,
    host: str,
    port: int,
    expected_process_group: int,
    proc_root: Path = Path("/proc"),
) -> dict[int, tuple[int, int]]:
    """Return owners only when every listener belongs to the expected job session."""

    if expected_process_group <= 0:
        raise ListenerOwnershipError("expected process group must be positive")
    socket_inodes = _listener_inodes(proc_root, host, port)
    if not socket_inodes:
        raise ListenerNotReadyError(f"no listening socket found at {host}:{port}")

    expected_identity = (expected_process_group, expected_process_group)
    owners_by_inode = _listener_owners(proc_root, socket_inodes)
    if any(not owners for owners in owners_by_inode.values()):
        raise ListenerOwnershipError(
            f"listener at {host}:{port} is not attributable to process group/session "
            f"{expected_process_group}"
        )

    owners = {
        pid: identity
        for inode_owners in owners_by_inode.values()
        for pid, identity in inode_owners.items()
    }
    if any(identity != expected_identity for identity in owners.values()):
        raise ListenerOwnershipError(
            f"listener at {host}:{port} has an owner outside process group/session "
            f"{expected_process_group}"
        )
    return owners


def verify_listener_free(
    *,
    host: str,
    port: int,
    proc_root: Path = Path("/proc"),
) -> None:
    """Fail when any process is already listening on the target address."""

    if _listener_inodes(proc_root, host, port):
        raise ListenerOwnershipError(f"a listener already occupies {host}:{port}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that a loopback listener belongs to a process group/session"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    ownership = parser.add_mutually_exclusive_group(required=True)
    ownership.add_argument("--expected-pgid", type=int)
    ownership.add_argument("--expect-free", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.expect_free:
            verify_listener_free(host=args.host, port=args.port)
            print("listener_free=true")
            return 0
        owners = verify_listener_owner(
            host=args.host,
            port=args.port,
            expected_process_group=args.expected_pgid,
        )
    except ListenerNotReadyError as error:
        print(f"NOT_READY: {error}", file=sys.stderr)
        return 3
    except ListenerOwnershipError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print("listener_owner_verified=true")
    print(f"listener_owner_process_group={args.expected_pgid}")
    print(f"listener_owner_session={args.expected_pgid}")
    print(f"listener_owner_process_count={len(owners)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
