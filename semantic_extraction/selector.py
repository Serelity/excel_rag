from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .quality import case_content_quality_issue

PILOT_SCHEMA_VERSION = "qwen3-pilot-v2"


@dataclass(frozen=True, slots=True)
class Candidate:
    rank: int
    source_id: str
    source_row: int
    case_content: str
    content_sha256: str
    category1: str
    length_bucket: str


def length_bucket(length: int) -> str:
    if length <= 50:
        return "01_le_50"
    if length <= 200:
        return "02_51_200"
    if length <= 500:
        return "03_201_500"
    if length <= 2000:
        return "04_501_2000"
    return "05_gt_2000"


def _csv_limit() -> None:
    value = sys.maxsize
    while True:
        try:
            csv.field_size_limit(value)
            return
        except OverflowError:
            value //= 10


def select_pilot(
    input_path: Path,
    *,
    size: int,
    seed: int,
    excluded_quality: Counter[str] | None = None,
) -> list[Candidate]:
    if size < 1:
        raise ValueError("size must be positive")
    _csv_limit()
    # Keep ample candidates per category/length stratum, then round-robin the
    # strata. This pilot deliberately stresses rare/long cases; it is not a
    # population-weighted evaluation sample.
    per_stratum = max(32, math.ceil(size / 8))
    heaps: dict[tuple[str, str], list[tuple[int, int, Candidate]]] = defaultdict(list)
    seen_content: set[str] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t", strict=True)
        required = {"id", "case_content", "case_accord_type_one_name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"input TSV must contain {sorted(required)}")
        for source_row, row in enumerate(reader, start=1):
            source_id = (row.get("id") or "").strip()
            content = row.get("case_content") or ""
            if (
                not source_id
                or not content.strip()
                or content.strip().casefold() in {"null", "nan"}
            ):
                continue
            quality_issue = case_content_quality_issue(content)
            if quality_issue is not None:
                if excluded_quality is not None:
                    excluded_quality[quality_issue] += 1
                continue
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            category = (row.get("case_accord_type_one_name") or "").strip() or "__missing__"
            bucket = length_bucket(len(content))
            digest = hashlib.sha256(f"{seed}\0{source_id}\0{content_hash}".encode()).digest()
            rank = int.from_bytes(digest, "big")
            candidate = Candidate(
                rank=rank,
                source_id=source_id,
                source_row=source_row,
                case_content=content,
                content_sha256=content_hash,
                category1=category,
                length_bucket=bucket,
            )
            key = (category, bucket)
            heap = heaps[key]
            item = (-rank, source_row, candidate)
            if len(heap) < per_stratum:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    pools = {
        key: [item[2] for item in sorted(heap, key=lambda value: -value[0])]
        for key, heap in heaps.items()
    }
    selected: list[Candidate] = []
    keys = sorted(pools)
    index = 0
    while len(selected) < size:
        made_progress = False
        for key in keys:
            pool = pools[key]
            if index < len(pool):
                selected.append(pool[index])
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            break
        index += 1
    if len(selected) < size:
        raise ValueError(f"only {len(selected)} unique non-empty records are available")
    return selected


def write_pilot(path: Path, records: list[Candidate], *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            os.chmod(temporary, 0o600)
            for position, record in enumerate(records, start=1):
                value = {
                    "pilot_schema_version": PILOT_SCHEMA_VERSION,
                    "source_id": record.source_id,
                    "source_row": record.source_row,
                    "case_content": record.case_content,
                    "content_sha256": record.content_sha256,
                    "selection": {
                        "pilot_position": position,
                        "category1_stratum": record.category1,
                        "length_bucket": record.length_bucket,
                        "seed": seed,
                    },
                }
                target.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic Qwen3 extraction pilot")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {args.output}; use --overwrite")
    excluded_quality: Counter[str] = Counter()
    records = select_pilot(
        args.input,
        size=args.size,
        seed=args.seed,
        excluded_quality=excluded_quality,
    )
    write_pilot(args.output, records, seed=args.seed)
    counts = Counter((item.category1, item.length_bucket) for item in records)
    print(f"pilot_records={len(records)}")
    print(f"pilot_strata={len(counts)}")
    for code, count in sorted(excluded_quality.items()):
        print(f"excluded_quality_{code.lower()}={count}")
    print(f"pilot_output={args.output.resolve()}")


if __name__ == "__main__":
    main()
