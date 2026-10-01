from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

from .common import VERSION as DATASET_VERSION
from .common import file_hash, readonly, write_json

VERSION = "historical-address-index-v1"
PARSER_VERSION = "literal-address-v1"
ANCHORS = {"named_place", "road"}
_CJK = "\\u3400-\\u9fff"
_NAME = rf"[{_CJK}A-Za-z0-9·]{{1,24}}?"
_SUFFIXES = {
    "named_place": (
        r"小区|花园|花苑|新村|家园|公寓|广场|大厦|小学|中学|学校|医院|"
        r"卫生院|超市|商场|市场|园区|馨苑|体育馆|幼儿园|大学|政务服务中心"
    ),
    "road": r"大道|大街|胡同|路|街|巷",
    "city": "市",
    "district": r"区|县",
    "town": r"街道|镇|乡",
}
_PATTERNS = tuple(
    (kind, re.compile(_NAME + "(?:" + suffix + ")")) for kind, suffix in _SUFFIXES.items()
) + (
    (
        "building",
        re.compile(
            r"(?<![0-9A-Za-z])(?:第)?(?:[0-9]{1,6}|[一二三四五六七八九十百零〇两]{1,6})\s*(?:号楼|栋|幢)"
        ),
    ),
    (
        "house_number",
        re.compile(
            r"(?<![0-9A-Za-z])(?:[0-9]{1,6}|[一二三四五六七八九十百零〇两]{1,6})\s*号(?!\s*(?:楼|栋|幢|室|单元))"
        ),
    ),
)
_INTRO = re.compile(
    r"市民反映|市民投诉|投诉人反映|服务对象反映|居民反映|居住在|地址为|"
    r"反映|投诉|位于|地址|住在|咨询|请问|关于|发现|表示|希望|要求|建议"
)
_ADMIN_END = re.compile(r"市|(?<!小)(?<!园)(?<!社)区|县|镇|乡|街道")
_HARD_BREAK = re.compile(r"[，,。；;！？!?\n\r\u2028\u2029]")
_UNSUPPORTED_QUALIFIER = re.compile(
    r"(?<![0-9A-Za-z])(?:第)?(?:[A-Za-z0-9]+|[一二三四五六七八九十百零〇两]+)"
    r"\s*(?:单元|楼层|期|室|层|楼|座)"
)
_SMALL_ZONE = re.compile(
    r"^[ \t,，:：的]*(?:[A-Za-z0-9]+|[一二三四五六七八九十百零〇两]+|"
    r"东北|西北|东南|西南|东|西|南|北|中)\s*(?:区|组团|座)"
)
_GENERIC = {
    "这个小区",
    "那个小区",
    "所在小区",
    "附近小区",
    "同一小区",
    "居民小区",
    "住宅小区",
    "老旧小区",
    "某某小区",
    "某小区",
    "其他小区",
    "小区道路",
    "一条路",
    "这条路",
    "那条路",
    "某某路",
    "某条路",
    "马路",
    "公路",
    "道路",
    "人行道",
    "主干道",
    "附近学校",
    "附近医院",
    "附近超市",
    "附近市场",
}


def _implementation() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: file_hash(root / name) for name in ("address.py", "common.py")}


def _verified_dataset(dataset: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != DATASET_VERSION:
        raise ValueError("unsupported dataset version")
    if file_hash(dataset / "dataset.sqlite3") != manifest.get("artifacts", {}).get(
        "dataset.sqlite3"
    ):
        raise ValueError("dataset artifact changed: dataset.sqlite3")
    return manifest


def _normalized_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    parts, offsets = [], []
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", char)
        parts.append(normalized)
        offsets.extend([(index, index + 1)] * len(normalized))
    return "".join(parts), offsets


def _trim(raw: str, kind: str, *, connector: bool = False) -> int:
    start = 0
    intros = list(_INTRO.finditer(raw))
    if intros:
        start = intros[-1].end()
    if start < len(raw) and (
        (intros and raw[start] == "在") or (connector and not intros and raw[start] == "与")
    ):
        start += 1
    if kind in ANCHORS:
        endings = list(_ADMIN_END.finditer(raw[start:]))
        if endings:
            start += endings[-1].end()
        if kind == "named_place":
            road = list(re.finditer(r"大道|大街|胡同|路|街|巷", raw[start:]))
            if road and road[-1].end() < len(raw[start:]):
                start += road[-1].end()
    elif kind == "district":
        endings = list(re.finditer(r"市", raw[start:]))
        if endings:
            start += endings[-1].end()
    elif kind == "town":
        endings = list(_ADMIN_END.finditer(raw[start:-1]))
        if endings:
            start += endings[-1].end()
    return start


def parse_address(text: str) -> list[dict]:
    """Conservative suffix rules, not a geographic resolver or a full address parser."""
    if not isinstance(text, str):
        raise TypeError("address text must be a string")
    normalized, offsets = _normalized_with_offsets(text)
    entities, seen = [], set()
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(normalized):
            start, end = match.span()
            surface = match.group()
            if kind in ANCHORS or kind in {"city", "district", "town"}:
                # Trim only the prefix, preserving evidence as an exact source substring.
                suffix = re.search("(?:" + _SUFFIXES[kind] + ")$", surface)
                assert suffix is not None
                prefix = surface[: suffix.start()]
                connector = any(
                    entity["kind"] in ANCHORS and entity["end"] == offsets[start][0]
                    for entity in entities
                )
                trimmed = _trim(prefix, kind, connector=connector)
                if (
                    kind == "named_place"
                    and surface[suffix.start() :] == "政务服务中心"
                    and trimmed == len(prefix)
                ):
                    # A stated district is part of this literal institution name, not inferred.
                    cities = list(re.finditer("市", prefix))
                    trimmed = cities[-1].end() if cities else 0
                start += trimmed
                surface = normalized[start:end]
                stem_length = suffix.start() - (start - match.start())
                minimum = 2 if kind in ANCHORS else 1
                if stem_length < minimum or stem_length > 12:
                    continue
                if kind == "district" and surface.endswith(("小区", "园区", "社区")):
                    continue
                if surface in _GENERIC:
                    continue
                if kind == "road" and surface.endswith("道路"):
                    continue
            key = kind, start, end
            if key in seen:
                continue
            seen.add(key)
            original_start, original_end = offsets[start][0], offsets[end - 1][1]
            entities.append(
                {
                    "kind": kind,
                    "text": text[original_start:original_end],
                    "normalized": re.sub(r"\s+", "", surface),
                    "start": original_start,
                    "end": original_end,
                    "address_group": None,
                }
            )
    entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["kind"]))
    anchors = [entity for entity in entities if entity["kind"] in ANCHORS]
    groups: list[list[dict]] = []
    for anchor in anchors:
        if groups:
            previous = groups[-1][-1]
            gap = text[previous["end"] : anchor["start"]]
            overlap = anchor["start"] < previous["end"]
            adjacent_parts = (
                previous["kind"] != anchor["kind"]
                and anchor["start"] - previous["end"] <= 16
                and not _HARD_BREAK.search(gap)
            )
            if overlap or adjacent_parts:
                groups[-1].append(anchor)
                anchor["address_group"] = len(groups) - 1
                continue
        groups.append([anchor])
        anchor["address_group"] = len(groups) - 1
    for entity in entities:
        if entity["kind"] in ANCHORS:
            continue
        if entity["kind"] in {"building", "house_number"}:
            prior = [anchor for anchor in anchors if anchor["end"] <= entity["start"]]
            if prior:
                anchor = prior[-1]
                gap = text[anchor["end"] : entity["start"]]
                if len(gap) <= 20 and re.fullmatch(r"[ \t,，:：的]*", gap):
                    entity["address_group"] = anchor["address_group"]
        else:
            enclosing = [
                anchor
                for anchor in anchors
                if anchor["start"] <= entity["start"] and anchor["end"] >= entity["end"]
            ]
            if enclosing:
                entity["address_group"] = enclosing[0]["address_group"]
                continue
            following = [anchor for anchor in anchors if anchor["start"] >= entity["end"]]
            if following:
                anchor = following[0]
                gap = text[entity["end"] : anchor["start"]]
                if len(gap) <= 24 and not _HARD_BREAK.search(gap):
                    entity["address_group"] = anchor["address_group"]
    return entities


def build_address_index(dataset: Path, output: Path) -> dict:
    implementation = _implementation()
    dataset, output = Path(dataset), Path(output)
    manifest = _verified_dataset(dataset)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.perf_counter()
    source = readonly(dataset / "dataset.sqlite3")
    target = sqlite3.connect(output / "index.sqlite3")
    count = entity_count = cases_with_addresses = 0
    try:
        target.executescript("""
            PRAGMA user_version=1;
            CREATE TABLE cases(docid INTEGER PRIMARY KEY, source_id TEXT UNIQUE NOT NULL,
                               source_row INTEGER NOT NULL);
            CREATE TABLE entities(docid INTEGER NOT NULL, address_group INTEGER NOT NULL,
                                  kind TEXT NOT NULL, normalized TEXT NOT NULL,
                                  text TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL);
            CREATE INDEX entity_lookup ON entities(kind,normalized,docid,address_group);
            CREATE INDEX entity_document_lookup ON entities(docid,address_group);
        """)
        for rid, sid, row, content in source.execute(
            "SELECT r.rid,r.source_id,r.source_row,r.content FROM corpus c "
            "JOIN records r ON r.rid=c.rid ORDER BY r.rid"
        ):
            target.execute("INSERT INTO cases VALUES(?,?,?)", (rid, sid, row))
            entities = [
                entity for entity in parse_address(content) if entity["address_group"] is not None
            ]
            target.executemany(
                "INSERT INTO entities VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        rid,
                        entity["address_group"],
                        entity["kind"],
                        entity["normalized"],
                        entity["text"],
                        entity["start"],
                        entity["end"],
                    )
                    for entity in entities
                ],
            )
            count += 1
            entity_count += len(entities)
            cases_with_addresses += int(any(entity["kind"] in ANCHORS for entity in entities))
            if count % 25000 == 0:
                target.commit()
                print(f"address_indexed_cases={count}", flush=True)
        if not count:
            raise ValueError("historical corpus must be nonempty")
        target.commit()
    finally:
        source.close()
        target.close()
    if _implementation() != implementation:
        raise ValueError(
            "address implementation changed during index build; rebuild in a new directory"
        )
    result = {
        "version": VERSION,
        "parser_version": PARSER_VERSION,
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "source_input_sha256": manifest["input_sha256"],
        "index_sha256": file_hash(output / "index.sqlite3"),
        "implementation_sha256": implementation,
        "indexed_cases": count,
        "cases_with_addresses": cases_with_addresses,
        "indexed_entities": entity_count,
        "seconds": time.perf_counter() - started,
        "limitations": [
            "Rule-based literal surfaces from historical case_content only; "
            "extraction may miss addresses.",
            "Same name does not verify the same geographic location; "
            "no coordinates or proximity claims.",
            "No inferred administrative divisions, aliases, fuzzy spelling, "
            "or building-number equivalence.",
            "Explicit query address parts must occur in one local address group; "
            "missing parts do not match.",
            "Province-level parsing and phase/block/unit/room/floor qualifiers are unsupported; "
            "use a coarser named-place or road query.",
        ],
    }
    write_json(output / "manifest.json", result)
    return result


class AddressSearcher:
    def __init__(self, dataset: Path, index: Path):
        dataset, index = Path(dataset), Path(index)
        _verified_dataset(dataset)
        self.manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
        if (
            self.manifest.get("version") != VERSION
            or self.manifest.get("parser_version") != PARSER_VERSION
            or self.manifest.get("dataset_manifest_sha256") != file_hash(dataset / "manifest.json")
            or self.manifest.get("index_sha256") != file_hash(index / "index.sqlite3")
            or self.manifest.get("implementation_sha256") != _implementation()
        ):
            raise ValueError("address index provenance or hash mismatch")
        self.db = sqlite3.connect(
            (index / "index.sqlite3").resolve().as_uri() + "?mode=ro&immutable=1", uri=True
        )
        try:
            expected = {
                "cases": ["docid", "source_id", "source_row"],
                "entities": [
                    "docid",
                    "address_group",
                    "kind",
                    "normalized",
                    "text",
                    "start",
                    "end",
                ],
            }
            if self.db.execute("PRAGMA user_version").fetchone()[0] != 1 or any(
                [row[1] for row in self.db.execute(f"PRAGMA table_info({table})")] != names
                for table, names in expected.items()
            ):
                raise ValueError("unsupported address index schema")
            if self.db.execute("SELECT COUNT(*) FROM cases").fetchone()[0] != self.manifest.get(
                "indexed_cases"
            ):
                raise ValueError("address index case count mismatch")
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> AddressSearcher:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _matching_groups(self, entities: list[dict]) -> set[tuple[int, int]]:
        matches = None
        for kind, normalized in sorted(
            {(entity["kind"], entity["normalized"]) for entity in entities}
        ):
            current = set(
                self.db.execute(
                    "SELECT DISTINCT docid,address_group FROM entities "
                    "WHERE kind=? AND normalized=?",
                    (kind, normalized),
                )
            )
            matches = current if matches is None else matches & current
            if not matches:
                break
        return matches or set()

    def search(self, query: str, *, limit: int = 50, allow_broader: bool = False) -> list[dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("address result limit must be a positive integer")
        if not isinstance(allow_broader, bool):
            raise ValueError("allow_broader must be boolean")
        if not isinstance(query, str):
            raise TypeError("address query must be a string")
        if not query.strip():
            return []
        parsed = parse_address(query)
        anchors = [entity for entity in parsed if entity["kind"] in ANCHORS]
        if not anchors:
            raise ValueError(
                "address query needs a recognizable named place or road; "
                "use an explicit address override"
            )
        normalized_query = unicodedata.normalize("NFKC", query)
        if _UNSUPPORTED_QUALIFIER.search(normalized_query) or any(
            _SMALL_ZONE.match(unicodedata.normalize("NFKC", query[entity["end"] :]))
            for entity in anchors
        ):
            raise ValueError(
                "phase/block/unit/room/floor address qualifier is unsupported; "
                "use a coarser named-place/road query or explicit address override"
            )
        building_values = {
            entity["normalized"] for entity in parsed if entity["kind"] == "building"
        }
        if len(building_values) > 1:
            raise ValueError(
                "multiple building values in query; specify one building "
                "or use a coarser named-place query"
            )
        groups = {entity["address_group"] for entity in anchors}
        if len(groups) != 1:
            raise ValueError(
                "multiple address groups in query; provide one explicit address override"
            )
        group = next(iter(groups))
        if any(
            entity["kind"] not in ANCHORS and entity["address_group"] != group for entity in parsed
        ):
            raise ValueError(
                "address qualifier cannot be linked safely; provide one explicit address override "
                "or use a coarser named-place/road query"
            )
        required = [entity for entity in parsed if entity["address_group"] == group]
        matches = self._matching_groups(required)
        broader = False
        relaxed = []
        if not matches and allow_broader:
            relaxed = [
                entity for entity in required if entity["kind"] in {"building", "house_number"}
            ]
            required = [
                entity for entity in required if entity["kind"] not in {"building", "house_number"}
            ]
            matches = self._matching_groups(required) if relaxed else set()
            broader = bool(matches)
        if not matches:
            return []
        doc_groups: dict[int, set[int]] = {}
        for docid, address_group in matches:
            doc_groups.setdefault(docid, set()).add(address_group)
        documents = []
        docids = sorted(doc_groups)
        for offset in range(0, len(docids), 500):
            batch = docids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            documents.extend(
                self.db.execute(
                    f"SELECT docid,source_id,source_row FROM cases WHERE docid IN ({placeholders})",
                    batch,
                )
            )
        documents.sort(key=lambda row: row[1])
        has_place = any(entity["kind"] == "named_place" for entity in required)
        has_building = any(entity["kind"] == "building" for entity in required)
        has_house = any(entity["kind"] == "house_number" for entity in required)
        scope = "named_place" if has_place else "road_surface"
        if has_building:
            scope = "named_place_and_building" if has_place else "road_and_building"
        elif has_house:
            scope = "named_place_and_house_number" if has_place else "road_and_house_number"
        if broader:
            scope = "broader_" + scope
        result = []
        for rank, (docid, sid, row) in enumerate(documents[:limit], 1):
            matched_group = min(doc_groups[docid])
            source_entities = self.db.execute(
                "SELECT kind,normalized,text,start,end FROM entities "
                "WHERE docid=? AND address_group=?",
                (docid, matched_group),
            ).fetchall()
            evidence = []
            for entity in required:
                matching = [
                    item
                    for item in source_entities
                    if item[:2] == (entity["kind"], entity["normalized"])
                ]
                item = min(matching, key=lambda item: (item[3], item[4]))
                evidence.append(
                    {
                        "kind": item[0],
                        "normalized": item[1],
                        "query_text": entity["text"],
                        "text": item[2],
                        "start": item[3],
                        "end": item[4],
                        "source_field": "case_content",
                    }
                )
            result.append(
                {
                    "source_id": sid,
                    "source_row": row,
                    "rank": rank,
                    "address_match": {
                        "scope": scope,
                        "broader_match": broader,
                        "geographic_identity": "not_verified",
                        "source_field": "case_content",
                        "query_entities": parsed,
                        "matched_evidence": evidence,
                        "relaxed_query_entities": relaxed,
                    },
                }
            )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CPU literal-address index from historical case_content"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_address_index(args.dataset, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
