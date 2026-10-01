from __future__ import annotations

import json
import os
import sqlite3
import stat

import pytest

import retrieval_baseline.address as address_module
from retrieval_baseline.address import AddressSearcher, build_address_index, parse_address
from retrieval_baseline.common import file_hash
from retrieval_baseline.dataset import build_dataset
from retrieval_baseline.tests.test_baseline import record, ref, write_source


@pytest.fixture
def address_fixture(tmp_path):
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "address")]
    texts = {
        "old-1": "常州市武进区幸福小区3号楼，垃圾清运不及时。",
        "old-2": "常州市武进区幸福小区23号楼漏水。",
        "old-3": "常州市武进区幸福花园3号楼，保洁不到位。",
        "old-4": "常州市天宁区劳动东路3号有积水。",
        "old-5": "常州市天宁区劳动东路23号，路灯损坏。",
        "old-6": "幸福小区4号楼有噪音；幸福花园3号楼有积水。",
        "old-7": "南京市鼓楼区幸福小区3号楼有垃圾。",
        "old-8": "市民反映常州市武进区幸福小区 ３号楼卫生差。",
        "old-9": "楼道保洁不及时，要求清理垃圾。",
    }
    rows = [
        record(sid, text, refs=[ref("clean", "不会用于地址检索的幸福新村")])
        for sid, text in texts.items()
    ]
    rows += [
        record("future", "未来花园3号楼有积水", date="2026-01-01 10:00:00"),
        record("future-test", "未来小区3号楼有垃圾", date="2026-02-01 10:00:00"),
        record("duplicate", texts["old-1"]),
    ]
    write_source(source, rows)
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    return dataset, index, texts


def test_parser_preserves_normalized_literal_evidence_and_numeric_boundaries():
    text = "市民反映常州市武进区幸福小区 ３号楼；劳动东路23号。"
    entities = parse_address(text)
    assert {(item["kind"], item["normalized"]) for item in entities} >= {
        ("city", "常州市"),
        ("district", "武进区"),
        ("named_place", "幸福小区"),
        ("building", "3号楼"),
        ("road", "劳动东路"),
        ("house_number", "23号"),
    }
    assert not any(
        item["normalized"] in {"3号", "3号楼"} and item["start"] > text.index("劳动东路")
        for item in entities
    )
    assert all(text[item["start"] : item["end"]] == item["text"] for item in entities)
    assert parse_address("楼道长期没人打扫，公交车疲劳驾驶") == []
    assert parse_address("道路积水，要求修复道路") == []
    assert parse_address("") == []


def test_index_only_historical_case_content_and_exact_buildings(address_fixture):
    dataset, index, texts = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        hits = searcher.search("幸福小区3号楼", limit=200)
        assert [hit["source_id"] for hit in hits] == ["old-1", "old-7", "old-8"]
        assert [hit["rank"] for hit in hits] == [1, 2, 3]
        assert [hit["source_id"] for hit in searcher.search("常州市武进区幸福小区3号楼")] == [
            "old-1",
            "old-8",
        ]
        assert [hit["source_id"] for hit in searcher.search("劳动东路3号")] == ["old-4"]
        assert [hit["source_id"] for hit in searcher.search("劳动东路23号")] == ["old-5"]
        assert searcher.search("未来花园3号楼") == []
        assert searcher.search("未来小区3号楼") == []
        assert searcher.search("幸福新村") == []  # Present only in knowledge title metadata.
        assert searcher.search("不存在花园") == []
        assert searcher.search("") == []
        for hit in hits:
            assert hit["address_match"]["scope"] == "named_place_and_building"
            assert hit["address_match"]["geographic_identity"] == "not_verified"
            assert hit["address_match"]["broader_match"] is False
            for evidence in hit["address_match"]["matched_evidence"]:
                assert evidence["source_field"] == "case_content"
                assert (
                    texts[hit["source_id"]][evidence["start"] : evidence["end"]] == evidence["text"]
                )


def test_evidence_lookup_uses_document_index_instead_of_corpus_scan(address_fixture):
    _, index, _ = address_fixture
    with sqlite3.connect(index / "index.sqlite3") as db:
        plan = db.execute(
            "EXPLAIN QUERY PLAN SELECT kind,normalized,text,start,end FROM entities "
            "WHERE docid=? AND address_group=?", (1, 0),
        ).fetchall()
    assert any("entity_document_lookup" in row[3] for row in plan)


def test_query_requires_one_named_address_and_never_cross_matches_source_groups(address_fixture):
    dataset, index, _ = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        for text in ("3号楼", "常州市", "楼道没人打扫"):
            with pytest.raises(ValueError, match="named place or road"):
                searcher.search(text)
        with pytest.raises(ValueError, match="multiple address"):
            searcher.search("幸福小区3号楼；幸福花园3号楼")
        with pytest.raises(ValueError, match="qualifier"):
            searcher.search("3号楼幸福小区")
        with pytest.raises(ValueError, match="qualifier"):
            searcher.search("幸福小区卫生差，其他位置3号楼")
        with pytest.raises(ValueError, match="qualifier"):
            searcher.search("幸福小区 南京市鼓楼区")
        assert [hit["source_id"] for hit in searcher.search("幸福小区，3号楼")] == [
            "old-1",
            "old-7",
            "old-8",
        ]
        assert "old-6" not in {hit["source_id"] for hit in searcher.search("幸福小区3号楼")}
        assert "old-6" in {hit["source_id"] for hit in searcher.search("幸福小区4号楼")}


def test_broader_matching_is_explicit_fallback_only(address_fixture):
    dataset, index, _ = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        assert searcher.search("幸福小区99号楼") == []
        broader = searcher.search("幸福小区99号楼", allow_broader=True)
        assert [hit["source_id"] for hit in broader] == [
            "old-1",
            "old-2",
            "old-6",
            "old-7",
            "old-8",
        ]
        for hit in broader:
            match = hit["address_match"]
            assert match["broader_match"] is True
            assert match["scope"] == "broader_named_place"
            assert [item["normalized"] for item in match["relaxed_query_entities"]] == ["99号楼"]
            assert all(item["kind"] == "named_place" for item in match["matched_evidence"])
        assert searcher.search("幸福小区3号楼", allow_broader=True) == searcher.search(
            "幸福小区3号楼"
        )
        assert searcher.search("不存在小区99号楼", allow_broader=True) == []


def test_road_and_place_parts_need_same_local_group(tmp_path):
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "index")]
    write_source(
        source,
        [
            record("one", "常州市武进区人民路幸福小区3号楼漏水", refs=[ref("a", "维修")]),
            record("two", "人民路有积水；幸福小区3号楼有垃圾。"),
        ],
    )
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    with AddressSearcher(dataset, index) as searcher:
        assert [hit["source_id"] for hit in searcher.search("人民路幸福小区3号楼")] == ["one"]


def test_data_specific_place_suffixes_preserve_source_and_named_identity(tmp_path):
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "index")]
    places = [
        "花语馨苑",
        "凯旋商务广场",
        "中天钢铁体育馆",
        "天宁区政务服务中心",
        "幸福幼儿园",
        "常州大学",
    ]
    write_source(
        source,
        [
            record(str(number), "市民反映常州市" + place + "存在问题", refs=[ref("a", "办理")])
            for number, place in enumerate(places)
        ],
    )
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    with AddressSearcher(dataset, index) as searcher:
        for number, place in enumerate(places):
            parsed = parse_address(place)
            assert [item["normalized"] for item in parsed if item["kind"] == "named_place"] == [
                place
            ]
            hits = searcher.search(place)
            assert [hit["source_id"] for hit in hits] == [str(number)]
            evidence = hits[0]["address_match"]["matched_evidence"]
            assert any(item["text"] == place for item in evidence)
        assert [hit["source_id"] for hit in searcher.search("常州市天宁区政务服务中心")] == ["3"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_address_output_directory_is_private_without_changing_existing_paths(address_fixture):
    dataset, index, _ = address_fixture
    assert stat.S_IMODE(index.stat().st_mode) == 0o700
    index.chmod(0o750)
    with pytest.raises(FileExistsError):
        build_address_index(dataset, index)
    assert stat.S_IMODE(index.stat().st_mode) == 0o750


def test_provenance_and_schema_fail_closed(address_fixture):
    dataset, index, _ = address_fixture
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with sqlite3.connect(index / "index.sqlite3") as db:
        db.execute("ALTER TABLE entities ADD COLUMN unexpected TEXT")
    with pytest.raises(ValueError, match="provenance or hash"):
        AddressSearcher(dataset, index)
    manifest["index_sha256"] = file_hash(index / "index.sqlite3")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        AddressSearcher(dataset, index)


def test_dataset_and_implementation_fingerprint_checked(address_fixture):
    dataset, index, _ = address_fixture
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_sha256"]["address.py"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance or hash"):
        AddressSearcher(dataset, index)
    with (dataset / "dataset.sqlite3").open("ab") as target:
        target.write(b"tampered")
    with pytest.raises(ValueError, match="dataset artifact changed"):
        AddressSearcher(dataset, index)


def test_bad_limits_rejected_and_build_refuses_overwrite(address_fixture):
    dataset, index, _ = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        for limit in (0, -1, 2.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                searcher.search("幸福小区", limit=limit)
        with pytest.raises(ValueError, match="boolean"):
            searcher.search("幸福小区", allow_broader="yes")
    with pytest.raises(FileExistsError):
        build_address_index(dataset, index)


def test_legitimate_names_keep_leading_characters_and_context_connectors(tmp_path):
    for text in ("和平路", "和丰花园"):
        for prefix in ("", "市民反映", "关于"):
            entities = [
                item
                for item in parse_address(prefix + text)
                if item["kind"] in {"road", "named_place"}
            ]
            assert [item["normalized"] for item in entities] == [text]
    entities = parse_address("市民反映在和丰花园3号楼")
    assert any(item["normalized"] == "和丰花园" for item in entities)
    entities = parse_address("人民路与和平路")
    assert [item["normalized"] for item in entities if item["kind"] == "road"] == [
        "人民路",
        "和平路",
    ]
    entities = parse_address("劳动东路和平路")
    assert [item["normalized"] for item in entities if item["kind"] == "road"] == [
        "劳动东路",
        "和平路",
    ]
    entities = parse_address("劳动东路和丰花园")
    assert [
        (item["kind"], item["normalized"])
        for item in entities
        if item["kind"] in {"road", "named_place"}
    ] == [
        ("road", "劳动东路"),
        ("named_place", "和丰花园"),
    ]
    entities = parse_address("幸福小区和丰花园")
    assert [item["normalized"] for item in entities if item["kind"] == "named_place"] == [
        "幸福小区",
        "和丰花园",
    ]
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "index")]
    write_source(
        source,
        [
            record("road", "和平路3号积水", refs=[ref("a", "排水")]),
            record("road-intro", "市民反映和平路4号积水"),
            record("place", "和丰花园3号楼漏水"),
            record("place-intro", "市民反映和丰花园4号楼漏水"),
        ],
    )
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    with AddressSearcher(dataset, index) as searcher:
        assert [item["source_id"] for item in searcher.search("和平路3号")] == ["road"]
        assert [item["source_id"] for item in searcher.search("和丰花园3号楼")] == ["place"]
        assert [item["source_id"] for item in searcher.search("和平路4号")] == ["road-intro"]
        assert [item["source_id"] for item in searcher.search("和丰花园4号楼")] == ["place-intro"]
        with pytest.raises(ValueError, match="multiple address"):
            searcher.search("劳动东路和平路")
        with pytest.raises(ValueError, match="multiple address"):
            searcher.search("幸福小区和丰花园")


@pytest.mark.parametrize(
    "query",
    [
        "常州市武进区幸福小区3号楼2单元401室",
        "幸福小区1期3号楼",
        "幸福小区一期3号楼",
        "幸福小区A区3号楼",
        "幸福小区北区3号楼",
        "幸福小区3号楼5层",
        "幸福小区3楼",
        "幸福小区Ａ区３号楼",
        "幸福小区A座",
    ],
)
def test_unsupported_address_qualifiers_fail_closed_with_coarser_query_hint(address_fixture, query):
    dataset, index, _ = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        with pytest.raises(ValueError, match="unsupported.*coarser"):
            searcher.search(query)
        with pytest.raises(ValueError, match="unsupported.*coarser"):
            searcher.search(query, allow_broader=True)


def test_multiple_building_queries_require_explicit_selection(address_fixture):
    dataset, index, _ = address_fixture
    with AddressSearcher(dataset, index) as searcher:
        with pytest.raises(ValueError, match="multiple building"):
            searcher.search("幸福小区3号楼与4号楼")


def test_numeric_qualifiers_never_attach_through_narration_or_unicode_paragraphs(tmp_path):
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "index")]
    texts = {
        "safe": "幸福小区的3号楼漏水",
        "safe-format": "幸福小区，3号楼漏水",
        "unsafe": "幸福小区隔壁的3号楼漏水",
        "unsafe-other": "幸福小区附近另一小区的3号楼漏水",
        "line": "幸福小区\u20283号楼漏水",
        "paragraph": "幸福小区\u20293号楼漏水",
    }
    write_source(
        source, [record(sid, text, refs=[ref("a", "维修")]) for sid, text in texts.items()]
    )
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    with AddressSearcher(dataset, index) as searcher:
        assert [item["source_id"] for item in searcher.search("幸福小区3号楼")] == [
            "safe",
            "safe-format",
        ]
        for text in ("幸福小区隔壁的3号楼", "幸福小区\u20283号楼", "幸福小区\u20293号楼"):
            with pytest.raises(ValueError, match="qualifier.*coarser"):
                searcher.search(text)


def test_changed_implementation_during_build_is_never_blessed(address_fixture, monkeypatch):
    dataset, index, _ = address_fixture
    initial = address_module._implementation()
    changed = {**initial, "address.py": "0" * 64}
    snapshots = iter([initial, changed])
    monkeypatch.setattr(address_module, "_implementation", lambda: next(snapshots))
    output = index.parent / "changed-build"
    with pytest.raises(ValueError, match="implementation changed during index build"):
        build_address_index(dataset, output)
    assert not (output / "manifest.json").exists()


def test_short_explicit_administrative_names_are_literal_same_group_constraints(tmp_path):
    source, dataset, index = [tmp_path / name for name in ("source.tsv", "dataset", "index")]
    texts = {
        "match": "郊区新镇幸福小区3号楼漏水",
        "other-district": "城区新镇幸福小区3号楼漏水",
        "other-town": "郊区老镇幸福小区3号楼漏水",
        "missing-admin": "幸福小区3号楼漏水",
        "different-group": "郊区新镇和平路积水；城区老镇幸福小区3号楼漏水",
    }
    write_source(
        source, [record(sid, text, refs=[ref("a", "维修")]) for sid, text in texts.items()]
    )
    build_dataset(source, dataset)
    build_address_index(dataset, index)
    query = "郊区新镇幸福小区3号楼"
    parsed = parse_address(query)
    assert {(item["kind"], item["normalized"]) for item in parsed} >= {
        ("district", "郊区"),
        ("town", "新镇"),
    }
    with AddressSearcher(dataset, index) as searcher:
        assert [item["source_id"] for item in searcher.search(query)] == ["match"]
        hits = searcher.search("郊区幸福小区3号楼")
        assert [item["source_id"] for item in hits] == ["match", "other-town"]
        hits = searcher.search("新镇幸福小区3号楼")
        assert [item["source_id"] for item in hits] == ["match", "other-district"]
        assert searcher.search("郊区不存在镇幸福小区3号楼") == []
        evidence = searcher.search(query)[0]["address_match"]["matched_evidence"]
        assert {(item["kind"], item["text"]) for item in evidence} >= {
            ("district", "郊区"),
            ("town", "新镇"),
        }
