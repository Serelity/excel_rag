"""Retrieve historical complaint cases, without knowledge voting or query labels."""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from pathlib import Path

from .address import AddressSearcher, parse_address
from .common import VERSION, file_hash, readonly, tokens
from .hybrid import fuse_cases
from .lexical import TOKENIZER, match_query, verified_manifest

SEARCH_VERSION = "case-search-v1"


def _chunks(values: list, size: int = 500):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _write_private_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


class CaseSearcher:
    """Keep verified data, optional model and indexes resident across queries."""

    def __init__(
        self, dataset: Path, lexical_index: Path, *, dense_index: Path | None = None,
        encoder=None, encoder_factory=None, address_index: Path | None = None, threads: int = 4,
    ):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be positive")
        self.dataset, self.lexical_index = dataset, lexical_index
        self.dense_index, self.encoder, self.threads = dense_index, encoder, threads
        self.encoder_factory = encoder_factory
        manifest = verified_manifest(dataset, ("dataset.sqlite3",))
        self.corpus_count = manifest["counts"]["corpus_unique_texts"]
        self.dataset_hash = file_hash(dataset / "manifest.json")
        self.lexical_hash = file_hash(lexical_index / "manifest.json")
        lexical = json.loads((lexical_index / "manifest.json").read_text(encoding="utf-8"))
        if (
            lexical.get("version") != VERSION or lexical.get("tokenizer") != TOKENIZER
            or lexical.get("dataset_manifest_sha256") != self.dataset_hash
            or lexical.get("index_sha256") != file_hash(lexical_index / "index.sqlite3")
            or lexical.get("indexed_cases") != self.corpus_count
        ):
            raise ValueError("Lexical index provenance/hash mismatch")
        self.source = readonly(dataset / "dataset.sqlite3")
        try:
            self.db = readonly(lexical_index / "index.sqlite3")
            self.db.execute("PRAGMA temp_store=MEMORY")
            self.db.execute("CREATE TEMP TABLE candidate_scope(docid INTEGER PRIMARY KEY)")
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.source.close()
            raise
        self.address_index = address_index
        self.address_searcher = None
        self.index = None
        self.dense_manifest = None
        self.dense_hash = self.address_hash = None
        self.implementation = {name: file_hash(Path(__file__).with_name(name)) for name in (
            "search.py", "address.py", "hybrid.py", "lexical.py", "common.py",
        )}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if self.address_searcher is not None:
            self.address_searcher.close()
            self.address_searcher = None
        self.source.close()
        self.db.close()

    def _address(self) -> AddressSearcher:
        if self.address_searcher is None:
            if self.address_index is None:
                raise ValueError("Address/combined mode requires a prepared --address-index")
            self.address_searcher = AddressSearcher(self.dataset, self.address_index)
            self.address_hash = file_hash(self.address_index / "manifest.json")
        return self.address_searcher

    def _load_dense(self) -> None:
        if self.index is not None:
            return
        if self.dense_index is None or (self.encoder is None and self.encoder_factory is None):
            raise ValueError("Dense/hybrid retrieval requires an existing dense index and encoder")
        import faiss

        from .dense import DENSE_VERSION, implementation_hashes

        manifest = json.loads((self.dense_index / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        if self.encoder is None:
            self.encoder = self.encoder_factory()
        # Batch size and device may change without changing the representation profile.
        runtime_keys = {"batch_size", "device"}
        recorded = {k: v for k, v in config["encoder"].items() if k not in runtime_keys}
        actual = {k: v for k, v in self.encoder.identity.items() if k not in runtime_keys}
        if (
            manifest.get("version") != DENSE_VERSION or recorded != actual
            or config.get("implementation_sha256") != implementation_hashes()
            or config.get("dataset_manifest_sha256") != self.dataset_hash
            or config.get("input_field") != "case_content"
            or config.get("faiss_version") != faiss.__version__
            or manifest.get("indexed_cases") != self.corpus_count
            or manifest.get("index_sha256") != file_hash(self.dense_index / "index.faiss")
        ):
            raise ValueError("Dense index provenance/model/settings/hash mismatch")
        faiss.omp_set_num_threads(self.threads)
        index = faiss.read_index(str(self.dense_index / "index.faiss"))
        if (
            index.ntotal != self.corpus_count or index.d != self.encoder.identity["dimension"]
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or not isinstance(index, faiss.IndexIDMap2)
        ):
            raise ValueError("Dense index shape/metric/type mismatch")
        actual_ids = faiss.vector_to_array(index.id_map).tolist()
        expected_ids = [row[0] for row in self.source.execute(
            "SELECT rid FROM corpus ORDER BY rid",
        )]
        if actual_ids != expected_ids:
            raise ValueError("Dense document IDs differ from the frozen historical corpus")
        self.index, self.dense_manifest = index, manifest
        self.dense_hash = file_hash(self.dense_index / "manifest.json")

    def _metadata(self, *, source_ids: list[str] | None = None,
                  docids: list[int] | None = None) -> dict[str, tuple]:
        if (source_ids is None) == (docids is None):
            raise ValueError("Specify either source IDs or document IDs")
        values = source_ids if source_ids is not None else docids
        field = "source_id" if source_ids is not None else "rid"
        rows = []
        for batch in _chunks(values):
            marks = ",".join("?" for _ in batch)
            # CTAS gives corpus.rid no affinity; unary + keeps its integer lookup index usable.
            rows.extend(self.source.execute(
                "SELECT r.rid,r.source_id,r.source_row,r.group_id FROM records r "
                f"JOIN corpus c ON c.rid=+r.rid WHERE r.{field} IN ({marks})", batch,
            ).fetchall())
        if len(rows) != len(values):
            raise ValueError("Candidates must belong to the frozen historical corpus")
        metadata = {row[1]: row for row in rows}
        indexed = []
        for batch in _chunks([row[0] for row in rows]):
            marks = ",".join("?" for _ in batch)
            indexed.extend(self.db.execute(
                f"SELECT docid,source_id,source_row,group_id FROM cases WHERE docid IN ({marks})",
                batch,
            ).fetchall())
        if {row[1]: row for row in indexed} != metadata:
            raise ValueError("Case metadata differs from the frozen historical corpus")
        return metadata

    def _lexical(self, query: str, case_k: int, max_terms: int,
                 allowed: dict[str, tuple] | None) -> list[dict]:
        expression = match_query(self.db, query, "case", max_terms)
        if not expression:
            return []
        join = ""
        if allowed is not None:
            self.db.execute("DELETE FROM candidate_scope")
            self.db.executemany("INSERT INTO candidate_scope VALUES(?)",
                                ((row[0],) for row in allowed.values()))
            join = " JOIN candidate_scope s ON s.docid=c.docid "
        rows = self.db.execute(
            "SELECT c.docid,c.source_id,c.source_row,bm25(case_fts) FROM case_fts "
            "JOIN cases c ON c.docid=case_fts.rowid " + join
            + "WHERE case_fts MATCH ? ORDER BY bm25(case_fts),c.source_id LIMIT ?",
            (expression, case_k),
        ).fetchall()
        return [{"source_id": sid, "source_row": row, "rank": rank, "score": score}
                for rank, (_, sid, row, score) in enumerate(rows, 1)]

    def _dense(self, query: str, case_k: int,
               allowed: dict[str, tuple] | None) -> tuple[list[dict], dict]:
        import numpy as np

        from .encoder import normalized_vectors

        self._load_dense()
        vectors, stats = self.encoder.encode([query])
        vector = normalized_vectors(vectors, 1, self.index.d)
        if allowed is None:
            scores, ids = self.index.search(vector, min(case_k, self.index.ntotal))
            metadata = self._metadata(docids=[int(rid) for rid in ids[0]])
            by_docid = {row[0]: row for row in metadata.values()}
            scored = [(by_docid[int(rid)], float(score))
                      for rid, score in zip(ids[0], scores[0], strict=True)]
        else:
            # Score every address-qualified case, not just matches in a global Top50.
            # Existing vectors are reconstructed in bounded batches; no corpus re-encoding.
            scored = []
            for batch in _chunks(list(allowed.values()), size=1024):
                ids = np.array([row[0] for row in batch], dtype=np.int64)
                stored = self.index.reconstruct_batch(ids)
                scores = stored @ vector[0]
                scored.extend(zip(batch, map(float, scores), strict=True))
                scored = heapq.nsmallest(case_k, scored, key=lambda item: (-item[1], item[0][1]))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0][1]))
        return ([{"source_id": row[1], "source_row": row[2], "rank": rank, "score": score}
                 for rank, (row, score) in enumerate(ranked[:case_k], 1)], stats)

    def _records(self, hits: list[dict], traces: dict[str, dict],
                 address_hits: dict[str, dict]) -> list[dict]:
        metadata = self._metadata(source_ids=[hit["source_id"] for hit in hits])
        rows = {}
        for batch in _chunks([row[0] for row in metadata.values()]):
            marks = ",".join("?" for _ in batch)
            rows.update({row[0]: row[1:] for row in self.source.execute(
                f"SELECT source_id,content,call_time FROM records WHERE rid IN ({marks})", batch,
            )})
        results = []
        for rank, hit in enumerate(hits, 1):
            sid = hit["source_id"]
            if metadata[sid][2] != hit["source_row"]:
                raise ValueError("Candidate source_row differs from the historical corpus")
            content, call_time = rows[sid]
            matching = {"routes": traces.get(sid, {}), "address": None}
            if "rrf_score" in hit:
                matching["rrf_score"] = hit["rrf_score"]
            if sid in address_hits:
                matching["address"] = address_hits[sid]["address_match"]
            results.append({
                "source_id": sid, "source_row": hit["source_row"], "rank": rank,
                "case_content": content, "call_time": call_time, "matching": matching,
            })
        return results

    def search(
        self, query: str, *, mode: str = "problem", retriever: str = "hybrid",
        address: str | None = None, top_k: int = 10, case_k: int = 50,
        max_terms: int = 32, allow_broader: bool = False,
    ) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        if mode not in {"problem", "address", "combined"} or retriever not in {
            "bm25", "dense", "hybrid",
        }:
            raise ValueError("Unknown search mode or retriever")
        if (
            any(type(v) is not int or v < 1 for v in (top_k, case_k, max_terms))
            or top_k > case_k
        ):
            raise ValueError("Require positive budgets and top_k <= case_k")
        if mode != "combined" and address is not None:
            raise ValueError("--address is a constraint for combined mode only")
        if mode == "problem" and allow_broader:
            raise ValueError("--allow-broader applies only to address matching")
        query = query.strip()
        address_query = None
        if mode == "address":
            address_query = query
        elif mode == "combined":
            address_query = address.strip() if isinstance(address, str) else query
            if not address_query or not parse_address(address_query):
                raise ValueError("Combined mode needs a named address; specify --address")
        started = time.perf_counter()
        allowed, address_hits = None, {}
        if address_query is not None:
            found = self._address().search(
                address_query, limit=self.corpus_count, allow_broader=allow_broader,
            )
            address_hits = {hit["source_id"]: hit for hit in found}
            allowed = self._metadata(source_ids=list(address_hits))
        routes, encoding, traces = {}, {}, {}
        if mode == "address":
            hits = list(address_hits.values())[:case_k]
        elif allowed == {}:
            hits = []
        else:
            if retriever in {"bm25", "hybrid"}:
                routes["bm25"] = self._lexical(query, case_k, max_terms, allowed)
            if retriever in {"dense", "hybrid"}:
                routes["dense"], encoding = self._dense(query, case_k, allowed)
            for name, ranking in routes.items():
                for hit in ranking:
                    traces.setdefault(hit["source_id"], {})[name] = {
                        "rank": hit["rank"], "score": hit["score"],
                    }
            if retriever == "hybrid":
                hits = fuse_cases(routes["bm25"], routes["dense"], case_k=case_k)
            else:
                hits = routes[retriever]
        results = self._records(hits[:top_k], traces, address_hits)
        if mode != "address":
            for result in results:
                result["matching"]["shared_keywords"] = sorted(
                    set(tokens(query)) & set(tokens(result["case_content"]))
                )[:10]
        return {
            "version": SEARCH_VERSION, "mode": mode,
            "retriever": "address_surface" if mode == "address" else retriever,
            "query": query, "address_query": address_query,
            "results": results, "result_count": len(results),
            "config": {"input_field": "case_content", "top_k": top_k, "case_k": case_k,
                       "max_terms": max_terms, "rrf_k": 60, "allow_broader": allow_broader},
            "corpus_count": self.corpus_count,
            "address_candidate_count": len(address_hits) if allowed is not None else None,
            "route_candidate_counts": {name: len(hits) for name, hits in routes.items()},
            "query_encoding_stats": encoding,
            "dataset_manifest_sha256": self.dataset_hash,
            "lexical_index_manifest_sha256": self.lexical_hash,
            "dense_index_manifest_sha256": self.dense_hash if "dense" in routes else None,
            "address_index_manifest_sha256": self.address_hash if allowed is not None else None,
            "model_sha256": (
                self.encoder.identity.get("model", {}).get("sha256")
                if "dense" in routes else None
            ),
            "implementation_sha256": self.implementation,
            "seconds_search_this_query": time.perf_counter() - started,
            "limitations": [
                "Results are historical complaint cases, not knowledge IDs or verified answers.",
                "Scores/ranks are retrieval signals, not calibrated relevance probabilities.",
                "Address matches concern written names, not verified location identity "
                "or distance.",
                "Problem matching uses raw text; separate LLM problem/address views are not used.",
                "The corpus is the frozen historical baseline, not an up-to-date production feed.",
                "First-query time can include optional index loading and validation.",
                "Dense equal scores at a global candidate cutoff can have ambiguous membership.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "index"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("dense-index", "address-index", "model", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mode", choices=("problem", "address", "combined"), default="problem")
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query")
    inputs.add_argument("--query-file", type=Path)
    inputs.add_argument("--interactive", action="store_true")
    parser.add_argument("--address")
    parser.add_argument("--allow-broader", action="store_true")
    for name, default in (("top-k", 10), ("case-k", 50), ("max-terms", 32), ("threads", 4),
                          ("max-length", 8192), ("batch-size", 8)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    if args.interactive and args.output:
        parser.error("--output is for a single query, not --interactive")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new private output file")
    encoder_factory = None
    if args.mode != "address" and args.retriever in {"dense", "hybrid"}:
        if args.model is None or args.dense_index is None:
            parser.error("Dense/hybrid needs --model and --dense-index")
        def encoder_factory():
            from .encoder import BGEEncoder

            return BGEEncoder(args.model, max_length=args.max_length, batch_size=args.batch_size,
                              device=args.device, dtype=args.dtype)
    settings = {name: getattr(args, name) for name in (
        "mode", "retriever", "address", "top_k", "case_k", "max_terms", "allow_broader",
    )}
    with CaseSearcher(args.dataset, args.index, dense_index=args.dense_index,
                      encoder_factory=encoder_factory,
                      address_index=args.address_index, threads=args.threads) as searcher:
        if args.interactive:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    report = searcher.search(line, **settings)
                    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
                except ValueError as error:
                    print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        else:
            query = args.query if args.query_file is None else args.query_file.read_text(
                encoding="utf-8-sig",
            )
            report = searcher.search(query, **settings)
            if args.output:
                _write_private_json(args.output, report)
                print(json.dumps({"output": str(args.output),
                                  "result_count": report["result_count"]}))
            else:
                print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
