# Knowledge-title retrieval baseline

This document records the first reproducible baseline for the sanitized civic
service ticket snapshot. It is a ticket-to-knowledge-title recommendation task,
not a full RAG result. The source only contains observed knowledge identifiers
and titles; knowledge body text, authoritative versions, and effective dates
remain unavailable.

## Frozen source and split

```text
source size:    655,769,345 bytes
source sha256:  1b778548b618de5f051e749232e47648323979898397588ac79b53d2123aac0c
train:          2024-05-01 through 2025-06-30
dev:            2025-07-01 through 2025-07-31
test:           2025-08-01 through 2025-08-29
```

The build completed on 2026-09-18 with no rejected source rows:

| Item | Count |
|---|---:|
| Physical source rows | 982,435 |
| Canonical parent tickets | 937,409 |
| Knowledge IDs (`type:value`) | 7,516 |
| Parent-level observed citation edges | 609,859 |
| Knowledge IDs with multiple observed titles | 43 |
| Missing/invalid `call_time` rows | 20,449 |
| Exact duplicate parents excluded across splits | 88,367 |
| Train queries / qrels | 247,281 / 273,641 |
| Dev queries / qrels | 37,391 / 39,826 |
| Test queries / qrels | 32,253 / 34,074 |

Parent conflicts were audited rather than silently resolved: 53 content, 6
goal, 2 taxonomy, 24 location, 7 knowledge-set, and 0 call-date conflicts. A
parent can have more than one conflict flag.

## Zero-model results

All metrics use observed citations as positives. Unobserved items remain
unjudged, so these values measure reproduction of historical citations rather
than complete semantic relevance.

| Split | Baseline | Hit@1 | Hit@5 | Hit@10 | Hit@50 | MRR@10 | nDCG@10 | Macro item R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Dev | Global popularity | 0.1915 | 0.3947 | 0.4764 | 0.6609 | 0.2757 | 0.3191 | 0.0047 |
| Dev | Category popularity | 0.4342 | 0.7402 | 0.8281 | 0.9326 | 0.5615 | 0.6210 | 0.2891 |
| Test | Global popularity | 0.2198 | 0.3931 | 0.4790 | 0.6613 | 0.2922 | 0.3321 | 0.0049 |
| Test | Category popularity | 0.4146 | 0.7294 | 0.8227 | 0.9317 | 0.5464 | 0.6085 | 0.2825 |

Category popularity is an oracle-style diagnostic until it is confirmed that
the taxonomy is available at ticket intake. It must not replace the text-only
main result. Its large gain shows that taxonomy routing is strong, while its
test Recall@10 remains only 0.0263 for training-frequency 1-4 knowledge items
and 0 for unseen knowledge. Long-tail retrieval therefore remains a primary
target rather than being hidden by the aggregate Hit@K.

## Next comparisons

The fixed comparison order is:

1. Pyserini 1.2.0 Chinese BM25 on content, goal, and joint query views.
2. BGE-M3 dense retrieval on the same three views.
3. Dense and sparse hybrid retrieval.
4. Reranking of a fixed Top-50 or Top-100 candidate set.
5. Raw query plus V4 extraction, only after the raw-text baselines are frozen.

The same qrels, split, and evaluator must be used for every comparison. A later
versioned knowledge-body corpus will require a new dataset version rather than
silently changing this title-level benchmark.
