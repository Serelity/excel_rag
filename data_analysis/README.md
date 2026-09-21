# Data analysis stage

This directory owns the first stage of the rebuild: a reproducible,
privacy-aware profile of the source data. It does not clean, rewrite, or move
anything under `data/raw/`.

## Inputs

- `t_order_master.sanitized.v1_9.tsv`: the only file used for value-level
  analysis.
- `t_order_master.tsv`: used only to compare schema, row keys, and per-field
  change counts. Raw values are never written to the report.
- `t_order_master_100.sanitized.v1_9.tsv`: checked as a prefix fixture.

The parser treats the files as UTF-8 TSV with quoted multiline fields. Shell
line counts are not record counts. The sanitized input is parsed strictly. The
historical 129-column export is parsed permissively because it contains legacy
short rows and quote sequences that strict CSV rejects; its row widths and
unavailable trailing fields are reported explicitly.

## Run

No third-party package is required:

```bash
python3 data_analysis/analyze.py
```

For a faster sanitized-only run:

```bash
python3 data_analysis/analyze.py --skip-raw-comparison --skip-raw-hash
```

Outputs are written to `data_analysis/output/`:

- `profile.json`: complete machine-readable profile;
- `columns.csv`: flat field-level structural and quality table;
- `field_dictionary.csv`: machine-readable business meaning, lifecycle,
  availability, RAG role, evidence status, and field risks;
- `data_dictionary.md`: human-readable semantic dictionary for every field;
- `profile.md`: semantic, structural, quality, privacy, and modelling conclusions.

The repository does not contain the source system's official field manual.
Consequently, the semantic dictionary distinguishes observations supported by
the data from name/value-based inferences and unresolved business definitions.
An inferred definition must not be treated as production truth until it is
confirmed by the data owner.

The output directory is ignored by Git. Reports contain aggregate values, but
they must still receive a disclosure review before publication.

## Guarantees

- Source files are opened read-only.
- Empty strings and literal null tokens are counted separately.
- `order_id` association-group counts and within-group variations are exact and
  calculated in temporary SQLite storage. The profiler does not assume that an
  `order_id` is a parent ticket or that all rows in a group should be merged.
- High-cardinality field counts use HyperLogLog p=12 and are marked as
  estimates.
- Text, IDs, addresses, department names, knowledge titles, and PII matches
  are never emitted.
- PII-like values and digit sequences of seven or more characters are redacted
  even when malformed data places them in an otherwise safe categorical field.
- PII regexes are screening signals, not proof of successful anonymization.
- `case_content` values containing tab characters are counted as embedded TSV
  contamination. They remain structurally parseable but are excluded from the
  semantic-extraction pilot because multiple source fields or records have been
  swallowed into one narrative field.
