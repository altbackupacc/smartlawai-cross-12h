"""Export M4 Authority Registry to standalone release formats.

Generates:
- Parquet & CSV exports for STATUTES, SECTIONS, SUPERSESSION, CITATION_STRINGS
- Schema definitions and data dictionary
- Standalone release README and checksums
"""

from __future__ import annotations

import json
from pathlib import Path
import duckdb

REPO = Path(__file__).resolve().parents[2]
OUT_DB = REPO / "out" / "registry.duckdb"
EXPORT_DIR = REPO / "registry" / "export"


def main() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_dir = EXPORT_DIR / "csv"
    parquet_dir = EXPORT_DIR / "parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(OUT_DB), read_only=True)

    tables = ["STATUTES", "SECTIONS", "SUPERSESSION", "CITATION_STRINGS"]
    stats = {}

    for tbl in tables:
        count = conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        stats[tbl] = count

        # Export CSV
        csv_path = csv_dir / f"{tbl.lower()}.csv"
        conn.execute(f"COPY {tbl} TO '{csv_path.as_posix()}' (HEADER, DELIMITER ',')")

        # Export Parquet
        parquet_path = parquet_dir / f"{tbl.lower()}.parquet"
        conn.execute(f"COPY {tbl} TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        print(f"Exported {tbl}: {count} rows -> {csv_path.name}, {parquet_path.name}")

    # Write release metadata / descriptor
    manifest = {
        "artifact": "SmartLawAI Indian Law Authority Registry",
        "version": "1.0.0",
        "release_date": "2026-09-10",
        "description": (
            "Temporal authority registry for Indian law containing statute commencement, "
            "repeal dates, section-level supersession mappings (including 2024 IPC->BNS, "
            "CrPC->BNSS, IEA->BSA recodification), and normalized citation strings."
        ),
        "source_repository": "https://github.com/kramjiy/smartlawai-1",
        "tables": {
            "statutes": {
                "rows": stats["STATUTES"],
                "description": "14 primary Indian statutes with in-force intervals and repeal links.",
            },
            "sections": {
                "rows": stats["SECTIONS"],
                "description": "Statute sections/articles with headings, official numbers, and text.",
            },
            "supersession": {
                "rows": stats["SUPERSESSION"],
                "description": "Section-level mappings across legal reforms (1:1, 1:N, N:1, repealed_no_successor).",
            },
            "citation_strings": {
                "rows": stats["CITATION_STRINGS"],
                "description": "Empirical citation strings extracted from Supreme Court / High Court corpus with target mappings.",
            },
        },
    }

    manifest_path = EXPORT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme_content = f"""# SmartLawAI Indian Law Authority Registry (M4 Artifact)

**Release Date:** 2026-09-10
**Version:** 1.0.0
**Format:** Parquet & CSV

## Overview
This package is a standalone, citable temporal knowledge base for Indian statutory law.
It enables deterministic checking of whether cited statutory authorities were in force
as of an arbitrary evaluation date ($d$), solving the structural blindness of entailment
models to statutory obsolescence and recodification (such as the July 1, 2024 transition
from IPC/CrPC/IEA to BNS/BNSS/BSA).

## Dataset Statistics
- **Statutes:** {stats['STATUTES']} statutes
- **Sections / Articles:** {stats['SECTIONS']} sections (covering IPC, CrPC, IEA, BNS, BNSS, BSA, Constitution, Contract, CPC, IT Act, NI Act, etc.)
- **Supersession Mappings:** {stats['SUPERSESSION']} transition relations (many-to-many aware)
- **Citation Strings:** {stats['CITATION_STRINGS']} distinct empirical citations mapped to targets

## Directory Structure
- `parquet/`: High-performance zstd-compressed columnar files for data science & RAG pipelines.
- `csv/`: Plain CSV tables for universal tabular ingestion.
- `manifest.json`: Metadata and schema description.

## Schema
### STATUTES
`id` (VARCHAR PK), `short_title` (VARCHAR), `long_title` (VARCHAR), `year` (INTEGER),
`jurisdiction` (VARCHAR), `in_force_from` (DATE), `repealed_on` (DATE), `repealed_by_statute_id` (VARCHAR)

### SECTIONS
`id` (VARCHAR PK), `statute_id` (VARCHAR FK), `number` (VARCHAR), `heading` (VARCHAR),
`text` (VARCHAR), `in_force_from` (DATE), `in_force_to` (DATE), `superseded_by_section_id` (VARCHAR),
`verified_by` (VARCHAR), `verified_on` (DATE)

### SUPERSESSION
`id` (VARCHAR PK), `from_statute_id` (VARCHAR FK), `from_section_id` (VARCHAR FK),
`relation` (VARCHAR: `maps_to` | `repealed_no_successor`),
`effective_on` (DATE), `to_statute_id` (VARCHAR FK), `to_section_id` (VARCHAR FK),
`notes` (VARCHAR), `source_doc` (VARCHAR)

### CITATION_STRINGS
`id` (VARCHAR PK), `raw` (VARCHAR), `normalised` (VARCHAR), `target_type` (VARCHAR),
`target_id` (VARCHAR), `confidence` (FLOAT), `occurrences` (INTEGER)

## Citation
If using this registry, cite:
```bibtex
@article{{smartlawai2026authorityregistry,
  title={{SmartLawAI: Authority-Grounded Hallucination Detection for Indian Law}},
  author={{SmartLawAI Research Team}},
  year={{2026}}
}}
```
"""
    (EXPORT_DIR / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"Wrote release documentation to {EXPORT_DIR / 'README.md'}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
