# SmartLawAI Indian Law Authority Registry (M4 Artifact)

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
- **Statutes:** 14 statutes
- **Sections / Articles:** 3817 sections (covering IPC, CrPC, IEA, BNS, BNSS, BSA, Constitution, Contract, CPC, IT Act, NI Act, etc.)
- **Supersession Mappings:** 1289 transition relations (many-to-many aware)
- **Citation Strings:** 19784 distinct empirical citations mapped to targets

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
@article{smartlawai2026authorityregistry,
  title={SmartLawAI: Authority-Grounded Hallucination Detection for Indian Law},
  author={SmartLawAI Research Team},
  year={2026}
}
```
