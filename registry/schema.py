"""The authority registry database (DuckDB).

Schema follows PLAN.md M4. Storage style follows
`src/smartlawai/adapters/local.py`: `CREATE TABLE IF NOT EXISTS`, named-column
inserts, never positional (CLAUDE.md 4). This is a standalone `.duckdb` file,
not a table bolted onto the pipeline's database (M4_ONBOARDING.md 4).

Three deliberate departures from PLAN.md's literal DDL, each recorded here
because the registry is the artifact the paper is judged on:

1. **`SUPERSESSION` is its own table.** PLAN.md puts
   `superseded_by_section_id` on `sections` -- a single foreign key, i.e. a
   1:1 mapping. The actual MHA correspondence table is many-to-many: BNS
   1(1)-1(6) replaces IPC 1-5, and IPC 23 is split across BNS 2(36), 2(37) and
   2(38). A single column would silently drop every mapping after the first.
   `sections.superseded_by_section_id` is kept and populated ONLY where the
   mapping is unambiguously 1:1, so PLAN.md's stated schema and the frozen M5
   contract both still hold; `SUPERSESSION` is the authoritative record.

2. **`kind` on sections.** The corpus cites Articles (Constitution), Orders and
   Rules (CPC), not only Sections. Without `kind`, `constitution-1950-art32`
   and a hypothetical section 32 collide.

3. **Provenance columns** (`source_url`, `source_doc`, `verified_by`,
   `verified_on`). Threat T10: we are building the registry we score ourselves
   against, so every row must be traceable to a primary source and a
   verification event.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB = Path(__file__).resolve().parents[1] / "out" / "registry.duckdb"


class RegistryDB:
    """Thin wrapper over a DuckDB connection holding the authority registry."""

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.path))
        self._init_schema()

    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS STATUTES(
            id VARCHAR PRIMARY KEY,
            short_title VARCHAR NOT NULL,
            long_title VARCHAR,
            year INTEGER,
            jurisdiction VARCHAR DEFAULT 'IN',
            in_force_from DATE,
            repealed_on DATE,
            repealed_by_statute_id VARCHAR,
            source_url VARCHAR,
            created_at TIMESTAMP DEFAULT now())""")

        self.db.execute("""CREATE TABLE IF NOT EXISTS SECTIONS(
            id VARCHAR PRIMARY KEY,
            statute_id VARCHAR NOT NULL,
            kind VARCHAR DEFAULT 's',
            number VARCHAR NOT NULL,
            heading VARCHAR,
            text VARCHAR,
            in_force_from DATE,
            in_force_to DATE,
            -- 'omitted' means a source states the provision was removed but we
            -- do not have the effective date. Needed because neither a NULL
            -- `in_force_to` (which reads as still-in-force) nor a guessed date
            -- would be truthful. Constitution articles come in this way: the
            -- arrangement of articles marks "[31. Omitted.]" but the amendment
            -- date lives in body footnotes.
            status VARCHAR,
            superseded_by_section_id VARCHAR,
            source_url VARCHAR,
            source_doc VARCHAR,
            verified_by VARCHAR,
            verified_on DATE,
            created_at TIMESTAMP DEFAULT now())""")

        # Authoritative many-to-many supersession record. See module docstring.
        self.db.execute("""CREATE TABLE IF NOT EXISTS SUPERSESSION(
            id VARCHAR PRIMARY KEY,
            from_section_id VARCHAR NOT NULL,
            to_section_id VARCHAR,
            from_statute_id VARCHAR NOT NULL,
            to_statute_id VARCHAR,
            effective_on DATE NOT NULL,
            relation VARCHAR,
            note VARCHAR,
            source_doc VARCHAR,
            verified_by VARCHAR,
            verified_on DATE)""")

        self.db.execute("""CREATE TABLE IF NOT EXISTS CITATION_STRINGS(
            id VARCHAR PRIMARY KEY,
            raw VARCHAR NOT NULL,
            normalised VARCHAR,
            target_type VARCHAR,
            target_id VARCHAR,
            confidence DOUBLE,
            occurrences INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT now())""")

        for stmt in (
            "CREATE INDEX IF NOT EXISTS idx_sections_statute ON SECTIONS(statute_id)",
            "CREATE INDEX IF NOT EXISTS idx_sections_lookup ON SECTIONS(statute_id, kind, number)",
            "CREATE INDEX IF NOT EXISTS idx_supersession_from ON SUPERSESSION(from_section_id)",
            "CREATE INDEX IF NOT EXISTS idx_citation_norm ON CITATION_STRINGS(normalised)",
        ):
            self.db.execute(stmt)

    # ------------------------------------------------------------------ #
    def add_statute(self, **kw) -> None:
        """Insert or replace a statute. Named columns only (CLAUDE.md 4)."""
        cols = ["id", "short_title", "long_title", "year", "jurisdiction",
                "in_force_from", "repealed_on", "repealed_by_statute_id", "source_url"]
        vals = [kw.get(c) for c in cols]
        self.db.execute(
            "INSERT OR REPLACE INTO STATUTES ({}) VALUES ({})".format(", ".join(cols), ", ".join("?" * len(cols))),
            vals,
        )

    def add_section(self, **kw) -> None:
        cols = ["id", "statute_id", "kind", "number", "heading", "text",
                "in_force_from", "in_force_to", "status", "superseded_by_section_id",
                "source_url", "source_doc", "verified_by", "verified_on"]
        vals = [kw.get(c) for c in cols]
        self.db.execute(
            "INSERT OR REPLACE INTO SECTIONS ({}) VALUES ({})".format(", ".join(cols), ", ".join("?" * len(cols))),
            vals,
        )

    def add_supersession(self, **kw) -> None:
        cols = ["id", "from_section_id", "to_section_id", "from_statute_id",
                "to_statute_id", "effective_on", "relation", "note",
                "source_doc", "verified_by", "verified_on"]
        vals = [kw.get(c) for c in cols]
        self.db.execute(
            "INSERT OR REPLACE INTO SUPERSESSION ({}) VALUES ({})".format(", ".join(cols), ", ".join("?" * len(cols))),
            vals,
        )

    def counts(self) -> dict[str, int]:
        out = {}
        for t in ("STATUTES", "SECTIONS", "SUPERSESSION", "CITATION_STRINGS"):
            out[t] = self.db.execute("SELECT count(*) FROM " + t).fetchone()[0]
        return out

    def close(self) -> None:
        self.db.close()
