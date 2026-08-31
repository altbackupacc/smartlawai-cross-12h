"""LocalBackend: DuckDB + local filesystem + on-disk FAISS.
Primary backend for VSCode and Kaggle development. Zero cloud dependencies."""
from __future__ import annotations

import json
import os
import shutil

import duckdb

from smartlawai.scope import Scope

from .base import (
    AuditEvent,
    BackendInterface,
    Chunk,
    Clause,
    Document,
    EvalResult,
    IngestionRecord,
    Summary,
)

DATA_DIR = os.environ.get("SMARTLAW_LOCAL_DIR", "./.smartlaw_local")


class LocalBackend(BackendInterface):
    def __init__(self) -> None:
        os.makedirs(f"{DATA_DIR}/docs", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/faiss", exist_ok=True)
        self.db = duckdb.connect(f"{DATA_DIR}/smartlaw.duckdb")
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.execute("""CREATE TABLE IF NOT EXISTS DOCUMENTS(
            doc_id VARCHAR PRIMARY KEY, filename VARCHAR, doc_type VARCHAR,
            language VARCHAR, source VARCHAR, storage_uri VARCHAR,
            num_pages INTEGER, ocr_applied BOOLEAN, raw_text VARCHAR,
            session_id VARCHAR, owner_id VARCHAR,
            uploaded_at TIMESTAMP DEFAULT now())""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS CHUNKS(
            chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR, parent_chunk_id VARCHAR,
            chunk_index INTEGER, chunk_text VARCHAR, char_start INTEGER,
            char_end INTEGER, embedding VARCHAR, bm25_indexed BOOLEAN,
            owner_id VARCHAR)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS CLAUSES(
            clause_id VARCHAR PRIMARY KEY, doc_id VARCHAR, chunk_id VARCHAR,
            clause_type VARCHAR, clause_text VARCHAR, span_start INTEGER,
            span_end INTEGER, risk_tier VARCHAR, risk_score DOUBLE,
            statute_ref VARCHAR)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS SUMMARIES(
            summary_id VARCHAR PRIMARY KEY, doc_id VARCHAR,
            section_label VARCHAR, level VARCHAR, summary_text VARCHAR,
            faithfulness_score DOUBLE, model VARCHAR,
            created_at TIMESTAMP DEFAULT now())""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS INGESTION_LOG(
            ingest_id VARCHAR PRIMARY KEY, doc_id VARCHAR, stage_path VARCHAR,
            ocr_engine VARCHAR, ocr_status VARCHAR, lang_detected VARCHAR,
            preprocess_status VARCHAR, error_msg VARCHAR,
            ingested_at TIMESTAMP DEFAULT now())""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS AUDIT_LOG(
            event_id VARCHAR PRIMARY KEY, session_id VARCHAR,
            event_type VARCHAR, endpoint VARCHAR, query_text VARCHAR,
            response_text VARCHAR, faithfulness_score DOUBLE,
            pii_redacted BOOLEAN, created_at TIMESTAMP DEFAULT now())""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS EVAL_RESULTS(
            eval_id VARCHAR PRIMARY KEY, task VARCHAR, system_name VARCHAR,
            metric VARCHAR, value DOUBLE, split VARCHAR,
            n_examples INTEGER, run_at TIMESTAMP DEFAULT now())""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS ANNOTATIONS(
            annotation_id VARCHAR PRIMARY KEY, doc_id VARCHAR,
            category VARCHAR, task VARCHAR, annotator_id VARCHAR,
            annotation_json VARCHAR, gold_summary VARCHAR, split VARCHAR,
            kappa DOUBLE, created_at TIMESTAMP DEFAULT now())""")


    # ---------- relational ops ----------
    def upload_doc(self, local_path: str, doc: Document) -> str:
        dest = f"{DATA_DIR}/docs/{doc.doc_id}_{doc.filename}"
        shutil.copy(local_path, dest)
        self.db.execute(
            """INSERT INTO DOCUMENTS(doc_id,filename,doc_type,language,source,
               storage_uri,num_pages,ocr_applied,raw_text,session_id,owner_id,
               uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,now())
               ON CONFLICT DO NOTHING""",
            [doc.doc_id, doc.filename, doc.doc_type, doc.language, doc.source,
             dest, doc.num_pages, doc.ocr_applied, doc.raw_text, doc.session_id,
             doc.owner_id])
        return dest

    def store_chunks(self, chunks: list[Chunk]) -> None:
        self.db.executemany(
            """INSERT INTO CHUNKS(chunk_id,doc_id,parent_chunk_id,chunk_index,
               chunk_text,char_start,char_end,embedding,bm25_indexed,owner_id)
               VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
            [[c.chunk_id, c.doc_id, c.parent_chunk_id, c.chunk_index, c.chunk_text,
              c.char_start, c.char_end,
              json.dumps(c.embedding) if c.embedding else None, c.bm25_indexed,
              c.owner_id]
             for c in chunks])

    def store_clauses(self, clauses: list[Clause]) -> None:
        self.db.executemany(
            """INSERT INTO CLAUSES(clause_id,doc_id,chunk_id,clause_type,clause_text,
               span_start,span_end,risk_tier,risk_score,statute_ref)
               VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
            [[c.clause_id, c.doc_id, c.chunk_id, c.clause_type, c.clause_text,
              c.span_start, c.span_end, c.risk_tier, c.risk_score, c.statute_ref]
             for c in clauses])

    def store_summaries(self, summaries: list[Summary]) -> None:
        self.db.executemany(
            """INSERT INTO SUMMARIES(summary_id,doc_id,section_label,level,
               summary_text,faithfulness_score,model,created_at)
               VALUES (?,?,?,?,?,?,?,now()) ON CONFLICT DO NOTHING""",
            [[s.summary_id, s.doc_id, s.section_label, s.level, s.summary_text,
              s.faithfulness_score, s.model] for s in summaries])

    def store_audit(self, e: AuditEvent) -> None:
        self.db.execute(
            """INSERT INTO AUDIT_LOG(event_id,session_id,event_type,endpoint,
               query_text,response_text,faithfulness_score,pii_redacted,created_at)
               VALUES (?,?,?,?,?,?,?,?,now()) ON CONFLICT DO NOTHING""",
            [e.event_id, e.session_id, e.event_type, e.endpoint, e.query_text,
             e.response_text, e.faithfulness_score, e.pii_redacted])

    def store_ingestion(self, r: IngestionRecord) -> None:
        self.db.execute(
            """INSERT INTO INGESTION_LOG(ingest_id,doc_id,stage_path,ocr_engine,
               ocr_status,lang_detected,preprocess_status,error_msg,ingested_at)
               VALUES (?,?,?,?,?,?,?,?,now()) ON CONFLICT DO NOTHING""",
            [r.ingest_id, r.doc_id, r.stage_path, r.ocr_engine, r.ocr_status,
             r.lang_detected, r.preprocess_status, r.error_msg])

    def store_eval(self, results: list[EvalResult]) -> None:
        self.db.executemany(
            """INSERT INTO EVAL_RESULTS(eval_id,task,system_name,metric,value,
               split,n_examples,run_at) VALUES (?,?,?,?,?,?,?,now())
               ON CONFLICT DO NOTHING""",
            [[r.eval_id, r.task, r.system_name, r.metric, r.value, r.split,
              r.n_examples] for r in results])

    def mark_preprocessed(self, doc_id: str, status: str) -> None:
        self.db.execute("UPDATE INGESTION_LOG SET preprocess_status=? WHERE doc_id=?",
                        [status, doc_id])

    def fetch_document_text(self, doc_id: str) -> str:
        r = self.db.execute("SELECT raw_text FROM DOCUMENTS WHERE doc_id=?",
                            [doc_id]).fetchone()
        return r[0] if r else ""

    def _row_to_chunk(self, r) -> Chunk:
        return Chunk(chunk_id=r[0], doc_id=r[1], parent_chunk_id=r[2], chunk_index=r[3],
                     chunk_text=r[4], char_start=r[5], char_end=r[6],
                     embedding=json.loads(r[7]) if r[7] else None, bm25_indexed=r[8],
                     owner_id=r[9] or "")

    def fetch_chunks(self, doc_id: str) -> list[Chunk]:
        rows = self.db.execute(
            "SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,char_start,"
            "char_end,embedding,bm25_indexed,owner_id FROM CHUNKS WHERE doc_id=? "
            "ORDER BY chunk_index", [doc_id]).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def fetch_all_chunks(self) -> list[Chunk]:
        rows = self.db.execute(
            "SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,char_start,"
            "char_end,embedding,bm25_indexed,owner_id FROM CHUNKS").fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        ph = ",".join(["?"] * len(chunk_ids))
        rows = self.db.execute(
            f"SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,char_start,"
            f"char_end,embedding,bm25_indexed,owner_id FROM CHUNKS WHERE chunk_id IN ({ph})",
            chunk_ids).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def fetch_chunks_scoped(self, scope: Scope) -> list[Chunk]:
        """I1 enforcement point: the SQL itself cannot return out-of-scope rows."""
        ph = ",".join(["?"] * len(scope.doc_ids))
        rows = self.db.execute(
            f"SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,char_start,"
            f"char_end,embedding,bm25_indexed,owner_id FROM CHUNKS "
            f"WHERE doc_id IN ({ph}) AND owner_id=? ORDER BY doc_id, chunk_index",
            [*scope.doc_ids, scope.owner_id]).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def list_doc_ids(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT doc_id FROM DOCUMENTS").fetchall()]

    def update_chunk_embeddings(self, pairs) -> None:
        self.db.executemany(
            "UPDATE CHUNKS SET embedding=?, bm25_indexed=TRUE WHERE chunk_id=?",
            [[json.dumps(v), cid] for cid, v in pairs])

    def fetch_annotations(self, split: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT annotation_id,doc_id,category,task,annotation_json,gold_summary,"
            "split,kappa FROM ANNOTATIONS WHERE split=?", [split]).fetchall()
        cols = ["annotation_id", "doc_id", "category", "task", "annotation_json",
                "gold_summary", "split", "kappa"]
        out = []
        for r in rows:
            d = dict(zip(cols, r))
            if d.get("annotation_json"):
                try:
                    d["annotation_json"] = json.loads(d["annotation_json"])
                except Exception:
                    pass
            out.append(d)
        return out

    # ---------- FAISS to disk ----------
    def _path(self, key: str) -> str:
        return f"{DATA_DIR}/faiss/{key}.bin"

    def _write_index_bytes(self, key: str, data: bytes) -> None:
        with open(self._path(key), "wb") as f:
            f.write(data)

    def _read_index_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def faiss_index_exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))
