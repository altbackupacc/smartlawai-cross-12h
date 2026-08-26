"""GCloudBackend: Cloud SQL Postgres + Google Cloud Storage (blobs + FAISS).
Production backend. Same interface as LocalBackend, so the core never changes.
Requires: pip install "smartlawai[gcloud]" and GCS/Postgres env vars (.env)."""
from __future__ import annotations

import json
import os
import tempfile

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


class GCloudBackend(BackendInterface):
    def __init__(self) -> None:
        import psycopg2
        from google.cloud import storage

        self.bucket_name = os.environ["GCS_BUCKET"]
        self._gcs = storage.Client(project=os.environ.get("GCP_PROJECT") or None)
        self.bucket = self._gcs.bucket(self.bucket_name)
        self.conn = psycopg2.connect(
            host=os.environ["PG_HOST"], port=int(os.environ.get("PG_PORT", 5432)),
            dbname=os.environ["PG_DB"], user=os.environ["PG_USER"],
            password=os.environ["PG_PASSWORD"])
        self.conn.autocommit = True
        self._init_schema()

    def _cur(self):
        return self.conn.cursor()

    def _init_schema(self) -> None:
        ddl = [
            """CREATE TABLE IF NOT EXISTS documents(
                doc_id TEXT PRIMARY KEY, filename TEXT, doc_type TEXT, language TEXT,
                source TEXT, storage_uri TEXT, num_pages INTEGER, ocr_applied BOOLEAN,
                raw_text TEXT, session_id TEXT, uploaded_at TIMESTAMP DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS chunks(
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, parent_chunk_id TEXT,
                chunk_index INTEGER, chunk_text TEXT, char_start INTEGER, char_end INTEGER,
                embedding JSONB, bm25_indexed BOOLEAN)""",
            """CREATE TABLE IF NOT EXISTS clauses(
                clause_id TEXT PRIMARY KEY, doc_id TEXT, chunk_id TEXT, clause_type TEXT,
                clause_text TEXT, span_start INTEGER, span_end INTEGER, risk_tier TEXT,
                risk_score DOUBLE PRECISION, statute_ref TEXT)""",
            """CREATE TABLE IF NOT EXISTS summaries(
                summary_id TEXT PRIMARY KEY, doc_id TEXT, section_label TEXT, level TEXT,
                summary_text TEXT, faithfulness_score DOUBLE PRECISION, model TEXT,
                created_at TIMESTAMP DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS ingestion_log(
                ingest_id TEXT PRIMARY KEY, doc_id TEXT, stage_path TEXT, ocr_engine TEXT,
                ocr_status TEXT, lang_detected TEXT, preprocess_status TEXT, error_msg TEXT,
                ingested_at TIMESTAMP DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS audit_log(
                event_id TEXT PRIMARY KEY, session_id TEXT, event_type TEXT, endpoint TEXT,
                query_text TEXT, response_text TEXT, faithfulness_score DOUBLE PRECISION,
                pii_redacted BOOLEAN, created_at TIMESTAMP DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS eval_results(
                eval_id TEXT PRIMARY KEY, task TEXT, system_name TEXT, metric TEXT,
                value DOUBLE PRECISION, split TEXT, n_examples INTEGER,
                run_at TIMESTAMP DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS annotations(
                annotation_id TEXT PRIMARY KEY, doc_id TEXT, category TEXT, task TEXT,
                annotator_id TEXT, annotation_json JSONB, gold_summary TEXT, split TEXT,
                kappa DOUBLE PRECISION, created_at TIMESTAMP DEFAULT now())""",
        ]
        with self._cur() as cur:
            for stmt in ddl:
                cur.execute(stmt)

    # ---------- relational ops ----------
    def upload_doc(self, local_path: str, doc: Document) -> str:
        blob_path = f"docs/{doc.doc_id}/{doc.filename}"
        self.bucket.blob(blob_path).upload_from_filename(local_path)
        uri = f"gs://{self.bucket_name}/{blob_path}"
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO documents(doc_id,filename,doc_type,language,source,
                   storage_uri,num_pages,ocr_applied,raw_text,session_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (doc_id) DO NOTHING""",
                (doc.doc_id, doc.filename, doc.doc_type, doc.language, doc.source,
                 uri, doc.num_pages, doc.ocr_applied, doc.raw_text, doc.session_id))
        return uri

    def store_chunks(self, chunks: list[Chunk]) -> None:
        with self._cur() as cur:
            cur.executemany(
                """INSERT INTO chunks(chunk_id,doc_id,parent_chunk_id,chunk_index,
                   chunk_text,char_start,char_end,embedding,bm25_indexed)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (chunk_id) DO NOTHING""",
                [(c.chunk_id, c.doc_id, c.parent_chunk_id, c.chunk_index, c.chunk_text,
                  c.char_start, c.char_end,
                  json.dumps(c.embedding) if c.embedding else None, c.bm25_indexed)
                 for c in chunks])

    def store_clauses(self, clauses: list[Clause]) -> None:
        with self._cur() as cur:
            cur.executemany(
                """INSERT INTO clauses(clause_id,doc_id,chunk_id,clause_type,clause_text,
                   span_start,span_end,risk_tier,risk_score,statute_ref)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (clause_id) DO NOTHING""",
                [(c.clause_id, c.doc_id, c.chunk_id, c.clause_type, c.clause_text,
                  c.span_start, c.span_end, c.risk_tier, c.risk_score, c.statute_ref)
                 for c in clauses])

    def store_summaries(self, summaries: list[Summary]) -> None:
        with self._cur() as cur:
            cur.executemany(
                """INSERT INTO summaries(summary_id,doc_id,section_label,level,
                   summary_text,faithfulness_score,model)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (summary_id) DO NOTHING""",
                [(s.summary_id, s.doc_id, s.section_label, s.level, s.summary_text,
                  s.faithfulness_score, s.model) for s in summaries])

    def store_audit(self, e: AuditEvent) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO audit_log(event_id,session_id,event_type,endpoint,
                   query_text,response_text,faithfulness_score,pii_redacted)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (e.event_id, e.session_id, e.event_type, e.endpoint, e.query_text,
                 e.response_text, e.faithfulness_score, e.pii_redacted))

    def store_ingestion(self, r: IngestionRecord) -> None:
        with self._cur() as cur:
            cur.execute(
                """INSERT INTO ingestion_log(ingest_id,doc_id,stage_path,ocr_engine,
                   ocr_status,lang_detected,preprocess_status,error_msg)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (r.ingest_id, r.doc_id, r.stage_path, r.ocr_engine, r.ocr_status,
                 r.lang_detected, r.preprocess_status, r.error_msg))

    def store_eval(self, results: list[EvalResult]) -> None:
        with self._cur() as cur:
            cur.executemany(
                """INSERT INTO eval_results(eval_id,task,system_name,metric,value,
                   split,n_examples) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                [(r.eval_id, r.task, r.system_name, r.metric, r.value, r.split,
                  r.n_examples) for r in results])

    def mark_preprocessed(self, doc_id: str, status: str) -> None:
        with self._cur() as cur:
            cur.execute("UPDATE ingestion_log SET preprocess_status=%s WHERE doc_id=%s",
                        (status, doc_id))

    def fetch_document_text(self, doc_id: str) -> str:
        with self._cur() as cur:
            cur.execute("SELECT raw_text FROM documents WHERE doc_id=%s", (doc_id,))
            row = cur.fetchone()
            return row[0] if row else ""

    def _row_to_chunk(self, r) -> Chunk:
        emb = r[7]
        if isinstance(emb, str):
            emb = json.loads(emb)
        return Chunk(chunk_id=r[0], doc_id=r[1], parent_chunk_id=r[2], chunk_index=r[3],
                     chunk_text=r[4], char_start=r[5], char_end=r[6],
                     embedding=emb, bm25_indexed=r[8])

    def fetch_chunks(self, doc_id: str) -> list[Chunk]:
        with self._cur() as cur:
            cur.execute(
                "SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,"
                "char_start,char_end,embedding,bm25_indexed FROM chunks "
                "WHERE doc_id=%s ORDER BY chunk_index", (doc_id,))
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    def fetch_all_chunks(self) -> list[Chunk]:
        with self._cur() as cur:
            cur.execute("SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,"
                        "char_start,char_end,embedding,bm25_indexed FROM chunks")
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    def fetch_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self._cur() as cur:
            cur.execute(
                "SELECT chunk_id,doc_id,parent_chunk_id,chunk_index,chunk_text,"
                "char_start,char_end,embedding,bm25_indexed FROM chunks "
                "WHERE chunk_id = ANY(%s)", (chunk_ids,))
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    def list_doc_ids(self) -> list[str]:
        with self._cur() as cur:
            cur.execute("SELECT doc_id FROM documents")
            return [r[0] for r in cur.fetchall()]

    def update_chunk_embeddings(self, pairs) -> None:
        with self._cur() as cur:
            cur.executemany(
                "UPDATE chunks SET embedding=%s, bm25_indexed=TRUE WHERE chunk_id=%s",
                [(json.dumps(v), cid) for cid, v in pairs])

    def fetch_annotations(self, split: str) -> list[dict]:
        with self._cur() as cur:
            cur.execute("SELECT annotation_id,doc_id,category,task,annotation_json,"
                        "gold_summary,split,kappa FROM annotations WHERE split=%s",
                        (split,))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    # ---------- FAISS on GCS ----------
    def _blob_path(self, key: str) -> str:
        return f"faiss/{key}.bin"

    def _write_index_bytes(self, key: str, data: bytes) -> None:
        self.bucket.blob(self._blob_path(key)).upload_from_string(data)

    def _read_index_bytes(self, key: str) -> bytes:
        with tempfile.NamedTemporaryFile() as tmp:
            self.bucket.blob(self._blob_path(key)).download_to_filename(tmp.name)
            with open(tmp.name, "rb") as f:
                return f.read()

    def faiss_index_exists(self, key: str) -> bool:
        return self.bucket.blob(self._blob_path(key)).exists()
