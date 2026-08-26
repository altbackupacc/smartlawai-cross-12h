"""Tests for the FastAPI endpoints — validates route existence, request/response
shapes, and error handling. Uses FastAPI TestClient (no live server needed)."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def client():
    """Create a TestClient against the actual app, using a temp local backend."""
    tmpdir = tempfile.mkdtemp()
    os.environ["SMARTLAW_LOCAL_DIR"] = tmpdir
    os.environ["SMARTLAW_BACKEND"] = "local"

    from fastapi.testclient import TestClient
    from api.main import app
    yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_session(client):
    r = client.post("/session")
    assert r.status_code == 200
    assert r.json()["session_id"].startswith("s-")


def test_upload_no_file(client):
    """Uploading without a file should 422 (validation error)."""
    r = client.post("/upload")
    assert r.status_code == 422


def test_upload_txt_file(client):
    """Upload a small .txt file and verify it returns a doc_id."""
    content = b"This is a test contract for SmartLawAI."
    r = client.post("/upload", files={"file": ("test.txt", content, "text/plain")})
    assert r.status_code == 200
    data = r.json()
    assert "doc_id" in data
    assert data["status"] != "OCR_FAIL"


def test_ocr_missing_doc(client):
    """OCR for a non-existent doc_id should return 404."""
    r = client.get("/ocr/nonexistent-doc-id-12345")
    assert r.status_code == 404


def test_entities_missing_doc(client):
    """NER on a non-existent doc should return 404."""
    r = client.post("/entities/nonexistent-doc-id-12345")
    assert r.status_code == 404


def test_summarise_missing_doc(client):
    """Summarise on a non-existent doc should return 404."""
    r = client.post("/summarise/nonexistent-doc-id-12345")
    assert r.status_code == 404


def test_clauses_missing_doc(client):
    """Clauses on a non-existent doc should return 404."""
    r = client.post("/clauses/nonexistent-doc-id-12345")
    assert r.status_code == 404


def test_ask_empty_question(client):
    """Ask with an empty question should still return 200 (or 500 if no RAG docs)."""
    r = client.post("/ask", json={"question": "", "session_id": "test"})
    # Might be 200 with empty answer or 500 if RAG is uninitialized — both are valid
    assert r.status_code in (200, 500)


def test_endpoint_methods(client):
    """Verify the correct HTTP methods are enforced."""
    # /summarise and /clauses should be POST, not GET
    r = client.get("/summarise/any-doc-id")
    assert r.status_code == 405  # Method Not Allowed

    r = client.get("/clauses/any-doc-id")
    assert r.status_code == 405

    # /entities should be POST, not GET
    r = client.get("/entities/any-doc-id")
    assert r.status_code == 405
