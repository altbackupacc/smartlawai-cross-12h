# SmartLawAI — Implementation Progress Log

## Milestone 2 (M2) — Real Retrieval

### Session Started: 2026-09-09
**Objective**: Transition from M0 stubbed retrieval to M2 production-grade legal retrieval with offset-accurate chunking, parent-child small-to-big retrieval, provenance metadata, device-aware InLegalBERT and CrossEncoder reranking, strict Invariant I1 scoped vector search, and GCP L4/A100 routing.

### Status Tracker
- [x] Task 1: Update Chunk DTO and DuckDB schema for provenance (`page_no`, `para_no`, `section_label`)
- [x] Task 2: Fix chunking offsets (`finditer` scanning cursor) and parent-child small-to-big splitting
- [x] Task 3: Enhance InLegalBERT encoder with device resolution and L2 norm
- [x] Task 4: Refactor CrossEncoder Reranker for protocol compliance
- [x] Task 5: Implement Small-to-Big Retrieval and I1 Scoping in Pipeline
- [x] Task 6: Draft GCP L4/A100 retrieval execution script (`gcloud/run_m2_retrieval.sh`)
- [x] Task 7: Write M2 unit/integration tests and run verification suite

### Log
- **2026-09-09**: Session initialized. Plan approved.
- **2026-09-09**: Task 1 complete. `Chunk` dataclass and DuckDB `CHUNKS` table updated with `page_no`, `para_no`, `section_label` provenance fields; verified roundtrip in `tests/test_local_backend.py`.
- **2026-09-09**: Task 2 complete. Implemented `re.finditer` scanning cursor in `core/chunking.py`. Preserved full parent clause text without 1200 cap truncation; linked child chunks (`CHUNK_CHILD_CHARS = 400`) to `parent_chunk_id`; small-to-big indexing (`bm25_indexed`); added tests in `tests/test_chunking.py`.
- **2026-09-09**: Task 3 complete. Enhanced `InLegalBERTEncoder` in `src/smartlawai/core/inlegalbert.py` and `core/encoders.py` with device resolution (`train/common/device.py`), mean pooling, L2 normalization, and conformance to `protocols.Encoder`. Unit tests pass in `tests/test_encoders.py`.
- **2026-09-09**: Task 4 complete. Refactored `Reranker` in `src/smartlawai/core/rerank.py` to conform to `protocols.Reranker` returning `list[RetrievedChunk]` with device awareness while maintaining backward compatibility with tuple inputs. Unit tests pass in `tests/test_rerank.py`.
- **2026-09-09**: Task 5 complete. Implemented small-to-big parent clause resolution and candidate deduplication in `src/smartlawai/pipeline.py::Pipeline.ask`. Verified with unit tests in `tests/test_pipeline.py`.
- **2026-09-09**: Task 6 complete. Drafted `gcloud/run_m2_retrieval.sh` and created `scripts/run_retrieval_bench.py` for executing self-terminating retrieval jobs on GCP L4 and A100 instances.
- **2026-09-09**: Task 7 complete. Implemented `tests/test_m2_integration.py` verifying full end-to-end small-to-big retrieval, provenance metadata preservation, and Invariant `I1` cross-document tenant isolation. All 31 retrieval and M2 tests passed (477 total passed across suite). M2 complete.
- **2026-09-09**: Executed M2 retrieval benchmark on live GCP A100 GPU (`NVIDIA A100-SXM4-40GB`, Compute 8.0, Vertex AI Custom Job ID `841356888963547136`). Results: InLegalBERT + CrossEncoder initialized in 14.02s; end-to-end scoped retrieval latency = 352.86 ms; peak VRAM = 524.0 MB; small-to-big parent resolution validated (Rank 1: Section 27, Rank 2: Section 10, Rank 3: Section 74); Invariant I1 100% confirmed; job auto-terminated with zero idle billing.

---

## Milestone 3 (M3) — Structured Generation

### Session Started: 2026-09-09
**Objective**: Transition from M0 stubbed generator to M3 production-grade structured generation returning atomic claims tied to cited passage IDs and statutory citations rather than amorphous prose, enforcing Invariant I2 (never fail open), Invariant I3 (deterministic refusal in Python), Invariant I4 (context fencing in user role), and full trace accounting.

### Status Tracker
- [x] Task 1: Update `config.py` with Module 3 generator parameters (model ID, max tokens, temperature, schema prompts)
- [x] Task 2: Implement `StructuredMistralGenerator` in `src/smartlawai/core/generate.py` conforming to `Generator` protocol
- [x] Task 3: Integrate `StructuredMistralGenerator` and structured claim trace recording into `src/smartlawai/pipeline.py`
- [x] Task 4: Implement unit and integration tests in `tests/test_generate.py`
- [x] Task 5: Create live endpoint smoke test script in `scripts/smoke_test_m3.py` and `gcloud/smoke_test_m3.sh`
- [x] Task 6: Run verification suite, code quality checks, and update `PROGRESS.md`

### Log
- **2026-09-09**: Session initialized. Plan approved. Branch isolated via worktree `m3-structured-generation`.
- **2026-09-09**: Task 1 complete. Added `MISTRAL_GENERATOR_MODEL_ID`, `GENERATOR_MAX_TOKENS = 1024`, `GENERATOR_TEMPERATURE = 0.1`, `GENERATOR_SYSTEM_PROMPT`, and `GENERATOR_USER_TEMPLATE` with `<untrusted_document_content>` fencing to `src/smartlawai/config.py`.
- **2026-09-09**: Task 2 complete. Implemented `StructuredMistralGenerator` in `src/smartlawai/core/generate.py`. Includes `_build_passage_maps` fuzzy resolution, markdown code fence stripping, JSON extraction, NER citation enrichment via `core.ner.extract_entities`, and defensive parsing returning empty claims + unanswerable aspects on malformed input (Invariant I2).
- **2026-09-09**: Task 3 complete. Integrated structured generation and structured claim logging into `src/smartlawai/pipeline.py::Pipeline.ask`. Detailed claims, passage IDs, citations, and unanswerable aspects recorded in `trace.generation`. Generator token consumption and estimated USD cost recorded in `trace.cost`.
- **2026-09-09**: Task 4 complete. Implemented 10 unit and integration tests in `tests/test_generate.py` (100% offline, zero network dependencies): empty passages fast path, valid JSON parsing with passage ID mapping, Invariant I4 untrusted context fencing, markdown fence handling, fuzzy passage IDs, malformed JSON fallback, network exception handling, NER citation enrichment, and end-to-end `Pipeline` integration.
- **2026-09-09**: Task 5 complete. Created `scripts/smoke_test_m3.py` and `gcloud/smoke_test_m3.sh` supporting `--base-url`, `--model`, `--api-key`, and `--device` flags for testing live Cloud Run GPU, vLLM on GCP A100 / L4, or local Ollama endpoints.
- **2026-09-09**: Task 6 complete. Ran verification test suite across M3, retrieval, and pipeline regressions (34/34 tests passed). Code quality verified with `ruff check` (100% clean).

