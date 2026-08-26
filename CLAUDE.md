# SmartLawAI — Project Context for Claude Code

Hallucination-aware legal RAG for Indian law. Academic project targeting a publication.
Read `PLAN.md` for the milestone you are working on. This file is the standing context.

---

## 1. NON-NEGOTIABLE INVARIANTS

Violating any of these is a bug, even if tests pass. Do not "improve" around them.

### I1 — Retrieval is ALWAYS scoped
Every retrieval call takes an explicit `scope` argument. There is no default and no
"search everything" path. `fetch_all_chunks()` must never be used for serving.

```python
# CORRECT
retrieve(query, scope=Scope(doc_ids=["doc-abc"], owner_id=uid))
# FORBIDDEN — this is a cross-document data leak
retrieve(query)
```

### I2 — Safety components NEVER fail open
A verifier that cannot run returns `status="unavailable"`, never a number.
Never substitute a default score (the old code returned `0.5` — that bug caused
fabricated metrics). Any `unavailable` component → the gate REFUSES.

```python
# CORRECT
return Score(value=None, status="unavailable", reason="hhem_load_failed")
# FORBIDDEN
return 0.5
```

### I3 — The refusal decision is application code, never an LLM
`gate.py` is deterministic. An LLM may produce evidence; only Python decides.

### I4 — Untrusted document text is fenced, never in the system role
Retrieved chunks are attacker-controlled. Wrap in `<untrusted_document_content>` tags
inside the user role, with an explicit instruction-hierarchy preamble. The LLM judge
must receive sanitized context (see `verify/sanitize.py`).

### I5 — Document-level splits, recorded, never regenerated
Train/dev/test splits live in `data/splits/*.json` keyed by `doc_id`. Never resplit.
Never let chunks of one document land in two splits. Near-duplicate dedup before split.

### I6 — Checkpoint before any run over 1 hour
Prove save AND reload works before launching. Push adapters to HF Hub every epoch.

### I7 — Secrets never in code, notebooks, or git
Local: `.env` (gitignored). Kaggle: Kaggle Secrets (`UserSecretsClient`). Never paste
a token into a notebook cell.

### I8 — Pin versions the moment a training run works
Write to `requirements.lock`. The ML stack (transformers/peft/trl/bitsandbytes) breaks
constantly. Our prior notebook logged 7 separate bugs from this.

---

## 2. HARDWARE ROUTING

We have three compute targets. **Every training/inference script must accept
`--device` and run unchanged on all of them.** Never hardcode `cuda:0`.

**GCP is already provisioned — there is no free local-GPU fallback left. Every
GPU-needing step in M2–M8 is now billed time.** Smoke-test before the real run, size
checkpoint frequency deliberately (`OPS.md` §4), and tear down instances the moment a
job finishes.

| Target | Use for | Do NOT use for |
|---|---|---|
| **Local (RTX 3050 / RTX 4060)** | Streamlit, API host, DuckDB, orchestration, CPU-only dev/tests, quick manual 7B Q4 sanity checks via Ollama on the 4060. | Anything used for a measured/reported result. |
| **GCP L4** | Default paid tier: encoder inference (InLegalBERT, rerank, HHEM, clause QA, doc-cls), all M8a encoder fine-tuning (5 seeds × 6 tasks), M2/M5 dev+test iteration, M3's serving endpoint (Cloud Run GPU, `gcloud/deploy.sh`), M7 baselines, QLoRA smoke tests, Arms S/C/M of the scale ladder. | Arm L; anything that OOMs or needs more bf16 headroom than 24 GB gives. |
| **GCP A100** | Heavy tier only: Arm L (Qwen2.5-14B) QLoRA seeds; L4 OOM fallback. | Anything L4 already handles — A100 is markedly more expensive per hour; reserve it for genuine headroom/speed need. |

**Routing rule:** L4 by default, A100 only when L4 falls short, smoke test before
real run. Total cash spend now tracked in `PLAN.md`'s `COMPUTE BUDGET` table, not
here — with the free local tier gone, keeping one number in one place matters more.
See `OPS.md` for how Claude Code drafts GCP commands (**draft-only**: Claude writes
the exact script, the human runs every one — Claude never creates or tears down a
billed resource itself) and how checkpoints sync to local disk, HF Hub, and GCS.

### Device abstraction (required in every script)
```python
# train/common/device.py
def resolve_device(arg: str | None) -> torch.device: ...
def supports_bf16() -> bool:  # torch.cuda.is_bf16_supported()
def dtype_for_device() -> torch.dtype:  # bf16 if supported else fp16
```
Never assume bf16. On T4 you get fp16 + fp32 master weights + loss scaling.

---

## 3. REPO LAYOUT

```
smartlawai/
├── CLAUDE.md, PLAN.md
├── pyproject.toml, requirements.lock, .claudeignore
├── src/smartlawai/
│   ├── config.py         # ALL thresholds, model IDs, paths. No magic numbers elsewhere.
│   ├── trace.py          # PipelineTrace — the observability + reproducibility record
│   ├── pipeline.py       # THE single pipeline. Serving and research both call this.
│   ├── scope.py          # Scope dataclass (I1)
│   ├── adapters/         # storage: base.py (ABC), local.py (DuckDB)
│   ├── core/             # ocr, preprocess, chunking, bm25, encoders, rerank, generate
│   ├── registry/         # citation registry: statutes, sections, in-force dates
│   ├── verify/           # structural, entailment, registry_check, judge, sanitize
│   ├── gate.py           # deterministic refusal decision (I3)
│   └── guardrails/       # pii, disclaimer
├── api/main.py           # FastAPI, thin handlers
├── ui/app.py             # Streamlit
├── train/
│   ├── common/           # device.py, seeding.py, checkpointing.py
│   ├── encoders/         # doc_cls.py, ner.py, clause_qa.py, risk.py, reranker.py
│   └── qlora/            # train_qlora.py (runs on 4060 / Kaggle / cloud unchanged)
├── kaggle/               # kernel-metadata.json + bootstrap notebooks per job
├── eval/
│   ├── gold/             # retrieval.jsonl, oos.jsonl, groundedness.jsonl
│   ├── metrics.py        # rouge-score + bert-score. NO hand-rolled metrics.
│   └── run_eval.py
├── data/splits/          # frozen document-level splits (I5)
└── tests/
```

---

## 4. CONVENTIONS

- Python 3.11. Type hints on public functions. `ruff` clean.
- **No magic numbers outside `config.py`.** Thresholds especially.
- Every pipeline stage appends to `PipelineTrace`. No silent steps.
- Dataclasses for DTOs, Pydantic only at the HTTP boundary.
- Named-column SQL inserts. Never positional.
- Tests for `scope`, `verify`, `gate`, `guardrails` are mandatory. Others best-effort.
- **`tests/test_scope.py::test_cross_document_isolation` must never be deleted or skipped.**

## 5. ONE DEFINITION ONLY

Hallucination rate = **% of generated claims not entailed by their cited passage**
(claim-level, HHEM-judged). This definition is used in serving AND eval AND the paper.
The old code had three conflicting definitions. If you find another, it is a bug.

## 6. WORKING WITH THIS REPO EFFICIENTLY

- One milestone per session. `/clear` between unrelated tasks.
- Use plan mode before refactors touching `pipeline.py` or `adapters/`.
- **Do not run training or model downloads inside the agent loop** — write the script,
  the human runs it on Kaggle/cloud, paste back the output.
- Paste error text directly; don't guess-and-rerun.
- `.claudeignore` must contain: `__pycache__/`, `*.ipynb`, `.smartlaw_local/`,
  `artifacts/`, `models/`, `node_modules/`, `data/raw/`, `*.duckdb`, `*.bin`

## 7. WHAT NOT TO BUILD

Do not add: agents/tool-calling, LangChain, Redis, multi-tenancy, Kubernetes,
a message queue, a React SPA (Streamlit is the UI), or the 41 CUAD clause types.
Scope is 12 Indian clause types. If a task seems to need one of these, stop and ask.
