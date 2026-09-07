# M7 — Baselines: Progress

Status as of this writing. Written in the same honest-reporting spirit as
`M6_TEAM_OVERVIEW.md` — this records what is actually built and verified,
not what the milestone eventually intends to produce.

**Headline: the engineering scaffolding is built and tested. The milestone's
actual output — a real baselines table with real numbers — has not started.**
No real model call has been made anywhere in this work; every test mocks the
network. See "What's NOT done" below before treating any of this as results.

---

## 1. What M7 is (recap)

Per `PLAN.md`'s M7 section: five baselines that need no training, set the
quality bar, and validate the eval harness on real model outputs before any
GPU training job is submitted. The load-bearing question is whether this
project's own hybrid+rerank retrieval actually beats "BM25 + a good prompt" —
if not, that's the headline finding, not a footnote.

## 2. What's built (code, ~90% of the engineering scope)

New top-level `baselines/` package:

| File | What it does | Status |
|---|---|---|
| `baselines/config.py` | Model IDs, endpoint defaults, cutoff-date table | Built; `MODEL_CUTOFF_DATES` intentionally empty (see #4) |
| `baselines/retrieval.py` | `scoped_bm25`, `scoped_hybrid` — I1-scoped retrieval | Built, tested |
| `baselines/frontier_client.py` | Anthropic API client, I4-safe | Built, tested (mocked SDK only) |
| `baselines/generators.py` | `MistralGenerator`, `SaulLMGenerator`, `FrontierGenerator` | Built, tested (mocked network only) |
| `baselines/citation_existence.py` | The existence-only baseline (reuses `core/ner.py`) | Built, tested against `sample_docs/sample_judgment.txt` |
| `baselines/contamination.py` | T1's date-resolution heuristic + cutoff split | Built, tested; produces real (non-fabricated) results on real text |
| `baselines/run_baselines.py` | Orchestrator | Built, tested; CLI entry point (`main()`) is a stub — see #4 |

Also: `gcloud/deploy_saullm.sh` (drafted, not run), one backward-compatible
extension to `src/smartlawai/core/mistral_client.py`, a new `[baselines]`
extras group in `pyproject.toml`, new `.env.example` entries.

## 3. What's verified

- 30 new tests across 6 files, all passing, all network/API calls mocked
- Full project suite: 91/91 passing after these changes (zero regressions)
- `ruff check` clean on every new/modified file (5 real issues were caught
  and fixed during verification: an unnecessary `dict()` call, two unsorted
  import blocks, two `import x as x` aliases)
- One **pre-existing, unrelated** flaky test found during verification:
  `tests/test_local_backend.py::test_faiss_lifecycle` fails ~1-in-5 runs due
  to unseeded `np.random.rand` in a FAISS nearest-neighbor assertion. Not
  touched by this work; already documented as a known flake in
  `M6_TEAM_OVERVIEW.md` section 9.

## 4. What's NOT done — the actual milestone output

PLAN.md's definition of "done" for M7 is a real comparison table and a
surfaced finding, not the existence of code. None of the following has
happened yet:

- **No real model call has been made anywhere.** Every generator/client is
  exercised only against mocked responses in tests.
- **`MODEL_CUTOFF_DATES` is empty.** `split_by_cutoff()` will raise `KeyError`
  until a human sources and cites each model's real training-cutoff date from
  its provider's model card/announcement.
- **`SAULLM_MODEL_ID` (`Equall/Saul-7B-Instruct-v1`) is an unverified guess**,
  and no SaulLM endpoint has been deployed. `gcloud/deploy_saullm.sh` is
  drafted but a human must confirm the exact HF repo ID/license and run it.
- **No frontier API key is configured**, so the Anthropic-backed conditions
  (`bm25_frontier`, `frontier_zero_shot`, `frontier_rag`) have never actually
  run.
- **`run_baselines.py`'s CLI (`main()`) can't load real questions** — there is
  no `eval/gold/*.jsonl` to read from, because M6 (the eval harness) is not
  actually present in this checkout (verified directly: no `eval/` directory
  exists on any branch). `run()` itself works and is tested with hand-built
  `(question, scope)` items; only the gold-set loading is missing, and it's
  missing because its dependency genuinely doesn't exist yet.
- **The citation-existence baseline hasn't been run against real repealed
  citations** — it works correctly against `sample_docs/sample_judgment.txt`,
  but the actual demonstration PLAN.md wants (reproducing arXiv 2606.00898's
  gap on IPC→BNS citations) needs `eval/gold/repealed.jsonl`, which doesn't
  exist yet either.
- **The headline "watch for" finding — hybrid+rerank vs. BM25+frontier — has
  not been produced.** This requires all of the above to exist first.
- **`citation_existence.py`'s output shape for `eval/metrics.py` is inferred**
  from a one-paragraph description in `M6_ONBOARDING.md`, not a confirmed
  interface (M4's real `RegistryResult` doesn't exist yet either).

## 5. Rough completion estimate

| Dimension | Estimate |
|---|---|
| Code written against PLAN.md's M7 spec | ~90% |
| Lint/test verification of that code | 100% of what's testable without real network/API access |
| Real baseline results produced | 0% |
| **Overall milestone** (PLAN.md's actual done-when: a real table + finding) | **~40%** |

## 6. What unblocks the rest

In rough dependency order:

1. M6 actually lands in this checkout (gold sets + `eval/metrics.py`/`eval/stats.py` real, not just specified).
2. A human sources and cites real `MODEL_CUTOFF_DATES` entries in `baselines/config.py`.
3. A human confirms `SAULLM_MODEL_ID` and its license/gating status.
4. A human runs `gcloud/deploy_saullm.sh` (billed) and sets `ANTHROPIC_API_KEY`/`FRONTIER_MODEL` (billed per call) — neither is something this agent triggers itself.
5. `run_baselines.py`'s `main()` gets wired to the real gold-set loader once (1) exists.
6. Run all seven conditions, produce `baselines/results/*.json`, and write up the headline comparison.
