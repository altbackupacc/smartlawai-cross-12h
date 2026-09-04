# RR <-> SUMM Alignment Investigation

**Verdict: Path 2 (frontier-generated section summaries + human validation) is the section-pair construction method for the whole corpus.**

## What was checked (in order, per M1_ONBOARDING.md #6)

1. **Real schema** (not assumed):
   - `summ`: `id, document, summary, num_doc_tokens, num_summ_tokens` (document-level, professional headnote as `summary`). Split counts: train=7030, test=100.
   - `rr`: `id, text, labels, expert_1, expert_2, expert_3` — `text` is a per-sentence list, `labels` a matching per-sentence rhetorical-role id list (0-12), `expert_N` carry per-annotator primary/secondary label sequences for inter-annotator agreement. Split counts: CL_train=40, CL_dev=5, CL_test=5, IT_train=40, IT_dev=5, IT_test=5.

2. **Shared joinable document id**: `summ` has **7130** unique ids (plain numeric strings, e.g. `'427'`, `'5037'`). `rr` has **100** unique ids, which are structured court-case identifiers (e.g. `CCI_All_India_Distillers_Association_vs_Haldyn_Glass_...`) drawn from Competition Commission of India / High Court antitrust matters — a different document universe and a different legal domain than `summ`'s Supreme Court judgments. **Overlap: 0 documents (0.00% of summ).**

3–4. **RR-span-to-headnote alignment quality, coverage**: not evaluated — moot. Per M1_ONBOARDING.md #6 point 2: *"If the two configs draw from disjoint document universes, Path 1 is dead immediately, regardless of alignment quality."* That is exactly this case: zero shared documents, so there is no pair of (rr spans, summ headnote) to test alignment on in the first place.

## Decision rule applied

Coverage = 0.00% (< 70% threshold) -> **Path 2 primary**.

No docs passed clean alignment (there are none to check), so there is no bonus gold subset to carry forward from Path 1 — this is a pure Path 2 corpus, not a hybrid one. That absence, not a judgment call, is what decided this.

## What Path 2 means concretely for this corpus

`summ` already provides a full-document (judgment, professional headnote) pair per document — this is *document-level* gold, not silver, and not itself in question. What Path 2 supplies is the missing piece RR was meant to give for free: a way to split that single document-level pair into several *section-level* pairs. `build_pairs.py` uses a frontier model to (a) segment the judgment text into sections and (b) produce a section-level summary aligned to the existing headnote, each pair tagged `provenance: path2-silver`. This is silver data by construction (RESEARCH.md T4) and must be human-validated (200-item sample, acceptance rate reported plainly) before being treated as usable training data.

## Cost implication — action required before spending anything

Path 2 triggering at full corpus scale (~7,030 train + 100 test `summ` docs) has **zero line item in `PLAN.md`'s COMPUTE BUDGET table**. Per M1_ONBOARDING.md #6, `build_pairs.py --dry-run` must report an estimated token volume and dollar cost, and that estimate must be surfaced to the user for explicit go-ahead before any real generation pass runs.
