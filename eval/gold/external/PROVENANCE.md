# `eval/gold/external/` — third-party evaluation sets

Sets in this directory are **read-only and never relabelled or modified**. Their
entire value is being independent of anything built in this project — the
paper's answer to "you built your own test set" (`RESEARCH.md` §2.5's Hard Rules
and threat T11). Any local edit destroys that independence.

This directory is currently **empty**. Why, for the one set `PLAN.md` M6 names:

---

## ClaimRAG-Law (968 validated claims) — NOT publicly released as of 2026-09-05

`PLAN.md` M6 lists "pull ClaimRAG-Law's 968 validated claims as the external
judge evaluation set" as a deliverable, used later by M8a as an out-of-domain
generalisation check for InLegalNLI. `M6_ONBOARDING.md` §10 scopes M6's part
narrowly: *locate and cache it, store it read-only, do not relabel it — and "if
it turns out not to be publicly released, say so explicitly rather than
substituting something else silently."*

**It is not publicly released.** Stating that explicitly, per those instructions.

### What was checked (2026-09-05)

| Source | Finding |
|---|---|
| arXiv abstract page [2605.21071](https://arxiv.org/abs/2605.21071) | No data-availability statement, no repository link, no dataset licence. |
| Full HTML, [v4](https://arxiv.org/html/2605.21071v4) | States *"The dataset and its generation code are publicly available"*, but the accompanying footnote says the links were **omitted for double-blind review**. No GitHub, HuggingFace, Zenodo or OSF URL appears anywhere in the paper. |
| Web search for a mirror/repo under the authors' names | No dataset repository located. |

### What the paper does establish

Useful for planning even without the data. ClaimRAG-Law is a bilingual,
bi-jurisdictional claim-level legal RAG benchmark from the University of
Luxembourg: 317 expert-validated QA pairs over the GDPR (English) and
Luxembourg civil law (French), carrying 968 manually validated claims, of which
the legal expert judged 838 (86.6%) correct with respect to their source legal
context.

### Consequences for this project — flag these, do not route around them

1. **The external generalisation check is blocked**, not delayed by a download.
   `RESEARCH.md` §2.5 lists it under Hard Rules and T11 answers a specific
   reviewer objection with it. That answer is currently unavailable.
2. **The domain gap is larger than "external" implies.** GDPR and Luxembourg
   civil law, in English and French, against a project scoped to Indian
   statutory and case law. Even once released, this measures cross-jurisdiction,
   cross-language transfer — a strong result if it holds, and a result that
   needs framing as such rather than as a like-for-like held-out set.
3. **No substitute has been chosen.** Silently swapping in another benchmark is
   exactly what `M6_ONBOARDING.md` §10 forbids. Picking a replacement is a
   research decision for M8a/M9, made explicitly and recorded here, not a gap
   for M6 to paper over.

### To resolve

Re-check on the paper's de-anonymised camera-ready version, where the omitted
links should be restored; or contact the authors directly. `RESEARCH.md` §1's
"re-run the literature search" discipline exists to catch exactly this kind of
external dependency early — it has been caught, and it is open.

When the data is obtained, place it here as `claimrag_law.jsonl`, record its
source URL, retrieval date, licence and checksum in this file, and tag its rows
with `judge_source: "claimrag-law"` (`EntailmentItem` already carries that
field) so external rows never mix silently with internally-produced ones.
