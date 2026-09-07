# M6 annotation app — Streamlit Community Cloud deployment

Scope: `eval/annotate/app.py` only. This does not change M6 evaluation logic
(`eval/run_eval.py`, `eval/metrics.py`, `eval/stats.py`, gold-set schemas) and
does not touch `src/smartlawai/`. `streamlit run eval/annotate/app.py` continues
to work locally exactly as before.

Findings below come from reading the actual repo (`pyproject.toml`, every
import in `eval/annotate/app.py`'s dependency chain, `eval/config.py`,
`.gitignore`, `.env.example`, and `git status`/`git remote`), not from
assumptions about Streamlit Cloud in general.

---

## 1. Deployment Readiness

**READY WITH MINOR CHANGES.**

The app itself is unusually cloud-friendly for a project this size:

- Its full import chain (`eval/annotate/app.py` → `eval.config`,
  `eval.schemas`, `eval.gold_sets`, `eval.annotate.store`,
  `eval.annotate.agreement`, `eval.annotate.guidelines`) imports **nothing
  outside the Python standard library except `streamlit`**. No `duckdb`,
  `faiss`, `torch`, `transformers`, `pytesseract`, etc. This was verified by
  grepping every `import`/`from` line in that chain, not inferred from
  `pyproject.toml`.
- Every path the app touches (`eval.config.GOLD_DIR`, `ANNOTATIONS_DIR`, etc.)
  is computed as `Path(__file__).resolve().parent...` — relative to the
  repo's own file layout, not to a local machine's absolute path or `cwd`.
  This works unchanged after a fresh `git clone` on Cloud's infrastructure.
- The app needs **no secrets, API keys, or database connection** to run in
  its current form — it never imports `smartlawai`, so none of
  `MISTRAL_BASE_URL`, `HF_TOKEN`, `GCS_BUCKET`, etc. from `.env.example` are
  touched.

What stops it from being flatly READY:

1. **No `eval/gold/*.jsonl` in git yet.** `git status` shows `eval/gold/` as
   untracked (`??`), along with all of `eval/`, `eval/annotate/`, and
   `docs/`. Cloud deploys from a GitHub branch — anything not committed and
   pushed simply doesn't exist there. This is the single biggest blocker
   right now, and it's a `git add`/`commit`/`push` away, not a code change.
2. **No dependency manifest Streamlit Cloud reads by default.** There was no
   `requirements.txt` (Cloud does not use `pyproject.toml` extras the way
   your local `pip install -e ".[eval,ui,dev]"` does). Added — see §3.
3. **Annotation storage is local-filesystem and will not survive Cloud's
   ephemeral disk.** This is the important one — see §4. It does not block a
   first deploy, but it means annotations collected on Cloud are not safe
   until you act on §4's recommendation.

One rendering fix in `app.py` was required — see the note at the end of §2.
Nothing else in `app.py`, `store.py`, or any other M6 logic file was changed;
the fix does not touch scoring, labels, schemas, or storage.

---

## 2. Required Repository Changes

| # | Change | Status |
|---|---|---|
| 1 | Commit and push `eval/`, `eval/gold/*.jsonl`, `eval/annotate/`, `docs/`, and the other currently-untracked M6 files to the branch you deploy from | **Not done — requires your `git add`/`commit`/`push`, see §6** |
| 2 | Add a root `requirements.txt` scoped to what `app.py` actually imports | **Done** — `requirements.txt` |
| 3 | Pin the Cloud Python version to match `pyproject.toml`'s `requires-python = ">=3.11"` (the app uses `datetime.UTC`, added in 3.11 — it will hard-fail on 3.9/3.10) | **Done** — `.python-version` (`3.11`) |
| 4 | Decide and (separately, later) implement persistent annotation storage | **Not done — recommendation only, see §4** |
| 5 | Fix `render_item()`'s bare ternary in `app.py` that crashed on every item render under `streamlit==1.63.0` | **Done** — see note below |

**One functional fix, discovered during verification, not part of the
original plan:** `eval/annotate/app.py`'s `render_item()` had a bare
multi-line ternary statement (`st.info(value) if field_name in (...) else
st.text_area(...)`). Streamlit's "magic" turns bare expression statements
into an implicit `st.write(...)` call; with `streamlit==1.63.0` (the version
installed locally, satisfying the `>=1.30` floor in `pyproject.toml`), that
implicit `st.write` fell through to a variable-name-inspection path that
tried to `ast.parse()` a source slice of the multi-line ternary and threw
`SyntaxError: '(' was never closed` on **every item render** — the app
crashed as soon as a rater reached the first question, before this fix. This
predates and is unrelated to the Cloud-deployment changes; it would have hit
local usage too under this streamlit version. Rewritten as an explicit
`if/else` (same two calls, same output, no longer a bare expression magic
picks up) — confirmed by re-running the app and completing a full
submit-and-next cycle end to end (item rendered, form submitted, sidebar
throughput/agreement updated, next item loaded). No other line in `app.py`
changed, and `eval/*.py`, `src/smartlawai/*`, and `pyproject.toml` were not
touched.

---

## 3. Dependencies

`app.py`'s only third-party import is `streamlit`. Everything else in its
chain (`eval.config`, `eval.schemas`, `eval.gold_sets`, `eval.annotate.store`,
`eval.annotate.agreement`, `eval.annotate.guidelines`) is stdlib
(`dataclasses`, `hashlib`, `json`, `pathlib`, `typing`, `datetime`, `random`,
`uuid`, `itertools`, `collections`) plus internal `eval.*` modules.

Added **`requirements.txt`** at the repo root:

```
streamlit>=1.30
```

Why not point Cloud at `pyproject.toml` or the `[ui]` extra directly: when a
`requirements.txt` is present, Streamlit Cloud uses it exclusively and ignores
`pyproject.toml`. Pointing it at the full project (`pip install -e .` or an
extras combination) would also pull in the base `dependencies` list —
`duckdb`, `faiss-cpu`, `pytesseract`, `pdf2image`, `pypdf`, `langdetect` —
none of which `app.py` needs, all of which cost cold-start time and count
against the free tier's resource cap. `requirements.txt` is a single line
because that's genuinely everything the app needs; it isn't a stand-in for
the full dependency set.

This does not duplicate `pyproject.toml` in a way that can drift unnoticed:
`requirements.txt` carries a comment pointing back at the `ui` extra as the
source of truth for the version floor, and CI (`eval-regression.yml`) still
installs from `pyproject.toml` for everything else, so the two files serve
different, non-overlapping audiences (Cloud vs. local/CI).

Also added **`.python-version`** (`3.11`) so Cloud's interpreter matches
`pyproject.toml`'s `requires-python` and `store.py`'s use of
`datetime.UTC` (Python 3.11+).

---

## 4. Persistent Annotation Storage

**This is the part that actually matters before you point real annotators at
a Cloud URL.**

### What happens today

`AnnotationStore` (`eval/annotate/store.py`) is, by its own docstring, "One
JSONL file per task under `eval/annotations/`" — an append-only local file,
deliberately not a database ("adding DuckDB here would put an eval concern in
the serving backend's territory"). `eval/config.ANNOTATIONS_DIR` resolves to
`<repo>/eval/annotations/`, and `.gitignore` explicitly excludes
`eval/annotations/` — correct for local dev (raw per-rater logs aren't meant
to be committed; only the *exported* `eval/gold/*.jsonl` is), but it means
there is no copy of this data anywhere except the container's local disk.

### Why that fails on Streamlit Community Cloud

Streamlit Community Cloud's filesystem is **not durable**:

- **App restart** (crash, resource limit, platform maintenance): local disk
  is reset. Anything written to `eval/annotations/*.jsonl` since the last
  redeploy is gone.
- **Redeployment** (a `git push` to the deployed branch, including pushing
  the `eval/gold/` commit from §2): Cloud rebuilds the container from a fresh
  clone. Same result — local writes are gone.
- **Inactivity sleep**: Community Cloud apps sleep after a period of no
  traffic and wake on the next visit; waking is not guaranteed to preserve
  the same container/disk.
- **Multiple users, same moment**: Community Cloud normally runs **one**
  container per app, shared by all concurrent visitors on that URL — so
  annotators working *at the same time*, before any restart, actually do
  share the same file and do see each other's submissions live (this is not
  the multi-instance, no-shared-disk problem some other PaaS have). The
  failure mode here is purely **durability over time**, not
  cross-user isolation in the moment.

Net effect: an annotator could run a full session, click "Export to gold
set," and then lose the underlying raw log (and, if the export itself hasn't
been downloaded off the container before the next sleep/redeploy, the export
too) with no warning from the app. For an academic gold set whose provenance
your `PROVENANCE.md` / paper depends on, that's a real risk, not a
theoretical one.

### Options considered

| Option | Fit for M6 now |
|---|---|
| **Supabase / Postgres** | Overkill. Annotation volume is "hundreds of rows" (store.py's own description). A managed relational DB adds a schema, a connection secret, and a new failure mode for what is currently eight lines of `json.dumps` per row. |
| **GitHub-based storage** (commit each annotation via the GitHub API) | Fits the existing convention (`eval/gold/*.jsonl` already lives in git) but is the wrong grain: committing on every single form submission is noisy (hundreds of commits for a pilot), adds GitHub API latency to the submit path, and the "current file SHA" precondition the Contents API requires means two raters submitting near-simultaneously can hit real write conflicts — exactly the multi-user case §5 needs to not break. |
| **Streamlit-compatible cloud storage (object storage)** | Good fit. `pyproject.toml` already has a `[gcloud]` extra with `google-cloud-storage`, and `.env.example` already has `GCS_BUCKET` — this repo already treats GCS as its sanctioned "not local disk" backend (`SMARTLAW_BACKEND=gcloud` is a first-class mode in `CLAUDE.md`). Reusing that means no new *kind* of infrastructure, just a new bucket/prefix. |
| **Other lightweight persistent storage (e.g. Streamlit `st.connection` to Google Sheets)** | Also viable and arguably even simpler to set up (no bucket, human-readable in a browser), but it's a dependency and pattern not used anywhere else in this repo, and it doesn't map cleanly onto the existing "one JSONL file, append a line" shape `AnnotationStore` already has. |

### Recommendation — now implemented

The recommendation above (back `AnnotationStore` with the repo's existing GCS
pattern, not a new database) has been **built, tested, and documented in
full** in a follow-up pass — see **`docs/M6_PERSISTENT_STORAGE.md`** for the
complete architecture, and **`docs/M6_PERSISTENT_STORAGE_VALIDATION.md`** for
the test evidence. Summary of what changed since this section was first
written (verified: the earlier recommendation's shape — one growing blob —
turned out to be the wrong grain; see that doc's design-question section for
why one-object-per-annotation is what actually shipped):

- New `eval/annotate/cloud_storage.py`: a `GCSAnnotationBackend` that writes
  **one immutable GCS object per annotation**, plus `resolve_backend()`,
  env-driven (`SMARTLAW_ANNOTATIONS_BACKEND=local|gcs`).
- `AnnotationStore.append()`/`.load()` in `store.py` gained a thin, additive
  hook into that backend — local-only behavior is byte-for-byte unchanged
  when no cloud backend is configured (the default), so everything in §1–§3
  of this document still holds as written.
- `app.py` gained a small, inert-unless-configured block that bridges a
  Streamlit Cloud secret into Application Default Credentials for GCS.

Cloud-collected annotations are no longer session-scoped once
`SMARTLAW_ANNOTATIONS_BACKEND=gcs` is configured on Cloud — see the other doc
for setup. **Deploying without that env var set still works** (local-file
behavior, same caveat as before: session-scoped only) — this document's
readiness verdict in §1 does not change, since making storage durable was a
separate, explicit follow-up request, not a blocker to the base app running.

---

## 5. Multi-User Safety

Checked `eval/annotate/store.py` and `tests/test_annotation_store.py`
directly — there is no locking, no transaction, and no annotation-store test
covering concurrent writers.

- **Annotation conflicts**: `AnnotationStore.append()` does a single
  `open(path, "a")` + one `write()` call per submission. For lines of this
  size, that's effectively atomic on the shared container's filesystem, so
  two raters submitting at the same instant won't corrupt the file or
  interleave partial lines.
- **Duplicate annotations**: the store is append-only by design — a rater
  resubmitting (e.g., two open browser tabs on the same item) doesn't
  overwrite anything; it appends a second row. `latest_by_rater()` then
  resolves by `created_at`, so the later submission is what counts toward
  agreement and export. This is intentional (`store.py`'s docstring: "a
  corrected judgment is a new row with a later timestamp"), not a bug, but it
  does mean a rater who double-submits leaves both rows in the raw log
  forever — expected and harmless for this use case.
- **Race conditions**: the one place this could matter is
  `export_gold_set()` reading `latest_by_rater()` while another rater's
  `append()` is mid-write — in practice not a real risk given Streamlit's
  single-threaded-per-session request handling and the size of these writes,
  but worth knowing this was never explicitly tested for.
- **Rater ID issues**: `rater_id` is a free-text sidebar field with no
  validation against a fixed roster (`app.py`'s own docstring: "opaque
  strings," deliberately annotator-pool-agnostic). Two annotators who
  accidentally type the same ID would be merged into one rater for agreement
  purposes — this is a **process** risk (assign IDs out of band, e.g. in the
  Slack/sheet you coordinate the pilot in), not something the app can catch,
  and not something this deployment pass should add validation for without
  touching M6 logic.

None of this needs a code change to deploy safely for a small pilot (the
scale `PILOT_SIZE = 20` and `TARGET_N` in `eval/config.py` imply — tens of
raters, hundreds of items, not concurrent high-throughput writes). It's
listed here so it's a documented, known limitation rather than a surprise.

---

## 6. Deployment Instructions

**Prerequisite — do this first (not yet done, per §2):**

```bash
git add eval/ docs/ requirements.txt .python-version .github/workflows/eval-regression.yml
git status   # review what's staged before committing
git commit -m "Add M6 eval harness, gold sets, annotation app, and Cloud deployment prep"
git push origin m1-data-foundation
```

(`git status` currently shows the repo on branch `m1-data-foundation` with
`eval/`, `eval/gold/`, `eval/annotate/`, and `docs/` all untracked — adjust
the branch name if you deploy from a different one, e.g. after merging to
`main`.) This is a repository-changing action — review the `git status`
output yourself before running `git commit`/`git push`; nothing here has been
committed or pushed on your behalf.

**Then, on Streamlit Community Cloud:**

1. **Repository requirements**: the GitHub repo (`kramjiy/smartlawai`, per
   `git remote -v`) must be visible to Streamlit Cloud — either public, or
   private with Cloud's GitHub app granted access to it. `requirements.txt`
   and `.python-version` must exist on the branch you select (done, once
   pushed per the prerequisite above).
2. Go to **share.streamlit.io** → **New app**.
3. **Branch selection**: choose the branch that has the commit from the
   prerequisite step above (e.g. `m1-data-foundation`, or `main` if you merge
   first).
4. **Application path**: set the main file path to
   `eval/annotate/app.py`.
5. **Python/dependency setup**: Cloud auto-detects `requirements.txt` at the
   repo root and `.python-version` for the interpreter — no manual entry
   needed for either. Under **Advanced settings**, double-check the Python
   version shows `3.11` before deploying.
6. Click **Deploy**. First build installs `streamlit` from
   `requirements.txt` — this is a small, fast install (no compiled ML
   wheels), so expect a short first build compared to a full `[eval,ui,dev]`
   install.
7. **Accessing the URL**: once the build finishes, Cloud assigns a URL of the
   form `https://<app-name>-<random>.streamlit.app` (or a custom subdomain
   you pick at creation, `https://<your-choice>.streamlit.app`). Share that
   URL with annotators — no additional server or port config needed, this
   replaces `localhost:8501`.
8. Redeploying after future changes is automatic: Cloud watches the deployed
   branch and rebuilds on every push to it.

---

## 7. Security

**Should the app be public?** Not as a default. It's an internal research
tool — the rater-facing screens show real gold-set item text (queries,
claims, passages) from `eval/gold/*.jsonl`/queue files, and an open URL with
no access control means anyone with the link (found, guessed, or leaked) can
submit annotations that pollute your inter-annotator-agreement numbers, or
just read unpublished eval content before the paper is out.

**Is authentication needed at this stage?** Recommend yes, at the lightest
tier Cloud offers, not a custom login system:

- Streamlit Community Cloud's built-in **viewer access control** (under the
  app's Settings → Sharing) can restrict the URL to a specific allow-list of
  email addresses / Streamlit accounts. This is a Cloud platform setting, not
  a code change, and it's the right amount of security for "a handful of
  named annotators," not the "public internet" the URL is reachable from by
  default.
- Do **not** build a custom password/login flow inside `app.py` for this —
  that would be new logic in the one module explicitly scoped to stay
  Streamlit-only (`eval/annotate/__init__.py`: "Nothing here imports
  `smartlawai` except app.py's optional smoke-input path"), and it's
  exactly the kind of infrastructure `CLAUDE.md` §7 says not to add for a
  research-stage tool when the platform already provides it for free.

**Secrets**: the app needs none today (§1). If you implement the GCS-backed
storage from §4 later, the *only* secret involved is a GCS service-account
key, and it belongs in Streamlit Cloud's **Secrets** manager
(`st.secrets`) — set via the Cloud dashboard, never committed to the repo,
never pasted into a notebook or `app.py`, consistent with `CLAUDE.md` I7.

---

## 8. Cost

Free-tier suitable. Streamlit Community Cloud's free tier gives each app
roughly 1 GB RAM / 1 CPU and sleeps the app after a period of inactivity
(auto-wakes on the next visit). Given the app's entire dependency footprint
is `streamlit` + stdlib — no torch, no vector index, no model weights loaded
at import time — it sits well inside those limits; there's nothing here that
would push you toward a paid tier.

The one thing that does have a cost if you implement §4's GCS
recommendation: a GCS bucket for a few hundred small JSONL rows is
effectively free-tier-eligible on GCP (a handful of KB stored, occasional
writes) — negligible next to the GPU spend `CLAUDE.md`'s COMPUTE BUDGET
table already tracks, but worth a line in that table once it exists, per
`CLAUDE.md`'s "one number in one place" rule.
