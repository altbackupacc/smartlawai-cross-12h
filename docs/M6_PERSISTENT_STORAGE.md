# M6 annotation app — persistent cloud storage

Companion to `docs/M6_STREAMLIT_CLOUD_DEPLOYMENT.md`. That document covers
deploying the app itself; this one covers making its data survive the cloud
platform's ephemeral filesystem. Test evidence for everything claimed here is
in `docs/M6_PERSISTENT_STORAGE_VALIDATION.md`.

Scope: `eval/annotate/store.py`, a new `eval/annotate/cloud_storage.py`, and
a small addition to `eval/annotate/app.py`. No M6 evaluation logic
(`eval/run_eval.py`, `eval/metrics.py`, gold-set schemas) changed.

---

## Why GCS, and why *this* shape (not the shape first suggested)

`docs/M6_STREAMLIT_CLOUD_DEPLOYMENT.md` §4 first floated "back the JSONL file
with the existing GCS bucket pattern," sketched as one growing blob synced up
and down. Before building that, it was checked against the repo and against
what GCS actually is:

- **Is GCS already supported here?** Yes, and not as a stub.
  `src/smartlawai/adapters/gcloud.py` already does
  `storage.Client(project=...)` → `client.bucket(GCS_BUCKET)` →
  `blob.upload_from_string(...)`, and `train/common/checkpointing.py`'s
  `sync_to_gcs()` uses the same pattern for training checkpoints.
  `pyproject.toml`'s `[gcloud]` extra (`google-cloud-storage>=2.16`) backs
  real, working code, not a placeholder, and the package is installed and
  importable in this repo's `.venv`.
- **Is there a bucket already available?** `.env.example` documents
  `GCS_BUCKET=smartlawai-docs`, but that's for source documents. No bucket
  exists yet for annotations — one needs to be created (§"Bucket setup"
  below; a human runs this, per `CLAUDE.md`/`OPS.md`, never this agent).
- **Does GCS support the append/read pattern `AnnotationStore` uses?** Not
  directly, and this is the finding that changed the design. GCS objects are
  immutable — there is no atomic byte-range append. Treating one blob as a
  growing logfile means download → modify → re-upload on every submission,
  which is only safe under concurrent writers if you add optimistic-
  concurrency preconditions (`if_generation_match`) and a retry loop. That's
  real complexity sitting directly on the path the user explicitly said must
  never lose data.
- **Could concurrent multi-user writes cause data loss under that shape?**
  Yes — get the precondition/retry logic even slightly wrong (or skip it, as
  the original one-line sketch in the deployment doc did) and two
  annotators submitting near-simultaneously can race: the second writer's
  upload can silently clobber the first's, because both are writing to the
  same key. That directly conflicts with "annotations must never disappear"
  and "raters must never silently overwrite each other."
- **A safer minimal alternative, still using GCS, not a new database**:
  store **one immutable object per annotation** instead of one growing
  blob — `annotations/<task>/<annotation_id>.json`, keyed by the
  `annotation_id` `store.py`'s `record()` already generates as a UUID.
  Every submission is a single independent `PUT` to a key nothing else ever
  writes to. There is no shared mutable state to race over, so two
  simultaneous annotators **structurally cannot** overwrite each other —
  both objects simply end up existing. This is exactly the append-only
  guarantee `store.py`'s own docstring already commits to on local disk
  ("Nothing here edits or deletes a submitted annotation"), expressed the
  way object storage is actually meant to be used, not fought.

**Conclusion**: GCS is suitable, and no new database (Supabase, Postgres,
Firebase) is needed — annotation volume is "hundreds of rows" per
`store.py`'s own docstring, comfortably within what this shape handles. But
it is suitable *only* in the one-object-per-annotation shape. The original
"sync a single file up and down" sketch has been superseded by this design;
that's a design correction, not a walk-back of GCS as the choice.

---

## Architecture

```
Streamlit Cloud (or local machine)
        │
        ▼
  AnnotationStore.append(row)              AnnotationStore.load()
        │                                          │
        ▼                                          ▼
  1. write row to local JSONL           1. sync_to_local(): list GCS objects
     eval/annotations/<task>.jsonl         under annotations/<task>/, download
     (exactly today's behaviour,           any not already in the local file,
      unchanged)                           append them (never rewrite existing
        │                                  lines)
        ▼                                       │
  2. IF a cloud backend is configured:           ▼
     upload_annotation(): PUT the same     2. parse the now-current local file
     row as one new, never-shared GCS         exactly as before (unchanged
     object:                                  line-numbered error reporting)
     annotations/<task>/<annotation_id>.json
        │
        ▼
  gs://<bucket>/annotations/<task>/ann-xxxxxxxxxxxx.json   <-- durable,
                                                                survives
                                                                restart/redeploy
```

The local JSONL file is always the thing read and written directly — that
code path in `store.py` did not change. What changed is that, when a cloud
backend is configured, that local file becomes a **synced cache of a durable
remote log** instead of the only copy: every write also lands in GCS, and
every read first pulls down anything the local cache is missing (e.g.
because a different container instance, or a previous container before a
redeploy, wrote it).

With no cloud backend configured (`SMARTLAW_ANNOTATIONS_BACKEND` unset or
`local`, the default), `google.cloud.storage` is never imported and behavior
is identical to before this feature existed.

### Why not a shared mutable file, restated

Two `AnnotationStore` instances (two Streamlit sessions, possibly on the same
running container, possibly across a redeploy) each call `append()`
independently. Each call's `upload_annotation()` writes to
`annotations/<task>/<their-own-annotation_id>.json` — a key only that one
call ever produces (UUIDs, not a counter or a fixed filename). There is
nothing to race over: no "who wrote last wins," no partial-write
interleaving, no precondition to get right. `sync_to_local()` only ever
*appends* newly-seen objects to the local file in a deterministic
(sorted-by-id) order and never touches a line already there — so the
line-numbered corrupt-log error `store.py` already gives (`load()` raising
`SchemaError` naming the exact line) still works exactly as before for
anything already on disk.

---

## Setup requirements

| Requirement | Where |
|---|---|
| `google-cloud-storage` | Already in `pyproject.toml`'s `[gcloud]` extra; only imported when `SMARTLAW_ANNOTATIONS_BACKEND=gcs` is actually set (lazy import in `cloud_storage.py`) |
| A GCS bucket dedicated to annotations | Not yet created — see "Bucket setup" below |
| GCS credentials | Local: standard Application Default Credentials (`gcloud auth application-default login`), same as `adapters/gcloud.py` already assumes. Streamlit Cloud: a service-account JSON in Cloud's Secrets manager (see "Streamlit Cloud setup" below) — never a committed file. |

## Environment variables

All optional; the app runs exactly as it did before this feature with none
of them set.

| Variable | Default | Meaning |
|---|---|---|
| `SMARTLAW_ANNOTATIONS_BACKEND` | `local` | `local` (today's filesystem-only behavior) or `gcs` (also durable via GCS). Any other value raises `ValueError` at startup rather than silently falling back to local — a misconfigured deployment should fail loudly, not quietly drop the durability the operator thought they'd turned on. |
| `GCS_BUCKET` | — (required when backend is `gcs`) | The bucket annotations are written to. **Use a dedicated bucket** (e.g. `smartlawai-annotations`), not `smartlawai-docs` — see "Bucket setup." |
| `GCP_PROJECT` | — (optional) | Passed to `storage.Client(project=...)`, same as `adapters/gcloud.py`; omit to use the project implied by the active credentials. |
| `SMARTLAW_ANNOTATIONS_GCS_PREFIX` | `annotations` | Object-name prefix inside the bucket, in case the bucket is later shared with other data. |

These are documented (commented out, defaulted to local) in `.env.example`.

## Bucket setup (draft commands — a human runs these, not this agent)

Per `CLAUDE.md`/`OPS.md`: GCP resource creation is a human action; Claude
drafts the exact commands and never runs anything that creates or tears down
billed resources itself.

```bash
# 1. Create a bucket dedicated to annotations (do NOT reuse smartlawai-docs --
#    annotations are authoritative research data with no other durable copy
#    until exported+committed to eval/gold/*.jsonl, unlike training
#    checkpoints where OPS.md explicitly treats GCS as "a cache" because HF
#    Hub is the real source of truth there).
gcloud storage buckets create gs://smartlawai-annotations \
  --project=$GCP_PROJECT --location=$REGION --uniform-bucket-level-access

# 2. Enable Object Versioning -- the same command OPS.md §4 already
#    documents for checkpoint buckets, applied here for the same reason:
#    protects against an accidental overwrite/delete of research data.
gcloud storage buckets update gs://smartlawai-annotations --versioning

# 3. A service account scoped to just this bucket (least privilege --
#    Storage Object Admin on this one bucket, not project-wide).
gcloud iam service-accounts create smartlawai-annotations-writer \
  --project=$GCP_PROJECT
gcloud storage buckets add-iam-policy-binding gs://smartlawai-annotations \
  --member="serviceAccount:smartlawai-annotations-writer@$GCP_PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 4. A key for that service account, to paste into Streamlit Cloud's Secrets
#    manager (never committed to the repo -- see "Streamlit Cloud setup").
gcloud iam service-accounts keys create smartlawai-annotations-key.json \
  --iam-account="smartlawai-annotations-writer@$GCP_PROJECT.iam.gserviceaccount.com"
```

## Streamlit Cloud setup

1. Deploy the app per `docs/M6_STREAMLIT_CLOUD_DEPLOYMENT.md` §6.
2. In the app's **Settings → Secrets**, add:

   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key_id = "..."
   private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
   client_email = "smartlawai-annotations-writer@....iam.gserviceaccount.com"
   client_id = "..."
   # ...the remaining fields from the JSON key downloaded in step 4 above
   ```

   Paste the full contents of the downloaded service-account JSON, reshaped
   as a `[gcp_service_account]` TOML table (each JSON key becomes a TOML
   key). This is the standard Streamlit-recommended way to hand a GCP
   service account to a Cloud app, and it's read by the new
   `_bridge_gcp_credentials_from_secrets()` in `app.py`, which writes it to a
   temp file and points `GOOGLE_APPLICATION_CREDENTIALS` at it before
   anything else runs — the same Application Default Credentials path
   `adapters/gcloud.py` already relies on.
3. Also add these to the app's **Settings → Secrets** (or Environment
   variables, depending on Cloud's current UI) as plain values:
   ```
   SMARTLAW_ANNOTATIONS_BACKEND = "gcs"
   GCS_BUCKET = "smartlawai-annotations"
   GCP_PROJECT = "<your project id>"
   ```
4. Redeploy (or reboot the app from Cloud's dashboard) so the new
   secrets/env vars take effect.

Delete the local `smartlawai-annotations-key.json` from disk once it's
pasted into Cloud's Secrets manager — it should not linger anywhere on a
local machine or in the repo (`CLAUDE.md` I7).

---

## Local vs. cloud behavior

| | Local dev (default) | Cloud with `SMARTLAW_ANNOTATIONS_BACKEND=gcs` |
|---|---|---|
| Where annotations live | `eval/annotations/<task>.jsonl` only | Same local file (as a synced cache) **and** `gs://<bucket>/annotations/<task>/*.json` (durable) |
| Survives process restart | Yes (same machine, same disk) | Yes — a fresh container syncs from GCS on first `load()` |
| Survives redeployment | N/A locally (no redeploy concept) | Yes — new container has no local file, but the first `load()` pulls every prior annotation down |
| Multiple concurrent users | Single machine, single file — not a concern | Structurally safe: distinct `annotation_id`s → distinct objects, never a shared write |
| Credentials needed | None | GCS service account via Streamlit Secrets (or ADC if run on a GCP VM) |
| `google-cloud-storage` import | Never triggered | Triggered only inside `resolve_backend()`, once, at startup |

---

## Backup / recovery procedure

Two independent layers, deliberately not just one:

1. **GCS itself is the durable log** — nothing needs to be done for
   annotations to survive a restart/redeploy once `SMARTLAW_ANNOTATIONS_BACKEND=gcs`
   is set; that's the point of this feature. Object Versioning (bucket setup
   step 2) additionally protects against an accidental delete/overwrite of
   an individual object.
2. **Manual sync-to-local, for archival or a bucket-independent copy**:

   ```bash
   SMARTLAW_ANNOTATIONS_BACKEND=gcs GCS_BUCKET=smartlawai-annotations \
     python -m eval.annotate.cloud_storage --task oos
   ```

   Pulls every annotation object for that task into
   `eval/annotations/oos.jsonl`, printing how many new rows it fetched. Run
   once per task (`oos`, `retrieval`, `groundedness`, `entailment`,
   `repealed`) to mirror everything locally.
3. **The long-term archival step, unchanged from before this feature**: use
   the app's existing sidebar "Export to gold set" button
   (`store.export_gold_set()`, untouched by this change) to fold annotations
   into `eval/gold/<task>.jsonl`, then `git add`/`commit`/`push` that file —
   exactly the convention `eval/README.md` and `OPS.md` already document for
   keeping gold sets "forever, in git." GCS is durable, but git is still the
   project's actual long-term record for anything meant to back a published
   number.

---

## Deployment instructions (delta from `M6_STREAMLIT_CLOUD_DEPLOYMENT.md`)

Follow that document's §6 for the base deployment (branch, app path,
`requirements.txt`, `.python-version`). This feature adds exactly one
decision point during Cloud setup: whether to configure
`SMARTLAW_ANNOTATIONS_BACKEND=gcs` and the accompanying secrets (this
document's "Streamlit Cloud setup" section) before or shortly after the
first deploy. **Recommended: do it before pointing real annotators at the
URL** — deploying without it first works (local-file behavior, session-scoped
only, as already documented), but there is no migration step needed either
way: turning `gcs` on later starts durability from that point forward, and
`sync_to_local()`'s first run on a fresh container will pick up anything
already in the bucket, so there's no data-loss risk in switching modes,
only in delaying when durability starts.
