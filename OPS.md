# SmartLawAI — Operations, Portability & Recovery

How the project survives a lost machine, a dead account, a new teammate, or a dataset
that arrives in month three.

**Design principle: the only thing that must be cloud is GPU-bound training.
Everything else runs anywhere.**

---

## 1. PORTABILITY MATRIX — what runs where

| Component | RTX 3050 (6 GB) | Any laptop (CPU) | GCP L4 | GCP A100 | RunPod / Modal | Notes |
|---|---|---|---|---|---|---|
| InLegalBERT embeddings | ✅ | 🐢 slow | ✅ | ✅ | ✅ | ~220 MB |
| Cross-encoder rerank | ✅ | 🐢 | ✅ | ✅ | ✅ | ~45 MB |
| **InLegalNLI** (judge) | ✅ | 🐢 | ✅ | ✅ | ✅ | ~110M, fast enough at serving |
| HHEM-2.1 | ✅ | 🐢 | ✅ | ✅ | ✅ | ~250M |
| Clause QA / NER / doc-cls | ✅ | 🐢 | ✅ | ✅ | ✅ | |
| BM25, chunking, RRF | ✅ | ✅ | ✅ | ✅ | ✅ | Pure CPU |
| OCR (Tesseract) | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-bound |
| PII (Presidio) | ✅ | ✅ | ✅ | ✅ | ✅ | CPU-bound |
| **Authority registry** | ✅ | ✅ | ✅ | ✅ | ✅ | It's a database |
| **All eval metrics** | ✅ | ✅ | ✅ | ✅ | ✅ | CPU: rouge, bertscore, bootstrap |
| Streamlit UI | ✅ | ✅ | — | — | — | |
| **Encoder fine-tuning** | 🐢 painful | ❌ | ✅ | ✅ | ✅ | Do it on GPU |
| **7B fp16 inference** | ❌ | ❌ | ✅ | ✅ | ✅ | Needs 24 GB |
| **7B/14B QLoRA** | ❌ | ❌ | ✅ (Arms S/C/M) | ✅ (Arm L, OOM fallback) | ✅ | Needs 24 GB + bf16 |

**Technical portability ≠ policy.** The 3050 can *run* most CPU/light-GPU components
— that column stays ✅ because nothing here is architecturally impossible on it. But
per `CLAUDE.md` §2, GCP is already provisioned and there's no more free local-GPU
tier: **L4 is the default for anything used in a measured/reported result**, A100 is
the heavy tier (Arm L, OOM fallback), and the 3050 stays a dev/UI/orchestration
machine only. Cloud is needed for every GPU step from M2 onward, not just training
and 7B inference.

---

## 2. THE THREE PORTABILITY GUARANTEES

Everything portable rests on exactly three things. Keep all three healthy and the
project cannot be stranded.

| Layer | Home | Guarantee |
|---|---|---|
| **Code** | Git (GitHub, private) | Any machine, `git clone` |
| **Environment** | `docker/Dockerfile.train` + `requirements.lock` | Byte-identical deps anywhere |
| **Artifacts** | **HF Hub** (models, datasets) + GCS (working copies) | Survives any cloud account dying |

> **HF Hub is the source of truth for anything trained. GCS is a cache.**
> GCS lives inside a billing account that can expire; HF Hub does not.

### Environment contract
```bash
# The same image runs on your laptop, Vertex, RunPod, Modal:
docker build -f docker/Dockerfile.train -t smartlaw-train:v1 .
docker run --rm smartlaw-train:v1 --device cpu  --model-id <tiny> --max-steps 5
docker run --rm --gpus all smartlaw-train:v1 --device cuda --seed 42
```
Every script takes `--device`. Never hardcode `cuda:0`. Never assume bf16 — probe
`torch.cuda.is_bf16_supported()` and fall back to fp16 + fp32 master weights.

---

## 3. WORKING ACROSS MACHINES AND TEAMMATES

Four people, four laptops, one project.

| Shared via | What |
|---|---|
| **Git** | code, configs, **frozen splits** (small JSON — commit them), `eval/results/*.json` |
| **HF Hub** (private org) | model checkpoints, LoRA adapters, processed datasets, the registry |
| **GCS bucket** | large raw corpora, intermediate artifacts, training working files |
| **`.env.example`** | committed. `.env` itself is gitignored and never shared |

### Onboarding a new machine (or teammate) — 5 commands
```bash
git clone <repo> && cd smartlawai
cp .env.example .env          # fill in HF_TOKEN, GCP project, bucket
pip install -r requirements.lock
python -m scripts.fetch_artifacts     # pulls models + datasets from HF Hub by manifest
pytest -q                             # green = you're set up correctly
```

`scripts/fetch_artifacts.py` reads `artifacts.yaml` (see §5) and hydrates everything
from HF Hub. **Nobody should ever be told "ask X for the checkpoint on their laptop."**

See §8 for the full parallel-work plan.

---

## 4. CHECKPOINTS AND ARTIFACTS — the run manifest

### Model checkpoints — synchronous triple-write

**A checkpoint isn't "saved" until it exists in three places: local disk, HF Hub, and
GCS — written in that order, each step blocking the next.** Rented pods get
reclaimed, sessions expire, laptops crash. A 20-hour run lost at hour 18 costs
nothing to prevent, and a single destination (even HF Hub) can itself be slow,
rate-limited, or briefly unreachable — the local write is what survives that.

1. **Local first** (fast, no network dependency) — the immediate resume point if the
   same VM survives.
2. **Then HF Hub, synchronously** — source of truth (§2).
3. **Then GCS, synchronously** — same-region cache, faster to pull from than HF Hub
   when resuming on a fresh VM in the same region.
4. Training only advances past a save once all three are confirmed.

```python
# train/common/checkpointing.py
class SyncCheckpointCallback(TrainerCallback):
    """Runs after every Trainer save. Blocks until local + HF Hub + GCS all confirm —
    a checkpoint is not considered saved until all three exist."""
    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        local_dir = f"{args.output_dir}/checkpoint-{step}"          # 1. already written by Trainer
        hub_rev = push_to_hf_hub(local_dir, HUB_MODEL_ID,           # 2. blocks
                                  revision=f"step-{step}")
        gcs_uri = sync_to_gcs(local_dir,                             # 3. blocks
                              f"gs://{GCS_BUCKET}/ckpt/{ARM}/seed{SEED}/step-{step}/")
        update_run_manifest(RUN_ID, step=step, locations={
            "local": local_dir, "hf_hub": hub_rev, "gcs": gcs_uri})
```

```python
trainer_args = TrainingArguments(
    save_strategy="steps", save_steps=500, save_total_limit=3,
    push_to_hub=True, hub_model_id=f"{ORG}/smartlaw-{ARM}-seed{SEED}",
    hub_strategy="every_save", hub_private_repo=True,
)
trainer = Trainer(..., args=trainer_args, callbacks=[SyncCheckpointCallback()])
```

**Resume order**: local disk (if the same VM is still alive) → HF Hub at the pinned
revision recorded in the manifest → GCS as a same-region fallback if HF Hub is slow
or unreachable.

**Cost tradeoff — size `save_steps` deliberately.** The synchronous triple-write adds
network-upload wall-clock to every save, and that time is still billed GPU-idle time
on L4/A100. Checkpointing every 50 steps on a 12-hour run means paying for dozens of
uploads that mostly get discarded (`save_total_limit=3`). Checkpoint on the cadence
I6 already implies — before any run over 1 hour — not more aggressively without a
specific reason (e.g., known-flaky preemptible capacity in the region).

**Local disk is ephemeral, with one caveat**: during an active run, local is briefly
the *newest* copy until the HF Hub/GCS sync completes a step behind it. The guarantee
is that it is never the *only* copy for longer than one save interval — not that it's
never authoritative for a moment.

### GCS Object Versioning — the checkpoint equivalent of §10's delayed git mirrors

The triple-write above protects against losing a checkpoint outright, but not against
a bug (or a human) overwriting or deleting an existing GCS blob at the same path.
Since each checkpoint already writes to a unique `step-{N}` path, this is defense in
depth rather than a gap in normal operation — but it's a one-line, GCS-native fix, so
there's no reason not to have it:

```bash
gcloud storage buckets update gs://$GCS_BUCKET --versioning
```

One-time, run before the first real training job. After this, an overwritten or
deleted object's prior version is still recoverable (`gcloud storage objects list
--all-versions`), the same protection §10's delayed mirrors give the code repo,
without needing a second bucket or a scheduled job to get it.

### `RunManifest` — written by every training run
`PipelineTrace` makes an *inference* reproducible. `RunManifest` does it for *training*.
Without it, a result is unreproducible six months later.

```json
{
  "run_id": "qlora-mistral7b-seed42-20260901T1412Z",
  "git_sha": "a1b2c3d",
  "image": "asia-south1-docker.pkg.dev/smartlawai-1/smartlaw/train:v7",
  "requirements_lock_sha256": "…",
  "split_version": "v1",
  "dataset_hashes": {"in_abs_sections": "sha256:…", "perturbations": "sha256:…"},
  "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
  "base_model_revision": "e0bc86c",
  "seed": 42,
  "config_sha256": "…",
  "hardware": "GCP g2-standard-8 / NVIDIA_L4",
  "dtype": "bfloat16",
  "output_uri": "hf://smartlawai/smartlaw-mistral7b-seed42",
  "checkpoint_locations": {
    "local": "/mnt/disks/train/ckpt/step-2500",
    "hf_hub": "smartlawai/smartlaw-mistral7b-seed42@step-2500",
    "gcs": "gs://smartlawai-1-train/ckpt/C/seed42/step-2500/"
  },
  "started_at": "…", "finished_at": "…",
  "final_metrics": {"eval_loss": 1.42}
}
```
Written to `runs/{run_id}.json`, **committed to git**, and updated after every
confirmed checkpoint (not just at the end) — `checkpoint_locations` records all three
sync targets so a resume never has to guess where the last good state lives. Every
number in the paper traces back to one of these.

**Pin base models by revision, not just name.** `Mistral-7B-Instruct-v0.3` can be
updated on the Hub; `revision="e0bc86c"` cannot.

### Retention
| Artifact | Keep | Where |
|---|---|---|
| Final adapter per seed | forever | HF Hub |
| Intermediate epoch checkpoints | until the run is validated | GCS, then delete |
| `RunManifest` | forever | git |
| `eval/results/*.json` | forever | git |
| Raw corpora | project lifetime | GCS + HF Hub dataset repo |

---

## 5. INTRODUCING A NEW DATASET MID-PROJECT

**The danger:** new data tempts you to re-split, and re-splitting invalidates every
result you already have. A model trained on split v1 evaluated on a v2 test set may
have seen those documents in training. That's silent, fatal leakage (`I5`).

### The rule: splits are versioned, never mutated
```
data/splits/v1/{train,dev,test}.json   ← frozen, referenced by all v1 results
data/splits/v2/{train,dev,test}.json   ← new, includes the new corpus
data/splits/PROVENANCE.md              ← what changed, when, why
```
Every `RunManifest` and every results file records `split_version`.
**Never compare a v1-trained model against a v2 evaluation.**

### Decide which of three paths you're taking

| Path | When | Cost |
|---|---|---|
| **A. Held-out generalisation set** ← *usually best* | New data is a different distribution (new court, new doc type, contracts) | **Zero retraining.** Becomes a paper result: "we test transfer to X" |
| **B. New split version, full retrain** | New data is the same distribution and materially larger | Retrain **every** arm and seed on v2. Nothing mixes. |
| **C. Reject for this paper** | Arrives after M8 | Note it in Future Work |

**Path A turns a disruption into a contribution.** A mid-project dataset used as an
out-of-distribution evaluation set is a *stronger* paper result than a slightly larger
training set — and it costs nothing.

### `artifacts.yaml` — the dataset manifest (commit this)
```yaml
datasets:
  - name: il_tur_summ
    source: hf://Exploration-Lab/IL-TUR
    config: summ
    revision: <commit-sha>
    sha256: …
    license: <verify>
    acquired: 2026-09-01
    used_in_splits: [v1, v2]
  - name: indian_contracts_v1
    source: gs://smartlawai-1-train/raw/contracts/
    sha256: …
    license: <verify>
    acquired: 2026-11-14
    used_in_splits: [v2]
    role: held_out_generalisation      # Path A
```
Every dataset gets a license field, and it gets filled in **before** the data is used —
not the week before submission.

---

## 6. MIGRATING GOOGLE CLOUD ACCOUNT 1 → ACCOUNT 2

Yes, and it's simpler than you'd expect: **you move the billing account, not the project.**

### What moves and what doesn't

| | Survives migration? |
|---|---|
| Project, resources, buckets, images | ✅ Everything stays — same project ID |
| **GPU quota approvals** | ✅ **Quota is per-project, so approvals survive** |
| Artifact Registry, GCS contents | ✅ |
| IAM, service accounts | ✅ |
| **Remaining credits** | ❌ **Credits belong to the billing account and do not transfer** |

That quota row is the important one — you don't re-request GPU quota after switching
billing accounts. That's the expensive part and it's preserved.

### Steps
```bash
# On account 2: find the new billing account ID
gcloud billing accounts list

# Requires: "Billing Account Administrator" on the NEW account
#           "Project Billing Manager" on the project
gcloud billing projects link smartlawai-1 --billing-account=XXXXXX-YYYYYY-ZZZZZZ

# Verify
gcloud billing projects describe smartlawai-1
```

If account 2 is a different person, first grant yourself access:
```bash
gcloud projects add-iam-policy-binding smartlawai-1 \
  --member="user:teammate@gmail.com" --role="roles/owner"
```

### Before you switch — re-arm the guardrails
Budget caps live on the **billing account**, not the project. **Your budget cap and
billing-disable function do not follow the migration.** Recreate them on account 2
*before* the first job runs, or you have GPU access with no kill switch.

### On chaining free trials
Google's free trial is limited per person/payment method. Moving a project to a
teammate's fresh trial account to harvest another $300 is against the terms — don't
plan around it. Legitimate reasons to migrate: moving to a college/department billing
account, a lab's account, or consolidating a team's spend. Those are fine.

---

## 7. DISASTER RECOVERY — what if X disappears

| Failure | Recovery | Prevention |
|---|---|---|
| **GCP credits expire / account lost** | Checkpoints are on HF Hub; rebuild image from git; retrain or serve on RunPod/Modal | HF Hub as source of truth, never GCS-only |
| **Laptop dies** | `git clone` + `fetch_artifacts` — 5 commands, §3 | Nothing lives only on a laptop |
| **Training pod preempted at hour 18** | Resume from the last HF Hub checkpoint | `hub_strategy="every_save"` |
| **HF Hub repo deleted** | GCS mirror | Push to both |
| **Dependency drift breaks the build** | `requirements.lock` + pinned base image | `I9` |
| **A result can't be reproduced** | `RunManifest` gives git SHA + image + split + seed + config hash | §4 |
| **Base model updated on the Hub** | Pinned `revision` | §4 |
| **Splits accidentally regenerated** | Committed to git — `git checkout` restores | Commit splits, never gitignore them |
| **Teammate leaves mid-project** | Everything is in git + HF Hub | Never "ask X for the file" |
| **Primary GitHub repo lost, or a bad force-push/history rewrite propagates** | `git clone` `smartlawai-backup-1` (6h behind) or `smartlawai-backup-2` (24h behind, if the problem predates backup-1's last sync too) | Delayed mirrors, §10 |

### The single test that proves recovery works
Once, in M8, **do this from a clean machine you've never used**:
```bash
git clone <repo> && cd smartlawai
cp .env.example .env && $EDITOR .env
pip install -r requirements.lock
python -m scripts.fetch_artifacts
python -m eval.run_eval --split dev --arm C
```
If the numbers match `eval/results/`, your recovery story is real. If they don't, you
have a reproducibility bug and you want to find it now — not when a reviewer asks.

Do this **before** you write the paper, not after.

---

## 8. PARALLEL WORK — what two (or more) people can do simultaneously

### The dependency graph

```
M0 Skeleton ──┬──→ M2 Retrieval ──┐
              │                    ├──→ M5 Verify+Gate ──┐
              └──→ M3 Generation ──┘                     │
                                                          ├──→ M7 Baselines ──→ M9
M1 Data ──────┬──→ M8b QLoRA ─────────────────────────────┤
              └──→ (feeds M2 index, M6 gold, M8a data)    │
                                                          │
M4 Registry ──────────────────────────────────────────────┤   ★ fully standalone
                                                          │
M6 Harness ───────────────────────────────────────────────┘   ★ fully standalone
   + gold sets
```

### The four independent tracks

| Track | Milestones | Depends on | Touches |
|---|---|---|---|
| **A — System** | M0 → M2 → M3 → M5 | M1 (data to index), M4 (registry) | `pipeline.py`, `core/`, `verify/`, `api/`, `ui/` |
| **B — Data** | M1 → M8b | nothing | `data/`, `train/qlora/` |
| **C — Registry** ★ | M4 | nothing | `registry/` |
| **D — Evaluation** ★ | M6 harness + gold sets | nothing | `eval/` |

**Tracks C and D are completely standalone.** The registry is a database plus a resolver;
the eval harness is pure functions over data structures. Neither imports the pipeline.
Track B needs nothing from anyone.

### Two-person plan

| Weeks | Person 1 — System | Person 2 — Data / Research | Conflicts? |
|---|---|---|---|
| 1–2 | **M0** skeleton | **M1** IN-Abs, section pairs, dedup, splits, OCR WER | None — different directories |
| 3–4 | **M2** retrieval, **M3** generation | **M4 registry** ★ | None |
| 5–6 | **M5** verify + gate (consumes M4) | **M6** harness + gold sets | None — M4 delivered as a module |
| 7 | UI, adversarial suite | **M8a** encoder training | None |
| 8 | **M7** baselines | **M8b** QLoRA ladder | Share GPU quota — coordinate |
| 9+ | **M9** — converge | **M9** — converge | Both |

**Person 2 owns both of the project's differentiators** (the registry and the
evaluation), which is the right concentration: those are what the paper is judged on.

### The enabler: freeze these interfaces on day one

Independent tracks only work if the contracts between them are fixed **before** anyone
starts. Spend two hours on this in week 1; it buys eight weeks of parallelism.

| Contract | Between | Define in |
|---|---|---|
| `Document`, `Chunk` (incl. `page_no`, `para_no`, `section_label`) | A ↔ B | `adapters/base.py` |
| `Citation` — what `ner.py` emits, what `registry` consumes | A ↔ C | `registry/types.py` |
| `Claim` — `{text, passage_ids, citations}` | A ↔ D | `core/types.py` |
| `Score` — `{value, status, model, revision}` | A ↔ D | `verify/types.py` |
| Split file format + `split_version` | B ↔ D | `data/splits/README.md` |
| `RunManifest` schema | B ↔ D | `runs/SCHEMA.md` |

**Write these as dataclasses with tests before splitting up.** Stub implementations
behind them. Then each track fills in its own side without ever touching the other's files.

### Parallel *within* a phase

| Phase | Parallel units | Note |
|---|---|---|
| **M4 registry** | 10 statutes | Split by statute; one person does the IPC→BNS mapping alone (it's the critical artifact) |
| **M6 gold sets** | 5 sets: retrieval, OOS, groundedness, repealed, entailment | Independent |
| **M8a encoders** | 6 tasks × 5 seeds = 30 jobs | All independent — fire concurrently up to quota |
| **M8b QLoRA** | 4 arms × seeds | All independent — parallel costs the same as sequential |
| **M9 human eval** | 4 studies | Independent |

### Where duplication is the point, not waste

**Annotation must be duplicated.** You need inter-annotator agreement (Cohen's κ /
Krippendorff's α) for every gold set and every human study — that requires ≥2 people
labelling the *same* items independently. Do not split the annotation to "save time";
you'd lose the agreement statistic the paper needs.

### Do NOT parallelize

| Anti-pattern | Why |
|---|---|
| **Two people on M0** | Merge hell on `pipeline.py` and `config.py`. One person, two days. |
| Two people editing `config.py` concurrently | Every threshold lives there — constant conflicts |
| Splitting annotation to go faster | Destroys the IAA statistic (see above) |
| Both people blocked on GPU quota | Track A needs no GPU until M7 |
| Two people on the IPC→BNS mapping | One owner, one reviewer. It is the paper's ground truth |

### If you get to four people

| Person | Track |
|---|---|
| 1 | System: M0, M2, M3, M5, UI |
| 2 | **Registry (M4)** + citation audit |
| 3 | **Evaluation (M6)**: harness, stats, gold sets |
| 4 | **Data + training**: M1, M8a, M8b, GCP ops |

Persons 2 and 3 double as the second annotator for each other's gold sets, which gives
you IAA without hiring anyone.

### The one sequencing rule

**Nobody submits a training job until Track D can measure it.** M6 gates M7 and M8's
usefulness — a trained model you cannot evaluate is not progress. If Track D falls
behind, pull someone onto it rather than training ahead.

---

## 9. CONNECTING CLAUDE CODE TO GCP — DRAFT-ONLY MODE

The GCP project, billing, and L4/A100 quota are already provisioned. "Connecting"
Claude Code to it is a contract, not an account-setup task: **Claude drafts the exact
command or a committed script; a human runs every single one.** Claude never creates
or tears down a billed resource itself. This preserves §6's project-portability model
(nothing lives only on one account) and mirrors the existing rule in `CLAUDE.md` §6:
"do not run training or model downloads inside the agent loop."

### One-time setup (human-run, not scriptable by Claude — these are interactive OAuth)
```bash
gcloud auth login                          # browser OAuth, authenticates the gcloud CLI
gcloud auth application-default login       # lets client libraries (Vertex AI SDK,
                                             # google-cloud-storage) used by training
                                             # scripts authenticate the same way
gcloud config set project <PROJECT_ID>
```
Confirm these APIs are enabled (`gcloud/deploy.sh` already assumes the first two):
`run.googleapis.com`, `artifactregistry.googleapis.com`, `aiplatform.googleapis.com`
(Vertex AI Custom Jobs — new, needed for M8 training), `storage.googleapis.com`.

### What already exists vs. what's new
| Script | Status | Purpose |
|---|---|---|
| `gcloud/deploy.sh` | **Exists** (v1) | Deploys Mistral-7B via vLLM to **Cloud Run with an L4 GPU** (`--gpu-type nvidia-l4`) for M3's serving endpoint. Reuse as-is. |
| `gcloud/train_l4.sh` | New (draft) | Submits a **Vertex AI Custom Job** on L4 — M8a encoders, Arms S/C/M. |
| `gcloud/train_a100.sh` | New (draft) | Same shape, A100 accelerator — Arm L only. |
| `gcloud/teardown.sh` | New (draft) | Tears down any Cloud Run/Compute Engine resource left running (e.g., the M3 endpoint when not actively demoed). |

**Why Vertex AI Custom Jobs for training, not a raw Compute Engine VM**: a Custom Job
self-terminates on completion or failure. A raw VM does not — it bills until someone
remembers to `gcloud compute instances delete` it. Given every GPU-hour is now paid
(`CLAUDE.md` §2), removing the "forgot to shut it down" failure mode by construction
is worth more than the extra flexibility a raw VM offers.

### Claude's role at each step
For every GCP-touching action (submitting a job, deploying an endpoint, tearing
something down), Claude writes the exact `gcloud` command or updates the relevant
script in the repo — and stops there. The human reads it, then runs it. This applies
even to idempotent or free actions (e.g., `gcloud services enable`) — draft-only
means draft-only, not "auto-run the safe-looking ones."

**Backstop Claude cannot provide**: a GCP budget alert. Set one in the console
(Billing → Budgets & alerts) once, now, before the first `train_l4.sh` run — it is a
human-configured safety net, not something meaningfully scriptable as a guardrail.

### Secrets
`HF_TOKEN` and GCP credentials live in `.env` (gitignored, `I7`) or as Application
Default Credentials from the login step above. Never a pasted service-account key,
and never a credential value typed into a chat message — if a script needs one,
reference the environment variable name, not the value.

---

## 10. DELAYED GITHUB MIRRORS

Two scheduled GitHub Actions workflows mirror `smartlawai`'s full history to
separate, private backup repos on a deliberate delay — not instant, so they survive
a bad `git push --force` or an accidental history rewrite on the primary repo rather
than propagating it. Defined in `.github/workflows/mirror-backup.yml` and
`mirror-backup-2.yml`.

| Backup | Repo | Lag | Trigger |
|---|---|---|---|
| 1 | `kramjiy/smartlawai-backup-1` | 6 hours | `cron: '0 */6 * * *'` |
| 2 | `kramjiy/smartlawai-backup-2` | 24 hours | `cron: '0 0 * * *'` |

**Both backup repos have GitHub Actions disabled on themselves** (Settings → Actions
→ General → Disable actions) — a full mirror push includes the workflow files
themselves, so without this each backup would also try to run its own copy of these
schedules pointlessly (and fail, having no secrets of its own).

Each backup's push credential is a fine-grained PAT scoped to **only that one backup
repo** (Contents + Workflows, Read and write — Workflows is required specifically
because the mirror push includes `.github/workflows/*.yml`), stored as a secret on
the **primary** repo (`smartlawai`), never on the backup repo itself:
`BACKUP_REPO_TOKEN_1`, `BACKUP_REPO_TOKEN_2`. Both workflows strip whitespace from
the token before use, since a manually-selected (rather than copy-button) token copy
can carry a trailing newline that breaks git's URL parsing outright.

**Recovery**: if `smartlawai` is ever lost or corrupted, `git clone` either backup
repo directly — each is an exact mirror as of its last scheduled run. Try backup-1
first (freshest, up to 6h stale); fall back to backup-2 (up to 24h stale) only if
backup-1 already reflects the same bad state, i.e. the problem predates backup-1's
last sync too.
