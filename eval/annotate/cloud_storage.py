"""Optional GCS durability for AnnotationStore, additive to the local JSONL log.

Why GCS, and why THIS shape: `docs/M6_PERSISTENT_STORAGE.md` has the full
writeup. In short -- `src/smartlawai/adapters/gcloud.py` and
`train/common/checkpointing.py` already talk to GCS in this project, so the
*service* is not new infrastructure. But GCS objects are immutable; there is
no atomic byte-append, so treating one GCS blob as a growing logfile (the
local design) would need download-modify-reupload with generation-match
preconditions to be safe under concurrent writers -- real complexity, and
exactly the kind of thing that can silently lose an annotation if the retry
logic is even slightly wrong. Instead, every annotation is its own immutable
object, named by its already-unique `annotation_id`. Two raters submitting at
the same instant write to two different keys -- there is no shared mutable
state to race over, so one annotation can never silently clobber another.
This is the same append-only guarantee `store.py` already gives on local
disk, expressed the way object storage is actually meant to be used.

Imports nothing from `smartlawai` (OPS.md §8) -- `eval/` never imports the
serving pipeline or its adapters, and `google-cloud-storage` itself is only
imported lazily, inside `resolve_backend()`, so a local-only run (the
default) never needs the package installed at all.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from eval import config


# --------------------------------------------------------------------------- #
# The minimal surface this module needs from google-cloud-storage -- as a
# Protocol so tests can hand in an in-memory fake with zero mocking of the
# real SDK and zero network/credentials (see tests/test_cloud_storage.py).
# --------------------------------------------------------------------------- #
class _BlobLike(Protocol):
    name: str

    def upload_from_string(self, data: str, content_type: str = ...) -> None: ...
    def download_as_text(self) -> str: ...


class _BucketLike(Protocol):
    def blob(self, name: str) -> _BlobLike: ...
    def list_blobs(self, prefix: str) -> Iterable[_BlobLike]: ...


DEFAULT_PREFIX = "annotations"


class GCSAnnotationBackend:
    """One immutable GCS object per annotation under
    `<prefix>/<task>/<annotation_id>.json`.

    Constructed with an already-built bucket object -- this class never talks
    to `google.cloud.storage` itself, so it is trivially testable with a fake.
    """

    def __init__(self, bucket: _BucketLike, prefix: str = DEFAULT_PREFIX) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _blob_path(self, task: str, annotation_id: str) -> str:
        return f"{self._prefix}/{task}/{annotation_id}.json"

    def upload_annotation(self, task: str, row: dict[str, Any]) -> None:
        """Write one annotation as its own object. Never overwrites another
        annotation's object -- `annotation_id` is unique per submission
        (store.py's `record()`), so this key is never shared."""
        annotation_id = row["annotation_id"]
        blob = self._bucket.blob(self._blob_path(task, annotation_id))
        blob.upload_from_string(
            json.dumps(row, ensure_ascii=False, sort_keys=True),
            content_type="application/json")

    def sync_to_local(self, task: str, local_path: Path) -> int:
        """Append any annotation objects not yet in the local mirror.

        Only ever appends -- existing local lines are never rewritten or
        reordered, so `AnnotationStore.load()`'s line-numbered error
        reporting for anything already on disk is unaffected. Returns the
        number of rows pulled down. Downloaded in a deterministic
        (sorted-by-annotation_id) order so two syncs of the same remote state
        produce byte-identical appends.
        """
        known_ids = _existing_annotation_ids(local_path)
        prefix = f"{self._prefix}/{task}/"
        new_rows: list[tuple[str, str]] = []
        for blob in self._bucket.list_blobs(prefix=prefix):
            annotation_id = Path(blob.name).stem
            if annotation_id in known_ids:
                continue
            new_rows.append((annotation_id, blob.download_as_text()))
        new_rows.sort(key=lambda pair: pair[0])

        if not new_rows:
            return 0
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("a", encoding="utf-8") as fh:
            for _, text in new_rows:
                fh.write(text.strip() + "\n")
        return len(new_rows)


def _existing_annotation_ids(local_path: Path) -> set[str]:
    if not local_path.exists():
        return set()
    ids: set[str] = set()
    with local_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A corrupt line is load()'s problem to raise on with a line
                # number -- sync only needs to not double-download over it.
                continue
            annotation_id = row.get("annotation_id")
            if annotation_id:
                ids.add(annotation_id)
    return ids


# --------------------------------------------------------------------------- #
# Backend selection -- env-driven, no credential or bucket name hardcoded
# (CLAUDE.md I7). Mirrors the SMARTLAW_BACKEND=local|gcloud convention
# factory.py already uses for the serving pipeline, kept as its own variable
# so an annotator session's storage choice is independent of that pipeline's.
# --------------------------------------------------------------------------- #
def resolve_backend(
    bucket_factory: Callable[[str], _BucketLike] | None = None,
) -> GCSAnnotationBackend | None:
    """`SMARTLAW_ANNOTATIONS_BACKEND` unset or "local" -> None (today's
    filesystem-only behaviour; google-cloud-storage is never imported on this
    path). "gcs" -> a GCS-backed instance, requiring GCS_BUCKET. Any other
    value raises rather than silently falling back to local -- a
    misconfigured deployment should fail loudly, not quietly drop the
    durability the operator thought they'd turned on (I2's spirit).

    `bucket_factory` lets callers (tests) supply a bucket without this
    function ever importing the real SDK.
    """
    backend = os.environ.get("SMARTLAW_ANNOTATIONS_BACKEND", "local").strip().lower()
    if backend in ("", "local"):
        return None
    if backend != "gcs":
        raise ValueError(
            f"unknown SMARTLAW_ANNOTATIONS_BACKEND={backend!r}; use 'local' or 'gcs'")

    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise ValueError("SMARTLAW_ANNOTATIONS_BACKEND=gcs requires GCS_BUCKET to be set")

    if bucket_factory is None:
        bucket_factory = _real_gcs_bucket

    prefix = os.environ.get("SMARTLAW_ANNOTATIONS_GCS_PREFIX", DEFAULT_PREFIX)
    return GCSAnnotationBackend(bucket_factory(bucket_name), prefix=prefix)


def _real_gcs_bucket(bucket_name: str) -> _BucketLike:
    from google.cloud import storage  # lazy: only needed when gcs mode is live

    client = storage.Client(project=os.environ.get("GCP_PROJECT") or None)
    return client.bucket(bucket_name)


# --------------------------------------------------------------------------- #
# Manual backup/recovery: pull every annotation object for a task down to its
# local mirror on demand, independent of the Streamlit app.
# --------------------------------------------------------------------------- #
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync a task's GCS-backed annotations into its local JSONL mirror.")
    parser.add_argument("--task", required=True, help="gold set name, e.g. oos")
    args = parser.parse_args()

    backend = resolve_backend()
    if backend is None:
        raise SystemExit(
            "SMARTLAW_ANNOTATIONS_BACKEND=gcs (and GCS_BUCKET) must be set to sync from GCS")
    local_path = config.ANNOTATIONS_DIR / f"{args.task}.jsonl"
    pulled = backend.sync_to_local(args.task, local_path)
    print(f"pulled {pulled} new annotation(s) into {local_path}")


if __name__ == "__main__":
    _main()
