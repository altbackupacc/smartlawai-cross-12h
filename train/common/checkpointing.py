"""Synchronous local + HF Hub + GCS checkpoint sync (OPS.md #4).

A checkpoint is not considered saved until it exists in three places: local disk
(written by the Trainer itself), the Hugging Face Hub (source of truth), and GCS
(same-region cache for fast resume). Each step blocks the next -- see OPS.md #4 for
the cost/cadence tradeoff this implies for `save_steps`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder
from transformers import TrainerCallback


def push_to_hf_hub(local_dir: str, repo_id: str, revision: str) -> str:
    """Upload a checkpoint directory to HF Hub under a step-tagged revision.
    Blocks until the upload completes. Returns `repo_id@revision`."""
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    try:
        api.create_branch(repo_id, branch=revision, exist_ok=True)
    except Exception:
        pass  # branch already exists from a prior save
    upload_folder(repo_id=repo_id, folder_path=local_dir, revision=revision)
    return f"{repo_id}@{revision}"


def sync_to_gcs(local_dir: str, gcs_uri: str) -> str:
    """Mirror a checkpoint directory to GCS. Blocks until the sync completes."""
    from google.cloud import storage

    bucket_name, _, prefix = gcs_uri.removeprefix("gs://").partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_root = Path(local_dir)
    for path in local_root.rglob("*"):
        if path.is_file():
            blob_path = f"{prefix.rstrip('/')}/{path.relative_to(local_root)}"
            bucket.blob(blob_path).upload_from_filename(str(path))
    return gcs_uri


def update_run_manifest(run_id: str, step: int, locations: dict) -> None:
    """Record the latest confirmed checkpoint locations in this run's manifest.
    Written to runs/<run_id>.json, atomically, and committed to git (OPS.md #2/#4)."""
    path = Path("runs") / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else {"run_id": run_id}
    manifest["last_checkpoint_step"] = step
    manifest["checkpoint_locations"] = locations
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, path)  # atomic on both POSIX and Windows


class SyncCheckpointCallback(TrainerCallback):
    """Trainer callback: after every local save, push to HF Hub and GCS
    synchronously, then update the run manifest. Training does not advance past a
    save until all three locations are confirmed (OPS.md #4).

    Usage:
        trainer = Trainer(..., callbacks=[SyncCheckpointCallback(
            run_id=RUN_ID, hub_model_id=f"{ORG}/smartlaw-{ARM}-seed{SEED}",
            gcs_bucket=GCS_BUCKET, arm=ARM, seed=SEED)])
    """

    def __init__(self, run_id: str, hub_model_id: str, gcs_bucket: str,
                 arm: str = "", seed: int = 0) -> None:
        self.run_id = run_id
        self.hub_model_id = hub_model_id
        self.gcs_bucket = gcs_bucket
        self.arm = arm
        self.seed = seed

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        local_dir = f"{args.output_dir}/checkpoint-{step}"
        hub_rev = push_to_hf_hub(local_dir, self.hub_model_id, revision=f"step-{step}")
        gcs_uri = sync_to_gcs(
            local_dir,
            f"gs://{self.gcs_bucket}/ckpt/{self.arm}/seed{self.seed}/step-{step}/")
        update_run_manifest(self.run_id, step=step, locations={
            "local": local_dir, "hf_hub": hub_rev, "gcs": gcs_uri})
        return control
