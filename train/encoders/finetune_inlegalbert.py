"""Fine-tune law-ai/InLegalBERT for M8a encoder tasks (GCP L4, all seeds -- CLAUDE.md #2).

Adapted from v1's scripts/finetune_inlegalbert.py -- same seq/token-classification
core, now with real --device/bf16 handling and checkpoint sync instead of a hardcoded
fp16=True. Still one combined script gated by --task rather than the five separate
files CLAUDE.md #3 eventually wants (doc_cls.py, ner.py, clause_qa.py, risk.py,
reranker.py) -- splitting those out is real M8a work with its own per-task data
requirements, not something to fabricate as part of importing v1's code.

--task seq    -> document classification
--task token  -> token classification (NER / clause-boundary BIO)

Run: python -m train.encoders.finetune_inlegalbert data.jsonl --task seq --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from train.common.device import resolve_device, supports_bf16
from train.common.seeding import set_seed

MODEL_ID = "law-ai/InLegalBERT"


def _load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    return {"accuracy": float((preds[mask] == labels[mask]).mean())}


def main() -> None:
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification,
                              AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    from train.common.checkpointing import SyncCheckpointCallback

    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--task", choices=["seq", "token"], default="seq")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET"))
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "local-run"))
    ap.add_argument("--hub-org", default=os.environ.get("HF_HUB_ORG", "smartlawai"))
    ap.add_argument("--out", default="./ckpt-inlegalbert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)  # noqa: F841 -- Trainer places the model itself

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = _load_jsonl(args.data)
    coll = None

    if args.task == "seq":
        labels = sorted({r["label"] for r in rows})
        ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=args.seed)

        def enc(b):
            out = tok(b["text"], truncation=True, padding="max_length", max_length=512)
            out["labels"] = b["label"]
            return out
        ds = ds.map(enc, batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(labels))
    else:
        n_tags = max(max(r["tags"]) for r in rows) + 1
        ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=args.seed)

        def enc(b):
            out = tok(b["tokens"], truncation=True, is_split_into_words=True,
                      max_length=512)
            labs = []
            for i, tags in enumerate(b["tags"]):
                word_ids = out.word_ids(batch_index=i)
                labs.append([-100 if w is None else tags[w] for w in word_ids])
            out["labels"] = labs
            return out
        ds = ds.map(enc, batched=True)
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=n_tags)
        coll = DataCollatorForTokenClassification(tok)

    hub_model_id = f"{args.hub_org}/inlegalbert-{args.task}-seed{args.seed}"
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.bs,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, logging_steps=50, bf16=supports_bf16(),
        fp16=not supports_bf16(), report_to="none", seed=args.seed)

    callbacks = []
    if args.gcs_bucket:
        callbacks.append(SyncCheckpointCallback(
            run_id=args.run_id, hub_model_id=hub_model_id,
            gcs_bucket=args.gcs_bucket, arm=args.task, seed=args.seed))

    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["test"], compute_metrics=_metrics,
                      data_collator=coll, tokenizer=tok, callbacks=callbacks)
    trainer.train()
    trainer.save_model(args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
