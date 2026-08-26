"""LoRA fine-tune of the arm's base model for legal summarisation (4-bit QLoRA).

Adapted from v1's scripts/finetune_mistral_lora.py -- same core LoRA recipe, but now:
  - picks the base model from --arm (S/C/M/L, per RESEARCH.md #4.1) instead of a
    single hardcoded model
  - uses train.common.device for real --device/bf16 handling instead of hardcoding
    fp16=True (CLAUDE.md #2 -- L4/A100 are Ampere+, bf16-capable; don't leave that on
    the table by assuming fp16 everywhere)
  - accepts --seed/--arm/--device/--gcs-bucket/--run-id to match what
    gcloud/train_l4.sh (Arms S/C/M) and gcloud/train_a100.sh (Arm L) already pass
  - checkpoints sync to local + HF Hub + GCS via SyncCheckpointCallback (OPS.md #4)
    whenever --gcs-bucket is set

Run: python -m train.qlora.train_qlora data.jsonl --arm C --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

from train.common.device import dtype_for_device, resolve_device, supports_bf16
from train.common.seeding import set_seed

_PROMPT = ("<s>[INST] Summarise this Indian legal document in plain language, "
           "using ONLY the information present:\n\n{src} [/INST] {tgt}</s>")

# RESEARCH.md #4.1 -- Arm C is the controlled comparison (same base as SaulLM-7B),
# never cut below 5 seeds regardless of budget pressure.
_BASE_MODELS = {
    "S": "Qwen/Qwen2.5-3B-Instruct",
    "C": "mistralai/Mistral-7B-Instruct-v0.3",
    "M": "Qwen/Qwen2.5-7B-Instruct",
    "L": "Qwen/Qwen2.5-14B-Instruct",
}


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments)
    from trl import SFTTrainer

    from train.common.checkpointing import SyncCheckpointCallback

    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data/qlora/train.jsonl",
                    help='JSONL rows: {"src","tgt"}')
    ap.add_argument("--arm", required=True, choices=list(_BASE_MODELS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gcs-bucket", default=os.environ.get("GCS_BUCKET"))
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", "local-run"))
    ap.add_argument("--hub-org", default=os.environ.get("HF_HUB_ORG", "smartlawai"))
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=500)
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = dtype_for_device()
    base = _BASE_MODELS[args.arm]

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    ds = Dataset.from_list([{"text": _PROMPT.format(src=r["src"][:6000], tgt=r["tgt"])}
                            for r in rows])

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map={"": str(device)})
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    hub_model_id = f"{args.hub_org}/smartlaw-{args.arm}-seed{args.seed}"
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=8,
        learning_rate=2e-4, bf16=supports_bf16(), fp16=not supports_bf16(),
        logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=3, report_to="none", seed=args.seed)

    callbacks = []
    if args.gcs_bucket:
        callbacks.append(SyncCheckpointCallback(
            run_id=args.run_id, hub_model_id=hub_model_id,
            gcs_bucket=args.gcs_bucket, arm=args.arm, seed=args.seed))

    trainer = SFTTrainer(model=model, args=targs, train_dataset=ds,
                         dataset_text_field="text", max_seq_length=2048,
                         tokenizer=tok, callbacks=callbacks)
    trainer.train()
    trainer.save_model(args.out)
    print(f"Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
