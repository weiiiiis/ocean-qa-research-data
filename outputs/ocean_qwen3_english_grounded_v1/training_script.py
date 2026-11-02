"""Explicit, versioned QLoRA training for source-grounded English examples.

This script never moves or overwrites the existing application adapter.
Training and checkpoint choice are separate from the locked public test set.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_data(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if not row.get("id") or row["id"] in seen:
            raise ValueError("Every training example needs a unique ID")
        seen.add(row["id"])
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["system", "user", "assistant"] or any(not m.get("content", "").strip() for m in messages):
            raise ValueError(f"{row['id']}: expected complete system/user/assistant messages")
        if not row.get("source_group") or not row.get("provenance"):
            raise ValueError(f"{row['id']}: source provenance is required")
        provenance = json.dumps(row["provenance"], ensure_ascii=False).casefold()
        if any(x in provenance for x in ["paulopirozelli/pira", "pira-doc-", "oceansci-", "pira-test-"]):
            raise ValueError("Public evaluation QA or their source material cannot enter this training run")
    if not rows:
        raise ValueError("Empty training dataset")
    return rows


def encode(tokenizer, row: dict, max_length: int) -> dict:
    import torch
    prefix = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    complete = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError("Chat-template prefix mismatch; refusing to train misaligned labels")
    if len(complete) > max_length:
        raise ValueError(f"{row['id']}: {len(complete)} tokens exceed {max_length}; no answer truncation is allowed")
    if len(prefix) >= len(complete):
        raise ValueError("No assistant target tokens")
    return {"input_ids": torch.tensor(complete), "attention_mask": torch.ones(len(complete), dtype=torch.long),
            "labels": torch.tensor([-100] * len(prefix) + complete[len(prefix):])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="Optional family-separated validation examples; never used for gradient updates")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT / "models/Qwen3-4B-Instruct")
    parser.add_argument("--start-adapter", type=Path, help="Omit to initialize a new LoRA on the clean base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-data", action="store_true", help="Only count full-example tokens; no GPU or training")
    args = parser.parse_args()
    if args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("Epochs and accumulation must be positive")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}; select a new version directory")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_data(args.train)
    validation_rows = load_data(args.validation) if args.validation else []
    if validation_rows:
        for key in ("id", "family_id", "source_group"):
            train_values = {r[key] for r in rows if r.get(key)}
            validation_values = {r[key] for r in validation_rows if r.get(key)}
            if train_values & validation_values:
                raise ValueError(f"Training/validation overlap in {key}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode(tokenizer, row, args.max_length) for row in rows]
    validation_encoded = [encode(tokenizer, row, args.max_length) for row in validation_rows]
    lengths = [len(row["input_ids"]) for row in encoded]
    summary = {"examples": len(rows), "source_groups": len({r['source_group'] for r in rows}), "min_tokens": min(lengths),
        "median_tokens": float(np.median(lengths)), "p95_tokens": float(np.percentile(lengths, 95)), "max_tokens": max(lengths),
        "example_types": {kind: sum(r.get('example_type') == kind for r in rows) for kind in sorted({r.get('example_type', 'unspecified') for r in rows})},
        "assistant_only_loss": True, "truncated_examples": 0,
        "validation_examples": len(validation_rows), "validation_used_for_updates": False}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.inspect_data:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import bitsandbytes as bnb
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    args.out.mkdir(parents=True, exist_ok=False)
    base_files = [p for p in sorted(args.base.iterdir()) if p.is_file() and p.suffix in {".safetensors", ".json", ".model"}]
    configuration = {"base_model": str(args.base.resolve()), "base_model_files_sha256": {p.name: sha256(p) for p in base_files},
        "start_adapter": str(args.start_adapter.resolve()) if args.start_adapter else None,
        "train_file": str(args.train.resolve()), "train_sha256": sha256(args.train), "script_sha256": sha256(Path(__file__)),
        "validation_file": str(args.validation.resolve()) if args.validation else None,
        "validation_sha256": sha256(args.validation) if args.validation else None,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "max_seq_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation, "seed": args.seed, "data_summary": summary,
        "quantization": "4-bit NF4, double quantization, FP16 compute", "fresh_lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "checkpoint_selection": "Save every epoch. Select on document-separated development data, never on the locked test. No epoch is automatically declared best."}
    configuration["packages"] = {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "bitsandbytes", "numpy"]}
    configuration["gpu"] = torch.cuda.get_device_name(0)
    if args.start_adapter:
        configuration["starting_adapter_files_sha256"] = {p.name: sha256(p) for p in args.start_adapter.iterdir() if p.is_file() and p.suffix in {".json", ".safetensors"}}
    save_json(args.out / "training_config.json", configuration)
    (args.out / "training_script.py").write_bytes(Path(__file__).read_bytes())
    (args.out / "train_data.jsonl").write_bytes(args.train.read_bytes())
    if args.validation:
        (args.out / "validation_data.jsonl").write_bytes(args.validation.read_bytes())
    model = AutoModelForCausalLM.from_pretrained(str(args.base), local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, str(args.start_adapter), is_trainable=True) if args.start_adapter else get_peft_model(model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def collate(batch):
        return {key: pad_sequence([r[key] for r in batch], batch_first=True,
                padding_value=tokenizer.pad_token_id if key == "input_ids" else (-100 if key == "labels" else 0))
                for key in ["input_ids", "attention_mask", "labels"]}

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(encoded, batch_size=1, shuffle=True, generator=generator, num_workers=0, collate_fn=collate)
    optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    total_steps = math.ceil(len(loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.05)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    global_step = 0
    skipped_updates = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        started = time.perf_counter()
        for index, batch in enumerate(loader):
            # Correctly scale the final partial accumulation group.
            group_start = (index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                raw_loss = model(**batch).loss
                loss = raw_loss / group_size
            if not torch.isfinite(raw_loss):
                raise FloatingPointError("Non-finite loss; do not publish this checkpoint")
            scaler.scale(loss).backward()
            loss_sum += float(raw_loss.detach())
            if (index + 1) % args.gradient_accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
                if (index + 1) % (5 * args.gradient_accumulation) == 0 or index + 1 == len(loader):
                    print(f"epoch={epoch}/{args.epochs} sample={index+1}/{len(loader)} updates={global_step}/{total_steps} skipped={skipped_updates} train_loss={loss_sum/(index+1):.4f}", flush=True)
        checkpoint = args.out / f"epoch_{epoch}"
        checkpoint.mkdir(exist_ok=False)
        model.save_pretrained(str(checkpoint))
        tokenizer.save_pretrained(str(checkpoint))
        # Read-only validation after each epoch, weighted by assistant target tokens.
        validation_loss = None
        validation_target_tokens = 0
        if validation_encoded:
            model.eval()
            total_validation_loss = 0.0
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for example in validation_encoded:
                    validation_batch = {key: value.unsqueeze(0).to("cuda") for key, value in example.items()}
                    target_tokens = int((validation_batch["labels"][:, 1:] != -100).sum())
                    value = model(**validation_batch).loss
                    if not torch.isfinite(value):
                        raise FloatingPointError("Non-finite validation loss")
                    total_validation_loss += float(value) * target_tokens
                    validation_target_tokens += target_tokens
            validation_loss = total_validation_loss / validation_target_tokens
        row = {"epoch": epoch, "train_loss": loss_sum / len(loader), "global_step": global_step,
            "skipped_optimizer_updates_cumulative": skipped_updates,
            "validation_assistant_token_loss": validation_loss, "validation_target_tokens": validation_target_tokens,
            "seconds": time.perf_counter() - started, "adapter_sha256": sha256(checkpoint / "adapter_model.safetensors")}
        history.append(row)
        save_json(args.out / "training_metrics.json", history)
        print(f"Saved {checkpoint}. Development evaluation is still required.", flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
