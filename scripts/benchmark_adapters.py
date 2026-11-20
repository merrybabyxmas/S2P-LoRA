#!/usr/bin/env python3
"""Compare LoRA and S2P-LoRA on the same short causal-LM workload."""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from s2p_lora import S2PLoraTrainer, inject_s2p_adapter
from s2p_lora.run_utils import set_seed


class TimedS2PTrainer(S2PLoraTrainer):
    switch_seconds = 0.0

    def _execute_switch(self, progress: float) -> None:
        if self.args.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        super()._execute_switch(progress)
        if self.args.device.type == "cuda":
            torch.cuda.synchronize()
        self.switch_seconds = time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("lora", "s2p"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, default=Path("examples/tiny_instructions.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    model.config.use_cache = False
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.method == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.rank, target_modules=targets
            ),
            autocast_adapter_dtype=False,
        )
        trainer_cls = Trainer
    else:
        inject_s2p_adapter(model, targets, r=args.rank, lora_alpha=args.rank)
        trainer_cls = TimedS2PTrainer

    raw = load_dataset("json", data_files=str(args.data), split="train")
    data = raw.train_test_split(test_size=0.2, seed=args.seed)

    def tokenize(batch):
        text = [
            f"{instruction}\n{output}{tokenizer.eos_token or ''}"
            for instruction, output in zip(batch["instruction"], batch["output"], strict=True)
        ]
        return tokenizer(text, truncation=True, max_length=128, return_token_type_ids=False)

    encoded = data.map(tokenize, batched=True, remove_columns=raw.column_names)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        bf16=args.bf16,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=True,
    )
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if train_args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak = (
        torch.cuda.max_memory_allocated() / 1024**2 if train_args.device.type == "cuda" else None
    )
    metrics = {
        "method": args.method,
        "model": args.model,
        "steps": args.steps,
        "rank": args.rank,
        "seed": args.seed,
        "bf16": args.bf16,
        "train_seconds": result.metrics["train_runtime"],
        "eval_loss": trainer.evaluate()["eval_loss"],
        "switch_seconds": getattr(trainer, "switch_seconds", None),
        "peak_mib": peak,
        "trainable_parameters_before": before,
        "trainable_parameters_after": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
