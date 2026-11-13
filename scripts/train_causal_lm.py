#!/usr/bin/env python3
"""Instruction-tune a causal language model with S2P-LoRA only."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from s2p_lora import BestAdapterCallback, S2PLoraTrainer, inject_s2p_adapter, save_adapter
from s2p_lora.run_utils import (
    make_run_dir,
    parse_args_with_config,
    runtime_versions,
    set_seed,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML file providing argument defaults")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", default="fxmeng/pissa-dataset")
    parser.add_argument("--data-file", default="")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--dataset-data-dir", default="metamath")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--task-name", default="metamath")
    parser.add_argument("--prompt-field", default="instruction")
    parser.add_argument("--response-field", default="output")
    parser.add_argument("--text-field", default="")
    parser.add_argument(
        "--prompt-template", default="### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--overwrite-output", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=128.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--switch-ratio", type=float, default=0.5)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--no-momentum-reset", action="store_true")
    return parser


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = parse_args_with_config(build_parser())
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    set_seed(args.seed)
    run_dir = make_run_dir(
        args.output_root,
        family="nlg",
        task=args.task_name,
        model=args.model,
        rank=args.rank,
        switch_ratio=args.switch_ratio,
        seed=args.seed,
        overwrite=args.overwrite_output,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.data_file:
        load_kwargs = {
            "path": "json",
            "data_files": args.data_file,
            "split": args.train_split,
        }
    else:
        load_kwargs = {"path": args.dataset, "split": args.train_split}
        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        if args.dataset_data_dir:
            load_kwargs["data_dir"] = args.dataset_data_dir
    raw = load_dataset(**load_kwargs)
    if args.max_samples and len(raw) > args.max_samples:
        raw = raw.select(range(args.max_samples))
    split = raw.train_test_split(test_size=args.validation_ratio, seed=args.seed)

    def render(example):
        if args.text_field:
            text = str(example[args.text_field])
        else:
            text = args.prompt_template.format(
                prompt=example[args.prompt_field],
                response=example[args.response_field],
            )
        return {"text": text + (tokenizer.eos_token or "")}

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        )

    rendered = split.map(render)
    encoded = rendered.map(
        tokenize,
        batched=True,
        remove_columns=rendered["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_dtype(args.precision),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    report = inject_s2p_adapter(
        model,
        args.target_modules.split(","),
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    best_callback = BestAdapterCallback(run_dir, "loss", greater_is_better=False)
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_cuda and args.precision == "bfloat16",
        fp16=use_cuda and args.precision == "float16",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = S2PLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[best_callback],
        switch_ratio=args.switch_ratio,
        momentum_reset=not args.no_momentum_reset,
    )

    manifest = {
        "arguments": vars(args),
        "adapter": report.to_dict(),
        "runtime": runtime_versions(),
        "train_examples": len(encoded["train"]),
        "eval_examples": len(encoded["test"]),
    }
    write_json(run_dir / "config.json", manifest)
    train_result = trainer.train()
    final_eval = trainer.evaluate()
    save_adapter(
        model,
        run_dir / "final_adapter.pt",
        metadata={"selection": "final_training_step", "global_step": trainer.state.global_step},
    )
    write_json(
        run_dir / "metrics.json",
        {
            "best_validation": best_callback.summary(),
            "final_validation": final_eval,
            "train": train_result.metrics,
            "switch_events": trainer.switch_events,
        },
    )
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
