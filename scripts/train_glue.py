#!/usr/bin/env python3
"""Fine-tune a GLUE task with S2P-LoRA only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
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

TASKS = {
    "cola": {
        "keys": ("sentence", None),
        "labels": 2,
        "metric": "matthews_correlation",
        "epochs": 15,
    },
    "mnli": {"keys": ("premise", "hypothesis"), "labels": 3, "metric": "accuracy", "epochs": 10},
    "mrpc": {"keys": ("sentence1", "sentence2"), "labels": 2, "metric": "f1", "epochs": 15},
    "qnli": {"keys": ("question", "sentence"), "labels": 2, "metric": "accuracy", "epochs": 10},
    "rte": {"keys": ("sentence1", "sentence2"), "labels": 2, "metric": "accuracy", "epochs": 15},
    "sst2": {"keys": ("sentence", None), "labels": 2, "metric": "accuracy", "epochs": 15},
    "stsb": {"keys": ("sentence1", "sentence2"), "labels": 1, "metric": "pearson", "epochs": 15},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML file providing argument defaults")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--task", choices=sorted(TASKS), default="cola")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--overwrite-output", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=20)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--switch-ratio", type=float, default=0.5)
    parser.add_argument("--target-modules", default="query_proj,key_proj,value_proj")
    parser.add_argument("--modules-to-save", default="classifier,pooler")
    parser.add_argument("--no-momentum-reset", action="store_true")
    return parser


def _limit(dataset, maximum: int):
    if maximum and len(dataset) > maximum:
        return dataset.select(range(maximum))
    return dataset


def main() -> None:
    args = parse_args_with_config(build_parser())
    set_seed(args.seed)
    task = TASKS[args.task]
    epochs = args.epochs if args.epochs is not None else task["epochs"]
    run_dir = make_run_dir(
        args.output_root,
        family="nlu",
        task=args.task,
        model=args.model,
        rank=args.rank,
        switch_ratio=args.switch_ratio,
        seed=args.seed,
        overwrite=args.overwrite_output,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw = load_dataset("glue", args.task)
    first_key, second_key = task["keys"]

    def tokenize(batch):
        values = (
            (batch[first_key],)
            if second_key is None
            else (
                batch[first_key],
                batch[second_key],
            )
        )
        return tokenizer(*values, truncation=True, max_length=args.max_seq_length)

    encoded = raw.map(tokenize, batched=True)
    train_dataset = _limit(encoded["train"], args.max_train_samples)
    eval_split = "validation_matched" if args.task == "mnli" else "validation"
    eval_dataset = _limit(encoded[eval_split], args.max_eval_samples)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=task["labels"],
    )
    report = inject_s2p_adapter(
        model,
        args.target_modules.split(","),
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        modules_to_save=args.modules_to_save.split(","),
    )

    metric = load_metric("glue", args.task)

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        if task["labels"] == 1:
            predictions = predictions.squeeze(-1)
        else:
            predictions = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    best_callback = BestAdapterCallback(run_dir, task["metric"])
    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = S2PLoraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[best_callback],
        switch_ratio=args.switch_ratio,
        momentum_reset=not args.no_momentum_reset,
    )

    manifest = {
        "arguments": {**vars(args), "resolved_epochs": epochs},
        "adapter": report.to_dict(),
        "runtime": runtime_versions(),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
    }
    write_json(run_dir / "config.json", manifest)
    train_result = trainer.train()
    final_eval = trainer.evaluate()
    save_adapter(
        model,
        run_dir / "final_adapter.pt",
        metadata={"selection": "final_training_step", "global_step": trainer.state.global_step},
    )
    metrics = {
        "best_validation": best_callback.summary(),
        "final_validation": final_eval,
        "train": train_result.metrics,
        "switch_events": trainer.switch_events,
    }
    write_json(run_dir / "metrics.json", metrics)
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
