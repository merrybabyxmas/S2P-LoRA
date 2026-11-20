"""Export completed S2P adapters in PEFT's standard LoRA format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from .checkpoint import _read_adapter, _validate_factors

_TASK_TYPES = (
    "CAUSAL_LM",
    "FEATURE_EXTRACTION",
    "QUESTION_ANS",
    "SEQ_2_SEQ_LM",
    "SEQ_CLS",
    "TOKEN_CLS",
)


def export_peft_adapter(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    revision: str | None = None,
    task_type: str | None = None,
    modules_to_save: tuple[str, ...] | None = None,
) -> Path:
    """Write a parallel S2P checkpoint as a standard LoRA adapter in a new directory."""
    payload = _read_adapter(checkpoint)
    metadata = payload.get("metadata", {})
    base_model = base_model or metadata.get("base_model_name_or_path")
    revision = revision or metadata.get("revision")
    task_type = task_type or metadata.get("task_type")
    if not base_model or not isinstance(base_model, str):
        raise ValueError("Specify base_model or save its identifier in checkpoint metadata")
    if task_type is not None and task_type not in _TASK_TYPES:
        raise ValueError(f"Unknown PEFT task type: {task_type}")

    weights = {}
    defaults = None
    rank_pattern = {}
    alpha_pattern = {}
    for name, saved in payload["layers"].items():
        if not name:
            raise ValueError("PEFT export requires adapters attached to named model submodules")
        _validate_factors(name, saved)
        if saved["mode"] != "parallel":
            raise ValueError(f"{name} is still sequential; export requires a parallel checkpoint")
        rank = saved["r"]
        alpha = saved["scaling"] * rank
        if defaults is None:
            defaults = rank, alpha
        pattern = "^" + re.escape(name)
        if rank != defaults[0]:
            rank_pattern[pattern] = rank
        if alpha != defaults[1]:
            alpha_pattern[pattern] = alpha
        for key in ("lora_A", "lora_B"):
            weights[f"base_model.model.{name}.{key}.weight"] = saved[key]

    extras = {**payload.get("extra_trainable", {}), **payload.get("extra_buffers", {})}
    roots = modules_to_save
    if roots is None:
        roots = tuple(payload.get("modules_to_save", ()))
    if extras and not roots:
        raise ValueError("Specify modules_to_save for this checkpoint's extra tensors")
    for root in roots:
        if not root or not any(name.startswith(root + ".") for name in extras):
            raise ValueError(f"No extra tensors saved for modules_to_save entry {root!r}")
        if any(
            name == root or name.startswith(root + ".") or root.startswith(name + ".")
            for name in payload["layers"]
        ):
            raise ValueError(f"modules_to_save overlaps adapted module {root!r}")
    for name, tensor in extras.items():
        if not any(name.startswith(root + ".") for root in roots):
            raise ValueError(f"Extra tensor {name!r} does not belong to a modules_to_save entry")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid tensor for {name!r}")
        weights[f"base_model.model.{name}"] = tensor

    config = {
        "base_model_name_or_path": base_model,
        "revision": revision,
        "task_type": task_type,
        "peft_type": "LORA",
        "inference_mode": True,
        "target_modules": "^(?:" + "|".join(re.escape(name) for name in payload["layers"]) + ")$",
        "r": defaults[0],
        "lora_alpha": defaults[1],
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": list(roots) or None,
    }
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Export directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".peft-export-") as directory:
        staging = Path(directory) / "adapter"
        staging.mkdir()
        (staging / "adapter_config.json").write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        save_file(
            {key: tensor.detach().contiguous().clone() for key, tensor in weights.items()},
            staging / "adapter_model.safetensors",
            metadata={"format": "pt"},
        )
        staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--revision")
    parser.add_argument("--task-type", choices=_TASK_TYPES)
    parser.add_argument("--modules-to-save", nargs="+")
    args = parser.parse_args()
    destination = export_peft_adapter(
        args.checkpoint,
        args.output_dir,
        base_model=args.base_model,
        revision=args.revision,
        task_type=args.task_type,
        modules_to_save=args.modules_to_save,
    )
    print(destination)
