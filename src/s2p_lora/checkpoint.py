"""Compact S2P-LoRA adapter checkpoint and merge helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from torch import nn
from transformers.utils.import_utils import check_torch_load_is_safe

from .layer import S2PLoraLinear

FORMAT_VERSION = 1


def _adapter_layers(model: nn.Module) -> dict[str, S2PLoraLinear]:
    return {
        name: module for name, module in model.named_modules() if isinstance(module, S2PLoraLinear)
    }


def save_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save only S2P-LoRA factors and lightweight metadata."""
    layers = _adapter_layers(model)
    if not layers:
        raise RuntimeError("No S2PLoraLinear layers were found")

    details = dict(metadata or {})
    config = getattr(model, "config", None)
    base_model = getattr(config, "_name_or_path", None)
    revision = getattr(config, "_commit_hash", None)
    if base_model:
        details.setdefault("base_model_name_or_path", base_model)
    if revision:
        details.setdefault("revision", revision)
    json.dumps(details, allow_nan=False)

    payload_layers = {}
    for name, layer in layers.items():
        payload_layers[name] = {
            "mode": layer.mode,
            "r": layer.r,
            "d_in": layer.d_in,
            "d_out": layer.d_out,
            "scaling": layer.scaling,
            "dropout": layer.dropout.p if isinstance(layer.dropout, nn.Dropout) else 0.0,
            "lora_A": layer.lora_A.weight.detach().cpu(),
            "lora_B": layer.lora_B.weight.detach().cpu(),
        }

    factor_ids = {
        id(parameter)
        for layer in layers.values()
        for parameter in (layer.lora_A.weight, layer.lora_B.weight)
    }
    extra_trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in factor_ids
    }

    saved_modules = tuple(getattr(model, "_s2p_modules_to_save", ()))
    extra_buffers = {
        f"{name}.{key}": buffer.detach().cpu()
        for name in saved_modules
        for key, buffer in model.get_submodule(name).state_dict().items()
        if f"{name}.{key}" not in extra_trainable
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, prefix=".adapter-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        payload = {
            "format": "s2p_lora_adapter",
            "format_version": FORMAT_VERSION,
            "metadata": details,
            "layers": payload_layers,
            "extra_trainable": extra_trainable,
            "extra_buffers": extra_buffers,
            "modules_to_save": saved_modules,
        }
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _read_adapter(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    check_torch_load_is_safe()
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "s2p_lora_adapter":
        raise ValueError("Not an S2P-LoRA adapter checkpoint")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported adapter format version: {payload.get('format_version')}")
    if not isinstance(payload.get("layers"), dict) or not payload["layers"]:
        raise ValueError("The S2P-LoRA checkpoint contains no adapter layers")
    return payload


def _validate_factors(name: str, saved: dict[str, Any]) -> None:
    rank, width_in, width_out = saved["r"], saved["d_in"], saved["d_out"]
    mode = saved["mode"]
    if mode not in ("parallel", "sequential") or not 0 < rank <= min(width_in, width_out):
        raise ValueError(f"Invalid S2P phase or rank in checkpoint for {name}")
    width = width_in if mode == "parallel" else width_out
    for key, shape in (("lora_A", (rank, width)), ("lora_B", (width_out, rank))):
        factor = saved[key]
        if not isinstance(factor, torch.Tensor) or factor.shape != shape:
            raise ValueError(f"Expected {key} with shape {shape} for {name}")
        if not factor.is_floating_point() or not torch.isfinite(factor).all():
            raise ValueError(f"Invalid values in {key} for {name}")
    if not math.isfinite(saved["scaling"]):
        raise ValueError(f"Invalid scaling for {name}")


@torch.no_grad()
def load_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load an adapter into a model prepared with ``inject_s2p_adapter``."""
    payload = _read_adapter(path, map_location)

    model_layers = _adapter_layers(model)
    saved_layers = payload["layers"]
    if strict and set(model_layers) != set(saved_layers):
        missing = sorted(set(saved_layers) - set(model_layers))
        extra = sorted(set(model_layers) - set(saved_layers))
        raise RuntimeError(f"Adapter layer mismatch; missing={missing}, extra={extra}")

    factors = []
    for name, saved in saved_layers.items():
        if name not in model_layers:
            continue
        layer = model_layers[name]
        expected = (saved["d_in"], saved["d_out"], saved["r"])
        actual = (layer.d_in, layer.d_out, layer.r)
        if expected != actual:
            raise RuntimeError(f"Shape metadata mismatch for {name}: {expected} != {actual}")
        _validate_factors(name, saved)
        if strict and not math.isclose(layer.scaling, saved["scaling"]):
            raise RuntimeError(
                f"Scaling mismatch for {name}: {saved['scaling']} != {layer.scaling}"
            )
        dropout = layer.dropout.p if isinstance(layer.dropout, nn.Dropout) else 0.0
        if "dropout" in saved and strict and not math.isclose(dropout, saved["dropout"]):
            raise RuntimeError(f"Dropout mismatch for {name}: {saved['dropout']} != {dropout}")
        factors.append((layer, saved))
    if not factors:
        raise RuntimeError("No adapter layers were loaded")

    extra = {**payload.get("extra_trainable", {}), **payload.get("extra_buffers", {})}
    destinations = model.state_dict(keep_vars=True)
    copies = []
    for name, value in extra.items():
        if name not in destinations:
            if strict:
                raise RuntimeError(f"Missing saved tensor in model: {name}")
            continue
        tensor = destinations[name]
        if tensor.shape != value.shape:
            raise RuntimeError(
                f"Saved tensor shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(tensor.shape)}"
            )
        copies.append((tensor, value))

    for layer, saved in factors:
        layer._set_mode(saved["mode"])
        layer.scaling = saved["scaling"]
        layer.lora_A.weight.copy_(saved["lora_A"])
        layer.lora_B.weight.copy_(saved["lora_B"])
    for target, value in copies:
        target.copy_(value)
    return dict(payload.get("metadata", {}))


@torch.no_grad()
def merge_adapters(model: nn.Module, *, safe_merge: bool = True) -> tuple[str, ...]:
    """Replace parallel S2P-LoRA wrappers with merged ``nn.Linear`` layers."""
    layers = sorted(
        _adapter_layers(model).items(), key=lambda item: item[0].count("."), reverse=True
    )
    replacements = []
    for name, layer in layers:
        if not name:
            raise ValueError("Call layer.merge() to merge a standalone S2PLoraLinear")
        if layer.mode != "parallel":
            raise RuntimeError(f"Layer {name} is still sequential; switch before merging")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, layer.merge(safe_merge=safe_merge)))
    for parent, child_name, merged in replacements:
        setattr(parent, child_name, merged)
    return tuple(name for name, _ in reversed(layers))
