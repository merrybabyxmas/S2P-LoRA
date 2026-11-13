"""Compact S2P-LoRA adapter checkpoint and merge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

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

    payload_layers = {}
    for name, layer in layers.items():
        payload_layers[name] = {
            "mode": layer.mode,
            "r": layer.r,
            "d_in": layer.d_in,
            "d_out": layer.d_out,
            "scaling": layer.scaling,
            "lora_A": layer.lora_A.weight.detach().cpu(),
            "lora_B": layer.lora_B.weight.detach().cpu(),
        }

    extra_trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_A." not in name and ".lora_B." not in name
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "s2p_lora_adapter",
            "format_version": FORMAT_VERSION,
            "metadata": metadata or {},
            "layers": payload_layers,
            "extra_trainable": extra_trainable,
        },
        destination,
    )
    return destination


@torch.no_grad()
def load_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load an adapter into a model prepared with ``inject_s2p_adapter``."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") != "s2p_lora_adapter":
        raise ValueError("Not an S2P-LoRA adapter checkpoint")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported adapter format version: {payload.get('format_version')}")

    model_layers = _adapter_layers(model)
    saved_layers = payload["layers"]
    if strict and set(model_layers) != set(saved_layers):
        missing = sorted(set(saved_layers) - set(model_layers))
        extra = sorted(set(model_layers) - set(saved_layers))
        raise RuntimeError(f"Adapter layer mismatch; missing={missing}, extra={extra}")

    loaded = 0
    for name, saved in saved_layers.items():
        if name not in model_layers:
            continue
        layer = model_layers[name]
        expected = (saved["d_in"], saved["d_out"], saved["r"])
        actual = (layer.d_in, layer.d_out, layer.r)
        if expected != actual:
            raise RuntimeError(f"Shape metadata mismatch for {name}: {expected} != {actual}")

        if saved["mode"] == "parallel" and layer.mode != "parallel":
            layer.switch_to_parallel()
        elif saved["mode"] == "sequential" and layer.mode != "sequential":
            raise RuntimeError(f"Cannot load sequential factors into already-parallel layer {name}")

        layer.lora_A.weight.copy_(
            saved["lora_A"].to(device=layer.lora_A.weight.device, dtype=layer.lora_A.weight.dtype)
        )
        layer.lora_B.weight.copy_(
            saved["lora_B"].to(device=layer.lora_B.weight.device, dtype=layer.lora_B.weight.dtype)
        )
        loaded += 1

    named_parameters = dict(model.named_parameters())
    for name, value in payload.get("extra_trainable", {}).items():
        if name not in named_parameters:
            if strict:
                raise RuntimeError(f"Missing trainable parameter in model: {name}")
            continue
        parameter = named_parameters[name]
        if parameter.shape != value.shape:
            raise RuntimeError(
                f"Trainable parameter shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(parameter.shape)}"
            )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    if loaded == 0:
        raise RuntimeError("No adapter layers were loaded")
    return dict(payload.get("metadata", {}))


@torch.no_grad()
def merge_adapters(model: nn.Module) -> tuple[str, ...]:
    """Replace parallel S2P-LoRA wrappers with merged ``nn.Linear`` layers."""
    layers = sorted(
        _adapter_layers(model).items(), key=lambda item: item[0].count("."), reverse=True
    )
    merged_names: list[str] = []
    for name, layer in layers:
        if layer.mode != "parallel":
            raise RuntimeError(f"Layer {name} is still sequential; switch before merging")
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, layer.merge())
        merged_names.append(name)
    return tuple(reversed(merged_names))
