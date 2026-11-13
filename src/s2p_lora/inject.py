"""Utilities for attaching S2P-LoRA adapters to PyTorch models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from torch import nn

from .layer import S2PLoraLinear


def _matches(name: str, patterns: set[str]) -> bool:
    parts = name.split(".")
    return any(
        name == pattern or name.endswith(f".{pattern}") or ("." not in pattern and pattern in parts)
        for pattern in patterns
    )


@dataclass(frozen=True)
class InjectionReport:
    """Summary returned by :func:`inject_s2p_adapter`."""

    adapted_modules: tuple[str, ...]
    modules_to_save: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters

    def to_dict(self) -> dict[str, object]:
        report = asdict(self)
        report["trainable_fraction"] = self.trainable_fraction
        return report


def inject_s2p_adapter(
    model: nn.Module,
    target_modules: Sequence[str],
    *,
    r: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    modules_to_save: Iterable[str] = (),
) -> InjectionReport:
    """Freeze ``model`` and wrap selected ``nn.Linear`` modules with S2P-LoRA.

    Target names may be leaf names such as ``"query_proj"`` or qualified
    suffixes such as ``"attention.self.query_proj"``. Only S2P-LoRA adapter
    parameters and parameters matched by ``modules_to_save`` remain trainable.

    Args:
        model: Model to modify in place.
        target_modules: Linear-module names or qualified suffixes to adapt.
        r: Adapter rank.
        lora_alpha: Adapter scaling numerator.
        lora_dropout: Dropout applied to the adapter input.
        modules_to_save: Additional module names kept trainable, for example
            ``("classifier", "pooler")`` for sequence classification.
    """
    targets = {name.strip() for name in target_modules if name.strip()}
    saved = {name.strip() for name in modules_to_save if name.strip()}
    if not targets:
        raise ValueError("target_modules must contain at least one module name")

    for parameter in model.parameters():
        parameter.requires_grad = False

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and _matches(name, targets)
    ]
    if not candidates:
        available = sorted(
            name for name, module in model.named_modules() if isinstance(module, nn.Linear)
        )
        preview = ", ".join(available[:20])
        raise ValueError(
            f"No nn.Linear modules matched {sorted(targets)}. "
            f"First available linear modules: {preview}"
        )

    adapted_names: list[str] = []
    for name, linear in candidates:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = S2PLoraLinear(
            base_layer=linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        wrapper.train(linear.training)
        setattr(parent, child_name, wrapper)
        adapted_names.append(name)

    saved_names: list[str] = []
    if saved:
        for name, module in model.named_modules():
            if name and _matches(name, saved):
                for parameter in module.parameters():
                    parameter.requires_grad = True
                saved_names.append(name)
        missing = sorted(
            pattern
            for pattern in saved
            if not any(_matches(name, {pattern}) for name in saved_names)
        )
        if missing:
            raise ValueError(f"modules_to_save did not match model modules: {missing}")

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return InjectionReport(
        adapted_modules=tuple(adapted_names),
        modules_to_save=tuple(sorted(set(saved_names))),
        trainable_parameters=trainable,
        total_parameters=total,
    )
