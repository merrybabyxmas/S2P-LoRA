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
    if isinstance(target_modules, str) or isinstance(modules_to_save, str):
        raise TypeError("target_modules and modules_to_save must be sequences of module names")
    targets = {name.strip() for name in target_modules if name.strip()}
    saved = {name.strip() for name in modules_to_save if name.strip()}
    if not targets:
        raise ValueError("target_modules must contain at least one module name")
    if r <= 0 or not 0.0 <= lora_dropout < 1.0:
        raise ValueError("r must be positive and lora_dropout must be in [0, 1)")

    named_modules = dict(model.named_modules())
    if any(isinstance(module, S2PLoraLinear) for module in named_modules.values()):
        raise ValueError("The model already contains S2P-LoRA adapters")
    candidates = [
        (name, module)
        for name, module in named_modules.items()
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
    missing_targets = sorted(
        target for target in targets if not any(_matches(name, {target}) for name, _ in candidates)
    )
    if missing_targets:
        raise ValueError(f"No nn.Linear modules matched target entries {missing_targets}")

    for name, linear in candidates:
        if r > min(linear.in_features, linear.out_features):
            raise ValueError(f"r={r} exceeds the smaller dimension of {name}")
        if not linear.weight.is_floating_point() or linear.weight.device.type == "meta":
            raise ValueError(f"S2P-LoRA requires a materialized floating-point weight for {name}")

    saved_names = [name for name in named_modules if name and _matches(name, saved)]
    missing = sorted(
        pattern for pattern in saved if not any(_matches(name, {pattern}) for name in saved_names)
    )
    if missing:
        raise ValueError(f"modules_to_save did not match model modules: {missing}")
    saved_roots: list[str] = []
    for name in sorted(saved_names):
        if not any(name.startswith(root + ".") for root in saved_roots):
            saved_roots.append(name)
    saved_parameters = {
        parameter for name in saved_roots for parameter in named_modules[name].parameters()
    }
    for name, linear in candidates:
        if any(
            name == root or name.startswith(root + ".") or root.startswith(name + ".")
            for root in saved_roots
        ) or not saved_parameters.isdisjoint(linear.parameters()):
            raise ValueError(f"modules_to_save overlaps adapted module {name}")

    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    wrappers = []
    try:
        for name, linear in candidates:
            wrapper = S2PLoraLinear(linear, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            wrapper.train(linear.training)
            wrappers.append((name, wrapper))
    except Exception:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)
        raise
    for parameter in parameters:
        parameter.requires_grad_(parameter in saved_parameters)
    for name, wrapper in wrappers:
        parent_name, _, child_name = name.rpartition(".")
        setattr(named_modules[parent_name], child_name, wrapper)
    model._s2p_modules_to_save = tuple(saved_roots)

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return InjectionReport(
        adapted_modules=tuple(name for name, _ in candidates),
        modules_to_save=tuple(sorted(set(saved_names))),
        trainable_parameters=trainable,
        total_parameters=total,
    )
