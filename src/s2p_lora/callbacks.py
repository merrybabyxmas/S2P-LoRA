"""Trainer callbacks used by the clean S2P-LoRA entry points."""

from __future__ import annotations

import math
from pathlib import Path

from transformers import TrainerCallback

from .checkpoint import save_adapter
from .layer import S2PLoraLinear
from .run_utils import write_json


class BestAdapterCallback(TrainerCallback):
    """Save a compact adapter whenever the selected validation metric improves."""

    def __init__(
        self,
        output_dir: str | Path,
        metric_name: str,
        *,
        greater_is_better: bool = True,
        require_parallel: bool = True,
        adapter_metadata: dict[str, object] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        self.greater_is_better = greater_is_better
        self.require_parallel = require_parallel
        self.adapter_metadata = dict(adapter_metadata or {})
        self.best_value: float | None = None
        self.best_step: int | None = None

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.greater_is_better:
            return value > self.best_value
        return value < self.best_value

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        metrics = metrics or {}
        if not state.is_world_process_zero or model is None or self.metric_name not in metrics:
            return control
        if self.require_parallel and any(
            module.mode != "parallel"
            for module in model.modules()
            if isinstance(module, S2PLoraLinear)
        ):
            return control
        value = float(metrics[self.metric_name])
        if not math.isfinite(value) or not self._improved(value):
            return control

        step = int(state.global_step)
        metadata = {
            **self.adapter_metadata,
            "selection": "best_validation_checkpoint",
            "metric": self.metric_name,
            "value": value,
            "global_step": step,
        }
        save_adapter(model, self.output_dir / "best_adapter.pt", metadata=metadata)
        write_json(self.output_dir / "best_metric.json", metadata)
        self.best_value = value
        self.best_step = step
        return control

    def summary(self) -> dict[str, object]:
        return {
            "metric": self.metric_name,
            "value": self.best_value,
            "global_step": self.best_step,
        }
