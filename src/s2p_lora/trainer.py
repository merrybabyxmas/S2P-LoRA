"""Training-step scheduling for the standalone S2P-LoRA implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import Trainer, TrainerCallback

from .layer import S2PLoraLinear
from .run_utils import write_json

_TRAINING_STATE = "s2p_training_state.json"


class _SwitchCallback(TrainerCallback):
    def __init__(self, trainer: S2PLoraTrainer) -> None:
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._initialize_schedule(state)

    def on_step_begin(self, args, state, control, **kwargs):
        if not self.trainer.has_switched and state.global_step >= self.trainer.switch_step:
            self.trainer._execute_switch(state.global_step / state.max_steps)


class S2PLoraTrainer(Trainer):
    """Train sequentially for ceil(switch_ratio * max_steps) Trainer steps."""

    def __init__(
        self,
        *args,
        switch_ratio: float = 0.5,
        momentum_reset: bool = True,
        **kwargs,
    ):
        self.switch_ratio = float(switch_ratio)
        if not 0.0 < self.switch_ratio < 1.0:
            raise ValueError("switch_ratio must be in (0, 1) for sequential-to-parallel training")
        super().__init__(*args, **kwargs)
        self.momentum_reset = momentum_reset
        self.has_switched = False
        self.switch_step = 0
        self._restored_schedule: dict | None = None
        self.s2p_layers: list[S2PLoraLinear] = []
        self.switch_events: list[dict[str, object]] = []
        self.add_callback(_SwitchCallback(self))

    def _ensure_layers_cached(self) -> None:
        if not self.s2p_layers:
            self.s2p_layers = [
                module for module in self.model.modules() if isinstance(module, S2PLoraLinear)
            ]
        if not self.s2p_layers:
            raise RuntimeError("S2PLoraTrainer found no compatible S2P-LoRA layers")

    def _initialize_schedule(self, state) -> None:
        if (
            self.args.world_size != 1
            or self.args.n_gpu > 1
            or self.args.torch_compile
            or self.is_deepspeed_enabled
            or self.is_fsdp_enabled
        ):
            raise ValueError("S2PLoraTrainer requires one device without sharding or torch.compile")
        self.s2p_layers.clear()
        self._ensure_layers_cached()
        self.switch_step = math.ceil(self.switch_ratio * state.max_steps)
        if not 0 < self.switch_step < state.max_steps:
            raise ValueError("The schedule must reserve at least one training step for each phase")
        if self._restored_schedule is not None:
            previous = self._restored_schedule
            if (
                previous["max_steps"] != state.max_steps
                or previous["switch_step"] != self.switch_step
                or previous["momentum_reset"] != self.momentum_reset
            ):
                raise ValueError("Resume S2P-LoRA with the original training and switch schedule")
            self.switch_events = list(previous["switch_events"])
            self._restored_schedule = None

        expected = "parallel" if state.global_step > self.switch_step else "sequential"
        if any(layer.mode != expected for layer in self.s2p_layers):
            raise RuntimeError(
                f"Checkpoint adapter phase does not match the schedule at step "
                f"{state.global_step}; expected {expected}. Resume with the original schedule."
            )
        self.has_switched = expected == "parallel"

    def _execute_switch(self, progress: float) -> None:
        # Compute all projections before updating a factor or an optimizer group.
        optimizer = self.optimizer
        locations = {}
        for group in optimizer.param_groups:
            for index, parameter in enumerate(group["params"]):
                locations[parameter] = (group["params"], index)
        for layer in self.s2p_layers:
            for parameter in (layer.lora_A.weight, layer.lora_B.weight):
                if parameter.requires_grad and parameter not in locations:
                    raise RuntimeError("A trainable S2P-LoRA factor is missing from the optimizer")

        factors = [layer._parallel_factors() for layer in self.s2p_layers]
        for layer, (b_parallel, a_parallel) in zip(self.s2p_layers, factors, strict=True):
            for old, new in layer._apply_parallel_factors(b_parallel, a_parallel):
                if old in locations:
                    group, index = locations[old]
                    group[index] = new
                optimizer.state.pop(old, None)
            if self.momentum_reset:
                optimizer.state.pop(layer.lora_A.weight, None)
                optimizer.state.pop(layer.lora_B.weight, None)

        self.has_switched = True
        self.switch_events.append(
            {"step": self.state.global_step, "progress": progress, "layers": len(self.s2p_layers)}
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # A user-provided optimizer may exist before raw model loading resizes A.
        target = self.model if model is None else model
        optimizer = self.optimizer
        before = dict(target.named_parameters()) if optimizer is not None else {}
        schedule_path = Path(resume_from_checkpoint) / _TRAINING_STATE
        self._restored_schedule = None
        if schedule_path.is_file():
            self._restored_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if optimizer is not None:
            replaced = {
                before[name]: parameter
                for name, parameter in target.named_parameters()
                if parameter is not before[name]
            }
            for group in optimizer.param_groups:
                group["params"][:] = [
                    replaced.get(parameter, parameter) for parameter in group["params"]
                ]
            for old in replaced:
                optimizer.state.pop(old, None)

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.switch_step:
            write_json(
                Path(output_dir or self.args.output_dir) / _TRAINING_STATE,
                {
                    "max_steps": self.state.max_steps,
                    "switch_step": self.switch_step,
                    "momentum_reset": self.momentum_reset,
                    "switch_events": self.switch_events,
                },
            )

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.model.training and self.s2p_layers:
            logs.setdefault("adapter_mode", float(self.has_switched))
        super().log(logs, *args, **kwargs)
