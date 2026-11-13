"""Hugging Face Trainer integration for S2P-LoRA.

``switch_ratio`` denotes normalized training progress in ``(0, 1)``.
The trainer uses ``global_step / max_steps`` when available, so its meaning is
identical for epoch- and step-based training.

At the switch, the function-preserving projection converts every compatible
layer from ``(B_s, A_s)`` to ``(B_p, A_p)`` while preserving
``B_p A_p = B_s A_s W0``. Optimizer states are reset after switching. For
non-square weights, ``lora_A`` changes shape and its parameter reference is
replaced in the optimizer automatically.
"""

from __future__ import annotations

from transformers import Trainer


class S2PLoraTrainer(Trainer):
    """HuggingFace Trainer with S2P-LoRA sequential→parallel switching."""

    def __init__(
        self,
        *args,
        switch_ratio: float = 0.5,
        momentum_reset: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.switch_ratio = float(switch_ratio)
        if not 0.0 < self.switch_ratio < 1.0:
            raise ValueError("switch_ratio must be in (0, 1) for sequential-to-parallel training")

        self.momentum_reset = momentum_reset
        self.has_switched = False
        self.s2p_layers: list = []

        self.loss_track: dict[str, float] = {}
        self.switch_events: list[dict[str, object]] = []
        self.mode_initialized = False

    # ── Layer discovery and initial mode ─────────────────────────────────────────

    def _ensure_layers_cached(self) -> None:
        if self.s2p_layers:
            return

        for module in self.model.modules():
            has_switch = callable(getattr(module, "switch_to_parallel", None))
            if has_switch:
                self.s2p_layers.append(module)

        if not self.s2p_layers:
            raise RuntimeError(
                "S2PLoraTrainer found no compatible S2P-LoRA layers. "
                "Wrap target nn.Linear modules with S2PLoraLinear before training."
            )

    def _validate_initial_mode(self) -> None:
        self._ensure_layers_cached()
        if any(getattr(layer, "mode", None) != "sequential" for layer in self.s2p_layers):
            raise RuntimeError(
                "S2PLoraTrainer requires every adapter to start in sequential mode. "
                "Create adapters with inject_s2p_adapter before training."
            )

    def _current_progress(self) -> float:
        max_steps = int(getattr(self.state, "max_steps", 0) or 0)
        if max_steps > 0:
            progress = float(self.state.global_step) / max_steps
        else:
            total_epochs = float(getattr(self.args, "num_train_epochs", 0.0) or 0.0)
            progress = float(self.state.epoch or 0.0) / max(total_epochs, 1.0)
        return min(max(progress, 0.0), 1.0)

    # ── Switch and optimizer lifecycle ───────────────────────────────────────────────

    def _switch_layer(self, layer) -> list:
        """Project one layer to parallel mode and return replaced parameters."""
        return layer.switch_to_parallel()

    def _update_optimizer_for_replaced_params(self, replaced_params: list) -> None:
        """Replace shape-changing ``lora_A`` parameters in optimizer groups."""
        if not replaced_params or self.optimizer is None:
            return

        for old_param, new_param in replaced_params:
            found = False
            for group in self.optimizer.param_groups:
                for index, param in enumerate(group["params"]):
                    if param is old_param:
                        group["params"][index] = new_param
                        found = True
            self.optimizer.state.pop(old_param, None)
            if old_param.requires_grad and not found:
                raise RuntimeError(
                    "A replaced trainable lora_A parameter was not present in "
                    "optimizer.param_groups."
                )

    def _execute_switch(self, model, progress: float) -> None:
        """Switch all cached layers to parallel and reset optimizer state."""
        replaced = []
        for layer in self.s2p_layers:
            replaced.extend(self._switch_layer(layer))

        self._update_optimizer_for_replaced_params(replaced)

        if self.optimizer is not None and self.momentum_reset:
            reset_count = 0
            for name, param in model.named_parameters():
                if (
                    param.requires_grad
                    and ("s2p" in name or "lora" in name)
                    and param in self.optimizer.state
                ):
                    self.optimizer.state[param] = {}
                    reset_count += 1
            print(f">>> [RESET] {reset_count} optimizer states cleared.")

        self.has_switched = True
        self.switch_events.append(
            {"progress": progress, "scope": "all", "layers": len(self.s2p_layers)}
        )
        print(f">>> [SWITCH] Progress {progress:.3f}: ALL layers Sequential → Parallel")

    # ── Training step ─────────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._ensure_layers_cached()

        if not self.mode_initialized:
            self._validate_initial_mode()
            self.mode_initialized = True

        if not self.has_switched:
            progress = self._current_progress()
            if progress >= self.switch_ratio:
                self._execute_switch(model, progress)

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        parallel_fraction = sum(
            getattr(layer, "mode", None) == "parallel" for layer in self.s2p_layers
        ) / len(self.s2p_layers)
        self.loss_track = {
            "task_loss": loss.detach().mean().item(),
            "adapter_mode": float(parallel_fraction),
        }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self.model.training and self.loss_track:
            logs.update(self.loss_track)
        super().log(logs, *args, **kwargs)
